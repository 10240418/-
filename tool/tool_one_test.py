import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归模型
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties

plt.rcParams['font.family'] = 'Microsoft YaHei'  # 设置支持中文的字体

# 1. 读取数据，没有表头
data = pd.read_excel("C:\\Users\\19176\\Desktop\\酒精预测2.xlsx", header=None)

# 2. 提取特征和目标变量
X = data.values  # 所有数据作为特征（没有表头，因此直接使用 .values）

# 3. 为每7行分配浓度标签
n_samples = X.shape[0]
y = np.repeat(np.linspace(0.0385, 0.5, 25), 7)[:n_samples]  # 3.85%, 7.41%, ..., 50.00%

# 4. 选择光谱数据（即所有列，去掉标签列）
X = X[:, 1:]  # 从第二列开始取光谱数据

# 5. 填充缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 6. MSC 标准化
def msc_correction(sdata):
    n = sdata.shape[0]
    msc_corrected_data = np.zeros_like(sdata)

    for i in range(n):
        y = sdata[i, :]
        mean_y = np.mean(y)
        std_y = np.std(y)
        msc_corrected_data[i, :] = (y - mean_y) / std_y

    return msc_corrected_data

# 7. 对数据进行 MSC 标准化处理
X_msc = msc_correction(X)  # MSC 标准化

# 8. PCA 降维，保留5个主成分
pca = PCA(n_components=5)  # 设置保留前5个主成分
X_pca = pca.fit_transform(X_msc)  # 将处理后的数据进行PCA降维

# 9. 对数据进行标准化处理
scaler = StandardScaler()
X_pca = scaler.fit_transform(X_pca)

# 10. 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 11. 设置随机森林回归模型的参数搜索空间
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 12. 使用 GridSearchCV 进行超参数搜索
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 13. 输出最佳超参数
print(f"最佳超参数: {grid_search.best_params_}")

# 14. 使用最佳参数训练模型
best_rf_model = grid_search.best_estimator_

# 15. 训练集和测试集的预测
y_train_pred = best_rf_model.predict(X_train)
y_test_pred = best_rf_model.predict(X_test)

# 16. 计算训练集和测试集的均方根误差（RMSE）和决定系数（R²）
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"训练集均方根误差（RMSE）：{train_rmse:.4f}")
print(f"测试集均方根误差（RMSE）：{test_rmse:.4f}")
print(f"训练集决定系数（R²）：{train_r2:.4f}")
print(f"测试集决定系数（R²）：{test_r2:.4f}")

# 17. 绘制实际值与预测值的散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='#1f77b4', label='预测值')
plt.plot([0.0385, 0.5], [0.0385, 0.5], color='red', linestyle='--', label='理想预测线')
plt.xlabel("实际掺伪度", fontsize=12)
plt.ylabel("预测掺伪度", fontsize=12)
font = FontProperties(fname=r"C:\Windows\Fonts\SimSun.ttc", size=12)
plt.legend(prop=font)
plt.show()

# -------------------------新增部分-------------------------

# 18. 预测新样品数据的掺伪度
def predict_sample(new_data):
    # 对新的样品数据进行与训练集相同的预处理
    new_data_imputed = imputer.transform(new_data)  # 填充缺失值
    new_data_msc = msc_correction(new_data_imputed)  # MSC标准化
    new_data_pca = pca.transform(new_data_msc)  # PCA降维
    new_data_scaled = scaler.transform(new_data_pca)  # 标准化

    # 使用训练好的模型进行预测
    predicted_value = best_rf_model.predict(new_data_scaled)
    return predicted_value

# -------------------------从文件导入新样品数据-------------------------
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

# 读取数据（确保路径正确）
new_sample = pd.read_excel('C:\\Users\\19176\\Desktop\\样品数据.xlsx', header=None)

# 只选择数值型数据列，忽略任何非数值数据（例如字符串或分类数据）
new_sample_numeric = new_sample.select_dtypes(include=[np.number])

# 如果包含非数值列，进行转换或处理：
# 你可以使用 LabelEncoder 或 OneHotEncoder 来转换非数值特征（如果有的话）
# 例如，如果你有一个列是字符串类别数据：
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# new_sample['column_name'] = le.fit_transform(new_sample['column_name'])

# 填充缺失值，只针对数值型数据
imputer = SimpleImputer(strategy='mean')
new_sample_numeric_imputed = imputer.fit_transform(new_sample_numeric)

# 如果你有非数值列并且需要重新合并它们，可以把数值列和处理过的非数值列合并：
# new_sample_processed = pd.concat([new_sample_numeric_imputed, new_sample_non_numeric], axis=1)

# 将数据转换为二维格式（1行n列）
new_sample_numeric_imputed = new_sample_numeric_imputed.reshape(1, -1)

# 使用模型进行预测
predicted_pseudo_degree = predict_sample(new_sample_numeric_imputed)
print(f"新样品的预测掺伪度为：{predicted_pseudo_degree[0]:.4f}")

