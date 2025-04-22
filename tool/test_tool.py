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
data = pd.read_excel("C:\\Users\\Stitch\\Desktop\\酒精预测2.xlsx", header=None)

# 2. 提取特征和目标变量
X = data.values  # 所有数据作为特征（没有表头，因此直接使用 .values）

# 3. 为每7行分配浓度标签
n_samples = X.shape[0]
# 使用linspace生成线性间隔的浓度值 (0.0385, 0.5)，长度为25
y = np.repeat(np.linspace(0.0385, 0.5, 25), 7)[:n_samples]  # 3.85%, 7.41%, ..., 50.00% 对应每7行

# 4. 选择光谱数据（即所有列，去掉标签列）
X = X[:, 1:]  # 从第二列开始取光谱数据

# 5. 填充缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 6. MSC 标准化
def msc_correction(sdata):
    """
    对光谱数据进行MSC（主成分标准化）。
    :param sdata: 原始光谱数据，行是样本，列是波长（特征）
    :return: MSC 标准化后的数据
    """
    n = sdata.shape[0]  # 样本数量
    msc_corrected_data = np.zeros_like(sdata)

    for i in range(n):
        y = sdata[i, :]
        mean_y = np.mean(y)
        std_y = np.std(y)
        msc_corrected_data[i, :] = (y - mean_y) / std_y  # 标准化处理

    return msc_corrected_data

# 7. 对数据进行 MSC 标准化处理
X_msc = msc_correction(X)  # MSC 标准化

# 8. PCA 降维，保留5个主成分
pca = PCA(n_components=5)  # 设置保留前5个主成分
X_pca = pca.fit_transform(X_msc)  # 将处理后的数据进行PCA降维

# 输出解释的方差比例
print("每个主成分解释的方差比例：", pca.explained_variance_ratio_)
print("累积解释的方差比例：", np.cumsum(pca.explained_variance_ratio_))

# 9. 对数据进行标准化处理
scaler = StandardScaler()
X_pca = scaler.fit_transform(X_pca)

# 10. 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 11. 设置随机森林回归模型的参数搜索空间
param_grid = {
    'n_estimators': [100, 200, 300],  # 树的数量
    'max_depth': [None, 10, 20, 30],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 分裂所需的最小样本数
    'min_samples_leaf': [1, 2, 4]  # 叶子节点的最小样本数
}

# 12. 使用 GridSearchCV 进行超参数搜索
rf_model = RandomForestRegressor(random_state=42)  # 使用随机森林回归模型
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

# 输出训练集和测试集的评价指标
print(f"训练集均方根误差（RMSE）：{train_rmse:.4f}")
print(f"测试集均方根误差（RMSE）：{test_rmse:.4f}")
print(f"训练集决定系数（R²）：{train_r2:.4f}")
print(f"测试集决定系数（R²）：{test_r2:.4f}")

# 17. 绘制实际值与预测值的散点图
plt.figure(figsize=(8, 6))

# 使用更好的蓝色
plt.scatter(y_test, y_test_pred, color='#1f77b4', label='预测值')

# 绘制理想预测线
plt.plot([0.0385, 0.5], [0.0385, 0.5], color='red', linestyle='--', label='理想预测线')

# 添加坐标轴标签和标题
plt.xlabel("实际掺伪度", fontsize=12)
plt.ylabel("预测掺伪度", fontsize=12)

# 设置图例字体并显示图表
font = FontProperties(fname=r"C:\Windows\Fonts\SimSun.ttc", size=12)  # 设置中文字体
plt.legend(prop=font)
plt.show()
