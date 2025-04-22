# 白酒品质检测系统

本系统是一个基于Python的白酒掺伪量检测应用程序，使用CustomTkinter构建现代化图形界面，并使用SQLite数据库存储用户信息和分析历史记录。

## 系统特点

- **现代化界面**：使用CustomTkinter构建的美观、现代的用户界面
- **数据库支持**：使用SQLite数据库存储用户信息和分析历史记录
- **多功能分析**：支持白酒掺伪量分析预测和白酒真伪分类
- **可视化展示**：使用Matplotlib绘制分析结果图表
- **多选项卡界面**：支持分析、历史记录和设置等功能的快速切换
- **主题切换**：支持深色和浅色模式，以及多种颜色主题
- **用户管理**：多用户登录系统，支持用户注册和权限控制

## 工作流程

```
+---------------+
|    启动应用    |
+-------+-------+
        |
        v
+---------------+
|   用户登录     | -------+
+-------+-------+        |
        |                v
        |        +---------------+
        |        |   用户注册     |
        |        +-------+-------+
        |                |
        v                v
+---------------+        |
|   主界面选择   | <------+
+-------+-------+
        |
        +-----------------+------------------+-------------------+
        |                 |                  |                   |
        v                 v                  v                   v
+----------------+ +----------------+ +----------------+ +----------------+
| 掺伪量分析预测 | | 白酒真伪分类   | |  历史记录查看  | |   系统设置     |
+--------+-------+ +--------+-------+ +--------+-------+ +----------------+
        |                 |                  |
        v                 v                  v
+----------------+ +----------------+ +----------------+
|  选择数据文件  | |  选择数据文件  | | 查看分析历史   |
+--------+-------+ +--------+-------+ +--------+-------+
        |                 |                  |
        v                 v                  v
+----------------+ +----------------+ +----------------+
|   数据分析     | |   数据分类     | | 导出分析记录   |
+--------+-------+ +--------+-------+ +----------------+
        |                 |
        v                 v
+----------------+ +----------------+
| 结果可视化展示 | | 结果置信度展示 |
+--------+-------+ +--------+-------+
        |                 |
        v                 v
+----------------+ +----------------+
|  保存分析结果  | |  保存分析结果  |
+----------------+ +----------------+
```

## 系统要求

- Python 3.8+ (建议使用Python 3.9或3.10)
- 以下Python库：
  - customtkinter
  - numpy
  - pandas
  - matplotlib
  - pillow

## 快速启动

1. 确保已安装Python和所需依赖：
   ```
   pip install customtkinter pandas numpy matplotlib pillow
   ```

2. 运行启动脚本：
   ```
   start_baijiu_app.bat
   ```
   或直接启动Python应用：
   ```
   python baijiu_app.py
   ```

3. 脚本会自动：
   - 创建必要的目录结构
   - 初始化SQLite数据库
   - 启动应用程序

## 登录与注册

### 默认用户
- 管理员：用户名 `admin`，密码 `admin123`
- 测试用户：用户名 `user1`，密码 `123456`

### 用户注册
系统支持新用户注册：
1. 在登录界面点击"注册新用户"按钮
2. 填写用户名、密码和可选的电子邮件
3. 用户名不能重复，密码长度不少于6个字符
4. 注册成功后可使用新账户登录系统

## 使用方法

1. **登录系统**：使用默认账户、已创建的账户登录，或注册新账户
2. **掺伪量分析预测**：
   - 选择"掺伪量分析预测"选项卡
   - 点击"浏览"选择CSV或Excel格式的光谱数据文件
   - 点击"开始分析"进行掺伪量检测
   - 系统会显示分析结果和图表
   - 可保存分析结果到文本文件
3. **白酒真伪分类**：
   - 选择"白酒真伪分类"选项卡
   - 点击"浏览"选择CSV或Excel格式的光谱数据文件
   - 点击"开始分类"进行真伪分类
   - 系统会显示分类结果和各类别的置信度
   - 可保存分类结果到文本文件
4. **查看历史记录**：
   - 选择"历史记录"选项卡查看过去的分析记录
   - 点击"刷新历史记录"更新列表
   - 可导出历史记录为TXT或CSV格式
5. **系统设置**：
   - 选择"设置"选项卡调整系统外观
   - 可以切换深色/浅色模式
   - 可以更改颜色主题

## 文件结构

- `baijiu_app.py` - 主应用程序
- `start_baijiu_app.bat` - 启动脚本
- `database/` - 数据库相关文件
  - `db_manager.py` - 数据库管理类
  - `init_db.py` - 数据库初始化脚本
  - `baijiu.db` - SQLite数据库文件
- `logs/` - 应用日志
- `history/` - 分析结果保存目录

## 疑难解答

1. **应用程序无法启动**：
   - 确保已安装所有必要的依赖库
   - 检查Python版本是否为3.8+
   - 查看日志文件 `logs/app.log` 获取详细错误信息

2. **数据库连接失败**：
   - 检查数据库文件是否存在
   - 确保有相应目录的读写权限
   - 尝试手动初始化数据库: `python database/init_db.py`

3. **界面显示问题**：
   - 尝试切换界面主题
   - 如果使用远程桌面，确保支持GUI显示

4. **分析失败**：
   - 确保上传的是有效的光谱数据文件
   - 检查文件格式是否为CSV或Excel
   - 查看日志文件 `logs/app.log` 获取详细错误信息

## 版本兼容性

- 建议使用Python 3.9或3.10版本
- 如果使用Python 3.11+，部分功能可能需要更新依赖库

## 开发者信息

四川农业大学白酒品质检测系统
© 2024 版权所有



@analysis_frame.py 我的理解出现了偏差,修改我的代码 1 关于这个页面,上面应该是两个选择本地文件的按钮,一个是完整的xlsx文件,一个是单个样品xlsx文件  2 @test_tool.py  参考代码@improved_test_tool.py 以及这个代码,要求类似画出一个图,和示例代码一样,以及现有代码一样,点击分析结果以后会和现有代码一样能出现一个预测掺伪量图表的图,这个不需要修改,得到上面的四个参数,不需要原来的平均预测掺伪度什么的东西,只需要得到这四个量  3 然后import matplotlib.pyplot as plt
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

 参考上面的示例代码,用来分析单个样品数据,和当前代码生成的光谱图类似,出现一个按钮显示光谱图和文件保存路径,然后得到一个新样品的预测掺伪度为：{predicted_pseudo_degree[0]:.4f}")
,然后将前面的四个量和这个量作为文字的分析结果显示出来,这样的话页面就可以出现这五个量的分析结果,和两个图的显示按钮以及图片保存路径,务必参考我给的示例算法,中文回答