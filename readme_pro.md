# 白酒品质检测与分析系统

## 快速启动指南

### 方法一：从源码运行

1. 确保您已安装Python 3.7或更高版本

2. 下载或克隆项目代码：
   ```bash
   git clone https://github.com/username/baijiu-analysis-system.git
   cd baijiu-analysis-system
   ```

3. 安装依赖（推荐方式）：
   ```bash
   python install_dependencies.py
   ```
   该脚本会自动安装所有必要的依赖库，并处理可能的错误。

   或者手动安装依赖：
   ```bash
   pip install numpy pandas matplotlib scikit-learn torch customtkinter joblib openpyxl xlrd pillow
   ```

4. 初始化数据库：
   ```bash
   python database/init_db.py
   ```

5. 检查数据库（可选）：
   ```bash
   python inspect_database.py
   ```
   这将显示数据库的结构和内容，确认是否正确初始化。

6. 启动应用：
   ```bash
   python baijiu_app.py
   ```

### 方法二：使用可执行文件

在Windows系统下：
1. 下载最新的发布版本（.exe文件）
2. 解压缩下载的文件
3. 双击运行 `baijiu_app.exe`

**注意**：首次运行可能需要一些时间来加载模型和初始化数据库。

### 初始账户

- 管理员账户：用户名 `admin`，密码 `admin123`
- 测试用户：用户名 `user1`，密码 `123456` 