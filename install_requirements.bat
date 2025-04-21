@echo off
echo 正在安装白酒品质检测系统所需的库...

REM 创建虚拟环境(可选)
REM python -m venv venv
REM call venv\Scripts\activate

REM 安装所需的库
echo 安装pandas、numpy和scikit-learn...
pip install pandas numpy scikit-learn

echo 安装Excel处理库...
pip install openpyxl xlrd

echo 安装图形界面和数据库库...
pip install customtkinter matplotlib joblib

echo 所有库已安装完成!
echo 现在可以运行 start_baijiu_app.bat 启动应用程序。

pause 