@echo off
echo 正在启动白酒品质检测系统...

REM 创建必要的目录
if not exist logs mkdir logs
if not exist history mkdir history
if not exist database mkdir database

REM 确保数据库目录存在
if not exist database\baijiu.db (
    echo 初始化数据库...
    python database\init_db.py
)

REM 启动应用程序
echo 启动应用程序...
python baijiu_app.py

echo 程序已退出。
pause 