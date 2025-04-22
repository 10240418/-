@echo off
echo ==================================
echo 白酒品质检测系统启动脚本
echo ==================================
echo.

REM 以UTF-8编码运行Python脚本
chcp 65001 > nul
python start.py

echo.
echo 如果应用程序启动失败，请检查logs目录下的日志文件

pause 