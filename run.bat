@echo off
echo ==================================
echo 白酒品质检测系统启动脚本
echo ==================================
echo.
echo 正在启动系统，请稍候...
echo.

REM 以UTF-8编码运行Python脚本
chcp 65001 > nul
python start.py

echo.
echo 应用程序已关闭。如果遇到问题，请检查logs目录下的日志文件。
pause 