@echo off
echo 正在启动白酒品质检测系统依赖安装程序...

REM 调用Python安装脚本，该脚本提供更完善的检查和错误处理
python install_dependencies.py

echo.
echo 如果所有依赖安装成功，现在可以运行 start_baijiu_app.bat 启动应用程序。
echo 如果安装失败，请查看上面的错误信息并解决问题。

pause  