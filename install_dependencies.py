import subprocess
import sys
import os
import platform

def check_python_version():
    """检查Python版本是否满足要求"""
    required_version = (3, 7)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"错误: Python版本必须是 {required_version[0]}.{required_version[1]} 或更高")
        print(f"当前版本: {current_version[0]}.{current_version[1]}")
        return False
    return True

def check_pip():
    """检查pip是否已安装"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        print("错误: pip未安装")
        return False

def install_package(package):
    """安装单个包并处理可能的错误"""
    print(f"正在安装 {package}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        print(f"✓ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 安装 {package} 失败")
        print(f"错误信息: {e}")
        return False

def install_requirements():
    """从requirements.txt安装所有依赖"""
    if not os.path.exists("requirements.txt"):
        print("错误: 未找到requirements.txt文件")
        return False
    
    print("开始安装依赖...")
    
    # 首先升级pip
    print("正在升级pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    with open("requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    
    failed_packages = []
    for req in requirements:
        if not install_package(req):
            failed_packages.append(req)
    
    if failed_packages:
        print("\n以下包安装失败:")
        for package in failed_packages:
            print(f"- {package}")
        print("\n请尝试手动安装这些包，或检查是否有特殊的系统要求。")
        return False
    
    return True

def main():
    """主函数"""
    print("=== 白酒分析系统依赖安装程序 ===")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version.split()[0]}")
    print("=" * 35 + "\n")
    
    if not check_python_version():
        return 1
    
    if not check_pip():
        return 1
    
    try:
        if install_requirements():
            print("\n✓ 所有依赖安装成功！")
            print("现在您可以运行白酒分析系统了。")
            return 0
        else:
            print("\n✗ 部分依赖安装失败。")
            print("请检查上述错误信息并手动解决。")
            return 1
    except Exception as e:
        print(f"\n发生未预期的错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 