#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
启动白酒品质检测系统的纯Python脚本
避免批处理文件中可能的编码问题
"""

import os
import sys
import subprocess
import platform

def check_models():
    """检查模型文件是否存在"""
    models_dir = "models"
    model_path = os.path.join(models_dir, "rf_model.pkl")
    pca_path = os.path.join(models_dir, "pca_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler_model.pkl")
    
    # 检查模型目录是否存在，不存在则创建
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        return False
    
    # 检查所有必要的模型文件是否存在
    return os.path.exists(model_path) and os.path.exists(pca_path) and os.path.exists(scaler_path)

def train_models():
    """运行模型训练脚本"""
    print("=" * 40)
    print("需要先训练模型才能使用预测功能")
    print("=" * 40)
    
    choice = input("是否立即训练模型? (y/n): ").strip().lower()
    if choice == 'y':
        print("\n启动模型训练脚本...\n")
        try:
            if platform.system() == "Windows":
                # Windows上使用subprocess运行命令
                result = subprocess.run(["python", "improved_test_tool.py"], check=True)
                if result.returncode == 0:
                    print("模型训练完成!")
                    return True
                else:
                    print("模型训练失败，请查看输出信息")
                    return False
            else:
                # 在其他系统上直接导入并运行
                try:
                    from improved_test_tool import main as train_main
                    train_main()
                    print("模型训练完成!")
                    return True
                except Exception as e:
                    print(f"模型训练失败: {e}")
                    return False
        except Exception as e:
            print(f"启动模型训练脚本失败: {e}")
            return False
    else:
        print("\n没有训练模型，掺伪度分析功能将不可用。")
        return False

def main():
    # 确保必要的目录存在
    os.makedirs('logs', exist_ok=True)
    os.makedirs('history', exist_ok=True)
    os.makedirs('history_img', exist_ok=True)
    os.makedirs('database', exist_ok=True)
    
    print("=" * 40)
    print("白酒品质检测系统启动脚本")
    print("=" * 40)
    
    # 检查Python版本
    py_version = platform.python_version()
    print(f"Python版本: {py_version}")
    
    # 检查模型文件
    models_exist = check_models()
    if not models_exist:
        print("\n警告: 预测模型文件不存在。掺伪度分析功能需要先训练模型。")
        train_choice = input("是否现在训练模型? (y/n，选择n将继续启动应用): ").strip().lower()
        if train_choice == 'y':
            if not train_models():
                print("继续启动应用程序，但掺伪度分析功能可能无法正常工作。")
        else:
            print("继续启动应用程序，但掺伪度分析功能可能无法正常工作。")
    
    # 导入并启动应用
    try:
        # 直接导入main函数并执行
        print("\n正在启动应用程序...")
        from app import main
        main()
        return True
    except ImportError as e:
        print(f"导入错误: {e}")
        print("无法导入应用程序模块。请确保app.py文件存在并且可以访问。")
    except Exception as e:
        print(f"启动失败: {e}")
        print("请检查logs目录下的日志文件以获取详细错误信息。")
    
    return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        # 在Windows上要求用户按任意键继续
        if platform.system() == "Windows":
            print("\n按回车键继续...")
            input() 