#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
白酒品质检测系统 - 组件化版本

基于原始baijiu_app.py改进的组件化版本
保留了原有功能，但将代码拆分为多个组件文件，提高可维护性
"""

import os
import sys
import logging
import customtkinter as ctk

# 设置日志
os.makedirs('logs', exist_ok=True)
os.makedirs('history', exist_ok=True)
os.makedirs('history_img', exist_ok=True)  # 创建图片保存目录

logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('baijiu_app')

# 设置CustomTkinter外观
ctk.set_appearance_mode("dark")  # 默认为深色模式
ctk.set_default_color_theme("blue")  # 默认颜色主题

# 导入组件
from components import LoginWindow

def check_database():
    """检查数据库是否存在，不存在则创建"""
    try:
        from database.db_manager import DatabaseManager
        db_manager = DatabaseManager()
        
        # 创建所有表（使用db_manager中的create_tables方法）
        if not db_manager.table_exists("users") or not db_manager.table_exists("analysis_history") or not db_manager.table_exists("settings"):
            logger.info("创建数据库表")
            db_manager.create_tables()
            
            # 创建默认管理员账户
            if not db_manager.check_username_exists("admin"):
                db_manager.add_user("admin", "admin123", "admin")
                logger.info("已创建默认管理员账户: admin/admin123")
        
        db_manager.close()
        return True
    except Exception as e:
        logger.error(f"检查/初始化数据库时出错: {str(e)}")
        return False

def main():
    """主函数"""
    try:
        # 检查数据库
        if not check_database():
            print("数据库初始化失败，请检查日志文件")
            return
        
        # 创建并运行登录窗口
        app = LoginWindow()
        app.mainloop()
    except Exception as e:
        logger.error(f"应用程序启动失败: {str(e)}")
        print(f"应用程序启动失败: {str(e)}")

if __name__ == "__main__":
    main() 