import os
import sys
import logging
from pprint import pprint

# 设置日志目录
os.makedirs('logs', exist_ok=True)

# 配置日志
logging.basicConfig(
    filename='logs/db_inspect.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_inspector')

# 将当前目录添加到路径中，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from database.db_manager import DatabaseManager
    
    def inspect_database():
        """使用DatabaseManager检查数据库内容"""
        try:
            # 创建数据库管理器实例
            db_manager = DatabaseManager()
            
            print("\n======== 数据库检查工具 ========\n")
            
            # 检查并显示用户表
            print("===== 用户信息 =====")
            users = db_manager.execute_query("SELECT * FROM users")
            user_columns = db_manager.execute_query("PRAGMA table_info(users)")
            
            # 显示用户表结构
            print("\n用户表结构:")
            for col in user_columns:
                col_id, name, dtype, notnull, default_val, pk = col
                print(f"  {name} ({dtype})", end="")
                if pk:
                    print(" PRIMARY KEY", end="")
                if notnull:
                    print(" NOT NULL", end="")
                if default_val is not None:
                    print(f" DEFAULT {default_val}", end="")
                print()
            
            # 显示用户数据
            print(f"\n用户数量: {len(users)}")
            if users:
                print("\n用户数据:")
                col_names = [col[1] for col in user_columns]
                print("  " + " | ".join(col_names))
                print("  " + "-" * 50)
                
                for user in users:
                    print("  " + " | ".join(str(item) for item in user))
            
            # 检查并显示设置表
            print("\n\n===== 用户设置 =====")
            settings = db_manager.execute_query("SELECT * FROM settings")
            settings_columns = db_manager.execute_query("PRAGMA table_info(settings)")
            
            # 显示设置表结构
            print("\n设置表结构:")
            for col in settings_columns:
                col_id, name, dtype, notnull, default_val, pk = col
                print(f"  {name} ({dtype})", end="")
                if pk:
                    print(" PRIMARY KEY", end="")
                if notnull:
                    print(" NOT NULL", end="")
                if default_val is not None:
                    print(f" DEFAULT {default_val}", end="")
                print()
            
            # 显示设置数据
            print(f"\n设置数量: {len(settings)}")
            if settings:
                print("\n设置数据:")
                col_names = [col[1] for col in settings_columns]
                print("  " + " | ".join(col_names))
                print("  " + "-" * 50)
                
                for setting in settings:
                    print("  " + " | ".join(str(item) for item in setting))
            
            # 检查并显示分析历史表
            print("\n\n===== 分析历史 =====")
            history = db_manager.execute_query("SELECT * FROM analysis_history")
            history_columns = db_manager.execute_query("PRAGMA table_info(analysis_history)")
            
            # 显示历史表结构
            print("\n历史表结构:")
            for col in history_columns:
                col_id, name, dtype, notnull, default_val, pk = col
                print(f"  {name} ({dtype})", end="")
                if pk:
                    print(" PRIMARY KEY", end="")
                if notnull:
                    print(" NOT NULL", end="")
                if default_val is not None:
                    print(f" DEFAULT {default_val}", end="")
                print()
            
            # 显示历史数据
            print(f"\n历史记录数量: {len(history)}")
            if history:
                print("\n历史数据 (最多显示5条):")
                col_names = [col[1] for col in history_columns]
                print("  " + " | ".join(col_names))
                print("  " + "-" * 80)
                
                for i, record in enumerate(history[:5]):
                    # 格式化记录以便于显示
                    formatted_record = []
                    for item in record:
                        if isinstance(item, str) and len(item) > 20:
                            formatted_record.append(f"{item[:20]}...")
                        else:
                            formatted_record.append(str(item))
                    print("  " + " | ".join(formatted_record))
                
                if len(history) > 5:
                    print(f"  ... 还有 {len(history) - 5} 条记录")
            
            # 关闭数据库连接
            db_manager.close()
            
            print("\n======== 数据库检查完成 ========\n")
            return True
        except Exception as e:
            logger.error(f"数据库检查失败: {str(e)}")
            print(f"数据库检查失败: {str(e)}")
            return False

    if __name__ == "__main__":
        inspect_database()
        
except Exception as e:
    logger.error(f"导入模块失败: {str(e)}")
    print(f"导入模块失败: {str(e)}") 