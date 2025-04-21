import os
import sys
import logging

# 设置日志目录
os.makedirs('logs', exist_ok=True)

# 配置日志
logging.basicConfig(
    filename='logs/init_db.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('init_db')

# 将当前目录添加到路径中，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from database.db_manager import DatabaseManager
    
    def init_database():
        """初始化数据库，创建表和添加默认用户"""
        try:
            # 创建数据库管理器实例
            db_manager = DatabaseManager()
            
            # 创建数据库表
            db_manager.create_tables()
            logger.info("数据库表创建成功")
            
            # 检查用户表是否有数据
            users = db_manager.execute_query("SELECT COUNT(*) FROM users")
            
            # 如果没有用户，添加默认用户
            if users[0][0] == 0:
                # 添加管理员用户
                db_manager.add_user("admin", "admin123", role="admin")
                
                # 添加测试用户
                db_manager.add_user("user1", "123456")
                
                logger.info("默认用户创建成功")
            else:
                logger.info("用户已存在，跳过添加默认用户")
            
            # 关闭数据库连接
            db_manager.close()
            
            print("数据库初始化成功!")
            return True
        except Exception as e:
            logger.error(f"数据库初始化失败: {str(e)}")
            print(f"数据库初始化失败: {str(e)}")
            return False

    if __name__ == "__main__":
        init_database()
        
except Exception as e:
    logger.error(f"导入模块失败: {str(e)}")
    print(f"导入模块失败: {str(e)}") 