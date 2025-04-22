import sqlite3
import os
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    filename='logs/db.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('db_manager')

class DatabaseManager:
    """SQLite数据库管理类，负责处理数据库连接和操作"""
    
    def __init__(self, db_file='database/baijiu.db'):
        """初始化数据库连接
        
        Args:
            db_file: SQLite数据库文件路径
        """
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(db_file), exist_ok=True)
        
        self.db_file = db_file
        self.connection = None
        self.cursor = None
        
        try:
            self.connect()
            logger.info("数据库连接成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            raise
    
    def connect(self):
        """建立数据库连接"""
        self.connection = sqlite3.connect(self.db_file)
        self.cursor = self.connection.cursor()
    
    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")
    
    def execute_query(self, query, params=None):
        """执行SQL查询
        
        Args:
            query: SQL查询语句
            params: 查询参数(可选)
            
        Returns:
            查询结果
        """
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            self.connection.commit()
            return self.cursor.fetchall()
        except Exception as e:
            self.connection.rollback()
            logger.error(f"查询执行失败: {str(e)}, 查询: {query}, 参数: {params}")
            raise
    
    def table_exists(self, table_name):
        """检查表是否存在
        
        Args:
            table_name: 要检查的表名
            
        Returns:
            如果表存在返回True，否则返回False
        """
        try:
            result = self.execute_query(
                "SELECT count(name) FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            )
            return result[0][0] > 0
        except Exception as e:
            logger.error(f"检查表是否存在时出错: {str(e)}")
            return False
    
    def create_tables(self):
        """创建数据库表"""
        try:
            # 用户表
            self.execute_query('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT,
                role TEXT DEFAULT 'user',
                created_at TEXT,
                last_login TEXT
            )
            ''')
            
            # 分析历史记录表
            self.execute_query('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                filename TEXT,
                result TEXT,
                analysis_type TEXT,
                created_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # 系统设置表
            self.execute_query('''
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                theme TEXT DEFAULT 'dark',
                color_theme TEXT DEFAULT 'blue',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            logger.info("数据库表创建成功")
        except Exception as e:
            logger.error(f"创建表失败: {str(e)}")
            raise
    
    def add_user(self, username, password, email=None, role='user'):
        """添加新用户
        
        Args:
            username: 用户名
            password: 密码
            email: 电子邮件(可选)
            role: 用户角色(默认为'user')
            
        Returns:
            新用户ID
        """
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            self.execute_query(
                "INSERT INTO users (username, password, email, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, password, email, role, created_at)
            )
            user_id = self.cursor.lastrowid
            logger.info(f"添加用户成功: {username}, ID: {user_id}")
            return user_id
        except Exception as e:
            logger.error(f"添加用户失败: {str(e)}")
            raise
    
    def check_username_exists(self, username):
        """检查用户名是否已存在
        
        Args:
            username: 要检查的用户名
            
        Returns:
            如果用户名存在返回True，否则返回False
        """
        try:
            result = self.execute_query(
                "SELECT COUNT(*) FROM users WHERE username = ?",
                (username,)
            )
            return result[0][0] > 0
        except Exception as e:
            logger.error(f"检查用户名时出错: {str(e)}")
            return False
    
    def verify_user(self, username, password):
        """验证用户登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            用户ID和角色的元组，验证失败则返回(None, None)
        """
        try:
            result = self.execute_query(
                "SELECT id, role FROM users WHERE username = ? AND password = ?",
                (username, password)
            )
            
            if result:
                user_id, role = result[0]
                # 更新最后登录时间
                last_login = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.execute_query(
                    "UPDATE users SET last_login = ? WHERE id = ?",
                    (last_login, user_id)
                )
                logger.info(f"用户验证成功: {username}")
                return user_id, role
            else:
                logger.warning(f"用户验证失败: {username}")
                return None, None
        except Exception as e:
            logger.error(f"用户验证过程中出错: {str(e)}")
            return None, None
    
    def add_analysis_record(self, user_id, filename, result, analysis_type):
        """添加分析记录
        
        Args:
            user_id: 用户ID
            filename: 分析的文件名
            result: 分析结果
            analysis_type: 分析类型
            
        Returns:
            记录ID
        """
        created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            self.execute_query(
                "INSERT INTO analysis_history (user_id, filename, result, analysis_type, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, filename, result, analysis_type, created_at)
            )
            record_id = self.cursor.lastrowid
            logger.info(f"添加分析记录成功: ID: {record_id}, 用户ID: {user_id}")
            return record_id
        except Exception as e:
            logger.error(f"添加分析记录失败: {str(e)}")
            raise
    
    def get_user_history(self, user_id):
        """获取用户的分析历史记录
        
        Args:
            user_id: 用户ID
            
        Returns:
            分析记录列表
        """
        try:
            records = self.execute_query(
                "SELECT id, filename, result, analysis_type, created_at FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            )
            return records
        except Exception as e:
            logger.error(f"获取用户历史记录失败: {str(e)}")
            return []
    
    def get_user_settings(self, user_id):
        """获取用户设置
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户设置字典
        """
        try:
            result = self.execute_query(
                "SELECT theme, color_theme FROM settings WHERE user_id = ?",
                (user_id,)
            )
            
            if result:
                theme, color_theme = result[0]
                return {"theme": theme, "color_theme": color_theme}
            else:
                # 如果没有找到设置，创建默认设置
                self.execute_query(
                    "INSERT INTO settings (user_id, theme, color_theme) VALUES (?, 'dark', 'blue')",
                    (user_id,)
                )
                return {"theme": "dark", "color_theme": "blue"}
        except Exception as e:
            logger.error(f"获取用户设置失败: {str(e)}")
            return {"theme": "dark", "color_theme": "blue"}
    
    def update_user_settings(self, user_id, theme=None, color_theme=None):
        """更新用户设置
        
        Args:
            user_id: 用户ID
            theme: 主题(可选)
            color_theme: 颜色主题(可选)
            
        Returns:
            更新是否成功
        """
        try:
            settings = self.get_user_settings(user_id)
            
            if theme is not None:
                settings["theme"] = theme
            if color_theme is not None:
                settings["color_theme"] = color_theme
            
            self.execute_query(
                "UPDATE settings SET theme = ?, color_theme = ? WHERE user_id = ?",
                (settings["theme"], settings["color_theme"], user_id)
            )
            
            logger.info(f"更新用户设置成功: 用户ID: {user_id}")
            return True
        except Exception as e:
            logger.error(f"更新用户设置失败: {str(e)}")
            return False
    
    def get_user_by_id(self, user_id):
        """根据用户ID获取用户完整信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户信息字典，如果用户不存在则返回None
        """
        try:
            result = self.execute_query(
                "SELECT id, username, email, role, created_at, last_login FROM users WHERE id = ?",
                (user_id,)
            )
            
            if result:
                user_data = {
                    "id": result[0][0],
                    "username": result[0][1],
                    "email": result[0][2],
                    "role": result[0][3],
                    "created_at": result[0][4],
                    "last_login": result[0][5]
                }
                logger.info(f"获取用户信息成功: 用户ID: {user_id}")
                return user_data
            else:
                logger.warning(f"未找到用户: 用户ID: {user_id}")
                return None
        except Exception as e:
            logger.error(f"获取用户信息失败: {str(e)}")
            return None
    
    def update_user(self, user_id, email=None):
        """更新用户信息
        
        Args:
            user_id: 用户ID
            email: 用户电子邮件(可选)
            
        Returns:
            更新是否成功
        """
        try:
            fields_to_update = []
            params = []
            
            if email is not None:
                fields_to_update.append("email = ?")
                params.append(email)
            
            if not fields_to_update:
                return True  # 没有需要更新的字段
            
            # 添加用户ID作为WHERE条件的参数
            params.append(user_id)
            
            # 构建UPDATE语句
            update_query = f"UPDATE users SET {', '.join(fields_to_update)} WHERE id = ?"
            
            self.execute_query(update_query, params)
            logger.info(f"更新用户信息成功: 用户ID: {user_id}")
            return True
        except Exception as e:
            logger.error(f"更新用户信息失败: {str(e)}")
            return False
    
    def get_all_users(self):
        """获取所有用户信息
        
        Returns:
            所有用户信息的列表，每个用户为一个字典
        """
        try:
            result = self.execute_query(
                "SELECT id, username, password, email, role, created_at, last_login FROM users ORDER BY id"
            )
            
            users = []
            for row in result:
                user_data = {
                    "id": row[0],
                    "username": row[1],
                    "password": row[2],
                    "email": row[3],
                    "role": row[4],
                    "created_at": row[5],
                    "last_login": row[6]
                }
                users.append(user_data)
                
            logger.info(f"成功获取全部用户信息，共 {len(users)} 条记录")
            return users
        except Exception as e:
            logger.error(f"获取全部用户信息失败: {str(e)}")
            return []
            
    def get_all_users_except(self, exclude_user_id):
        """获取除指定用户外的所有用户信息
        
        Args:
            exclude_user_id: 要排除的用户ID
            
        Returns:
            用户信息元组列表 [(id, username, role, created_at), ...]
        """
        try:
            result = self.execute_query(
                "SELECT id, username, role, created_at FROM users WHERE id != ? ORDER BY id",
                (exclude_user_id,)
            )
            
            logger.info(f"成功获取除用户 {exclude_user_id} 外的所有用户信息")
            return result
        except Exception as e:
            logger.error(f"获取用户列表失败: {str(e)}")
            return []
            
    def update_user_role(self, user_id, new_role):
        """更新用户角色
        
        Args:
            user_id: 用户ID
            new_role: 新角色('admin'或'user')
            
        Returns:
            更新是否成功
        """
        try:
            self.execute_query(
                "UPDATE users SET role = ? WHERE id = ?",
                (new_role, user_id)
            )
            logger.info(f"更新用户角色成功: 用户ID: {user_id}, 新角色: {new_role}")
            return True
        except Exception as e:
            logger.error(f"更新用户角色失败: {str(e)}")
            return False
    
    def delete_recent_history(self, user_id, count=50):
        """删除用户最近的N条历史记录
        
        Args:
            user_id: 用户ID
            count: 要删除的记录数量(默认为50)
            
        Returns:
            成功删除的记录数量
        """
        try:
            # 首先获取该用户最近的N条记录的ID
            record_ids = self.execute_query(
                "SELECT id FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, count)
            )
            
            if not record_ids:
                logger.info(f"用户 {user_id} 没有历史记录可删除")
                return 0
                
            # 将ID列表转换为元组中的字符串形式，例如 (1,2,3)
            id_list = ','.join(str(record[0]) for record in record_ids)
            
            # 执行删除操作
            self.execute_query(
                f"DELETE FROM analysis_history WHERE id IN ({id_list})"
            )
            
            deleted_count = len(record_ids)
            logger.info(f"成功删除用户 {user_id} 的 {deleted_count} 条最近历史记录")
            return deleted_count
            
        except Exception as e:
            logger.error(f"删除用户历史记录失败: {str(e)}")
            return 0 