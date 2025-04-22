import logging
import customtkinter as ctk
import tkinter as tk

logger = logging.getLogger('baijiu_app')

class UserManagementFrame(ctk.CTkFrame):
    """用户管理页面（仅管理员可访问）"""
    
    def __init__(self, parent, user_id, db_manager):
        super().__init__(parent, corner_radius=0)
        
        self.user_id = user_id
        self.db_manager = db_manager
        self.users = []
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建用户管理页面组件"""
        
        # 页面标题
        self.title_label = ctk.CTkLabel(
            self, 
            text="用户管理", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 20), padx=30, anchor="w")
        
        # 操作按钮框架
        self.action_frame = ctk.CTkFrame(self)
        self.action_frame.pack(fill="x", padx=30, pady=(0, 20))
        
        # 刷新按钮
        self.refresh_button = ctk.CTkButton(
            self.action_frame, 
            text="刷新用户列表", 
            width=150,
            command=self.refresh_users
        )
        self.refresh_button.pack(side="left", padx=10, pady=10)
        
        # 用户信息表格框架
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # 创建表格标题
        self.create_table_header()
        
        # 创建表格内容框架（使用滚动视图）
        self.table_content_frame = ctk.CTkScrollableFrame(self.table_frame)
        self.table_content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # 加载用户列表
        self.refresh_users()
    
    def create_table_header(self):
        """创建表格标题"""
        header_frame = ctk.CTkFrame(self.table_frame)
        header_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        # ID列
        id_label = ctk.CTkLabel(
            header_frame, 
            text="ID", 
            font=ctk.CTkFont(weight="bold"),
            width=50
        )
        id_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # 用户名列
        username_label = ctk.CTkLabel(
            header_frame, 
            text="用户名", 
            font=ctk.CTkFont(weight="bold"),
            width=150
        )
        username_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # 角色列
        role_label = ctk.CTkLabel(
            header_frame, 
            text="用户角色", 
            font=ctk.CTkFont(weight="bold"),
            width=120
        )
        role_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        # 注册时间列
        time_label = ctk.CTkLabel(
            header_frame, 
            text="注册时间", 
            font=ctk.CTkFont(weight="bold"),
            width=150
        )
        time_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # 操作列
        action_label = ctk.CTkLabel(
            header_frame, 
            text="操作", 
            font=ctk.CTkFont(weight="bold"),
            width=150
        )
        action_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        
        # 配置网格
        for i in range(5):
            header_frame.grid_columnconfigure(i, weight=1)
    
    def refresh_users(self):
        """刷新用户列表"""
        try:
            # 清除旧记录
            for widget in self.table_content_frame.winfo_children():
                widget.destroy()
            
            # 获取用户列表（除了当前用户）
            self.users = self.db_manager.get_all_users_except(self.user_id)
            
            if not self.users:
                # 显示无用户提示
                no_data_label = ctk.CTkLabel(
                    self.table_content_frame, 
                    text="没有其他用户",
                    font=ctk.CTkFont(size=14)
                )
                no_data_label.pack(pady=50)
                return
            
            # 显示用户列表
            for i, user in enumerate(self.users):
                user_id, username, role, created_at = user
                
                # 创建用户行
                row_frame = ctk.CTkFrame(self.table_content_frame)
                row_frame.pack(fill="x", pady=2)
                
                # 设置行背景颜色（奇偶行不同颜色）
                if i % 2 == 0:
                    row_frame.configure(fg_color=("gray90", "gray20"))
                
                # ID列
                id_label = ctk.CTkLabel(
                    row_frame, 
                    text=str(user_id),
                    width=50
                )
                id_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
                
                # 用户名列
                username_label = ctk.CTkLabel(
                    row_frame, 
                    text=username,
                    width=150
                )
                username_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
                
                # 角色列
                role_label = ctk.CTkLabel(
                    row_frame, 
                    text=role,
                    width=120
                )
                role_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
                
                # 注册时间列
                time_label = ctk.CTkLabel(
                    row_frame, 
                    text=created_at,
                    width=150
                )
                time_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")
                
                # 操作列 - 角色切换按钮
                action_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
                action_frame.grid(row=0, column=4, padx=5, pady=5, sticky="w")
                
                # 不同角色切换按钮
                if role == "admin":
                    role_button = ctk.CTkButton(
                        action_frame,
                        text="设为普通用户",
                        width=120,
                        height=25,
                        fg_color="orange",
                        hover_color="darkorange",
                        command=lambda uid=user_id: self._change_user_role(uid, "user")
                    )
                else:
                    role_button = ctk.CTkButton(
                        action_frame,
                        text="设为管理员",
                        width=120,
                        height=25,
                        fg_color="green",
                        hover_color="darkgreen",
                        command=lambda uid=user_id: self._change_user_role(uid, "admin")
                    )
                role_button.pack(side="left", padx=(0, 10))
                
                # 配置网格
                for j in range(5):
                    row_frame.grid_columnconfigure(j, weight=1)
            
            logger.info(f"已加载 {len(self.users)} 个用户记录")
            
        except Exception as e:
            logger.error(f"加载用户列表时出错: {str(e)}")
            self.show_error("加载错误", f"无法加载用户列表: {str(e)}")
    
    def _change_user_role(self, user_id, new_role):
        """更改用户角色"""
        try:
            success = self.db_manager.update_user_role(user_id, new_role)
            
            if success:
                logger.info(f"用户ID {user_id} 的角色已更改为 {new_role}")
                self.show_info("角色更改", f"用户角色已更改为 {new_role}")
                self.refresh_users()  # 刷新用户列表
            else:
                self.show_error("角色更改失败", "无法更改用户角色")
                
        except Exception as e:
            logger.error(f"更改用户角色时出错: {str(e)}")
            self.show_error("角色更改错误", f"更改用户角色时出错: {str(e)}")
    
    def show_error(self, title, message):
        """显示错误对话框"""
        error_window = ctk.CTkToplevel(self)
        error_window.title(title)
        error_window.geometry("300x200")
        error_window.resizable(False, False)
        error_window.transient(self)
        error_window.grab_set()
        
        # 错误图标和消息
        error_label = ctk.CTkLabel(
            error_window, 
            text="错误", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="red"
        )
        error_label.pack(pady=(20, 10))
        
        message_label = ctk.CTkLabel(
            error_window, 
            text=message,
            wraplength=250
        )
        message_label.pack(pady=10, padx=20)
        
        # 确定按钮
        ok_button = ctk.CTkButton(
            error_window, 
            text="确定", 
            width=100,
            command=error_window.destroy
        )
        ok_button.pack(pady=(10, 20))
    
    def show_info(self, title, message):
        """显示信息对话框"""
        info_window = ctk.CTkToplevel(self)
        info_window.title(title)
        info_window.geometry("300x200")
        info_window.resizable(False, False)
        info_window.transient(self)
        info_window.grab_set()
        
        # 信息图标和消息
        info_label = ctk.CTkLabel(
            info_window, 
            text="信息", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="green"
        )
        info_label.pack(pady=(20, 10))
        
        message_label = ctk.CTkLabel(
            info_window, 
            text=message,
            wraplength=250
        )
        message_label.pack(pady=10, padx=20)
        
        # 确定按钮
        ok_button = ctk.CTkButton(
            info_window, 
            text="确定", 
            width=100,
            command=info_window.destroy
        )
        ok_button.pack(pady=(10, 20)) 