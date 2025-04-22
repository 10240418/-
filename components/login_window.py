import os
import logging
import tkinter as tk
import customtkinter as ctk
from components.register_window import RegisterWindow
from components.main_application import MainApplication

logger = logging.getLogger('baijiu_app')

class LoginWindow(ctk.CTk):
    """登录窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 配置窗口
        self.title("白酒品质检测系统 - 登录")
        self.geometry("1000x700")  # 修改窗口尺寸与主界面一致
        self.resizable(False, False)
        
        # 初始化数据库管理器
        try:
            from database.db_manager import DatabaseManager
            self.db_manager = DatabaseManager()
            logger.info("数据库连接成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            self.show_error("数据库连接失败", f"无法连接到数据库: {str(e)}")
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建登录窗口UI组件"""
        
        # 创建主布局框架
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True)
        
        # 左侧装饰区域
        self.left_frame = ctk.CTkFrame(self.main_frame, width=500, corner_radius=0)
        self.left_frame.pack(side="left", fill="both", expand=True)
        
        # 标题标签
        self.title_label = ctk.CTkLabel(
            self.left_frame, 
            text="白酒品质检测系统", 
            font=ctk.CTkFont(size=36, weight="bold")
        )
        self.title_label.pack(pady=(150, 20))
        
        # 系统描述
        self.desc_label = ctk.CTkLabel(
            self.left_frame, 
            text="专业的白酒质量检测与分析平台", 
            font=ctk.CTkFont(size=18)
        )
        self.desc_label.pack(pady=(0, 40))
        
        # 版权信息
        self.copyright_label = ctk.CTkLabel(
            self.left_frame, 
            text="© 2024 四川农业大学 白酒品质检测系统",
            font=ctk.CTkFont(size=12)
        )
        self.copyright_label.pack(side="bottom", pady=20)
        
        # 右侧登录区域
        self.right_frame = ctk.CTkFrame(self.main_frame, width=500, corner_radius=0)
        self.right_frame.pack(side="right", fill="both")
        
        # 创建登录框架
        self.login_frame = ctk.CTkFrame(self.right_frame, width=350, height=400)
        self.login_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # 登录标签
        self.login_label = ctk.CTkLabel(
            self.login_frame, 
            text="用户登录", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.login_label.pack(pady=(30, 30))
        
        # 用户名输入
        self.username_label = ctk.CTkLabel(self.login_frame, text="用户名:", font=ctk.CTkFont(size=14))
        self.username_label.pack(pady=(10, 0), padx=40, anchor="w")
        
        self.username_entry = ctk.CTkEntry(self.login_frame, width=270, height=35, placeholder_text="请输入用户名")
        self.username_entry.insert(0, "admin")  # 默认设置用户名为admin
        self.username_entry.pack(pady=(5, 15), padx=40)
        
        # 密码输入
        self.password_label = ctk.CTkLabel(self.login_frame, text="密码:", font=ctk.CTkFont(size=14))
        self.password_label.pack(pady=(10, 0), padx=40, anchor="w")
        
        self.password_entry = ctk.CTkEntry(self.login_frame, width=270, height=35, placeholder_text="请输入密码", show="*")
        self.password_entry.insert(0, "admin123")  # 默认设置密码为admin123
        self.password_entry.pack(pady=(5, 20), padx=40)
        
        # 登录按钮
        self.login_button = ctk.CTkButton(
            self.login_frame, 
            text="登录", 
            width=270, 
            height=40,
            font=ctk.CTkFont(size=14),
            command=self.login
        )
        self.login_button.pack(pady=(15, 15), padx=40)
        
        # 注册按钮
        self.register_button = ctk.CTkButton(
            self.login_frame, 
            text="注册新用户", 
            width=270, 
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="transparent",  # 透明背景
            text_color=("gray10", "#DCE4EE"),  # 适应深色/浅色模式
            border_width=2,  # 添加边框
            hover_color=("gray70", "gray30"),  # 悬停颜色
            command=self.open_register_window
        )
        self.register_button.pack(pady=(5, 30), padx=40)
    
    def login(self):
        """处理登录逻辑"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            self.show_error("登录失败", "用户名和密码不能为空")
            return
        
        try:
            user_id, role = self.db_manager.verify_user(username, password)
            
            if user_id:
                logger.info(f"用户 {username} 登录成功")
                self.withdraw()  # 隐藏登录窗口
                # 打开主应用窗口，并传递登录窗口引用
                app = MainApplication(user_id, username, role, self.db_manager, self)
                app.mainloop()
                # 如果主窗口直接关闭，则也关闭登录窗口
                if not app.winfo_exists():
                    self.destroy()
            else:
                logger.warning(f"用户 {username} 登录失败：用户名或密码错误")
                self.show_error("登录失败", "用户名或密码错误")
        except Exception as e:
            logger.error(f"登录过程中出错: {str(e)}")
            self.show_error("登录错误", f"登录过程中出错: {str(e)}")
    
    def open_register_window(self):
        """打开注册窗口"""
        self.withdraw()  # 隐藏登录窗口
        register_window = RegisterWindow(self, self.db_manager)
        register_window.grab_set()  # 设置为模态窗口
    
    def show_error(self, title, message):
        """显示错误对话框"""
        try:
            # 确保先销毁进度窗口（如果存在）
            if hasattr(self, 'progress_window') and self.progress_window is not None:
                try:
                    if self.progress_window.winfo_exists():
                        self.progress_window.grab_release()
                        self.progress_window.destroy()
                    self.progress_window = None
                except Exception:
                    pass  # 忽略窗口可能已经销毁的错误
            
            # 创建新的错误窗口
            try:
                # 使用self.winfo_toplevel()作为父窗口而不是self
                parent_window = self.winfo_toplevel()
                error_window = ctk.CTkToplevel(parent_window)
                error_window.title(title)
                error_window.geometry("300x200")
                error_window.resizable(False, False)
                error_window.transient(parent_window)
                
                # 确保程序退出前，所有窗口都能正确关闭
                error_window.protocol("WM_DELETE_WINDOW", error_window.destroy)
                
                # 允许窗口绘制并显示
                error_window.update()
                
                # 然后才尝试设置grab
                try:
                    error_window.grab_set()
                except Exception:
                    logger.warning("无法设置错误窗口为模态窗口")
                
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
                
                # 强制窗口更新
                error_window.update()
                
            except Exception as window_error:
                logger.error(f"创建错误窗口失败: {str(window_error)}")
                # 如果创建窗口失败，至少在控制台输出错误信息
                print(f"错误: {title} - {message}")
        
        except Exception as e:
            logger.error(f"显示错误对话框时发生异常: {str(e)}")
            print(f"错误: {title} - {message}")
            
    def hide_progress(self):
        """隐藏进度窗口"""
        try:
            if hasattr(self, 'progress_window') and self.progress_window is not None:
                try:
                    if self.progress_window.winfo_exists():
                        self.progress_window.grab_release()
                        self.progress_window.destroy()
                    self.progress_window = None
                except Exception:
                    pass  # 忽略可能已经销毁的窗口错误
                self.progress_window = None
        except Exception as e:
            logger.error(f"隐藏进度窗口时出错: {str(e)}") 