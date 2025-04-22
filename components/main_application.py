import logging
import customtkinter as ctk
import tkinter as tk

# 导入各个页面组件
from components.user_info_frame import UserInfoFrame
from components.analysis_frame import AnalysisFrame
from components.classification_frame import ClassificationFrame
from components.history_frame import HistoryFrame
from components.settings_frame import SettingsFrame
from components.user_management_frame import UserManagementFrame

logger = logging.getLogger('baijiu_app')

class MainApplication(ctk.CTk):
    """主应用程序窗口类"""
    
    def __init__(self, user_id, username, role, db_manager, login_window=None):
        super().__init__()
        
        # 保存用户信息
        self.user_id = user_id
        self.username = username
        self.role = role
        self.db_manager = db_manager
        self.login_window = login_window  # 添加登录窗口引用
        
        # 获取用户设置
        self.user_settings = self.db_manager.get_user_settings(user_id)
        
        # 应用主题设置
        ctk.set_appearance_mode(self.user_settings["theme"])
        ctk.set_default_color_theme(self.user_settings["color_theme"])
        
        # 配置窗口
        self.title(f"白酒品质检测系统 - 欢迎 {username}")
        self.geometry("1000x700")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 创建UI组件
        self.create_widgets()
        
        logger.info(f"主应用程序窗口已为用户 {username} 创建")
    
    def create_widgets(self):
        """创建主窗口UI组件"""
        
        # 创建顶部菜单栏
        self.menu_frame = ctk.CTkFrame(self, height=50)
        self.menu_frame.pack(side="top", fill="x")
        
        # 应用标题
        self.app_title = ctk.CTkLabel(
            self.menu_frame, 
            text="白酒品质检测系统", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.app_title.pack(side="left", padx=20, pady=10)
        
        # 用户信息
        self.user_info = ctk.CTkLabel(
            self.menu_frame, 
            text=f"用户: {self.username} | 角色: {self.role}",
            font=ctk.CTkFont(size=12)
        )
        self.user_info.pack(side="right", padx=20, pady=10)
        
        # 创建左侧导航栏
        self.navigation_frame = ctk.CTkFrame(self, width=200)
        self.navigation_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        # 导航按钮 - 个人信息
        self.user_info_button = ctk.CTkButton(
            self.navigation_frame, 
            text="个人信息", 
            width=180,
            command=lambda: self.show_frame("user_info")
        )
        self.user_info_button.pack(pady=(20, 10), padx=10)
        
        # 导航按钮 - 掺伪量分析预测
        self.analysis_button = ctk.CTkButton(
            self.navigation_frame, 
            text="掺伪量分析预测", 
            width=180,
            command=lambda: self.show_frame("analysis")
        )
        self.analysis_button.pack(pady=10, padx=10)
        
        # 导航按钮 - 白酒真伪分类
        self.classification_button = ctk.CTkButton(
            self.navigation_frame, 
            text="白酒真伪分类", 
            width=180,
            command=lambda: self.show_frame("classification")
        )
        self.classification_button.pack(pady=10, padx=10)
        
        # 导航按钮 - 历史记录
        self.history_button = ctk.CTkButton(
            self.navigation_frame, 
            text="历史记录", 
            width=180,
            command=lambda: self.show_frame("history")
        )
        self.history_button.pack(pady=10, padx=10)
        
        # 只有管理员可以看到用户管理按钮
        if self.role == "admin":
            # 导航按钮 - 用户管理
            self.user_mgmt_button = ctk.CTkButton(
                self.navigation_frame, 
                text="用户管理", 
                width=180,
                command=lambda: self.show_frame("user_management")
            )
            self.user_mgmt_button.pack(pady=10, padx=10)
        
        # 导航按钮 - 设置
        self.settings_button = ctk.CTkButton(
            self.navigation_frame, 
            text="设置", 
            width=180,
            command=lambda: self.show_frame("settings")
        )
        self.settings_button.pack(pady=10, padx=10)
        
        # 导航按钮 - 退出
        self.exit_button = ctk.CTkButton(
            self.navigation_frame, 
            text="退出登录", 
            width=180,
            fg_color="#D35B58",
            hover_color="#C77C78",
            command=self.on_closing
        )
        self.exit_button.pack(pady=(50, 10), padx=10)
        
        # 创建主内容框架
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        # 创建不同页面的框架
        self.frames = {}
        
        # 个人信息页面
        self.frames["user_info"] = UserInfoFrame(self.main_frame, self.user_id, self.username, self.role, self.db_manager)
        
        # 掺伪量分析预测页面
        self.frames["analysis"] = AnalysisFrame(self.main_frame, self.user_id, self.db_manager)
        
        # 白酒真伪分类页面
        self.frames["classification"] = ClassificationFrame(self.main_frame, self.user_id, self.db_manager)
        
        # 历史记录页面
        self.frames["history"] = HistoryFrame(self.main_frame, self.user_id, self.db_manager)
        
        # 设置页面
        self.frames["settings"] = SettingsFrame(self.main_frame, self.user_id, self.db_manager, self.apply_settings)
        
        # 用户管理页面（仅管理员可见）
        if self.role == "admin":
            self.frames["user_management"] = UserManagementFrame(self.main_frame, self.user_id, self.db_manager)
        
        # 默认显示个人信息页面
        self.show_frame("user_info")
    
    def show_frame(self, frame_name):
        """显示指定的页面"""
        # 隐藏所有页面
        for frame in self.frames.values():
            frame.pack_forget()
        
        # 显示指定页面
        self.frames[frame_name].pack(fill="both", expand=True)
        
        # 刷新显示
        if frame_name == "history":
            self.frames[frame_name].refresh_history()
    
    def apply_settings(self, theme, color_theme):
        """应用设置变更"""
        ctk.set_appearance_mode(theme)
        ctk.set_default_color_theme(color_theme)
        
        # 保存设置到数据库
        self.db_manager.update_user_settings(self.user_id, theme, color_theme)
        
        # 更新当前设置
        self.user_settings["theme"] = theme
        self.user_settings["color_theme"] = color_theme
    
    def on_closing(self):
        """窗口关闭前执行的操作"""
        try:
            # 数据库连接仍然保持打开，以便登录窗口可以继续使用
            logger.info(f"用户 {self.username} 已退出登录")
            
            if self.login_window:
                # 关闭主应用并返回登录窗口
                self.destroy()
                self.login_window.deiconify()  # 显示登录窗口
                # 重置登录窗口的输入
                if hasattr(self.login_window, 'username_entry'):
                    self.login_window.username_entry.delete(0, tk.END)
                if hasattr(self.login_window, 'password_entry'):
                    self.login_window.password_entry.delete(0, tk.END)
            else:
                # 如果没有登录窗口引用，则关闭数据库连接并退出
                self.db_manager.close()
                self.destroy()
        except Exception as e:
            logger.error(f"关闭应用时出错: {str(e)}")
            self.destroy() 