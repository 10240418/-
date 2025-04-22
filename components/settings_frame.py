import logging
import tkinter as tk
import customtkinter as ctk

logger = logging.getLogger('baijiu_app')

class SettingsFrame(ctk.CTkFrame):
    """设置页面"""
    
    def __init__(self, parent, user_id, db_manager, apply_callback):
        super().__init__(parent, corner_radius=0)
        
        self.user_id = user_id
        self.db_manager = db_manager
        self.apply_callback = apply_callback
        
        # 获取当前设置
        self.user_settings = self.db_manager.get_user_settings(user_id)
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建设置页面组件"""
        
        # 页面标题
        self.title_label = ctk.CTkLabel(
            self, 
            text="系统设置", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 20), padx=30, anchor="w")
        
        # 设置容器
        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # 外观设置标题
        self.appearance_title = ctk.CTkLabel(
            self.settings_frame, 
            text="外观设置", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.appearance_title.pack(pady=(20, 15), padx=20, anchor="w")
        
        # 主题模式设置
        self.theme_frame = ctk.CTkFrame(self.settings_frame)
        self.theme_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        self.theme_label = ctk.CTkLabel(
            self.theme_frame, 
            text="界面主题:", 
            font=ctk.CTkFont(size=14),
            width=100,
            anchor="w"
        )
        self.theme_label.pack(side="left", padx=(20, 10), pady=10)
        
        # 主题选择
        self.theme_var = tk.StringVar(value=self.user_settings["theme"])
        
        self.theme_options = {
            "light": "浅色模式",
            "dark": "深色模式",
            "system": "跟随系统"
        }
        
        self.theme_menu = ctk.CTkOptionMenu(
            self.theme_frame,
            values=list(self.theme_options.values()),
            variable=self.theme_var,
            width=200,
            command=self.on_theme_change
        )
        self.theme_menu.pack(side="left", padx=10, pady=10)
        
        # 设置初始值
        for key, value in self.theme_options.items():
            if key == self.user_settings["theme"]:
                self.theme_menu.set(value)
                break
        
        # 颜色主题设置
        self.color_frame = ctk.CTkFrame(self.settings_frame)
        self.color_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        self.color_label = ctk.CTkLabel(
            self.color_frame, 
            text="颜色主题:", 
            font=ctk.CTkFont(size=14),
            width=100,
            anchor="w"
        )
        self.color_label.pack(side="left", padx=(20, 10), pady=10)
        
        # 颜色主题选择
        self.color_var = tk.StringVar(value=self.user_settings["color_theme"])
        
        self.color_options = {
            "blue": "蓝色主题",
            "green": "绿色主题",
            "dark-blue": "深蓝主题"
        }
        
        self.color_menu = ctk.CTkOptionMenu(
            self.color_frame,
            values=list(self.color_options.values()),
            variable=self.color_var,
            width=200,
            command=self.on_color_change
        )
        self.color_menu.pack(side="left", padx=10, pady=10)
        
        # 设置初始值
        for key, value in self.color_options.items():
            if key == self.user_settings["color_theme"]:
                self.color_menu.set(value)
                break
        
        # 应用按钮
        self.apply_frame = ctk.CTkFrame(self.settings_frame, fg_color="transparent")
        self.apply_frame.pack(pady=30)
        
        self.apply_button = ctk.CTkButton(
            self.apply_frame, 
            text="应用设置", 
            width=150,
            height=40,
            command=self.apply_settings
        )
        self.apply_button.pack()
    
    def on_theme_change(self, option):
        """主题选择变更处理"""
        # 保持选择值但不立即应用
        pass
    
    def on_color_change(self, option):
        """颜色主题选择变更处理"""
        # 保持选择值但不立即应用
        pass
    
    def apply_settings(self):
        """应用设置变更"""
        try:
            # 获取选择的主题
            theme_name = ""
            selected_theme = self.theme_menu.get()
            for key, value in self.theme_options.items():
                if value == selected_theme:
                    theme_name = key
                    break
            
            # 获取选择的颜色主题
            color_name = ""
            selected_color = self.color_menu.get()
            for key, value in self.color_options.items():
                if value == selected_color:
                    color_name = key
                    break
            
            # 如果选择有效，则应用设置
            if theme_name and color_name:
                # 调用主应用的应用设置回调函数
                self.apply_callback(theme_name, color_name)
                self.show_info("设置应用", "设置已成功应用")
            else:
                self.show_error("设置错误", "无法识别所选主题或颜色")
        
        except Exception as e:
            logger.error(f"应用设置时出错: {str(e)}")
            self.show_error("设置错误", f"应用设置时出错: {str(e)}")
    
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