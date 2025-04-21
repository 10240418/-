import os
import sys
import logging
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

# 设置日志
os.makedirs('logs', exist_ok=True)
os.makedirs('history', exist_ok=True)

logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('baijiu_app')

# 导入数据库管理器
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.db_manager import DatabaseManager

# 设置CustomTkinter外观
ctk.set_appearance_mode("dark")  # 默认为深色模式
ctk.set_default_color_theme("blue")  # 默认颜色主题

class LoginWindow(ctk.CTk):
    """登录窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 配置窗口
        self.title("白酒品质检测系统 - 登录")
        self.geometry("400x550")  # 稍微增加窗口高度以容纳注册按钮
        self.resizable(False, False)
        
        # 初始化数据库管理器
        try:
            self.db_manager = DatabaseManager()
            logger.info("数据库连接成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            self.show_error("数据库连接失败", f"无法连接到数据库: {str(e)}")
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建登录窗口UI组件"""
        
        # 标题标签
        self.title_label = ctk.CTkLabel(
            self, 
            text="白酒品质检测系统", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(50, 30))
        
        # 创建框架
        self.login_frame = ctk.CTkFrame(self, width=300, height=260)
        self.login_frame.pack(pady=20, padx=40, fill="both", expand=True)
        
        # 登录标签
        self.login_label = ctk.CTkLabel(
            self.login_frame, 
            text="用户登录", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.login_label.pack(pady=(20, 20))
        
        # 用户名输入
        self.username_label = ctk.CTkLabel(self.login_frame, text="用户名:")
        self.username_label.pack(pady=(10, 0), padx=10, anchor="w")
        
        self.username_entry = ctk.CTkEntry(self.login_frame, width=220, placeholder_text="请输入用户名")
        self.username_entry.insert(0, "admin")  # 默认设置用户名为admin
        self.username_entry.pack(pady=(5, 10), padx=10)
        
        # 密码输入
        self.password_label = ctk.CTkLabel(self.login_frame, text="密码:")
        self.password_label.pack(pady=(10, 0), padx=10, anchor="w")
        
        self.password_entry = ctk.CTkEntry(self.login_frame, width=220, placeholder_text="请输入密码", show="*")
        self.password_entry.insert(0, "admin123")  # 默认设置密码为admin123
        self.password_entry.pack(pady=(5, 15), padx=10)
        
        # 登录按钮
        self.login_button = ctk.CTkButton(
            self.login_frame, 
            text="登录", 
            width=220, 
            height=32,
            command=self.login
        )
        self.login_button.pack(pady=(10, 10), padx=10)
        
        # 注册按钮
        self.register_button = ctk.CTkButton(
            self.login_frame, 
            text="注册新用户", 
            width=220, 
            height=32,
            fg_color="transparent",  # 透明背景
            text_color=("gray10", "#DCE4EE"),  # 适应深色/浅色模式
            border_width=2,  # 添加边框
            hover_color=("gray70", "gray30"),  # 悬停颜色
            command=self.open_register_window
        )
        self.register_button.pack(pady=(5, 10), padx=10)
        
        # 版权信息
        self.copyright_label = ctk.CTkLabel(
            self, 
            text="© 2024 四川农业大学 白酒品质检测系统",
            font=ctk.CTkFont(size=10)
        )
        self.copyright_label.pack(pady=(10, 20))
    
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
                # 打开主应用窗口
                app = MainApplication(user_id, username, role, self.db_manager)
                app.mainloop()
                self.destroy()  # 关闭登录窗口
            else:
                logger.warning(f"用户 {username} 登录失败：用户名或密码错误")
                self.show_error("登录失败", "用户名或密码错误")
        except Exception as e:
            logger.error(f"登录过程中出错: {str(e)}")
            self.show_error("登录错误", f"登录过程中出错: {str(e)}")
    
    def open_register_window(self):
        """打开注册窗口"""
        register_window = RegisterWindow(self, self.db_manager)
        register_window.grab_set()  # 设置为模态窗口
    
    def show_error(self, title, message):
        """显示错误对话框"""
        error_window = ctk.CTkToplevel(self)
        error_window.title(title)
        error_window.geometry("300x200")
        error_window.resizable(False, False)
        
        # 设置为模态窗口
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
        
        # 设置焦点
        error_window.focus_set()


class MainApplication(ctk.CTk):
    """主应用程序窗口类"""
    
    def __init__(self, user_id, username, role, db_manager):
        super().__init__()
        
        # 保存用户信息
        self.user_id = user_id
        self.username = username
        self.role = role
        self.db_manager = db_manager
        
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
            text="退出", 
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
            self.db_manager.close()
            logger.info(f"用户 {self.username} 已关闭应用，数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接时出错: {str(e)}")
        
        self.destroy()


class UserInfoFrame(ctk.CTkFrame):
    """用户个人信息页面"""
    
    def __init__(self, parent, user_id, username, role, db_manager):
        super().__init__(parent, corner_radius=0)
        
        self.user_id = user_id
        self.username = username
        self.role = role
        self.db_manager = db_manager
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建用户信息页面组件"""
        
        # 页面标题
        self.title_label = ctk.CTkLabel(
            self, 
            text="个人信息", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 30), padx=30, anchor="w")
        
        # 用户信息容器
        self.info_frame = ctk.CTkFrame(self)
        self.info_frame.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # 用户ID信息
        self.id_label = ctk.CTkLabel(
            self.info_frame, 
            text="用户ID:", 
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        self.id_label.grid(row=0, column=0, padx=(20, 10), pady=(30, 10), sticky="w")
        
        self.id_value = ctk.CTkLabel(
            self.info_frame, 
            text=str(self.user_id),
            font=ctk.CTkFont(size=16),
            anchor="w"
        )
        self.id_value.grid(row=0, column=1, padx=10, pady=(30, 10), sticky="w")
        
        # 用户名信息
        self.username_label = ctk.CTkLabel(
            self.info_frame, 
            text="用户名:", 
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        self.username_label.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="w")
        
        self.username_value = ctk.CTkLabel(
            self.info_frame, 
            text=self.username,
            font=ctk.CTkFont(size=16),
            anchor="w"
        )
        self.username_value.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        # 用户角色信息
        self.role_label = ctk.CTkLabel(
            self.info_frame, 
            text="用户角色:", 
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        self.role_label.grid(row=2, column=0, padx=(20, 10), pady=10, sticky="w")
        
        self.role_value = ctk.CTkLabel(
            self.info_frame, 
            text=self.role,
            font=ctk.CTkFont(size=16),
            anchor="w"
        )
        self.role_value.grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        # 配置网格
        self.info_frame.grid_columnconfigure(0, weight=0)
        self.info_frame.grid_columnconfigure(1, weight=1)


class AnalysisFrame(ctk.CTkFrame):
    """掺伪量分析预测页面"""
    
    def __init__(self, parent, user_id, db_manager):
        super().__init__(parent, corner_radius=0)
        
        self.user_id = user_id
        self.db_manager = db_manager
        self.file_path = None
        self.analysis_result = None
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建掺伪量分析页面组件"""
        
        # 页面标题
        self.title_label = ctk.CTkLabel(
            self, 
            text="掺伪量分析预测", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 20), padx=30, anchor="w")
        
        # 文件选择区域
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(fill="x", padx=30, pady=(0, 20))
        
        self.file_label = ctk.CTkLabel(
            self.file_frame, 
            text="请选择CSV或Excel格式的光谱数据文件:",
            font=ctk.CTkFont(size=14)
        )
        self.file_label.pack(side="left", padx=10, pady=10)
        
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ctk.CTkEntry(
            self.file_frame, 
            width=300,
            textvariable=self.file_path_var,
            state="readonly"
        )
        self.file_path_entry.pack(side="left", padx=10, pady=10)
        
        self.browse_button = ctk.CTkButton(
            self.file_frame, 
            text="浏览", 
            width=80,
            command=self.browse_file
        )
        self.browse_button.pack(side="left", padx=10, pady=10)
        
        # 分析按钮
        self.analyze_button = ctk.CTkButton(
            self, 
            text="开始分析", 
            width=150,
            height=40,
            state="disabled",
            command=self.analyze_file
        )
        self.analyze_button.pack(pady=(0, 20))
        
        # 分析结果区域
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # 结果标签
        self.result_title = ctk.CTkLabel(
            self.result_frame, 
            text="分析结果", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.result_title.pack(pady=(20, 10))
        
        # 结果值
        self.result_value = ctk.CTkLabel(
            self.result_frame, 
            text="请上传文件并开始分析",
            font=ctk.CTkFont(size=16)
        )
        self.result_value.pack(pady=(0, 20))
        
        # 为图表创建一个框架
        self.chart_frame = ctk.CTkFrame(self.result_frame)
        self.chart_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # 保存结果按钮
        self.save_button = ctk.CTkButton(
            self.result_frame, 
            text="保存结果", 
            width=150,
            state="disabled",
            command=self.save_result
        )
        self.save_button.pack(pady=(0, 20))
    
    def browse_file(self):
        """选择文件"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.file_path = file_path
            self.file_path_var.set(file_path)
            self.analyze_button.configure(state="normal")
            logger.info(f"用户选择了文件: {file_path}")
    
    def analyze_file(self):
        """分析文件数据"""
        if not self.file_path:
            return
        
        try:
            # 显示进度条
            self.show_progress("正在分析文件，请稍候...")
            
            # 创建一个线程来执行分析，避免UI阻塞
            threading.Thread(target=self._perform_analysis, daemon=True).start()
        except Exception as e:
            logger.error(f"分析文件时出错: {str(e)}")
            self.show_error("分析错误", f"无法分析文件: {str(e)}")
    
    def _perform_analysis(self):
        """执行文件分析（在单独的线程中运行）"""
        try:
            # 导入数据
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(self.file_path, header=None)
            elif file_ext == '.csv':
                data = pd.read_csv(self.file_path, header=None)
            else:
                raise ValueError("不支持的文件格式")
            
            # 提取特征（所有数据作为特征）
            X = data.values
            
            # 数据预处理逻辑（实际应用中根据需要修改）
            # 这里仅作为示例，实际项目中需要根据test_tool.py中的具体逻辑实现
            
            # 假设的分析过程 (将来应当整合实际的机器学习模型)
            # 简化示例: 假设结果是工业酒精的体积百分比
            import random
            analysis_result = random.uniform(0.05, 0.5)
            
            # 记录结果到数据库
            result_str = f"工业酒精体积占比: {analysis_result:.4f} ({analysis_result * 100:.2f}%)"
            self.db_manager.add_analysis_record(
                self.user_id, 
                os.path.basename(self.file_path), 
                result_str,
                "掺伪量分析"
            )
            
            # 保存结果
            self.analysis_result = {
                'result': analysis_result,
                'filename': os.path.basename(self.file_path),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 更新UI（从主线程）
            self.after(0, self.update_results_ui)
            
        except Exception as e:
            logger.error(f"分析过程中出错: {str(e)}")
            self.after(0, lambda: self.show_error("分析错误", f"分析过程中出错: {str(e)}"))
            self.after(0, self.hide_progress)
    
    def update_results_ui(self):
        """更新结果UI"""
        # 隐藏进度条
        self.hide_progress()
        
        # 显示结果
        self.result_value.configure(
            text=f"工业酒精体积占比: {self.analysis_result['result']:.4f} ({self.analysis_result['result'] * 100:.2f}%)"
        )
        
        # 启用保存按钮
        self.save_button.configure(state="normal")
        
        # 绘制图表
        self.plot_chart()
    
    def plot_chart(self):
        """绘制分析结果图表"""
        # 清除旧图表
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # 假设数据（将来应该使用实际数据）
        labels = ['标准白酒', '样品']
        std_value = 0.00  # 标准白酒中工业酒精含量为0
        sample_value = self.analysis_result['result']
        values = [std_value, sample_value]
        
        # 设置条形图颜色
        colors = ['#2ca02c', '#d62728' if sample_value > 0.1 else '#1f77b4']
        
        # 绘制条形图
        bars = ax.bar(labels, values, color=colors, width=0.4)
        
        # 在条形上方显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom')
        
        # 设置图表标题和标签
        ax.set_title('白酒样品掺伪度分析')
        ax.set_ylabel('工业酒精体积占比')
        ax.set_ylim(0, max(values) * 1.2)  # 设置Y轴范围，留出空间显示数值
        
        # 添加警示线
        if sample_value > 0.1:
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.7)
            ax.text(0.5, 0.1 + 0.01, '警戒线 (10%)', color='red', ha='center')
        
        # 将图表嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def save_result(self):
        """保存分析结果到文件"""
        if not self.analysis_result:
            return
        
        try:
            # 选择保存路径
            save_path = filedialog.asksaveasfilename(
                title="保存分析结果",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
                initialdir="history"
            )
            
            if not save_path:
                return
            
            # 写入结果文件
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("白酒品质检测系统 - 掺伪量分析结果\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"分析时间: {self.analysis_result['datetime']}\n")
                f.write(f"文件名: {self.analysis_result['filename']}\n")
                f.write(f"工业酒精体积占比: {self.analysis_result['result']:.4f} ({self.analysis_result['result'] * 100:.2f}%)\n")
                
                # 添加结论
                if self.analysis_result['result'] > 0.1:
                    f.write("\n结论: 疑似假酒，工业酒精含量超过警戒值10%\n")
                else:
                    f.write("\n结论: 在可接受范围内\n")
            
            logger.info(f"分析结果已保存至: {save_path}")
            self.show_info("保存成功", f"分析结果已保存至: {save_path}")
            
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            self.show_error("保存错误", f"无法保存结果: {str(e)}")
    
    def show_progress(self, message):
        """显示进度窗口"""
        self.progress_window = ctk.CTkToplevel(self)
        self.progress_window.title("处理中")
        self.progress_window.geometry("300x150")
        self.progress_window.resizable(False, False)
        self.progress_window.transient(self)
        self.progress_window.grab_set()
        
        # 居中显示
        self.progress_window.update_idletasks()
        width = self.progress_window.winfo_width()
        height = self.progress_window.winfo_height()
        x = (self.progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.progress_window.winfo_screenheight() // 2) - (height // 2)
        self.progress_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # 进度消息
        message_label = ctk.CTkLabel(
            self.progress_window, 
            text=message,
            font=ctk.CTkFont(size=14)
        )
        message_label.pack(pady=(30, 20))
        
        # 进度条
        self.progress_bar = ctk.CTkProgressBar(self.progress_window, width=200)
        self.progress_bar.pack(pady=(0, 20))
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
    
    def hide_progress(self):
        """隐藏进度窗口"""
        if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
            self.progress_window.grab_release()
            self.progress_window.destroy()
    
    def show_error(self, title, message):
        """显示错误对话框"""
        if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
            self.progress_window.grab_release()
            self.progress_window.destroy()
            
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


class ClassificationFrame(ctk.CTkFrame):
    """白酒真伪分类页面"""
    
    def __init__(self, parent, user_id, db_manager):
        super().__init__(parent, corner_radius=0)
        
        self.user_id = user_id
        self.db_manager = db_manager
        self.file_path = None
        self.classification_result = None
        
        # 定义分类类别
        self.categories = {
            0: "真酒",
            1: "掺水假酒",
            2: "掺工业酒精假酒"
        }
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建白酒真伪分类页面组件"""
        
        # 页面标题
        self.title_label = ctk.CTkLabel(
            self, 
            text="白酒真伪分类", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 20), padx=30, anchor="w")
        
        # 文件选择区域
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(fill="x", padx=30, pady=(0, 20))
        
        self.file_label = ctk.CTkLabel(
            self.file_frame, 
            text="请选择CSV或Excel格式的光谱数据文件:",
            font=ctk.CTkFont(size=14)
        )
        self.file_label.pack(side="left", padx=10, pady=10)
        
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ctk.CTkEntry(
            self.file_frame, 
            width=300,
            textvariable=self.file_path_var,
            state="readonly"
        )
        self.file_path_entry.pack(side="left", padx=10, pady=10)
        
        self.browse_button = ctk.CTkButton(
            self.file_frame, 
            text="浏览", 
            width=80,
            command=self.browse_file
        )
        self.browse_button.pack(side="left", padx=10, pady=10)
        
        # 分类按钮
        self.classify_button = ctk.CTkButton(
            self, 
            text="开始分类", 
            width=150,
            height=40,
            state="disabled",
            command=self.classify_file
        )
        self.classify_button.pack(pady=(0, 20))
        
        # 分类结果区域
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # 结果标签
        self.result_title = ctk.CTkLabel(
            self.result_frame, 
            text="分类结果", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.result_title.pack(pady=(20, 10))
        
        # 结果值
        self.result_value = ctk.CTkLabel(
            self.result_frame, 
            text="请上传文件并开始分类",
            font=ctk.CTkFont(size=16)
        )
        self.result_value.pack(pady=(0, 20))
        
        # 分类结果详情区域
        self.details_frame = ctk.CTkFrame(self.result_frame)
        self.details_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # 保存结果按钮
        self.save_button = ctk.CTkButton(
            self.result_frame, 
            text="保存结果", 
            width=150,
            state="disabled",
            command=self.save_result
        )
        self.save_button.pack(pady=(0, 20))
    
    def browse_file(self):
        """选择文件"""
        file_path = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.file_path = file_path
            self.file_path_var.set(file_path)
            self.classify_button.configure(state="normal")
            logger.info(f"用户选择了文件: {file_path}")
    
    def classify_file(self):
        """分类文件数据"""
        if not self.file_path:
            return
        
        try:
            # 显示进度条
            self.show_progress("正在分类文件，请稍候...")
            
            # 创建一个线程来执行分类，避免UI阻塞
            threading.Thread(target=self._perform_classification, daemon=True).start()
        except Exception as e:
            logger.error(f"分类文件时出错: {str(e)}")
            self.show_error("分类错误", f"无法分类文件: {str(e)}")
    
    def _perform_classification(self):
        """执行文件分类（在单独的线程中运行）"""
        try:
            # 导入数据
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(self.file_path, header=None)
            elif file_ext == '.csv':
                data = pd.read_csv(self.file_path, header=None)
            else:
                raise ValueError("不支持的文件格式")
            
            # 提取特征（所有数据作为特征）
            X = data.values
            
            # 数据预处理逻辑（实际应用中根据需要修改）
            # 这里仅作为示例，实际项目中需要根据classify_tool.py中的具体逻辑实现
            
            # 假设的分类过程 (将来应当整合实际的机器学习模型)
            import random
            class_id = random.randint(0, 2)  # 随机选择一个类别作为示例
            class_name = self.categories[class_id]
            
            # 生成置信度分数（仅用于示例）
            confidence_scores = {}
            for i in range(3):
                if i == class_id:
                    confidence_scores[i] = random.uniform(0.7, 0.95)  # 预测类别的置信度较高
                else:
                    confidence_scores[i] = random.uniform(0.05, 0.3)  # 其他类别的置信度较低
            
            # 记录结果到数据库
            result_str = f"分类结果: {class_name} (置信度: {confidence_scores[class_id]:.2f})"
            self.db_manager.add_analysis_record(
                self.user_id, 
                os.path.basename(self.file_path), 
                result_str,
                "白酒真伪分类"
            )
            
            # 保存结果
            self.classification_result = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence_scores': confidence_scores,
                'filename': os.path.basename(self.file_path),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 更新UI（从主线程）
            self.after(0, self.update_results_ui)
            
        except Exception as e:
            logger.error(f"分类过程中出错: {str(e)}")
            self.after(0, lambda: self.show_error("分类错误", f"分类过程中出错: {str(e)}"))
            self.after(0, self.hide_progress)
    
    def update_results_ui(self):
        """更新结果UI"""
        # 隐藏进度条
        self.hide_progress()
        
        # 获取结果
        class_name = self.classification_result['class_name']
        confidence = self.classification_result['confidence_scores'][self.classification_result['class_id']]
        
        # 显示结果
        self.result_value.configure(
            text=f"分类结果: {class_name} (置信度: {confidence:.2f})",
            text_color=self.get_result_color(self.classification_result['class_id'])
        )
        
        # 启用保存按钮
        self.save_button.configure(state="normal")
        
        # 显示详细结果
        self.show_detailed_results()
    
    def get_result_color(self, class_id):
        """根据分类结果选择颜色"""
        colors = {
            0: "green",    # 真酒
            1: "orange",   # 掺水假酒
            2: "red"       # 掺工业酒精假酒
        }
        return colors.get(class_id, "white")
    
    def show_detailed_results(self):
        """显示详细的分类结果"""
        # 清除旧结果
        for widget in self.details_frame.winfo_children():
            widget.destroy()
        
        # 创建置信度条形显示
        confidence_scores = self.classification_result['confidence_scores']
        
        # 结果标题
        results_label = ctk.CTkLabel(
            self.details_frame,
            text="各类别置信度:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        results_label.pack(pady=(10, 15), anchor="w")
        
        # 为每个类别创建置信度显示
        for i, category in self.categories.items():
            # 创建框架
            category_frame = ctk.CTkFrame(self.details_frame)
            category_frame.pack(fill="x", pady=5)
            
            # 类别名称
            category_label = ctk.CTkLabel(
                category_frame,
                text=f"{category}:",
                width=120,
                anchor="w"
            )
            category_label.pack(side="left", padx=(10, 5))
            
            # 置信度条
            confidence = confidence_scores[i]
            
            # 进度条颜色
            progress_color = self.get_result_color(i)
            
            confidence_bar = ctk.CTkProgressBar(
                category_frame,
                width=300,
                height=15
            )
            confidence_bar.pack(side="left", padx=5)
            confidence_bar.set(confidence)  # 设置进度条值（0-1之间）
            confidence_bar.configure(progress_color=progress_color)
            
            # 置信度值
            confidence_value = ctk.CTkLabel(
                category_frame,
                text=f"{confidence:.2f}",
                width=50
            )
            confidence_value.pack(side="left", padx=5)
        
        # 添加分类结论
        conclusion_frame = ctk.CTkFrame(self.details_frame)
        conclusion_frame.pack(fill="x", pady=(20, 10))
        
        conclusion_label = ctk.CTkLabel(
            conclusion_frame,
            text="结论:",
            font=ctk.CTkFont(size=14, weight="bold"),
            width=50,
            anchor="w"
        )
        conclusion_label.pack(side="left", padx=10)
        
        # 根据分类结果生成结论
        class_id = self.classification_result['class_id']
        if class_id == 0:
            conclusion_text = "此样品为真酒，无掺假成分。"
        elif class_id == 1:
            conclusion_text = "此样品疑似掺水假酒，请注意。"
        else:  # class_id == 2
            conclusion_text = "此样品疑似掺工业酒精假酒，不建议饮用！"
        
        conclusion_value = ctk.CTkLabel(
            conclusion_frame,
            text=conclusion_text,
            font=ctk.CTkFont(weight="bold"),
            text_color=self.get_result_color(class_id)
        )
        conclusion_value.pack(side="left", padx=5)
    
    def save_result(self):
        """保存分类结果到文件"""
        if not self.classification_result:
            return
        
        try:
            # 选择保存路径
            save_path = filedialog.asksaveasfilename(
                title="保存分类结果",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")],
                initialdir="history"
            )
            
            if not save_path:
                return
            
            # 写入结果文件
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("白酒品质检测系统 - 白酒真伪分类结果\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"分析时间: {self.classification_result['datetime']}\n")
                f.write(f"文件名: {self.classification_result['filename']}\n\n")
                
                # 写入分类结果
                class_id = self.classification_result['class_id']
                class_name = self.classification_result['class_name']
                confidence = self.classification_result['confidence_scores'][class_id]
                
                f.write(f"分类结果: {class_name}\n")
                f.write(f"置信度: {confidence:.2f}\n\n")
                
                # 写入所有类别的置信度
                f.write("各类别置信度:\n")
                for i, category in self.categories.items():
                    score = self.classification_result['confidence_scores'][i]
                    f.write(f"  {category}: {score:.2f}\n")
                
                # 写入结论
                f.write("\n结论: ")
                if class_id == 0:
                    f.write("此样品为真酒，无掺假成分。\n")
                elif class_id == 1:
                    f.write("此样品疑似掺水假酒，请注意。\n")
                else:  # class_id == 2
                    f.write("此样品疑似掺工业酒精假酒，不建议饮用！\n")
            
            logger.info(f"分类结果已保存至: {save_path}")
            self.show_info("保存成功", f"分类结果已保存至: {save_path}")
            
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            self.show_error("保存错误", f"无法保存结果: {str(e)}")
    
    def show_progress(self, message):
        """显示进度窗口"""
        self.progress_window = ctk.CTkToplevel(self)
        self.progress_window.title("处理中")
        self.progress_window.geometry("300x150")
        self.progress_window.resizable(False, False)
        self.progress_window.transient(self)
        self.progress_window.grab_set()
        
        # 居中显示
        self.progress_window.update_idletasks()
        width = self.progress_window.winfo_width()
        height = self.progress_window.winfo_height()
        x = (self.progress_window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.progress_window.winfo_screenheight() // 2) - (height // 2)
        self.progress_window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
        # 进度消息
        message_label = ctk.CTkLabel(
            self.progress_window, 
            text=message,
            font=ctk.CTkFont(size=14)
        )
        message_label.pack(pady=(30, 20))
        
        # 进度条
        self.progress_bar = ctk.CTkProgressBar(self.progress_window, width=200)
        self.progress_bar.pack(pady=(0, 20))
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()
    
    def hide_progress(self):
        """隐藏进度窗口"""
        if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
            self.progress_window.grab_release()
            self.progress_window.destroy()
    
    def show_error(self, title, message):
        """显示错误对话框"""
        if hasattr(self, 'progress_window') and self.progress_window.winfo_exists():
            self.progress_window.grab_release()
            self.progress_window.destroy()
            
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


class HistoryFrame(ctk.CTkFrame):
    """历史记录页面"""
    
    def __init__(self, parent, user_id, db_manager):
        super().__init__(parent, corner_radius=0)
        
        self.user_id = user_id
        self.db_manager = db_manager
        self.history_records = []
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建历史记录页面组件"""
        
        # 页面标题
        self.title_label = ctk.CTkLabel(
            self, 
            text="历史记录", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 20), padx=30, anchor="w")
        
        # 操作按钮框架
        self.action_frame = ctk.CTkFrame(self)
        self.action_frame.pack(fill="x", padx=30, pady=(0, 20))
        
        # 刷新按钮
        self.refresh_button = ctk.CTkButton(
            self.action_frame, 
            text="刷新历史记录", 
            width=150,
            command=self.refresh_history
        )
        self.refresh_button.pack(side="left", padx=10, pady=10)
        
        # 导出按钮
        self.export_button = ctk.CTkButton(
            self.action_frame, 
            text="导出所有记录", 
            width=150,
            command=self.export_history
        )
        self.export_button.pack(side="left", padx=10, pady=10)
        
        # 历史记录表格框架
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # 创建表格标题
        self.create_table_header()
        
        # 创建表格内容框架（使用滚动视图）
        self.table_content_frame = ctk.CTkScrollableFrame(self.table_frame)
        self.table_content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # 加载历史记录
        self.refresh_history()
    
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
        
        # 文件名列
        filename_label = ctk.CTkLabel(
            header_frame, 
            text="文件名", 
            font=ctk.CTkFont(weight="bold"),
            width=150
        )
        filename_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # 结果列
        result_label = ctk.CTkLabel(
            header_frame, 
            text="分析结果", 
            font=ctk.CTkFont(weight="bold"),
            width=250
        )
        result_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        # 分析类型列
        type_label = ctk.CTkLabel(
            header_frame, 
            text="分析类型", 
            font=ctk.CTkFont(weight="bold"),
            width=120
        )
        type_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # 时间列
        time_label = ctk.CTkLabel(
            header_frame, 
            text="分析时间", 
            font=ctk.CTkFont(weight="bold"),
            width=150
        )
        time_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        
        # 配置网格
        for i in range(5):
            header_frame.grid_columnconfigure(i, weight=1)
    
    def refresh_history(self):
        """刷新历史记录"""
        try:
            # 清除旧记录
            for widget in self.table_content_frame.winfo_children():
                widget.destroy()
            
            # 获取历史记录
            self.history_records = self.db_manager.get_user_history(self.user_id)
            
            if not self.history_records:
                # 显示无记录提示
                no_data_label = ctk.CTkLabel(
                    self.table_content_frame, 
                    text="暂无历史记录",
                    font=ctk.CTkFont(size=14)
                )
                no_data_label.pack(pady=50)
                return
            
            # 显示历史记录
            for i, record in enumerate(self.history_records):
                record_id, filename, result, analysis_type, created_at = record
                
                # 创建记录行
                row_frame = ctk.CTkFrame(self.table_content_frame)
                row_frame.pack(fill="x", pady=2)
                
                # 设置行背景颜色（奇偶行不同颜色）
                if i % 2 == 0:
                    row_frame.configure(fg_color=("gray90", "gray20"))
                
                # ID列
                id_label = ctk.CTkLabel(
                    row_frame, 
                    text=str(record_id),
                    width=50
                )
                id_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
                
                # 文件名列
                filename_label = ctk.CTkLabel(
                    row_frame, 
                    text=filename,
                    width=150
                )
                filename_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
                
                # 结果列
                result_label = ctk.CTkLabel(
                    row_frame, 
                    text=result,
                    width=250
                )
                result_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
                
                # 分析类型列
                type_label = ctk.CTkLabel(
                    row_frame, 
                    text=analysis_type,
                    width=120
                )
                type_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")
                
                # 时间列
                time_label = ctk.CTkLabel(
                    row_frame, 
                    text=created_at,
                    width=150
                )
                time_label.grid(row=0, column=4, padx=5, pady=5, sticky="w")
                
                # 配置网格
                for j in range(5):
                    row_frame.grid_columnconfigure(j, weight=1)
            
            logger.info(f"已加载 {len(self.history_records)} 条历史记录")
            
        except Exception as e:
            logger.error(f"加载历史记录时出错: {str(e)}")
            self.show_error("加载错误", f"无法加载历史记录: {str(e)}")
    
    def export_history(self):
        """导出历史记录"""
        if not self.history_records:
            self.show_info("导出提示", "没有历史记录可导出")
            return
        
        try:
            # 选择保存路径
            save_path = filedialog.asksaveasfilename(
                title="导出历史记录",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv"), ("所有文件", "*.*")],
                initialdir="history"
            )
            
            if not save_path:
                return
            
            # 根据文件扩展名选择导出格式
            file_ext = os.path.splitext(save_path)[1].lower()
            
            if file_ext == ".csv":
                # 导出为CSV
                with open(save_path, 'w', encoding='utf-8', newline='') as f:
                    f.write("ID,文件名,分析结果,分析类型,分析时间\n")
                    for record in self.history_records:
                        record_id, filename, result, analysis_type, created_at = record
                        # 处理结果中可能出现的逗号
                        result = f'"{result}"'
                        f.write(f"{record_id},{filename},{result},{analysis_type},{created_at}\n")
            else:
                # 导出为TXT
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write("白酒品质检测系统 - 历史记录\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write(f"{'ID':<6} {'分析类型':<15} {'分析时间':<20} {'文件名':<30} {'分析结果'}\n")
                    f.write("-" * 100 + "\n")
                    
                    for record in self.history_records:
                        record_id, filename, result, analysis_type, created_at = record
                        f.write(f"{record_id:<6} {analysis_type:<15} {created_at:<20} {filename:<30} {result}\n")
            
            logger.info(f"历史记录已导出至: {save_path}")
            self.show_info("导出成功", f"历史记录已导出至: {save_path}")
            
        except Exception as e:
            logger.error(f"导出历史记录时出错: {str(e)}")
            self.show_error("导出错误", f"无法导出历史记录: {str(e)}")
    
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
                self.theme_var.set(value)
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
            "dark-blue": "深蓝主题",
            "sweetkind": "甜心主题"
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
                self.color_var.set(value)
                break
        
        # 分隔线
        self.separator = ctk.CTkFrame(self.settings_frame, height=1)
        self.separator.pack(fill="x", padx=20, pady=20)
        
        # 应用按钮
        self.apply_button = ctk.CTkButton(
            self.settings_frame, 
            text="应用设置", 
            width=150,
            command=self.apply_settings
        )
        self.apply_button.pack(pady=(10, 20))
    
    def on_theme_change(self, option):
        """主题变更回调"""
        # 找到对应的主题键
        theme_key = next((k for k, v in self.theme_options.items() if v == option), "dark")
        self.user_settings["theme"] = theme_key
    
    def on_color_change(self, option):
        """颜色主题变更回调"""
        # 找到对应的颜色主题键
        color_key = next((k for k, v in self.color_options.items() if v == option), "blue")
        self.user_settings["color_theme"] = color_key
    
    def apply_settings(self):
        """应用设置"""
        try:
            # 应用主题设置
            self.apply_callback(
                self.user_settings["theme"],
                self.user_settings["color_theme"]
            )
            
            logger.info(f"用户 {self.user_id} 更新了设置: theme={self.user_settings['theme']}, color_theme={self.user_settings['color_theme']}")
            
            # 显示成功消息
            self.show_info("设置已应用", "应用设置成功！")
            
        except Exception as e:
            logger.error(f"应用设置时出错: {str(e)}")
            self.show_error("设置错误", f"应用设置失败: {str(e)}")
    
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


class UserManagementFrame(ctk.CTkFrame):
    """用户管理界面"""
    
    def __init__(self, parent, user_id, db_manager):
        super().__init__(parent, corner_radius=0)
        
        self.user_id = user_id
        self.db_manager = db_manager
        self.user_list = []
        
        # 创建UI组件
        self.create_widgets()
        
        # 初始加载用户数据
        self.refresh_users()
    
    def create_widgets(self):
        """创建用户管理界面组件"""
        
        # 页面标题
        self.title_label = ctk.CTkLabel(
            self, 
            text="用户管理", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 20), padx=30, anchor="w")
        
        # 按钮容器
        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.pack(fill="x", padx=30, pady=(0, 10))
        
        # 刷新按钮
        self.refresh_button = ctk.CTkButton(
            self.button_frame, 
            text="刷新用户列表", 
            width=150,
            command=self.refresh_users
        )
        self.refresh_button.pack(side="left", padx=10, pady=10)
        
        # 注意提示
        self.note_label = ctk.CTkLabel(
            self.button_frame, 
            text="注意：此页面仅供调试使用，显示用户敏感信息",
            text_color="red"
        )
        self.note_label.pack(side="right", padx=10, pady=10)
        
        # 创建表格容器
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(fill="both", expand=True, padx=30, pady=(0, 30))
        
        # 创建表头
        self.create_table_header()
        
        # 创建表格内容滚动区域
        self.table_scroll = ctk.CTkScrollableFrame(self.table_frame)
        self.table_scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # 用于存放表格内容的框架
        self.table_content_frame = ctk.CTkFrame(self.table_scroll, fg_color="transparent")
        self.table_content_frame.pack(fill="both", expand=True)
    
    def create_table_header(self):
        """创建表格表头"""
        # 表头容器
        header_frame = ctk.CTkFrame(self.table_frame, height=40)
        header_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        # 保证表头高度固定
        header_frame.pack_propagate(False)
        
        # 列标题
        columns = ["ID", "用户名", "密码", "邮箱", "角色", "创建时间", "最后登录"]
        column_widths = [50, 120, 120, 150, 80, 180, 180]
        
        for i, (col, width) in enumerate(zip(columns, column_widths)):
            header_label = ctk.CTkLabel(
                header_frame, 
                text=col,
                font=ctk.CTkFont(weight="bold"),
                width=width
            )
            header_label.grid(row=0, column=i, padx=5, pady=5, sticky="w")
        
        # 配置网格
        for i in range(len(columns)):
            header_frame.grid_columnconfigure(i, weight=1)
    
    def refresh_users(self):
        """刷新用户列表"""
        try:
            # 清除旧记录
            for widget in self.table_content_frame.winfo_children():
                widget.destroy()
            
            # 获取用户列表
            self.user_list = self.db_manager.get_all_users()
            
            if not self.user_list:
                # 显示无记录提示
                no_data_label = ctk.CTkLabel(
                    self.table_content_frame, 
                    text="暂无用户记录",
                    font=ctk.CTkFont(size=14)
                )
                no_data_label.pack(pady=50)
                return
            
            # 显示用户记录
            for i, user in enumerate(self.user_list):
                # 创建记录行
                row_frame = ctk.CTkFrame(self.table_content_frame)
                row_frame.pack(fill="x", pady=2)
                
                # 设置行背景颜色（奇偶行不同颜色）
                if i % 2 == 0:
                    row_frame.configure(fg_color=("gray90", "gray20"))
                
                # 字段列表
                fields = ["id", "username", "password", "email", "role", "created_at", "last_login"]
                widths = [50, 120, 120, 150, 80, 180, 180]
                
                # 显示各个字段
                for j, (field, width) in enumerate(zip(fields, widths)):
                    value = user.get(field, "")
                    if value is None:
                        value = ""
                    
                    # 为密码字段添加特殊样式
                    text_color = "red" if field == "password" else None
                    
                    field_label = ctk.CTkLabel(
                        row_frame, 
                        text=str(value),
                        width=width,
                        text_color=text_color
                    )
                    field_label.grid(row=0, column=j, padx=5, pady=5, sticky="w")
                
                # 配置网格
                for j in range(len(fields)):
                    row_frame.grid_columnconfigure(j, weight=1)
            
            logger.info(f"已加载 {len(self.user_list)} 条用户记录")
            
        except Exception as e:
            logger.error(f"加载用户记录时出错: {str(e)}")
            self.show_error("加载错误", f"无法加载用户记录: {str(e)}")
    
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


class RegisterWindow(ctk.CTkToplevel):
    """注册窗口类"""
    
    def __init__(self, parent, db_manager):
        super().__init__(parent)
        
        self.parent = parent
        self.db_manager = db_manager
        
        # 配置窗口
        self.title("白酒品质检测系统 - 用户注册")
        self.geometry("400x500")
        self.resizable(False, False)
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建注册窗口UI组件"""
        
        # 标题标签
        self.title_label = ctk.CTkLabel(
            self, 
            text="用户注册", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 20))
        
        # 创建注册框架
        self.register_frame = ctk.CTkFrame(self, width=300, height=300)
        self.register_frame.pack(pady=20, padx=40, fill="both", expand=True)
        
        # 用户名输入
        self.username_label = ctk.CTkLabel(self.register_frame, text="用户名:")
        self.username_label.pack(pady=(20, 0), padx=10, anchor="w")
        
        self.username_entry = ctk.CTkEntry(self.register_frame, width=220, placeholder_text="请输入用户名")
        self.username_entry.pack(pady=(5, 10), padx=10)
        
        # 密码输入
        self.password_label = ctk.CTkLabel(self.register_frame, text="密码:")
        self.password_label.pack(pady=(10, 0), padx=10, anchor="w")
        
        self.password_entry = ctk.CTkEntry(self.register_frame, width=220, placeholder_text="请输入密码", show="*")
        self.password_entry.pack(pady=(5, 10), padx=10)
        
        # 确认密码输入
        self.confirm_password_label = ctk.CTkLabel(self.register_frame, text="确认密码:")
        self.confirm_password_label.pack(pady=(10, 0), padx=10, anchor="w")
        
        self.confirm_password_entry = ctk.CTkEntry(self.register_frame, width=220, placeholder_text="请再次输入密码", show="*")
        self.confirm_password_entry.pack(pady=(5, 10), padx=10)
        
        # 电子邮件输入
        self.email_label = ctk.CTkLabel(self.register_frame, text="电子邮件 (可选):")
        self.email_label.pack(pady=(10, 0), padx=10, anchor="w")
        
        self.email_entry = ctk.CTkEntry(self.register_frame, width=220, placeholder_text="请输入电子邮件")
        self.email_entry.pack(pady=(5, 20), padx=10)
        
        # 注册按钮
        self.register_button = ctk.CTkButton(
            self.register_frame, 
            text="注册", 
            width=220, 
            height=32,
            command=self.register
        )
        self.register_button.pack(pady=(10, 10), padx=10)
        
        # 取消按钮
        self.cancel_button = ctk.CTkButton(
            self.register_frame, 
            text="取消", 
            width=220, 
            height=32,
            fg_color="#D35B58",
            hover_color="#C77C78",
            command=self.destroy
        )
        self.cancel_button.pack(pady=(5, 10), padx=10)
    
    def register(self):
        """处理注册逻辑"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        confirm_password = self.confirm_password_entry.get()
        email = self.email_entry.get()
        
        # 验证输入
        if not username or not password or not confirm_password:
            self.show_error("注册失败", "用户名和密码不能为空")
            return
        
        if password != confirm_password:
            self.show_error("注册失败", "两次输入的密码不一致")
            return
        
        if len(password) < 6:
            self.show_error("注册失败", "密码长度不能少于6个字符")
            return
        
        try:
            # 检查用户名是否已存在
            if self.db_manager.check_username_exists(username):
                self.show_error("注册失败", "该用户名已被使用")
                return
            
            # 添加新用户
            user_id = self.db_manager.add_user(username, password, email)
            
            if user_id:
                logger.info(f"用户 {username} 注册成功")
                self.show_success("注册成功", "您已成功注册，现在可以登录系统")
                self.destroy()
            else:
                logger.warning(f"用户 {username} 注册失败")
                self.show_error("注册失败", "无法创建新用户")
        except Exception as e:
            logger.error(f"注册过程中出错: {str(e)}")
            self.show_error("注册错误", f"注册过程中出错: {str(e)}")
    
    def show_error(self, title, message):
        """显示错误对话框"""
        error_window = ctk.CTkToplevel(self)
        error_window.title(title)
        error_window.geometry("300x200")
        error_window.resizable(False, False)
        
        # 设置为模态窗口
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
    
    def show_success(self, title, message):
        """显示成功对话框"""
        success_window = ctk.CTkToplevel(self)
        success_window.title(title)
        success_window.geometry("300x200")
        success_window.resizable(False, False)
        
        # 设置为模态窗口
        success_window.transient(self)
        success_window.grab_set()
        
        # 成功图标和消息
        success_label = ctk.CTkLabel(
            success_window, 
            text="成功", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="green"
        )
        success_label.pack(pady=(20, 10))
        
        message_label = ctk.CTkLabel(
            success_window, 
            text=message,
            wraplength=250
        )
        message_label.pack(pady=10, padx=20)
        
        # 确定按钮
        ok_button = ctk.CTkButton(
            success_window, 
            text="确定", 
            width=100,
            command=success_window.destroy
        )
        ok_button.pack(pady=(10, 20))


def check_database():
    """检查并初始化数据库"""
    try:
        # 导入初始化模块
        from database.init_db import init_database
        
        # 初始化数据库
        init_result = init_database()
        
        if not init_result:
            logger.error("数据库初始化失败")
            return False
        
        return True
    except Exception as e:
        logger.error(f"检查数据库时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('history', exist_ok=True)
    
    # 检查并初始化数据库
    if not check_database():
        print("数据库初始化失败，请检查日志文件获取详细信息。")
        sys.exit(1)
    
    # 启动登录窗口
    app = LoginWindow()
    app.mainloop() 