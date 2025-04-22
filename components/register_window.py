import logging
import customtkinter as ctk

logger = logging.getLogger('baijiu_app')

class RegisterWindow(ctk.CTkToplevel):
    """注册窗口类"""
    
    def __init__(self, parent, db_manager):
        super().__init__(parent)
        
        # 保存父窗口和数据库管理器
        self.parent = parent
        self.db_manager = db_manager
        
        # 配置窗口
        self.title("白酒品质检测系统 - 用户注册")
        self.geometry("600x500")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.cancel)  # 关闭窗口时触发取消操作
        
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
        self.title_label.pack(pady=(30, 30))
        
        # 注册表单容器
        self.form_frame = ctk.CTkFrame(self)
        self.form_frame.pack(fill="both", expand=True, padx=50, pady=(0, 30))
        
        # 用户名
        self.username_label = ctk.CTkLabel(
            self.form_frame,
            text="用户名:",
            font=ctk.CTkFont(size=14),
            anchor="w"
        )
        self.username_label.pack(pady=(20, 0), padx=20, anchor="w")
        
        self.username_entry = ctk.CTkEntry(
            self.form_frame,
            width=400,
            height=35,
            placeholder_text="请输入用户名（3-20个字符）"
        )
        self.username_entry.pack(pady=(5, 15), padx=20)
        
        # 密码
        self.password_label = ctk.CTkLabel(
            self.form_frame,
            text="密码:",
            font=ctk.CTkFont(size=14),
            anchor="w"
        )
        self.password_label.pack(pady=(10, 0), padx=20, anchor="w")
        
        self.password_entry = ctk.CTkEntry(
            self.form_frame,
            width=400,
            height=35,
            placeholder_text="请输入密码（至少6个字符）",
            show="*"
        )
        self.password_entry.pack(pady=(5, 15), padx=20)
        
        # 确认密码
        self.confirm_label = ctk.CTkLabel(
            self.form_frame,
            text="确认密码:",
            font=ctk.CTkFont(size=14),
            anchor="w"
        )
        self.confirm_label.pack(pady=(10, 0), padx=20, anchor="w")
        
        self.confirm_entry = ctk.CTkEntry(
            self.form_frame,
            width=400,
            height=35,
            placeholder_text="请再次输入密码",
            show="*"
        )
        self.confirm_entry.pack(pady=(5, 15), padx=20)
        
        # 用户角色选择（只对新注册用户开放普通用户角色）
        self.role_label = ctk.CTkLabel(
            self.form_frame,
            text="注册为:",
            font=ctk.CTkFont(size=14),
            anchor="w"
        )
        self.role_label.pack(pady=(10, 0), padx=20, anchor="w")
        
        # 角色单选按钮组
        self.role_var = ctk.StringVar(value="user")  # 默认为普通用户
        
        self.role_frame = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        self.role_frame.pack(pady=(5, 15), padx=20, fill="x")
        
        self.user_radio = ctk.CTkRadioButton(
            self.role_frame,
            text="普通用户",
            variable=self.role_var,
            value="user"
        )
        self.user_radio.pack(side="left", padx=(20, 40))
        
        # 管理员选项（默认禁用）
        self.admin_radio = ctk.CTkRadioButton(
            self.role_frame,
            text="管理员（需要管理员审核）",
            variable=self.role_var,
            value="admin",
            state="disabled"
        )
        self.admin_radio.pack(side="left", padx=(0, 20))
        
        # 按钮区域
        self.button_frame = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        self.button_frame.pack(pady=20, fill="x")
        
        # 注册按钮
        self.register_button = ctk.CTkButton(
            self.button_frame,
            text="注册",
            width=180,
            height=40,
            font=ctk.CTkFont(size=14),
            command=self.register
        )
        self.register_button.pack(side="left", padx=20)
        
        # 取消按钮
        self.cancel_button = ctk.CTkButton(
            self.button_frame,
            text="取消",
            width=180,
            height=40,
            font=ctk.CTkFont(size=14),
            fg_color="transparent",
            text_color=("gray10", "#DCE4EE"),
            border_width=2,
            hover_color=("gray70", "gray30"),
            command=self.cancel
        )
        self.cancel_button.pack(side="right", padx=20)
    
    def register(self):
        """处理注册逻辑"""
        # 获取表单数据
        username = self.username_entry.get()
        password = self.password_entry.get()
        confirm = self.confirm_entry.get()
        role = self.role_var.get()
        
        # 表单验证
        if not username or not password or not confirm:
            self.show_error("注册失败", "所有字段都必须填写")
            return
        
        if len(username) < 3 or len(username) > 20:
            self.show_error("注册失败", "用户名长度必须在3-20个字符之间")
            return
        
        if len(password) < 6:
            self.show_error("注册失败", "密码长度必须至少6个字符")
            return
        
        if password != confirm:
            self.show_error("注册失败", "两次输入的密码不一致")
            return
        
        try:
            # 检查用户名是否已存在
            if self.db_manager.check_username_exists(username):
                self.show_error("注册失败", "用户名已被使用")
                return
            
            # 创建新用户
            user_id = self.db_manager.add_user(username, password, role)
            
            if user_id:
                logger.info(f"新用户注册成功: {username}, 角色: {role}")
                self.show_success("注册成功", f"用户 {username} 注册成功！请返回登录页面登录。")
                
                # 注册成功后返回登录窗口
                self.destroy()
                self.parent.deiconify()  # 显示登录窗口
            else:
                self.show_error("注册失败", "创建用户账户时出错")
        
        except Exception as e:
            logger.error(f"注册过程中出错: {str(e)}")
            self.show_error("注册错误", f"注册过程中出错: {str(e)}")
    
    def cancel(self):
        """取消注册"""
        self.destroy()
        self.parent.deiconify()  # 显示登录窗口
    
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
    
    def show_success(self, title, message):
        """显示成功对话框"""
        success_window = ctk.CTkToplevel(self)
        success_window.title(title)
        success_window.geometry("300x200")
        success_window.resizable(False, False)
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