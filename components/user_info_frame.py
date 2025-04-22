import customtkinter as ctk

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