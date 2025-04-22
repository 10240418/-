import os
import logging
from datetime import datetime
import customtkinter as ctk
from tkinter import filedialog

logger = logging.getLogger('baijiu_app')

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
        
        # 删除最近50条记录按钮
        self.delete_recent_button = ctk.CTkButton(
            self.action_frame, 
            text="删除最近50条记录", 
            width=150,
            fg_color="#E74C3C",  # 使用红色警示用户这是删除操作
            hover_color="#C0392B",
            command=self.delete_recent_history
        )
        self.delete_recent_button.pack(side="left", padx=10, pady=10)
        
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
    
    def delete_recent_history(self):
        """删除最近50条历史记录"""
        try:
            # 首先检查是否有记录
            if not self.history_records:
                self.show_info("删除提示", "没有历史记录可删除")
                return
                
            # 显示确认对话框
            self.show_confirmation(
                "确认删除", 
                "您确定要删除最近50条历史记录吗？\n此操作不可恢复！", 
                self._perform_delete
            )
                
        except Exception as e:
            logger.error(f"删除历史记录时出错: {str(e)}")
            self.show_error("删除错误", f"无法删除历史记录: {str(e)}")
    
    def _perform_delete(self):
        """执行删除操作"""
        try:
            # 获取要删除的记录数量（最多50条）
            count_to_delete = min(50, len(self.history_records))
            
            # 调用数据库管理器删除记录
            deleted_count = self.db_manager.delete_recent_history(self.user_id, count_to_delete)
            
            # 刷新历史记录
            self.refresh_history()
            
            # 显示成功消息
            self.show_info("删除成功", f"已成功删除 {deleted_count} 条最近的历史记录。")
            
            logger.info(f"用户 {self.user_id} 删除了 {deleted_count} 条最近的历史记录")
            
        except Exception as e:
            logger.error(f"执行删除历史记录时出错: {str(e)}")
            self.show_error("删除错误", f"执行删除操作时出错: {str(e)}")
    
    def show_confirmation(self, title, message, confirm_callback):
        """显示确认对话框"""
        confirm_window = ctk.CTkToplevel(self)
        confirm_window.title(title)
        confirm_window.geometry("350x200")
        confirm_window.resizable(False, False)
        confirm_window.transient(self)
        confirm_window.grab_set()
        
        # 警告图标和消息
        warning_label = ctk.CTkLabel(
            confirm_window, 
            text="警告", 
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#E74C3C"  # 红色文字
        )
        warning_label.pack(pady=(20, 10))
        
        message_label = ctk.CTkLabel(
            confirm_window, 
            text=message,
            wraplength=300
        )
        message_label.pack(pady=10, padx=20)
        
        # 按钮框架
        button_frame = ctk.CTkFrame(confirm_window, fg_color="transparent")
        button_frame.pack(pady=(10, 20))
        
        # 取消按钮
        cancel_button = ctk.CTkButton(
            button_frame, 
            text="取消", 
            width=100,
            fg_color="#95a5a6",  # 灰色按钮
            hover_color="#7f8c8d",
            command=confirm_window.destroy
        )
        cancel_button.pack(side="left", padx=10)
        
        # 确认按钮
        confirm_button = ctk.CTkButton(
            button_frame, 
            text="确认删除", 
            width=100,
            fg_color="#E74C3C",  # 红色按钮
            hover_color="#C0392B",
            command=lambda: [confirm_window.destroy(), confirm_callback()]
        )
        confirm_button.pack(side="left", padx=10)
    
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