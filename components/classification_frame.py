import os
import logging
import threading
import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
import customtkinter as ctk
from datetime import datetime
import pandas as pd
import numpy as np
import traceback
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tool.classify_model import ClassifierModel

# 设置日志
logger = logging.getLogger('baijiu_app')

# 尝试导入PyTorch，如果不可用则使用基于scikit-learn的模型
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    logger.info("PyTorch is available, using neural network model for classification")
except ImportError:
    TORCH_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier
    logger.warning("PyTorch not available, falling back to RandomForest model for classification")

class ClassificationFrame(ctk.CTkFrame):
    """白酒分类分析页面"""
    
    def __init__(self, parent, user_id, db_manager):
        super().__init__(parent, corner_radius=0)
        
        self.user_id = user_id
        self.db_manager = db_manager
        self.classifier = ClassifierModel()
        self.train_file_path = None
        self.sample_file_path = None
        self.analysis_result = None
        
        # 只在PyTorch可用时设置device
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 定义分类类别
        self.categories = {
            0: "真酒",
            1: "掺水假酒",
            2: "掺工业酒精假酒"
        }
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建分类分析页面组件"""
        
        # 页面标题
        self.title_label = ctk.CTkLabel(
            self, 
            text="白酒分类分析", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 10), padx=30, anchor="w")
        
        # === 文件选择区域 1: 训练数据文件 ===
        self.train_file_frame = ctk.CTkFrame(self)
        self.train_file_frame.pack(fill="x", padx=30, pady=(10, 5))
        
        self.train_file_label = ctk.CTkLabel(
            self.train_file_frame, 
            text="训练数据文件:",
            font=ctk.CTkFont(size=14)
        )
        self.train_file_label.pack(side="left", padx=10, pady=10)
        
        self.train_file_path_var = tk.StringVar()
        self.train_file_path_entry = ctk.CTkEntry(
            self.train_file_frame, 
            width=300,
            textvariable=self.train_file_path_var,
            state="readonly"
        )
        self.train_file_path_entry.pack(side="left", padx=10, pady=10)
        
        self.train_browse_button = ctk.CTkButton(
            self.train_file_frame, 
            text="浏览", 
            width=80,
            command=lambda: self._browse_file_generic(
                'train_file_path', 
                self.train_file_path_var, 
                self.train_file_path_entry, 
                "选择训练数据文件"
            )
        )
        self.train_browse_button.pack(side="left", padx=20, pady=10)
        
        # === 文件选择区域 2: 样本数据文件 ===
        self.sample_file_frame = ctk.CTkFrame(self)
        self.sample_file_frame.pack(fill="x", padx=30, pady=(5, 5))
        
        self.sample_file_label = ctk.CTkLabel(
            self.sample_file_frame, 
            text="样本数据文件:",
            font=ctk.CTkFont(size=14)
        )
        self.sample_file_label.pack(side="left", padx=10, pady=10)
        
        self.sample_file_path_var = tk.StringVar()
        self.sample_file_path_entry = ctk.CTkEntry(
            self.sample_file_frame, 
            width=300,
            textvariable=self.sample_file_path_var,
            state="readonly"
        )
        self.sample_file_path_entry.pack(side="left", padx=10, pady=10)
        
        self.sample_browse_button = ctk.CTkButton(
            self.sample_file_frame, 
            text="浏览", 
            width=80,
            command=lambda: self._browse_file_generic(
                'sample_file_path', 
                self.sample_file_path_var, 
                self.sample_file_path_entry, 
                "选择样本数据文件"
            )
        )
        self.sample_browse_button.pack(side="left", padx=20, pady=10)
        
        # 分析按钮容器
        self.button_container = ctk.CTkFrame(self, fg_color="transparent")
        self.button_container.pack(fill="x", padx=30, pady=(5, 5))
        
        # 分析按钮
        self.analyze_button = ctk.CTkButton(
            self.button_container, 
            text="开始分析", 
            width=150,
            height=30,
            state="disabled",
            command=self.analyze_sample
        )
        self.analyze_button.pack(side="right", padx=10, pady=0)
        
        # 分析结果区域
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.pack(fill="both", expand=True, padx=30, pady=(5, 30))
        
        # 结果标签
        self.result_title = ctk.CTkLabel(
            self.result_frame, 
            text="分析结果", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.result_title.pack(pady=(5, 5))
        
        # 结果值
        self.result_value = ctk.CTkLabel(
            self.result_frame, 
            text="请上传文件并开始分析",
            font=ctk.CTkFont(size=16)
        )
        self.result_value.pack(pady=(0, 20))
    
    def _browse_file_generic(self, path_attr, var_attr, entry_attr, title):
        """通用文件浏览函数"""
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("所有文件", "*.*")],
            initialdir=os.path.expanduser("~")
        )
        
        if file_path:
            setattr(self, path_attr, file_path)
            # Update display path (shortened)
            filename = os.path.basename(file_path)
            display_path = filename
            if len(filename) > 40:
                display_path = filename[:37] + "..."
            
            # Update StringVar and Entry
            var_attr.set(file_path)
            entry_attr.configure(state="normal")
            entry_attr.delete(0, tk.END)
            entry_attr.insert(0, display_path)
            entry_attr.configure(state="readonly")
            
            logger.info(f"用户选择了文件 '{title}': {file_path}")
            self._check_enable_analyze_button()
    
    def _check_enable_analyze_button(self):
        """检查是否两个文件都已选择，并更新分析按钮状态"""
        if self.train_file_path and self.sample_file_path:
            self.analyze_button.configure(state="normal")
        else:
            self.analyze_button.configure(state="disabled")
    
    def analyze_sample(self):
        """分析样本数据"""
        try:
            # 显示进度条
            self.show_progress("正在分析文件，请稍候...")
            
            # 创建一个线程来执行分析，避免UI阻塞
            threading.Thread(target=self._perform_analysis, daemon=True).start()
            
        except Exception as e:
            logger.error(f"分析样本时出错: {str(e)}")
            self.show_error("分析错误", str(e))
            self.hide_progress()
    
    def _perform_analysis(self):
        """执行分析（在单独的线程中运行）"""
        try:
            # 训练模型
            self.classifier.train(self.train_file_path)
            
            # 预测样本
            probabilities = self.classifier.predict(self.sample_file_path)
            
            # 准备结果
            labels = ["酒类A", "酒类B", "酒类C"]
            results = {label: prob for label, prob in zip(labels, probabilities)}
            
            # 保存结果
            self.analysis_result = {
                'probabilities': results,
                'train_filename': os.path.basename(self.train_file_path),
                'sample_filename': os.path.basename(self.sample_file_path),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_type': "白酒分类分析"
            }
            
            # 生成饼图
            plt.figure(figsize=(8, 8))
            plt.pie(
                probabilities, 
                labels=[f"{label}\n{prob:.1%}" for label, prob in zip(labels, probabilities)],
                autopct='%1.1f%%',
                colors=['#FF9999', '#66B2FF', '#99FF99']
            )
            plt.title("白酒分类分析结果")
            
            # 保存图表
            os.makedirs('history_img', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            chart_path = os.path.join('history_img', f'classify_{timestamp}.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.analysis_result['chart_path'] = chart_path
            
            # 在主线程中更新UI
            self.after(0, self.update_results_ui)
            
            # 记录到数据库
            result_str = ", ".join([f"{label}: {prob:.1%}" for label, prob in results.items()])
            self.db_manager.add_analysis_record(
                self.user_id,
                self.analysis_result['sample_filename'],
                result_str,
                self.analysis_result['analysis_type']
            )
            
        except Exception as e:
            logger.error(f"分析过程中出错: {str(e)}")
            self.after(0, lambda: self.show_error("分析错误", str(e)))
            self.after(0, self.hide_progress)
    
    def update_results_ui(self):
        """更新结果UI"""
        try:
            if not self.analysis_result:
                return
                
            # 清空结果框架
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            
            # === 结果标题 ===
            self.result_title = ctk.CTkLabel(
                self.result_frame, 
                text="分析结果", 
                font=ctk.CTkFont(size=18, weight="bold")
            )
            self.result_title.pack(pady=(20, 10))
            
            # === 分类结果 ===
            results = self.analysis_result['probabilities']
            result_text = "白酒分类结果:\n\n"
            for label, prob in results.items():
                result_text += f"{label}: {prob:.1%}\n"
            
            self.result_value = ctk.CTkLabel(
                self.result_frame, 
                text=result_text,
                font=ctk.CTkFont(size=16, family="Courier"),
                justify="left"
            )
            self.result_value.pack(pady=(0, 20))
            
            # === 显示饼图 ===
            try:
                img = Image.open(self.analysis_result['chart_path'])
                # 调整图像大小
                img = img.resize((400, 400), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                
                img_label = tk.Label(self.result_frame, image=img_tk)
                img_label.image = img_tk
                img_label.pack(pady=(0, 20))
            except Exception as img_err:
                logger.error(f"加载结果图表时出错: {str(img_err)}")
            
            # === 文件信息 ===
            info_text = f"训练数据文件: {self.analysis_result['train_filename']}\n"
            info_text += f"样本数据文件: {self.analysis_result['sample_filename']}\n"
            info_text += f"分析时间: {self.analysis_result['datetime']}"
            
            info_label = ctk.CTkLabel(
                self.result_frame,
                text=info_text,
                font=ctk.CTkFont(size=12),
                justify="left"
            )
            info_label.pack(pady=(0, 10))
            
            self.hide_progress()
            
        except Exception as e:
            logger.error(f"更新结果UI时出错: {str(e)}")
            self.show_error("错误", f"更新结果UI时出错: {str(e)}")
            self.hide_progress()
    
    def show_progress(self, message):
        """显示进度窗口"""
        try:
            # 创建新窗口
            self.progress_window = ctk.CTkToplevel(self)
            self.progress_window.title("处理中")
            self.progress_window.geometry("300x100")
            self.progress_window.resizable(False, False)
            self.progress_window.transient(self)
            
            # 进度消息
            message_label = ctk.CTkLabel(
                self.progress_window, 
                text=message,
                font=("Helvetica", 12)
            )
            message_label.pack(pady=(20, 10))
            
            # 进度条
            self.progress_bar = ttk.Progressbar(
                self.progress_window,
                orient="horizontal",
                length=200,
                mode="indeterminate"
            )
            self.progress_bar.pack(pady=(0, 20))
            self.progress_bar.start(10)
            
        except Exception as e:
            logger.error(f"显示进度窗口时出错: {str(e)}")
    
    def hide_progress(self):
        """隐藏进度窗口"""
        try:
            if hasattr(self, 'progress_window') and self.progress_window is not None:
                try:
                    if hasattr(self, 'progress_bar'):
                        self.progress_bar.stop()
                    self.progress_window.destroy()
                except Exception:
                    pass
                self.progress_window = None
                self.progress_bar = None
        except Exception as e:
            logger.error(f"隐藏进度窗口时出错: {str(e)}")
    
    def show_error(self, title, message):
        """显示错误对话框"""
        try:
            error_window = ctk.CTkToplevel(self)
            error_window.title(title)
            error_window.geometry("300x200")
            error_window.resizable(False, False)
            error_window.transient(self)
            
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
            
        except Exception as e:
            logger.error(f"显示错误对话框时出错: {str(e)}")
            print(f"错误: {title} - {message}") 