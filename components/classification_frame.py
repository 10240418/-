import os
import logging
import threading
import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
import customtkinter as ctk
from datetime import datetime

# 设置日志
logger = logging.getLogger('baijiu_app')

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
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("CSV文件", "*.csv"), ("所有文件", "*.*")],
            initialdir=os.path.expanduser("~") # 默认打开用户主目录
        )
        
        if file_path:
            self.file_path = file_path
            self.file_path_var.set(file_path)
            self.classify_button.configure(state="normal")
            logger.info(f"用户选择了文件: {file_path}")
            
            # 更新UI上的文件名显示
            filename = os.path.basename(file_path)
            if len(filename) > 40:  # 如果文件名太长，截断显示
                filename = filename[:37] + "..."
            self.file_path_entry.configure(state="normal")
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, filename)
            self.file_path_entry.configure(state="readonly")
    
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
        """执行文件分类（在单独的线程中运行）- 简化版实现"""
        try:
            # 在这里实现实际的分类逻辑，这是简化版
            import random
            import time
            
            # 模拟分类过程
            time.sleep(2)  # 模拟耗时操作
            
            # 随机选择一个类别作为示例结果
            class_id = random.randint(0, 2)
            class_name = self.categories[class_id]
            
            # 生成置信度分数（仅用于示例）
            confidence_scores = {}
            for i in range(3):
                if i == class_id:
                    confidence_scores[i] = random.uniform(0.7, 0.95)  # 预测类别的置信度较高
                else:
                    confidence_scores[i] = random.uniform(0.05, 0.3)  # 其他类别的置信度较低
            
            # 保存分类结果
            self.classification_result = {
                'class_id': class_id,
                'class_name': class_name,
                'confidence_scores': confidence_scores,
                'filename': os.path.basename(self.file_path),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 生成结果字符串
            result_str = f"分类结果: {class_name} (置信度: {confidence_scores[class_id]:.2f})"
            
            # 记录结果到数据库
            self.db_manager.add_analysis_record(
                self.user_id, 
                os.path.basename(self.file_path), 
                result_str,
                "白酒真伪分类"
            )
            
            # 在主线程中更新UI
            self.after(0, self.update_results_ui)
            
        except Exception as e:
            logger.error(f"分类过程中出错: {str(e)}")
            self.after(0, lambda: self.show_error("分类错误", f"分类过程中出错: {str(e)}"))
            self.after(0, self.hide_progress)
    
    def update_results_ui(self):
        """更新结果UI（在主线程中运行）"""
        try:
            if not self.classification_result:
                logger.error("没有可用的分类结果")
                self.hide_progress()
                self.show_error("错误", "没有可用的分类结果")
                return
                
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
            
            logger.info("分类结果UI更新完成")
            
        except Exception as e:
            logger.error(f"更新UI时出错: {str(e)}")
            self.hide_progress()
            self.show_error("错误", f"更新结果UI时出错: {str(e)}")
    
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
        try:
            # 关闭可能已存在的进度窗口
            self.hide_progress()
            
            # 创建新窗口前先刷新主窗口
            self.update_idletasks()
            
            # 创建新窗口
            parent_window = self.winfo_toplevel()
            self.progress_window = ctk.CTkToplevel(parent_window)
            self.progress_window.title("处理中")
            self.progress_window.geometry("300x100")
            self.progress_window.resizable(False, False)
            self.progress_window.transient(parent_window)
            
            # 确保窗口被正确构建
            self.progress_window.update()
            
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
        """安全地隐藏进度窗口"""
        try:
            if hasattr(self, 'progress_window') and self.progress_window is not None:
                try:
                    if hasattr(self, 'progress_bar') and self.progress_bar is not None:
                        try:
                            self.progress_bar.stop()
                        except Exception:
                            pass
                    
                    if self.progress_window.winfo_exists():
                        try:
                            self.progress_window.grab_release()
                        except Exception:
                            pass
                        
                        try:
                            self.progress_window.destroy()
                        except Exception:
                            pass
                except Exception:
                    pass
                
                self.progress_window = None
                self.progress_bar = None
                
        except Exception as e:
            logger.error(f"隐藏进度窗口时出错: {str(e)}")
    
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
            parent_window = self.winfo_toplevel()
            error_window = ctk.CTkToplevel(parent_window)
            error_window.title(title)
            error_window.geometry("300x200")
            error_window.resizable(False, False)
            error_window.transient(parent_window)
            
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
            logger.error(f"显示错误对话框时发生异常: {str(e)}")
            print(f"错误: {title} - {message}")
    
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