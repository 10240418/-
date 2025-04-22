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
    """白酒真伪分类页面"""
    
    def __init__(self, parent, user_id, db_manager):
        super().__init__(parent, corner_radius=0)
        
        self.user_id = user_id
        self.db_manager = db_manager
        self.file_path = None
        self.classification_result = None
        
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
        """创建白酒真伪分类页面组件"""
        
        # 页面标题
        self.title_label = ctk.CTkLabel(
            self, 
            text="白酒真伪分类", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 20), padx=30, anchor="w")
        
        # 文件选择区域框架
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(fill="x", padx=30, pady=(0, 20))
        
        # 文件选择部分
        self.file_frame = ctk.CTkFrame(self.input_frame)
        self.file_frame.pack(fill="x", padx=10, pady=10)
        
        self.file_label = ctk.CTkLabel(
            self.file_frame, 
            text="请选择Excel格式的光谱数据文件:",
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
        
        # 创建一个容纳分类按钮的中间框架
        self.action_frame = ctk.CTkFrame(self)
        self.action_frame.pack(fill="x", padx=30, pady=(0, 20))
        
        # 分类按钮 - 放在文件选择框下方，结果区域上方，并靠右对齐
        self.classify_button = ctk.CTkButton(
            self.action_frame, 
            text="开始分类", 
            width=150,
            height=30,
            state="disabled",
            command=self.classify_file
        )
        # 设置按钮自身的内边距为0
        self.classify_button.pack(side="right", padx=10, pady=0)
        
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
            # 在线程中创建数据库连接副本以防止SQLite线程问题
            threading.Thread(target=self._perform_classification_thread_safe, daemon=True).start()
        except Exception as e:
            logger.error(f"分类文件时出错: {str(e)}")
            self.show_error("分类错误", f"无法分类文件: {str(e)}")
    
    def _perform_classification_thread_safe(self):
        """线程安全的分类执行包装器"""
        try:
            # 执行分类
            result = self._perform_classification()
            
            # 在主线程中更新UI
            if result:
                self.after(0, self.update_results_ui)
        except Exception as e:
            logger.error(f"分类过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            self.after(0, lambda: self.show_error("分类错误", f"分类过程中出错: {str(e)}"))
            self.after(0, self.hide_progress)
    
    def _perform_classification(self):
        """执行文件分类（在单独的线程中运行）"""
        try:
            # 加载数据
            logger.info(f"开始处理文件: {self.file_path}")
            
            try:
                # 根据文件扩展名决定如何加载
                file_ext = os.path.splitext(self.file_path)[1].lower()
                if file_ext == '.csv':
                    data = pd.read_csv(self.file_path)
                elif file_ext in ['.xlsx', '.xls']:
                    data = pd.read_excel(self.file_path)
                else:
                    raise ValueError(f"不支持的文件格式: {file_ext}")
                
                # 数据预处理
                logger.info("预处理数据...")
                # 假设第一列是ID或标签列，不用于分类
                # 如果需要调整数据结构，可以在此处理
                if data.shape[1] > 1:
                    # 提取特征（假设第一列可能是ID或者标签）
                    features = data.iloc[:, 1:]
                else:
                    # 如果只有一列，就全部当作特征
                    features = data
                
                # 转换为numpy数组
                sample_data = np.array(features, dtype=np.float32)
                
                # 使用MinMaxScaler标准化数据
                scaler = MinMaxScaler()
                sample_data = scaler.fit_transform(sample_data)
                
                # 根据PyTorch可用性选择不同的模型和预测方法
                if TORCH_AVAILABLE:
                    # 转换为PyTorch张量
                    sample_tensor = torch.tensor(sample_data, dtype=torch.float32).to(self.device)
                    
                    logger.info(f"数据加载完成，特征形状: {sample_tensor.shape}")
                    
                    # 加载模型
                    logger.info("加载分类模型...")
                    model = self._load_model()
                    
                    # 进行预测
                    logger.info("执行分类预测...")
                    with torch.no_grad():
                        outputs = model(sample_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        
                        # 获取预测分类和置信度
                        batch_probs, batch_preds = torch.max(probabilities, dim=1)
                        
                        # 使用第一个样本的结果（或者可以对所有样本进行投票）
                        class_id = batch_preds[0].item()
                        confidence = batch_probs[0].item()
                        
                        # 获取所有类别的置信度
                        confidence_scores = {}
                        for i in range(len(self.categories)):
                            confidence_scores[i] = probabilities[0, i].item()
                else:
                    # 使用RandomForest模型
                    logger.info("使用RandomForest分类模型...")
                    model = self._load_sklearn_model()
                    
                    # 进行预测
                    logger.info("执行分类预测...")
                    class_probs = model.predict_proba(sample_data)
                    
                    # 获取预测分类和置信度
                    class_id = np.argmax(class_probs[0])
                    confidence = class_probs[0][class_id]
                    
                    # 获取所有类别的置信度
                    confidence_scores = {}
                    for i in range(len(self.categories)):
                        confidence_scores[i] = class_probs[0][i]
                
                class_name = self.categories[class_id]
                
                # 保存分类结果
                self.classification_result = {
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence_scores': confidence_scores,
                    'filename': os.path.basename(self.file_path),
                    'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # # 生成结果字符串
                result_str = f"分类结果: {class_name} (置信度: {confidence:.2f})"
                
                logger.info(f"分类完成: {result_str}")
                
                # 创建数据库连接的副本在此线程中使用
                from database.db_manager import DatabaseManager
                db_copy = DatabaseManager()
                db_copy.connect()
                db_copy.add_analysis_record(
                    self.user_id, 
                    os.path.basename(self.file_path), 
                    result_str,
                    "白酒真伪分类"
                )
                db_copy.close()
                
                return True
                
            except Exception as e:
                logger.error(f"数据处理或预测过程中出错: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
        except Exception as e:
            logger.error(f"分类过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_model(self):
        """加载PyTorch分类模型"""
        if not TORCH_AVAILABLE:
            logger.error("PyTorch模块不可用，无法加载神经网络模型")
            raise ImportError("PyTorch不可用")
            
        try:
            # 获取模型文件路径（假设模型文件保存在models目录下）
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            model_path = os.path.join(model_dir, "baijiu_classifier.pth")
            
            # 如果找不到预训练模型，创建一个新的模型
            if not os.path.exists(model_path):
                logger.warning(f"未找到模型文件: {model_path}，创建新模型")
                # 创建模型（使用classify_tool.py中的模型架构）
                model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(3, 128),  # 假设输入特征为3维（PCA降维后）
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 3),  # 3个分类类别
                )
            else:
                logger.info(f"加载预训练模型: {model_path}")
                # 创建模型架构（与保存时相同）
                model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(3, 128),  # 假设输入特征为3维（PCA降维后）
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 3),  # 3个分类类别
                )
                # 加载预训练权重
                model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # 将模型设置为评估模式
            model.eval()
            model = model.to(self.device)
            
            return model
            
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_sklearn_model(self):
        """加载scikit-learn分类模型"""
        try:
            # 创建一个随机森林分类器作为备选模型
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            rf_model_path = os.path.join(model_dir, "rf_model.pkl")
            
            if os.path.exists(rf_model_path):
                import joblib
                logger.info(f"加载随机森林模型: {rf_model_path}")
                model = joblib.load(rf_model_path)
            else:
                logger.warning(f"未找到随机森林模型: {rf_model_path}，创建新模型")
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                # 注意：这个模型未经训练，将会产生随机预测
            
            return model
        except Exception as e:
            logger.error(f"加载随机森林模型时出错: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
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
            
            # # 显示结果
            # self.result_value.configure(
            #     text=f"分类结果: {class_name} (置信度: {confidence:.2f})",
            #     text_color=self.get_result_color(self.classification_result['class_id'])
            # )
            
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
        
        # # 添加分类结论
        # conclusion_frame = ctk.CTkFrame(self.details_frame)
        # conclusion_frame.pack(fill="x", pady=(20, 10))
        
        # conclusion_label = ctk.CTkLabel(
        #     conclusion_frame,
        #     text="结论:",
        #     font=ctk.CTkFont(size=14, weight="bold"),
        #     width=50,
        #     anchor="w"
        # )
        # conclusion_label.pack(side="left", padx=10)
        
        # # 根据分类结果生成结论
        # class_id = self.classification_result['class_id']
        # if class_id == 0:
        #     conclusion_text = "此样品为真酒，无掺假成分。"
        # elif class_id == 1:
        #     conclusion_text = "此样品疑似掺水假酒，请注意。"
        # else:  # class_id == 2
        #     conclusion_text = "此样品疑似掺工业酒精假酒，不建议饮用！"
        
        # conclusion_value = ctk.CTkLabel(
        #     conclusion_frame,
        #     text=conclusion_text,
        #     font=ctk.CTkFont(weight="bold"),
        #     text_color=self.get_result_color(class_id)
        # )
        # conclusion_value.pack(side="left", padx=5)
    
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