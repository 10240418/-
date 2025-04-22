import os
import sys
import logging
import threading
import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
import customtkinter as ctk
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from PIL import Image, ImageTk
import matplotlib
import traceback # Import traceback module for detailed error logging
# Import necessary metric functions
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

matplotlib.use('Agg')  # 使用非交互式后端

# 设置日志
logger = logging.getLogger('baijiu_app')

class AnalysisFrame(ctk.CTkFrame):
    """掺伪量分析预测页面"""
    
    def __init__(self, parent, user_id, db_manager):
        super().__init__(parent, corner_radius=0)
        
        self.user_id = user_id
        self.db_manager = db_manager
        self.file_path = None # Prediction file path
        self.spectrum_file_path = None # Spectrum file path
        self.analysis_result = None
        
        # 创建UI组件
        self.create_widgets()
    
    def create_widgets(self):
        """创建掺伪量分析页面组件"""
        
        # 页面标题
        self.title_label = ctk.CTkLabel(
            self, 
            text="掺伪度分析预测", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 10), padx=30, anchor="w") # Reduced bottom padding
        
        # === 文件选择区域 1: 预测数据文件 ===
        self.pred_file_frame = ctk.CTkFrame(self)
        self.pred_file_frame.pack(fill="x", padx=30, pady=(10, 5)) # Add padding
        
        self.pred_file_label = ctk.CTkLabel(
            self.pred_file_frame, 
            text="预测分析文件(表头可有可无):",
            font=ctk.CTkFont(size=14)
        )
        self.pred_file_label.pack(side="left", padx=10, pady=10)
        
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ctk.CTkEntry(
            self.pred_file_frame, 
            width=300,
            textvariable=self.file_path_var,
            state="readonly"
        )
        self.file_path_entry.pack(side="left", padx=10, pady=10)
        
        self.pred_browse_button = ctk.CTkButton(
            self.pred_file_frame, 
            text="浏览", 
            width=80,
            command=lambda: self._browse_file_generic(
                'file_path', 
                self.file_path_var, 
                self.file_path_entry, 
                "选择预测数据文件"
            )
        )
        self.pred_browse_button.pack(side="left", padx=20, pady=10)
        
        # === 文件选择区域 2: 光谱数据文件 ===
        self.spectrum_file_frame = ctk.CTkFrame(self)
        self.spectrum_file_frame.pack(fill="x", padx=30, pady=(5, 5)) # 减小底部间距从20到5
        
        self.spectrum_file_label = ctk.CTkLabel(
            self.spectrum_file_frame, 
            text="样本光谱数据文件(需带表头):",
            font=ctk.CTkFont(size=14)
        )
        self.spectrum_file_label.pack(side="left", padx=10, pady=10)
        
        self.spectrum_file_path_var = tk.StringVar()
        self.spectrum_file_path_entry = ctk.CTkEntry(
            self.spectrum_file_frame, 
            width=300,
            textvariable=self.spectrum_file_path_var,
            state="readonly"
        )
        self.spectrum_file_path_entry.pack(side="left", padx=10, pady=10)
        
        self.spectrum_browse_button = ctk.CTkButton(
            self.spectrum_file_frame, 
            text="浏览", 
            width=80,
            command=lambda: self._browse_file_generic(
                'spectrum_file_path', 
                self.spectrum_file_path_var, 
                self.spectrum_file_path_entry, 
                "选择光谱数据文件 (样品数据)"
            )
        )
        self.spectrum_browse_button.pack(side="left", padx=20, pady=10)
        
        # 添加一个容器用于放置分析按钮，确保靠右对齐
        self.button_container = ctk.CTkFrame(self, fg_color="transparent")
        self.button_container.pack(fill="x", padx=30, pady=(5, 5)) # 缩小底部间距从20到5
        
        # 分析按钮 - 放在文件选择框下方，结果区域上方，并靠右对齐
        self.analyze_button = ctk.CTkButton(
            self.button_container, 
            text="开始分析", 
            width=150,
            height=30,
            state="disabled",
            command=self.analyze_file
        )
        # 设置按钮自身的内边距为0
        self.analyze_button.pack(side="right", padx=10, pady=0)
        
        # 分析结果区域
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.pack(fill="both", expand=True, padx=30, pady=(5, 30)) # 缩小顶部间距从0到5
        
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
        
        # 为图表创建一个框架
        self.chart_frame = ctk.CTkFrame(self.result_frame)
        self.chart_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
    
    def _browse_file_generic(self, path_attr, var_attr, entry_attr, title):
        """通用文件浏览函数"""
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Excel文件", "*.xlsx *.xls"), ("CSV文件", "*.csv"), ("所有文件", "*.*")],
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
            var_attr.set(file_path) # Store full path in var if needed elsewhere
            entry_attr.configure(state="normal")
            entry_attr.delete(0, tk.END)
            entry_attr.insert(0, display_path)
            entry_attr.configure(state="readonly")
            
            logger.info(f"用户选择了文件 '{title}': {file_path}")
            self._check_enable_analyze_button()

    def _check_enable_analyze_button(self):
        """检查是否两个文件都已选择，并更新分析按钮状态"""
        if self.file_path and self.spectrum_file_path:
            self.analyze_button.configure(state="normal")
        else:
            self.analyze_button.configure(state="disabled")
            
    def analyze_file(self):
        """分析文件数据"""
        # Check if both files are selected
        if not self.file_path or not self.spectrum_file_path:
            self.show_error("文件缺失", "请同时选择用于预测分析和光谱图的文件。")
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
        """执行文件分析（在单独的线程中运行）- 使用实际预测算法"""
        try:
            # 检查模型文件是否存在
            models_dir = "models"
            model_path = os.path.join(models_dir, "rf_model.pkl")
            pca_path = os.path.join(models_dir, "pca_model.pkl")
            scaler_path = os.path.join(models_dir, "scaler_model.pkl")
            
            if not (os.path.exists(model_path) and os.path.exists(pca_path) and os.path.exists(scaler_path)):
                # 如果模型文件不存在，显示错误
                self.after(0, lambda: self.show_error("模型错误", "预测模型文件不存在。请先运行训练工具来生成模型。"))
                self.after(0, self.hide_progress)
                return
            
            # 加载模型
            rf_model = joblib.load(model_path)
            pca_model = joblib.load(pca_path)
            scaler = joblib.load(scaler_path)
            
            # 确保目录存在
            os.makedirs('history_img', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
            # === 1. 光谱图绘制 (来自第二个文件) ===
            spectrum_path = None 
            raw_spectrum_data = None
            try:
                logger.info(f"加载光谱数据文件: {self.spectrum_file_path}")
                file_ext_spectrum = os.path.splitext(self.spectrum_file_path)[1].lower()
                if file_ext_spectrum in ['.xlsx', '.xls']:
                    raw_spectrum_data = pd.read_excel(self.spectrum_file_path, header=None)
                elif file_ext_spectrum == '.csv':
                    raw_spectrum_data = pd.read_csv(self.spectrum_file_path, header=None)
                else:
                    raise ValueError("光谱数据文件格式不支持") # Raise error to be caught below

                if raw_spectrum_data.shape[0] >= 2:
                    x_values = raw_spectrum_data.iloc[0].values
                    x_label = "波长/频率"
                    if isinstance(raw_spectrum_data.iloc[0, 0], str) and not self._is_numeric_string(raw_spectrum_data.iloc[0, 0]):
                        x_label = raw_spectrum_data.iloc[0, 0]
                        x_values = x_values[1:]
                    
                    y_values = raw_spectrum_data.iloc[1].values
                    y_label = "光谱值"
                    if isinstance(raw_spectrum_data.iloc[1, 0], str) and not self._is_numeric_string(raw_spectrum_data.iloc[1, 0]):
                        y_label = raw_spectrum_data.iloc[1, 0]
                        y_values = y_values[1:]

                    valid_x, valid_y = [], []
                    for x, y in zip(x_values, y_values):
                        try:
                            valid_x.append(float(x) if isinstance(x, str) else float(x))
                            valid_y.append(float(y) if isinstance(y, str) else float(y))
                        except (ValueError, TypeError):
                            continue
                    
                    if len(valid_x) > 1:
                        plt.figure(figsize=(12, 6))
                        plt.scatter(valid_x, valid_y, color='#1f77b4', s=30, alpha=0.8, label="数据点")
                        plt.plot(valid_x, valid_y, color='#1f77b4', linestyle='--', alpha=0.6, label="趋势线")
                        plt.title("白酒光谱数据可视化", fontsize=14)
                        plt.xlabel(x_label, fontsize=12)
                        plt.ylabel(y_label, fontsize=12)
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.legend(loc='best')
                        desc_text = f"光谱数据文件: {os.path.basename(self.spectrum_file_path)}\\n" \
                                    f"数据点数量: {len(valid_x)}\\n" \
                                    f"X轴范围: [{min(valid_x):.2f}, {max(valid_x):.2f}]\\n" \
                                    f"Y轴范围: [{min(valid_y):.2f}, {max(valid_y):.2f}]"
                        plt.annotate(desc_text, xy=(0.02, 0.98), xycoords='axes fraction',
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                                     fontsize=10, ha='left', va='top')
                        spectrum_path = os.path.join('history_img', f'spectrum_{timestamp}.png')
                        plt.tight_layout()
                        plt.savefig(spectrum_path, dpi=600)
                        plt.close()
                        logger.info(f"光谱图已保存至: {spectrum_path}")
                    else:
                        logger.warning("光谱数据文件中有效数值数据不足，无法绘制光谱图或用于单一样品预测。")
                else:
                    logger.warning("光谱数据文件行数不足，无法绘制光谱图或用于单一样品预测。")

            except Exception as spectrum_err:
                logger.error(f"处理光谱数据文件时出错: {str(spectrum_err)}")
                logger.error(traceback.format_exc())
                # Show error but continue if possible to predict on the first file
                self.after(0, lambda: self.show_error("光谱文件处理错误", f"无法处理光谱数据文件: {str(spectrum_err)}"))

            # === 2. 掺伪度预测 (来自第一个文件) ===
            logger.info(f"加载预测数据文件: {self.file_path}")
            file_ext_pred = os.path.splitext(self.file_path)[1].lower()
            if file_ext_pred in ['.xlsx', '.xls']:
                data = pd.read_excel(self.file_path, header=None)
            elif file_ext_pred == '.csv':
                data = pd.read_csv(self.file_path, header=None)
            else:
                self.after(0, lambda: self.show_error("预测文件错误", "预测数据文件格式不支持"))
                self.after(0, self.hide_progress)
                return

            # 预处理数据 - 检测并移除非数值数据
            # 检查第一行是否为标题行
            first_row_has_text = any(isinstance(col, str) and not self._is_numeric_string(col) for col in data.iloc[0])
            if first_row_has_text:
                data = data.iloc[1:].reset_index(drop=True)
            
            # 检查第一列是否为标签列
            first_col_has_text = any(isinstance(val, str) and not self._is_numeric_string(val) for val in data.iloc[:, 0])
            if first_col_has_text:
                data = data.iloc[:, 1:].reset_index(drop=True)
            
            # 转换所有数据为浮点数，非数值转为NaN
            def to_numeric_with_fallback(val):
                try: return pd.to_numeric(val)
                except: return np.nan
            data = data.applymap(to_numeric_with_fallback)
            
            # 移除包含太多NaN值的行列（超过50%）
            nan_rows = data.isna().mean(axis=1) > 0.5
            if nan_rows.any():
                data = data.loc[~nan_rows].reset_index(drop=True)
            
            nan_cols = data.isna().mean(axis=0) > 0.5
            if nan_cols.any():
                data = data.loc[:, ~nan_cols].reset_index(drop=True)
            
            # 填充剩余的NaN值
            data = data.fillna(data.mean())
            
            # 确保数据不为空
            if data.empty or data.shape[0] < 1:
                self.after(0, lambda: self.show_error("数据错误", "数据预处理后为空或数据量不足"))
                self.after(0, self.hide_progress)
                return
            
            # 参考improved_test_tool.py中的方法，生成浓度标签
            # 计算需要多少组标签（每7个样本为一组）
            n_samples = data.shape[0]
            n_groups = (n_samples + 6) // 7  # 向上取整
            
            # 使用linspace生成线性间隔的浓度值 (0.0385, 0.5)
            y_actual = np.repeat(np.linspace(0.0385, 0.5, n_groups), 7)[:n_samples]  # 3.85%, 7.41%, ..., 50.00% 对应每7行
            
            # 将数据转换为numpy数组
            X = data.values.astype(float)
            
            # 应用MSC标准化
            X_msc = self.msc_correction(X)
            
            # 应用PCA降维
            X_pca = pca_model.transform(X_msc)
            
            # 应用标准化
            X_pca = scaler.transform(X_pca)
            
            # 使用模型进行预测
            y_pred = rf_model.predict(X_pca)
            
            # 计算平均值
            mean_pred = np.mean(y_pred)
            mean_actual = np.mean(y_actual)
            
            # 计算R²和RMSE作为准确率指标
            r2 = r2_score(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            
            # 计算MAE和MRE
            mae = mean_absolute_error(y_actual, y_pred)
            # 处理y_actual为0的情况，避免除以零错误
            mre = np.mean(np.abs((y_pred - y_actual) / y_actual)[y_actual != 0]) * 100 if np.any(y_actual != 0) else 0
            
            # 准确率转为百分比
            accuracy = r2 * 100  # R²作为准确率，转为百分比
            
            # 生成对比图表 - 使用更大的尺寸并增加DPI
            plt.figure(figsize=(12, 9))  # 更大的图表尺寸
            
            # 绘制散点图
            plt.scatter(y_actual, y_pred, color='#1f77b4', alpha=0.7, s=60)  # 增大点的大小
            
            # 绘制理想预测线
            min_val = min(min(y_actual), min(y_pred))
            max_val = max(max(y_actual), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='理想预测线')
            
            # 添加坐标轴标签和标题
            plt.xlabel("实际掺伪度", fontsize=14)
            plt.ylabel("预测掺伪度", fontsize=14)
            plt.title("白酒掺伪度预测结果", fontsize=16)
            
            # 添加R²和RMSE的文本标注 - 增大字体
            text_str = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}'
            plt.annotate(text_str, xy=(0.05, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                        fontsize=12, ha='left', va='top')
            
            # 添加网格线
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存图表 - 使用更高的DPI
            chart_path = os.path.join('history_img', f'analysis_{timestamp}.png')
            plt.tight_layout()
            plt.savefig(chart_path, dpi=600)  # 增加DPI为600
            plt.close()
            
            # === 3. 单一样品预测 (来自第二个文件) ===
            single_sample_prediction = None
            if raw_spectrum_data is not None:
                try:
                    logger.info("开始对光谱文件数据进行单一样品预测...")
                    
                    # 1. 数据清洗 - 从原始数据中移除非数值数据
                    # 检查第一行是否包含非数值数据（作为列标题）
                    first_row_is_header = False
                    if raw_spectrum_data.shape[0] > 0:
                        first_row_is_header = any(isinstance(x, str) and not self._is_numeric_string(x) 
                                                for x in raw_spectrum_data.iloc[0])
                    
                    # 检查第一列是否包含非数值数据（作为行标题）
                    first_col_is_header = False
                    if raw_spectrum_data.shape[1] > 0:
                        first_col_is_header = any(isinstance(x, str) and not self._is_numeric_string(x) 
                                                for x in raw_spectrum_data.iloc[:, 0])
                    
                    # 根据头部信息剪裁数据
                    if first_row_is_header:
                        logger.info("检测到第一行为标题行，将跳过")
                        spectrum_data = raw_spectrum_data.iloc[1:].reset_index(drop=True)
                    else:
                        spectrum_data = raw_spectrum_data
                    
                    if first_col_is_header:
                        logger.info("检测到第一列为标题列，将跳过")
                        spectrum_data = spectrum_data.iloc[:, 1:].reset_index(drop=True)
                    
                    # 2. 将所有数据转换为数值，无法转换的变为NaN
                    logger.info("将光谱数据转换为数值类型")
                    spectrum_numeric = spectrum_data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
                    
                    # 3. 检查数据是否为空
                    if spectrum_numeric.empty:
                        logger.warning("预处理后的光谱数据为空")
                        raise ValueError("预处理后的光谱数据为空")
                    
                    # 4. 使用第一行作为样本数据
                    logger.info(f"使用光谱数据的第一行作为样本，数据形状: {spectrum_numeric.shape}")
                    sample_data = spectrum_numeric.iloc[0].values
                    
                    # 5. 处理样本中的NaN值
                    sample_data = np.nan_to_num(sample_data, nan=np.nanmean(sample_data))
                    
                    # 6. 确保样本数据是二维数组
                    sample_2d = sample_data.reshape(1, -1)
                    logger.info(f"重塑后的样本形状: {sample_2d.shape}, PCA模型期望特征数: {pca_model.n_features_in_}")
                    
                    # 7. 调整样本特征数与模型期望一致
                    if sample_2d.shape[1] != pca_model.n_features_in_:
                        if sample_2d.shape[1] > pca_model.n_features_in_:
                            # 特征数过多，截断
                            logger.warning(f"样本特征数 ({sample_2d.shape[1]}) 大于PCA期望 ({pca_model.n_features_in_})，进行截断")
                            sample_2d = sample_2d[:, :pca_model.n_features_in_]
                        else:
                            # 特征数不足，填充
                            logger.warning(f"样本特征数 ({sample_2d.shape[1]}) 小于PCA期望 ({pca_model.n_features_in_})，进行填充")
                            padding = np.zeros((1, pca_model.n_features_in_ - sample_2d.shape[1]))
                            sample_2d = np.hstack((sample_2d, padding))
                    
                    # 8. 执行与tool_one_test.py相同的预处理流程
                    # 8.1 填充缺失值
                    imputer = SimpleImputer(strategy='mean')
                    sample_imputed = imputer.fit_transform(sample_2d)
                    logger.info("已完成样本数据的缺失值填充")
                    
                    # 8.2 MSC标准化
                    sample_msc = self.msc_correction(sample_imputed)
                    logger.info("已完成样本数据的MSC标准化")
                    
                    # 8.3 PCA降维
                    sample_pca = pca_model.transform(sample_msc)
                    logger.info(f"已完成样本数据的PCA降维，形状变为: {sample_pca.shape}")
                    
                    # 8.4 标准化
                    sample_scaled = scaler.transform(sample_pca)
                    logger.info("已完成样本数据的标准化")
                    
                    # 8.5 预测
                    single_sample_prediction = rf_model.predict(sample_scaled)[0]
                    logger.info(f"单一样本预测结果: {single_sample_prediction:.4f}")
                    
                except Exception as single_pred_err:
                    logger.error(f"单样本预测失败: {str(single_pred_err)}")
                    logger.error(traceback.format_exc())
                    # 不显示错误弹窗，只在结果中标记预测失败
                    single_sample_prediction = None
            else:
                 logger.warning("没有从光谱数据文件中提取到有效数据进行单一样品预测。")

            # === 4. 保存结果 ===
            self.analysis_result = {
                # Overall metrics from prediction file
                'r2': r2, 'rmse': rmse, 'mae': mae, 'mre': mre,
                # Single prediction from spectrum file (if available)
                'single_sample_prediction': single_sample_prediction,
                # Other info
                'mean_pred': mean_pred, 'mean_actual': mean_actual, # Keep for popup reference
                'prediction_filename': os.path.basename(self.file_path),
                'spectrum_filename': os.path.basename(self.spectrum_file_path),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_type': "掺伪度分析",
                'result_str': f"R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MRE: {mre:.2f}%", # Base result str
                'chart_path': chart_path, 'spectrum_path': spectrum_path,
                'predictions': y_pred.tolist(), 'actuals': y_actual.tolist()
            }
            
            # 创建结果表格
            results_df = pd.DataFrame({
                "样本编号": range(1, len(y_actual) + 1),
                "实际掺伪度": y_actual,
                "预测掺伪度": y_pred,
                "误差": y_pred - y_actual,
                "相对误差(%)": np.abs((y_pred - y_actual) / y_actual) * 100
            })
            
            # 在主线程中更新UI
            self.after(0, self.update_results_ui)
            
        except Exception as e:
            logger.error(f"分析过程中出错: {str(e)}")
            logger.error(traceback.format_exc()) # 记录完整堆栈跟踪
            self.after(0, lambda: self.show_error("分析错误", f"分析过程中出错: {str(e)}"))
            self.after(0, self.hide_progress)
    
    def _is_numeric_string(self, val):
        """检查一个字符串是否可以被转换为数值"""
        if not isinstance(val, str):
            return True  # 非字符串直接返回True
        
        try:
            float(val)
            return True
        except:
            return False
    
    def msc_correction(self, sdata):
        """
        对光谱数据进行MSC（主成分标准化）。
        :param sdata: 原始光谱数据，行是样本，列是波长（特征）
        :return: MSC 标准化后的数据
        """
        n = sdata.shape[0]  # 样本数量
        msc_corrected_data = np.zeros_like(sdata)

        for i in range(n):
            y = sdata[i, :]
            mean_y = np.mean(y)
            std_y = np.std(y)
            if std_y == 0:  # 避免除以0
                msc_corrected_data[i, :] = y - mean_y
            else:
                msc_corrected_data[i, :] = (y - mean_y) / std_y  # 标准化处理

        return msc_corrected_data
    
    def update_results_ui(self):
        """更新结果UI（在主线程中运行）"""
        try:
            if not hasattr(self, 'analysis_result') or self.analysis_result is None:
                logger.error("没有可用的分析结果")
                self.hide_progress()
                self.show_error("错误", "没有可用的分析结果")
                return
            
            # 记录分析结果到数据库
            self.db_manager.add_analysis_record(
                self.user_id,
                self.analysis_result['prediction_filename'],
                self.analysis_result['result_str'],
                self.analysis_result['analysis_type']
            )
            
            # 先清空结果框架中的所有组件
            for widget in self.result_frame.winfo_children():
                widget.destroy()
                
            # === 第一部分：结果标题 ===
            self.result_title = ctk.CTkLabel(
                self.result_frame, 
                text="分析结果", 
                font=ctk.CTkFont(size=18, weight="bold")
            )
            self.result_title.pack(pady=(20, 10))
            
            # === 第二部分：简化的文字分析结果 ===
            text_frame = ctk.CTkFrame(self.result_frame)
            text_frame.pack(fill="x", padx=20, pady=(0, 15))
            
            # Display the four metrics
            result_text = f"决定系数 (R²) = {self.analysis_result['r2']:.4f}\n"
            result_text += f"均方根误差 (RMSE) = {self.analysis_result['rmse']:.4f}\n"
            result_text += f"平均绝对误差 (MAE) = {self.analysis_result['mae']:.4f}\n"
            result_text += f"平均相对误差 (MRE) = {self.analysis_result['mre']:.2f}%"
            
            # Add the single sample prediction if available
            single_pred = self.analysis_result.get('single_sample_prediction')
            if single_pred is not None:
                 result_text += f"\n光谱文件样品预测掺伪度 = {single_pred:.4f}" # Add the new line
            else:
                 result_text += f"\n光谱文件样品预测掺伪度 = 未能预测" # Indicate if prediction failed
            
            self.result_value = ctk.CTkLabel(
                text_frame, 
                text=result_text,
                font=ctk.CTkFont(size=16, family="Courier"),
                justify="left"
            )
            self.result_value.pack(pady=10, padx=20, anchor="w")
            
            # === 第四部分：图表按钮区 ===
            chart_button_frame = ctk.CTkFrame(self.result_frame, fg_color="transparent")
            chart_button_frame.pack(fill="x", padx=20, pady=(5, 10))
            
            # 创建左右布局
            button_left_frame = ctk.CTkFrame(chart_button_frame, fg_color="transparent")
            button_left_frame.pack(side="left", fill="y", padx=(0, 10))
            
            button_right_frame = ctk.CTkFrame(chart_button_frame, fg_color="transparent")
            button_right_frame.pack(side="right", fill="y", padx=(10, 0))
            
            # 掺伪度预测图表按钮
            view_pred_button = ctk.CTkButton(
                button_left_frame,
                text="查看预测掺伪度图表",
                width=150,
                height=30,
                font=ctk.CTkFont(size=14),
                command=self.show_prediction_popup
            )
            view_pred_button.pack(pady=5)
            
            # 光谱图表按钮 - 只有当生成了光谱图时才显示
            if 'spectrum_path' in self.analysis_result and self.analysis_result['spectrum_path']:
                view_spectrum_button = ctk.CTkButton(
                    button_right_frame,
                    text="查看光谱数据图表",
                    width=150,
                    height=30,
                    font=ctk.CTkFont(size=14),
                    command=self.show_spectrum_popup
                )
                view_spectrum_button.pack(pady=5)
            
            # === 第五部分：图表保存路径显示 ===
            path_frame = ctk.CTkFrame(self.result_frame, fg_color="transparent")
            path_frame.pack(fill="x", padx=20, pady=(5, 10))
            
            # 显示掺伪度图表路径
            chart_path = self.analysis_result['chart_path']
            pred_file_label = ctk.CTkLabel(
                path_frame,
                text=f"预测数据文件: {self.analysis_result['prediction_filename']}",
                font=ctk.CTkFont(size=12),
                justify="left"
            )
            pred_file_label.pack(pady=(0, 1), anchor="w")
            
            pred_path_label = ctk.CTkLabel(
                path_frame,
                text=f"预测图表路径: {chart_path}",
                font=ctk.CTkFont(size=12),
                text_color="gray",
                justify="left"
            )
            pred_path_label.pack(pady=(0, 1), anchor="w")
            
            # 如果有光谱图，显示光谱图路径
            if 'spectrum_path' in self.analysis_result and self.analysis_result['spectrum_path']:
                spectrum_path = self.analysis_result['spectrum_path']
                spectrum_file_label = ctk.CTkLabel(
                    path_frame,
                    text=f"光谱数据文件: {self.analysis_result['spectrum_filename']}",
                    font=ctk.CTkFont(size=12),
                    justify="left"
                )
                spectrum_file_label.pack(pady=(0, 1), anchor="w")
                
                spectrum_path_label = ctk.CTkLabel(
                    path_frame,
                    text=f"光谱图表路径: {spectrum_path}",
                    font=ctk.CTkFont(size=12),
                    text_color="gray",
                    justify="left"
                )
                spectrum_path_label.pack(pady=(0, 1), anchor="w")
            
            # 隐藏进度条
            self.hide_progress()
            
            logger.info("分析结果UI更新完成")
            
        except Exception as e:
            logger.error(f"更新UI时出错: {str(e)}")
            self.hide_progress()
            self.show_error("错误", f"更新结果UI时出错: {str(e)}")
    
    def _load_chart_image(self, parent_frame, loading_label):
        """延迟加载图表图像"""
        try:
            chart_path = self.analysis_result['chart_path']
            
            # 确保文件存在
            if not os.path.exists(chart_path):
                logger.warning(f"图表文件不存在: {chart_path}，等待1秒后重试")
                # 等待片刻，确保文件保存完成
                import time
                time.sleep(1)
                
                if not os.path.exists(chart_path):
                    loading_label.configure(text=f"图表文件不存在: {os.path.basename(chart_path)}")
                    return
            
            # 多次尝试加载图像
            max_attempts = 5  # 增加尝试次数
            attempt = 0
            success = False
            last_error = None
            
            while attempt < max_attempts and not success:
                try:
                    # 尝试加载图像 - 直接以更大尺寸显示
                    img = Image.open(chart_path)
                    
                    # 获取父框架大小来调整图像尺寸
                    parent_width = parent_frame.winfo_width() or 600  # 如果尚未布局，使用默认值600
                    parent_height = parent_frame.winfo_height() or 450  # 默认高度450
                    
                    # 如果框架尺寸不合理，使用预设值
                    if parent_width < 100:
                        parent_width = 600
                    if parent_height < 100:
                        parent_height = 450
                        
                    # 计算合适的大小，保持纵横比，并充分利用空间
                    img_width, img_height = img.size
                    ratio = min(parent_width / img_width, parent_height / img_height)
                    new_width = int(img_width * ratio * 0.95)  # 使用95%的空间
                    new_height = int(img_height * ratio * 0.95)
                    
                    # 调整图像大小
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    img_tk = ImageTk.PhotoImage(img)
                    
                    # 删除加载标签
                    loading_label.destroy()
                    
                    # 创建一个标签来显示图片
                    img_label = tk.Label(parent_frame, image=img_tk)
                    img_label.image = img_tk  # 保持引用以防止垃圾回收
                    img_label.pack(fill='both', expand=True, padx=5, pady=5)
                    
                    # 添加查看大图按钮（保留此按钮，以便查看更详细的图表）
                    view_button = ctk.CTkButton(
                        parent_frame, 
                        text="查看大图", 
                        width=100,
                        command=self.show_chart_popup
                    )
                    view_button.pack(pady=(5, 0))
                    
                    success = True
                    logger.info(f"成功加载图表: {chart_path}")
                    
                except Exception as img_err:
                    attempt += 1
                    last_error = str(img_err)
                    logger.warning(f"加载图表尝试 {attempt}/{max_attempts} 失败: {last_error}")
                    
                    # 更新加载标签显示尝试状态
                    loading_label.configure(text=f"正在加载图表... ({attempt}/{max_attempts})")
                    self.update_idletasks()  # 刷新UI
                    
                    # 等待时间随尝试次数增加
                    time.sleep(0.5 + 0.5 * attempt)
            
            # 如果多次尝试后仍失败，显示错误信息
            if not success:
                loading_label.configure(
                    text=f"无法加载图表\n{os.path.basename(chart_path)}\n请点击下方按钮查看",
                    font=ctk.CTkFont(size=12)
                )
                
                # 仍然添加查看大图按钮，使用户可以手动尝试
                view_button = ctk.CTkButton(
                    parent_frame, 
                    text="查看大图", 
                    width=100,
                    command=self.show_chart_popup
                )
                view_button.pack(pady=(15, 0))
                
                # 记录详细错误
                logger.error(f"多次尝试后无法加载图表: {last_error}")
                
        except Exception as e:
            logger.error(f"延迟加载图表时出错: {str(e)}")
            if loading_label.winfo_exists():
                loading_label.configure(text=f"加载图表出错: {str(e)}")
            
            # 添加查看大图按钮作为备选方案
            view_button = ctk.CTkButton(
                parent_frame, 
                text="查看大图", 
                width=100,
                command=self.show_chart_popup
            )
            view_button.pack(pady=(15, 0))
    
    def show_prediction_popup(self):
        """显示掺伪度预测图表大图弹窗"""
        try:
            if not hasattr(self, 'analysis_result') or 'chart_path' not in self.analysis_result:
                logger.error("没有可用的预测图表")
                self.show_error("错误", "没有可用的预测图表")
                return
                
            chart_path = self.analysis_result['chart_path']
                
            # 确保文件存在
            if not os.path.exists(chart_path):
                logger.warning(f"预测图表文件不存在: {chart_path}")
                self.show_error("错误", f"找不到预测图表文件: {chart_path}")
                return
            
            # 创建一个新窗口
            popup = tk.Toplevel()
            popup.title("掺伪度预测结果 - 详细图表")
            
            # 设置图标（如果有）
            try:
                popup.iconbitmap("assets/icon.ico")
            except:
                pass  # 忽略图标设置错误
                
            # 计算合适的窗口大小 - 使用屏幕尺寸的90%
            screen_width = popup.winfo_screenwidth()
            screen_height = popup.winfo_screenheight()
            window_width = int(screen_width * 0.9)
            window_height = int(screen_height * 0.9)
            
            # 设置窗口大小和位置
            popup.geometry(f"{window_width}x{window_height}+{int((screen_width-window_width)/2)}+{int((screen_height-window_height)/2)}")
            
            # 创建主框架
            main_frame = tk.Frame(popup)
            main_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            # 添加标题
            title_frame = tk.Frame(main_frame)
            title_frame.pack(fill='x', pady=(0, 20))
            
            title_label = tk.Label(
                title_frame, 
                text="白酒掺伪度预测结果", 
                font=("Helvetica", 16, "bold")
            )
            title_label.pack(side='left')
            
            # 添加一个提示标签
            loading_frame = tk.Frame(main_frame)
            loading_frame.pack(fill='x', pady=10)
            
            loading_label = tk.Label(loading_frame, text="正在加载高清图表...", font=("Helvetica", 12))
            loading_label.pack(pady=10)
            popup.update()  # 刷新窗口以显示加载标签
            
            # 尝试加载图像
            try:
                # 加载图像
                img = Image.open(chart_path)
                
                # 调整图像大小以适应窗口，保持宽高比
                img_width, img_height = img.size
                canvas_width = window_width - 40  # 减去padding
                canvas_height = window_height - 200  # 减去其他元素的高度
                
                ratio = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                # 使用高质量的调整方法
                img = img.resize((new_width, new_height), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                
                # 移除加载标签
                loading_frame.destroy()
                
                # 创建画布以显示图像
                canvas_frame = tk.Frame(main_frame)
                canvas_frame.pack(fill='both', expand=True)
                
                canvas = tk.Canvas(canvas_frame, width=new_width, height=new_height, bg='white', highlightthickness=0)
                canvas.pack(pady=10)
                
                # 在画布中央显示图像
                canvas.create_image(new_width//2, new_height//2, image=img_tk)
                canvas.image = img_tk  # 保持引用以防止垃圾回收
                
            except Exception as img_err:
                loading_label.config(text=f"加载图表失败: {str(img_err)}", fg="red")
                logger.error(f"加载大图时出错: {str(img_err)}")
                return
            
            # 添加统计信息面板
            info_frame = tk.Frame(main_frame, relief=tk.GROOVE, borderwidth=1)
            info_frame.pack(fill='x', pady=10)
            
            stats_label = tk.Label(info_frame, text="统计信息", font=("Helvetica", 12, "bold"))
            stats_label.pack(pady=(10, 5))
            
            # 添加R², RMSE等信息
            stats_content = f"""
            预测数据文件: {self.analysis_result['prediction_filename']}
            光谱数据文件: {self.analysis_result['spectrum_filename']}
            ------------------------------
            决定系数 (R²) = {self.analysis_result['r2']:.4f}
            均方根误差 (RMSE) = {self.analysis_result['rmse']:.4f}
            平均绝对误差 (MAE) = {self.analysis_result['mae']:.4f}
            平均相对误差 (MRE) = {self.analysis_result['mre']:.2f}%
            ------------------------------
            平均预测掺伪度 = {self.analysis_result['mean_pred']:.4f}
            平均实际掺伪度 = {self.analysis_result['mean_actual']:.4f}
            样本数量 = {len(self.analysis_result['predictions'])}
            分析时间 = {self.analysis_result['datetime']}
            """
            stats_text = tk.Label(
                info_frame, 
                text=stats_content.strip(), 
                font=("Courier", 11),
                justify="left",
                padx=20
            )
            stats_text.pack(pady=(0, 10))
            
            # 添加底部按钮面板
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill='x', pady=10)
            
            # 添加保存按钮
            save_button = tk.Button(
                button_frame, 
                text="保存图表", 
                command=lambda: self._save_chart_image(chart_path),
                font=("Helvetica", 10),
                padx=15
            )
            save_button.pack(side="left", padx=5)
            
            # 添加关闭按钮
            close_button = tk.Button(
                button_frame, 
                text="关闭", 
                command=popup.destroy,
                font=("Helvetica", 10),
                padx=15
            )
            close_button.pack(side="right", padx=5)
            
            # 添加图表路径信息
            path_label = tk.Label(
                main_frame, 
                text=f"图表文件: {os.path.basename(chart_path)}",
                font=("Helvetica", 8),
                fg="gray"
            )
            path_label.pack(side="bottom", pady=(5, 0), anchor='e')
            
            logger.info("预测图表弹窗已显示")
            
        except Exception as e:
            logger.error(f"显示预测图表弹窗时出错: {str(e)}")
            self.show_error("错误", f"显示预测图表弹窗时出错: {str(e)}")
            
    def show_spectrum_popup(self):
        """显示光谱数据图表大图弹窗"""
        try:
            if not hasattr(self, 'analysis_result') or 'spectrum_path' not in self.analysis_result or not self.analysis_result['spectrum_path']:
                logger.error("没有可用的光谱图表")
                self.show_error("错误", "没有可用的光谱图表")
                return
                
            spectrum_path = self.analysis_result['spectrum_path']
                
            # 确保文件存在
            if not os.path.exists(spectrum_path):
                logger.warning(f"光谱图表文件不存在: {spectrum_path}")
                self.show_error("错误", f"找不到光谱图表文件: {spectrum_path}")
                return
            
            # 创建一个新窗口
            popup = tk.Toplevel()
            popup.title("白酒光谱数据可视化")
            
            # 设置图标（如果有）
            try:
                popup.iconbitmap("assets/icon.ico")
            except:
                pass  # 忽略图标设置错误
                
            # 计算合适的窗口大小 - 使用屏幕尺寸的90%
            screen_width = popup.winfo_screenwidth()
            screen_height = popup.winfo_screenheight()
            window_width = int(screen_width * 0.9)
            window_height = int(screen_height * 0.9)
            
            # 设置窗口大小和位置
            popup.geometry(f"{window_width}x{window_height}+{int((screen_width-window_width)/2)}+{int((screen_height-window_height)/2)}")
            
            # 创建主框架
            main_frame = tk.Frame(popup)
            main_frame.pack(fill='both', expand=True, padx=20, pady=20)
            
            # 添加标题
            title_frame = tk.Frame(main_frame)
            title_frame.pack(fill='x', pady=(0, 20))
            
            title_label = tk.Label(
                title_frame, 
                text="白酒光谱数据可视化", 
                font=("Helvetica", 16, "bold")
            )
            title_label.pack(side='left')
            
            # 添加一个提示标签
            loading_frame = tk.Frame(main_frame)
            loading_frame.pack(fill='x', pady=10)
            
            loading_label = tk.Label(loading_frame, text="正在加载高清图表...", font=("Helvetica", 12))
            loading_label.pack(pady=10)
            popup.update()  # 刷新窗口以显示加载标签
            
            # 尝试加载图像
            try:
                # 加载图像
                img = Image.open(spectrum_path)
                
                # 调整图像大小以适应窗口，保持宽高比
                img_width, img_height = img.size
                canvas_width = window_width - 40  # 减去padding
                canvas_height = window_height - 200  # 减去其他元素的高度
                
                ratio = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                # 使用高质量的调整方法
                img = img.resize((new_width, new_height), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(img)
                
                # 移除加载标签
                loading_frame.destroy()
                
                # 创建画布以显示图像
                canvas_frame = tk.Frame(main_frame)
                canvas_frame.pack(fill='both', expand=True)
                
                canvas = tk.Canvas(canvas_frame, width=new_width, height=new_height, bg='white', highlightthickness=0)
                canvas.pack(pady=10)
                
                # 在画布中央显示图像
                canvas.create_image(new_width//2, new_height//2, image=img_tk)
                canvas.image = img_tk  # 保持引用以防止垃圾回收
                
            except Exception as img_err:
                loading_label.config(text=f"加载图表失败: {str(img_err)}", fg="red")
                logger.error(f"加载光谱图时出错: {str(img_err)}")
                return
            
            # 添加说明信息面板
            info_frame = tk.Frame(main_frame, relief=tk.GROOVE, borderwidth=1)
            info_frame.pack(fill='x', pady=10)
            
            info_label = tk.Label(info_frame, text="图表说明", font=("Helvetica", 12, "bold"))
            info_label.pack(pady=(10, 5))
            
            # 添加图表说明
            info_content = """
            此图表展示了样品的光谱数据可视化结果。
            X轴代表波长或频率值，Y轴代表对应的光谱强度值。
            蓝色点表示实际数据点，虚线表示趋势线。
            通过观察光谱曲线的特征峰和谷，可以辅助判断样品的纯度和成分。
            """
            info_text = tk.Label(
                info_frame, 
                text=info_content.strip(), 
                font=("Helvetica", 11),
                justify="left",
                padx=20
            )
            info_text.pack(pady=(0, 10))
            
            # 添加底部按钮面板
            button_frame = tk.Frame(main_frame)
            button_frame.pack(fill='x', pady=10)
            
            # 添加保存按钮
            save_button = tk.Button(
                button_frame, 
                text="保存图表", 
                command=lambda: self._save_chart_image(spectrum_path),
                font=("Helvetica", 10),
                padx=15
            )
            save_button.pack(side="left", padx=5)
            
            # 添加关闭按钮
            close_button = tk.Button(
                button_frame, 
                text="关闭", 
                command=popup.destroy,
                font=("Helvetica", 10),
                padx=15
            )
            close_button.pack(side="right", padx=5)
            
            # 添加图表路径信息
            path_label = tk.Label(
                main_frame, 
                text=f"图表文件: {os.path.basename(spectrum_path)}",
                font=("Helvetica", 8),
                fg="gray"
            )
            path_label.pack(side="bottom", pady=(5, 0), anchor='e')
            
            logger.info("光谱图表弹窗已显示")
            
        except Exception as e:
            logger.error(f"显示光谱图表弹窗时出错: {str(e)}")
            self.show_error("错误", f"显示光谱图表弹窗时出错: {str(e)}")
    
    def _save_chart_image(self, chart_path):
        """保存图表到用户指定位置"""
        try:
            if not os.path.exists(chart_path):
                self.show_error("错误", "原图表文件不存在，无法保存")
                return
                
            # 选择保存路径
            save_path = filedialog.asksaveasfilename(
                title="保存图表",
                defaultextension=".png",
                filetypes=[("PNG图像", "*.png"), ("所有文件", "*.*")],
                initialdir="history"
            )
            
            if not save_path:
                return
                
            # 复制图表文件
            import shutil
            shutil.copy2(chart_path, save_path)
            
            self.show_info("保存成功", f"图表已保存至:\n{save_path}")
            
        except Exception as e:
            logger.error(f"保存图表时出错: {str(e)}")
            self.show_error("保存错误", f"无法保存图表: {str(e)}")
    
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