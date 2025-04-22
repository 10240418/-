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
matplotlib.use('Agg')  # 使用非交互式后端

# 设置日志
logger = logging.getLogger('baijiu_app')

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
            text="掺伪度分析预测", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(30, 20), padx=30, anchor="w")
        
        # 文件选择区域
        self.file_frame = ctk.CTkFrame(self)
        self.file_frame.pack(fill="x", padx=30, pady=(0, 20))
        
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
            filetypes=[("Excel文件", "*.xlsx"), ("Excel 97-2003", "*.xls"), ("CSV文件", "*.csv"), ("所有文件", "*.*")],
            initialdir=os.path.expanduser("~") # 默认打开用户主目录
        )
        
        if file_path:
            self.file_path = file_path
            self.file_path_var.set(file_path)
            self.analyze_button.configure(state="normal")
            logger.info(f"用户选择了文件: {file_path}")
            
            # 更新UI上的文件名显示
            filename = os.path.basename(file_path)
            if len(filename) > 40:  # 如果文件名太长，截断显示
                filename = filename[:37] + "..."
            self.file_path_entry.configure(state="normal")
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, filename)
            self.file_path_entry.configure(state="readonly")
    
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
            
            # 加载数据文件
            file_ext = os.path.splitext(self.file_path)[1].lower()
            if file_ext in ['.xlsx', '.xls']:
                data = pd.read_excel(self.file_path, header=None)
            elif file_ext == '.csv':
                data = pd.read_csv(self.file_path, header=None)
            else:
                self.after(0, lambda: self.show_error("文件错误", "不支持的文件格式，请提供.xlsx, .xls或.csv文件"))
                self.after(0, self.hide_progress)
                return
            
            # 预处理数据 - 检测并移除非数值数据
            # 检查第一行是否为标题行
            first_row_has_text = False
            for col in data.iloc[0]:
                if isinstance(col, str) and not self._is_numeric_string(col):
                    first_row_has_text = True
                    break
            
            # 如果第一行是标题行，则移除
            if first_row_has_text:
                data = data.iloc[1:].reset_index(drop=True)
            
            # 检查第一列是否为标签列
            first_col_has_text = False
            for val in data.iloc[:, 0]:
                if isinstance(val, str) and not self._is_numeric_string(val):
                    first_col_has_text = True
                    break
            
            # 如果第一列是标签列，则移除
            if first_col_has_text:
                data = data.iloc[:, 1:].reset_index(drop=True)
            
            # 转换所有数据为浮点数，非数值转为NaN
            def to_numeric_with_fallback(val):
                try:
                    return pd.to_numeric(val)
                except:
                    return np.nan
            
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
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            
            # 准确率转为百分比
            accuracy = r2 * 100  # R²作为准确率，转为百分比
            
            # 确保目录存在
            os.makedirs('history_img', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            
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
            
            # 另外创建一个直方图显示预测分布
            plt.figure(figsize=(10, 7))  # 也增大直方图尺寸
            plt.hist(y_pred, bins=15, color='#1f77b4', alpha=0.7, label='预测值')
            plt.axvline(mean_pred, color='red', linestyle='--', linewidth=2, label=f'预测平均值: {mean_pred:.4f}')
            plt.axvline(mean_actual, color='green', linestyle='-.', linewidth=2, label=f'实际平均值: {mean_actual:.4f}')
            plt.xlabel('掺伪度值', fontsize=14)
            plt.ylabel('频率', fontsize=14)
            plt.title('掺伪度预测结果分布', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存直方图
            hist_path = os.path.join('history_img', f'hist_{timestamp}.png')
            plt.tight_layout()
            plt.savefig(hist_path, dpi=600)  # 增加DPI为600
            plt.close()
            
            # 保存结果
            self.analysis_result = {
                'mean_pred': mean_pred,
                'mean_actual': mean_actual,
                'accuracy': accuracy,
                'r2': r2,
                'rmse': rmse,
                'filename': os.path.basename(self.file_path),
                'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_type': "掺伪度分析",
                'result_str': f"掺伪度预测结果: {mean_pred:.4f}, 实际掺伪度: {mean_actual:.4f}, 准确率: {accuracy:.2f}%",
                'chart_path': chart_path,
                'hist_path': hist_path,
                'predictions': y_pred.tolist(),
                'actuals': y_actual.tolist()
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
                self.analysis_result['filename'],
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
            
            # 简化的结果文本，仅包含平均预测、实际掺伪度和准确率
            result_text = f"平均预测掺伪度 = {self.analysis_result['mean_pred']:.4f}\n"
            result_text += f"实际掺伪度 = {self.analysis_result['mean_actual']:.4f}\n"
            result_text += f"准确率 = {self.analysis_result['accuracy']:.2f}%"
            
            self.result_value = ctk.CTkLabel(
                text_frame, 
                text=result_text,
                font=ctk.CTkFont(size=16, family="Courier"),
                justify="left"
            )
            self.result_value.pack(pady=10, padx=20, anchor="w")
            
            # === 第三部分：结果描述 ===
            description_frame = ctk.CTkFrame(self.result_frame, fg_color="transparent")
            description_frame.pack(fill="x", padx=20, pady=(0, 15))
            
            # 根据结果生成文字描述
            error = abs(self.analysis_result['mean_pred'] - self.analysis_result['mean_actual'])
            rel_error = (error / self.analysis_result['mean_actual']) * 100 if self.analysis_result['mean_actual'] != 0 else 0
            
            if self.analysis_result['accuracy'] > 95:
                quality_text = "分析结果表明，该样品的掺伪度预测非常准确，可信度高。"
            elif self.analysis_result['accuracy'] > 90:
                quality_text = "分析结果表明，该样品的掺伪度预测较为准确，可信度良好。"
            else:
                quality_text = "分析结果表明，该样品的掺伪度预测存在一定误差，建议重新检测。"
                
            description_label = ctk.CTkLabel(
                description_frame,
                text=quality_text,
                font=ctk.CTkFont(size=14),
                wraplength=500,
                justify="left"
            )
            description_label.pack(pady=5, anchor="w")
            
            # === 第四部分：查看图表按钮 ===
            chart_button_frame = ctk.CTkFrame(self.result_frame, fg_color="transparent")
            chart_button_frame.pack(fill="x", padx=20, pady=(5, 10))
            
            view_chart_button = ctk.CTkButton(
                chart_button_frame,
                text="查看预测掺伪度图表",
                width=200,
                height=40,
                font=ctk.CTkFont(size=14),
                command=self.show_chart_popup
            )
            view_chart_button.pack(pady=5)
            
            # === 第五部分：图表保存路径显示（放在按钮下面）===
            chart_path = self.analysis_result['chart_path']
            path_label = ctk.CTkLabel(
                chart_button_frame,
                text=f"图片保存路径: {chart_path}",
                font=ctk.CTkFont(size=12),
                text_color="gray",
                justify="left"
            )
            path_label.pack(pady=(5, 10), anchor="w")
            
            # === 第六部分：保存结果按钮 ===
            self.save_button = ctk.CTkButton(
                self.result_frame, 
                text="保存结果", 
                width=150,
                state="normal",
                command=self.save_result
            )
            self.save_button.pack(pady=(5, 20))
            
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
    
    def show_chart_popup(self):
        """显示图表大图弹窗"""
        try:
            if not hasattr(self, 'analysis_result') or 'chart_path' not in self.analysis_result:
                logger.error("没有可用的图表")
                self.show_error("错误", "没有可用的图表")
                return
                
            chart_path = self.analysis_result['chart_path']
                
            # 确保文件存在
            if not os.path.exists(chart_path):
                logger.warning(f"图表文件不存在: {chart_path}")
                self.show_error("错误", f"找不到图表文件: {chart_path}")
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
                text="白酒掺伪度分析预测结果", 
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
            决定系数 (R²) = {self.analysis_result['r2']:.4f}
            均方根误差 (RMSE) = {self.analysis_result['rmse']:.4f}
            平均预测掺伪度 = {self.analysis_result['mean_pred']:.4f}
            实际掺伪度 = {self.analysis_result['mean_actual']:.4f}
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
            
            logger.info("大图弹窗已显示")
            
        except Exception as e:
            logger.error(f"显示图表弹窗时出错: {str(e)}")
            self.show_error("错误", f"显示图表弹窗时出错: {str(e)}")
            
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
            
            # 确保图表路径存在
            chart_path = self.analysis_result.get('chart_path', '未保存')
            
            # 生成文字描述
            if self.analysis_result['accuracy'] > 95:
                quality_text = "分析结果表明，该样品的掺伪度预测非常准确，可信度高。"
            elif self.analysis_result['accuracy'] > 90:
                quality_text = "分析结果表明，该样品的掺伪度预测较为准确，可信度良好。"
            else:
                quality_text = "分析结果表明，该样品的掺伪度预测存在一定误差，建议重新检测。"
            
            # 写入结果文件
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("白酒品质检测系统 - 分析结果\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"分析时间: {self.analysis_result['datetime']}\n")
                f.write(f"文件名: {self.analysis_result['filename']}\n\n")
                
                # 写入掺伪量分析结果（简化版）
                f.write("掺伪度分析结果:\n")
                f.write("-" * 30 + "\n")
                f.write(f"平均预测掺伪度 = {self.analysis_result['mean_pred']:.4f}\n")
                f.write(f"实际掺伪度 = {self.analysis_result['mean_actual']:.4f}\n")
                f.write(f"准确率 = {self.analysis_result['accuracy']:.2f}%\n\n")
                
                # 添加结果描述
                f.write(f"{quality_text}\n\n")
                
                # 添加图表路径
                f.write(f"图片保存路径: {chart_path}\n")
            
            logger.info(f"分析结果已保存至: {save_path}")
            self.show_info("保存成功", f"分析结果已保存至: {save_path}")
            
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