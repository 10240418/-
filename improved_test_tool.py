import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import sys
import joblib
from tkinter import filedialog
import tkinter as tk
import datetime

# 设置matplotlib支持中文
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")


def select_file():
    """使用文件对话框选择Excel文件"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择数据文件",
        filetypes=[("Excel文件", "*.xlsx *.xls"), ("CSV文件", "*.csv"), ("所有文件", "*.*")]
    )
    return file_path


def load_data(file_path):
    """加载并预处理数据文件"""
    print(f"正在读取文件: {file_path}")
    
    # 获取文件扩展名
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 根据扩展名读取文件
    if file_ext in ['.xlsx', '.xls']:
        data = pd.read_excel(file_path, header=None)
    elif file_ext == '.csv':
        data = pd.read_csv(file_path, header=None)
    else:
        raise ValueError("不支持的文件格式，请提供.xlsx, .xls或.csv文件")
    
    # 数据预处理 - 检测并移除非数值数据
    print(f"原始数据形状: {data.shape}")
    
    # 检查第一行是否为标题行（包含文本数据）
    first_row_has_text = False
    for col in data.iloc[0]:
        if isinstance(col, str) and not is_numeric_string(col):
            first_row_has_text = True
            break
    
    # 如果第一行是标题行，则移除
    if first_row_has_text:
        print("检测到标题行，将移除第一行")
        data = data.iloc[1:].reset_index(drop=True)
    
    # 检查第一列是否为标签列（包含文本数据）
    first_col_has_text = False
    for val in data.iloc[:, 0]:
        if isinstance(val, str) and not is_numeric_string(val):
            first_col_has_text = True
            break
    
    # 如果第一列是标签列，则移除
    if first_col_has_text:
        print("检测到标签列，将移除第一列")
        data = data.iloc[:, 1:].reset_index(drop=True)
    
    # 检查并移除任何其他包含非数值数据的行和列
    # 1. 首先尝试转换所有数据为浮点数
    def to_numeric_with_fallback(val):
        try:
            return pd.to_numeric(val)
        except:
            return np.nan
    
    data = data.applymap(to_numeric_with_fallback)
    
    # 2. 移除包含太多NaN值的行列（超过50%）
    nan_rows = data.isna().mean(axis=1) > 0.5
    if nan_rows.any():
        print(f"移除包含大量非数值数据的行: {nan_rows.sum()}行")
        data = data.loc[~nan_rows].reset_index(drop=True)
    
    nan_cols = data.isna().mean(axis=0) > 0.5
    if nan_cols.any():
        print(f"移除包含大量非数值数据的列: {nan_cols.sum()}列")
        data = data.loc[:, ~nan_cols].reset_index(drop=True)
    
    # 3. 填充剩余的NaN值
    data = data.fillna(data.mean())
    
    print(f"预处理后数据形状: {data.shape}")
    return data


def is_numeric_string(val):
    """检查一个字符串是否可以被转换为数值"""
    if not isinstance(val, str):
        return True  # 非字符串直接返回True
    
    try:
        float(val)
        return True
    except:
        return False


def msc_correction(sdata):
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


def predict_sample(model, sample_data, pca_model=None, scaler=None):
    """对单个样本进行掺伪度预测
    
    Args:
        model: 训练好的随机森林模型
        sample_data: 需要预测的样本数据，应为2D数组
        pca_model: 可选，PCA模型，如果提供则会进行PCA降维
        scaler: 可选，标准化模型，如果提供则会进行标准化
        
    Returns:
        预测的掺伪度值
    """
    # 复制数据避免修改原数据
    X = np.array(sample_data).copy()
    
    # 应用MSC标准化
    X_msc = msc_correction(X)
    
    # 应用PCA降维（如果提供了PCA模型）
    if pca_model is not None:
        X_msc = pca_model.transform(X_msc)
    
    # 应用标准化（如果提供了标准化模型）
    if scaler is not None:
        X_msc = scaler.transform(X_msc)
    
    # 进行预测
    prediction = model.predict(X_msc)
    
    return prediction


def main():
    """主函数"""
    print("=" * 50)
    print("白酒掺伪度分析模型训练工具")
    print("=" * 50)
    
    # 1. 选择文件
    print("\n请选择数据文件...")
    file_path = select_file()
    if not file_path:
        print("未选择文件，程序退出")
        return
    
    # 2. 加载原始数据（保留标题行）用于绘制光谱图
    print("加载原始数据用于绘制光谱图...")
    try:
        # 获取文件扩展名
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # 读取原始数据（不进行预处理）
        if file_ext in ['.xlsx', '.xls']:
            raw_data = pd.read_excel(file_path, header=None)
        elif file_ext == '.csv':
            raw_data = pd.read_csv(file_path, header=None)
        else:
            raise ValueError("不支持的文件格式，请提供.xlsx, .xls或.csv文件")
        
        # 检查数据是否至少有两行（标题行和数据行）
        if raw_data.shape[0] >= 2:
            print(f"找到可用于绘图的数据: {raw_data.shape[0]} 行, {raw_data.shape[1]} 列")
            
            # 创建一个新的图形
            plt.figure(figsize=(12, 6))
            
            try:
                # 使用第一行作为X轴数据（波长或频率）
                x_values = raw_data.iloc[0].values
                # 尝试确定X轴标签（如果第一列有标签）
                x_label = "波长/频率"
                if isinstance(raw_data.iloc[0, 0], str) and not is_numeric_string(raw_data.iloc[0, 0]):
                    x_label = raw_data.iloc[0, 0]
                    # 从x_values中移除标签
                    x_values = x_values[1:]
                
                # 使用第二行作为Y轴数据（光谱值）
                y_values = raw_data.iloc[1].values
                # 尝试确定Y轴标签
                y_label = "光谱值"
                if isinstance(raw_data.iloc[1, 0], str) and not is_numeric_string(raw_data.iloc[1, 0]):
                    y_label = raw_data.iloc[1, 0]
                    # 从y_values中移除标签
                    y_values = y_values[1:]
                
                # 过滤掉非数值数据点
                valid_indices = []
                valid_x = []
                valid_y = []
                
                for i, (x, y) in enumerate(zip(x_values, y_values)):
                    try:
                        # 尝试转换为浮点数
                        x_float = float(x) if isinstance(x, str) else float(x)
                        y_float = float(y) if isinstance(y, str) else float(y)
                        
                        # 只有当x和y都能成功转换时，才添加到有效数据点
                        valid_x.append(x_float)
                        valid_y.append(y_float)
                        valid_indices.append(i)
                    except (ValueError, TypeError):
                        continue  # 跳过无法转换的值
                
                # 检查是否有足够的有效数据点来绘图
                if len(valid_x) > 1:
                    print(f"找到 {len(valid_x)} 个有效数据点用于绘制光谱图")
                    
                    # 绘制散点图
                    plt.scatter(valid_x, valid_y, color='#1f77b4', s=30, alpha=0.8, label="数据点")
                    
                    # 用虚线连接数据点
                    plt.plot(valid_x, valid_y, color='#1f77b4', linestyle='--', alpha=0.6, label="趋势线")
                    
                    # 添加图表标题和轴标签
                    plt.title("白酒光谱数据可视化", fontsize=14)
                    plt.xlabel(x_label, fontsize=12)
                    plt.ylabel(y_label, fontsize=12)
                    
                    # 添加网格线
                    plt.grid(True, linestyle='--', alpha=0.7)
                    
                    # 添加图例
                    plt.legend(loc='best')
                    
                    # 添加数据描述文本框
                    desc_text = f"数据文件: {os.path.basename(file_path)}\n"
                    desc_text += f"数据点数量: {len(valid_x)}\n"
                    desc_text += f"X轴范围: [{min(valid_x):.2f}, {max(valid_x):.2f}]\n"
                    desc_text += f"Y轴范围: [{min(valid_y):.2f}, {max(valid_y):.2f}]"
                    
                    # 在图表右上角添加文本框
                    plt.annotate(desc_text, xy=(0.02, 0.98), xycoords='axes fraction',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                              fontsize=10, ha='left', va='top')
                    
                    # 调整布局并保存图表
                    plt.tight_layout()
                    
                    # 确保目录存在
                    results_dir = "results"
                    if not os.path.exists(results_dir):
                        os.makedirs(results_dir)
                    
                    # 保存光谱图
                    spectrum_plot_path = os.path.join(results_dir, "spectrum_plot.png")
                    plt.savefig(spectrum_plot_path, dpi=300)
                    print(f"光谱图已保存至: {spectrum_plot_path}")
                    
                    # 显示图表
                    plt.show()
                else:
                    print("警告: 没有找到足够的有效数据点来绘制光谱图")
            except Exception as inner_e:
                print(f"处理光谱数据时出错: {str(inner_e)}")
        else:
            print("警告: 数据行数不足，无法绘制光谱图")
            
    except Exception as e:
        print(f"绘制光谱图时出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 3. 加载数据（进行预处理用于模型训练）
    try:
        data = load_data(file_path)
    except Exception as e:
        print(f"加载数据出错: {str(e)}")
        return
    
    # 确保数据不为空
    if data.empty or data.shape[0] < 10 or data.shape[1] < 2:
        print("数据量不足，无法进行分析。需要至少10行数据和2列特征。")
        return
    
    # 4. 提取特征
    X = data.values.astype(float)
    
    # 5. 生成浓度标签 (在0.0385到0.5之间)
    n_samples = X.shape[0]
    print(f"样本数量: {n_samples}")
    
    # 计算需要多少组标签（每7个样本为一组）
    n_groups = (n_samples + 6) // 7  # 向上取整
    print(f"生成{n_groups}组浓度标签")
    
    # 使用linspace生成线性间隔的浓度值 (0.0385, 0.5)
    y = np.repeat(np.linspace(0.0385, 0.5, n_groups), 7)[:n_samples]  # 3.85%, 7.41%, ..., 50.00% 对应每7行
    
    # 6. MSC 标准化
    print("正在进行MSC标准化...")
    X_msc = msc_correction(X)
    
    # 7. PCA 降维，保留最多5个主成分
    print("正在进行PCA降维...")
    n_components = min(5, X_msc.shape[1])  # 确保主成分数不超过特征数
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_msc)
    
    # 输出解释的方差比例
    print("\n每个主成分解释的方差比例：")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  主成分 {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    
    print(f"累积解释的方差比例：{np.cumsum(pca.explained_variance_ratio_)[-1]:.4f}")
    
    # 8. 标准化处理
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_pca = scaler.fit_transform(X_pca)
    
    # 9. 分割数据为训练集和测试集
    print("正在分割训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    print(f"训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    
    # 10. 训练模型
    print("\n正在训练随机森林回归模型...")
    
    # 简化版超参数网格（用于快速测试）
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20]
    }
    
    # 完整版超参数网格（取消注释以使用，但会大大增加训练时间）
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }
    
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
    
    print("开始网格搜索（这可能需要一些时间）...")
    grid_search.fit(X_train, y_train)
    
    # 11. 输出最佳超参数
    print(f"\n最佳超参数: {grid_search.best_params_}")
    
    # 12. 使用最佳参数训练模型
    best_rf_model = grid_search.best_estimator_
    
    # 13. 在训练集和测试集上评估模型
    y_train_pred = best_rf_model.predict(X_train)
    y_test_pred = best_rf_model.predict(X_test)
    
    # 14. 计算评价指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 输出评价指标
    print("\n模型评价指标:")
    print(f"训练集均方根误差（RMSE）：{train_rmse:.4f}")
    print(f"测试集均方根误差（RMSE）：{test_rmse:.4f}")
    print(f"训练集决定系数（R²）：{train_r2:.4f}")
    print(f"测试集决定系数（R²）：{test_r2:.4f}")
    
    # 15. 保存模型
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    model_path = os.path.join(models_dir, "rf_model.pkl")
    joblib.dump(best_rf_model, model_path)
    print(f"\n模型已保存至: {model_path}")
    
    # 同时保存PCA和标准化模型，以便后续预测使用
    pca_path = os.path.join(models_dir, "pca_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler_model.pkl")
    joblib.dump(pca, pca_path)
    joblib.dump(scaler, scaler_path)
    print(f"PCA模型已保存至: {pca_path}")
    print(f"标准化模型已保存至: {scaler_path}")
    
    # 16. 绘制实际值与预测值的散点图
    plt.figure(figsize=(10, 8))
    
    # 绘制测试集的预测结果散点图
    plt.scatter(y_test, y_test_pred, color='#1f77b4', label='测试集预测值', alpha=0.7)
    
    # 绘制理想预测线
    min_val = min(min(y_test), min(y_test_pred))
    max_val = max(max(y_test), max(y_test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='理想预测线')
    
    # 添加坐标轴标签和标题
    plt.xlabel("实际掺伪度", fontsize=12)
    plt.ylabel("预测掺伪度", fontsize=12)
    plt.title("白酒掺伪度预测模型性能", fontsize=14)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加R²和RMSE的文本标注
    text_str = f'测试集R² = {test_r2:.4f}\n测试集RMSE = {test_rmse:.4f}'
    plt.annotate(text_str, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                 fontsize=10, ha='left', va='top')
    
    # 设置图例
    plt.legend(loc='lower right')
    
    # 保存图表
    plt.tight_layout()
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    plt.savefig(os.path.join(results_dir, "prediction_plot.png"), dpi=300)
    print(f"预测结果图表已保存至: {os.path.join(results_dir, 'prediction_plot.png')}")
    
    # 显示图表
    plt.show()
    
    # 17. 输出实际掺伪度与预测掺伪度对比表格
    print("\n测试集掺伪度预测结果对比:")
    print("-" * 60)
    print(f"{'样本编号':^10}{'实际掺伪度':^15}{'预测掺伪度':^15}{'误差':^10}{'相对误差(%)':^10}")
    print("-" * 60)
    
    # 创建对比结果数据框
    results_df = pd.DataFrame({
        "样本编号": range(1, len(y_test) + 1),
        "实际掺伪度": y_test,
        "预测掺伪度": y_test_pred,
        "误差": y_test_pred - y_test,
        "相对误差(%)": np.abs((y_test_pred - y_test) / y_test) * 100
    })
    
    # 打印结果表格
    for i, (actual, pred) in enumerate(zip(y_test, y_test_pred)):
        error = pred - actual
        rel_error = abs(error / actual) * 100 if actual != 0 else float('inf')
        print(f"{i+1:^10}{actual:^15.4f}{pred:^15.4f}{error:^10.4f}{rel_error:^10.2f}")
    
    # 计算平均指标
    mean_abs_error = np.mean(np.abs(y_test_pred - y_test))
    mean_rel_error = np.mean(np.abs((y_test_pred - y_test) / y_test)) * 100
    
    print("-" * 60)
    print(f"平均绝对误差: {mean_abs_error:.4f}")
    print(f"平均相对误差: {mean_rel_error:.2f}%")
    
    # 保存结果到CSV文件
    csv_path = os.path.join(results_dir, "prediction_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n预测结果已保存至CSV文件: {csv_path}")
    
    # 18. 提供一个简单的预测示例
    print("\n模型使用示例:")
    print("从测试集中选择第一个样本进行预测演示...")
    
    sample_x = X_test[0].reshape(1, -1)
    sample_y_actual = y_test[0]
    sample_y_pred = best_rf_model.predict(sample_x)[0]
    
    print(f"实际掺伪度: {sample_y_actual:.4f}")
    print(f"预测掺伪度: {sample_y_pred:.4f}")
    print(f"误差: {sample_y_pred - sample_y_actual:.4f}")
    print(f"相对误差: {abs((sample_y_pred - sample_y_actual) / sample_y_actual) * 100:.2f}%")
    
    # 添加平均掺伪度预测结果的明确输出
    print("\n" + "=" * 30)
    print("白酒掺伪度分析结果")
    print("=" * 30)
    mean_actual = np.mean(y_test)
    mean_pred = np.mean(y_test_pred)
    print(f"平均实际掺伪度 = {mean_actual:.4f}")
    print(f"平均预测掺伪度 = {mean_pred:.4f}")
    print(f"准确率 = {test_r2 * 100:.2f}%")
    print("=" * 30)
    
    # 输出所有生成的图表路径
    print("\n生成的图表文件:")
    spectrum_plot_path = os.path.join(results_dir, "spectrum_plot.png")
    prediction_plot_path = os.path.join(results_dir, "prediction_plot.png")
    
    if os.path.exists(spectrum_plot_path):
        print(f"1. 光谱数据可视化: {spectrum_plot_path}")
    if os.path.exists(prediction_plot_path):
        print(f"2. 掺伪度预测结果: {prediction_plot_path}")
    
    # 将图表路径添加到CSV结果文件中
    try:
        with open(csv_path, 'a', encoding='utf-8') as f:
            f.write("\n\n# 分析图表路径\n")
            if os.path.exists(spectrum_plot_path):
                f.write(f"光谱数据可视化图表: {spectrum_plot_path}\n")
            if os.path.exists(prediction_plot_path):
                f.write(f"掺伪度预测结果图表: {prediction_plot_path}\n")
    except Exception as e:
        print(f"保存图表路径到CSV时出错: {str(e)}")
    
    # 创建一个总结报告
    summary_path = os.path.join(results_dir, "analysis_summary.txt")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("白酒掺伪度分析 - 结果总结\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {os.path.basename(file_path)}\n\n")
            
            f.write("模型性能指标:\n")
            f.write("-" * 40 + "\n")
            f.write(f"测试集均方根误差(RMSE): {test_rmse:.4f}\n")
            f.write(f"测试集决定系数(R²): {test_r2:.4f}\n")
            f.write(f"平均绝对误差: {mean_abs_error:.4f}\n")
            f.write(f"平均相对误差: {mean_rel_error:.2f}%\n\n")
            
            f.write("掺伪度分析结果:\n")
            f.write("-" * 40 + "\n")
            f.write(f"平均实际掺伪度 = {mean_actual:.4f}\n")
            f.write(f"平均预测掺伪度 = {mean_pred:.4f}\n")
            f.write(f"准确率 = {test_r2 * 100:.2f}%\n\n")
            
            f.write("生成的图表文件:\n")
            f.write("-" * 40 + "\n")
            if os.path.exists(spectrum_plot_path):
                f.write(f"1. 光谱数据可视化: {spectrum_plot_path}\n")
            if os.path.exists(prediction_plot_path):
                f.write(f"2. 掺伪度预测结果: {prediction_plot_path}\n\n")
            
            f.write("详细结果数据文件:\n")
            f.write("-" * 40 + "\n")
            f.write(f"CSV数据文件: {csv_path}\n")
        
        print(f"\n分析总结已保存至: {summary_path}")
    except Exception as e:
        print(f"创建结果总结时出错: {str(e)}")
    
    print("\n程序执行完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc() 