import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging
import time

# 设置日志
logging.basicConfig(
    filename='logs/model.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_training')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

class BaijiuModel:
    """白酒掺伪量预测和分类模型"""
    
    def __init__(self):
        """初始化模型"""
        # 创建模型目录
        os.makedirs('models', exist_ok=True)
        
        # 回归模型（掺伪量预测）的文件路径
        self.regression_model_path = 'models/regression_model.joblib'
        # 分类模型（真伪分类）的文件路径
        self.classification_model_path = 'models/classification_model.joblib'
        # 数据预处理器的文件路径
        self.preprocessor_path = 'models/preprocessor.joblib'
        
        # 初始化模型
        self.regression_model = None
        self.classification_model = None
        self.preprocessor = {
            'imputer': SimpleImputer(strategy='mean'),
            'scaler': StandardScaler(),
            'pca': PCA(n_components=5)
        }
        
        # 尝试加载已有模型
        self.load_models()
    
    def load_models(self):
        """加载已有的模型和预处理器"""
        try:
            # 加载回归模型
            if os.path.exists(self.regression_model_path):
                self.regression_model = joblib.load(self.regression_model_path)
                logger.info("回归模型加载成功")
            else:
                logger.warning("回归模型文件不存在，需要训练")
            
            # 加载分类模型
            if os.path.exists(self.classification_model_path):
                self.classification_model = joblib.load(self.classification_model_path)
                logger.info("分类模型加载成功")
            else:
                logger.warning("分类模型文件不存在，需要训练")
            
            # 加载预处理器
            if os.path.exists(self.preprocessor_path):
                self.preprocessor = joblib.load(self.preprocessor_path)
                logger.info("预处理器加载成功")
            else:
                logger.warning("预处理器文件不存在，需要重新创建")
                
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
    
    def preprocess_data(self, X_data, is_training=False):
        """预处理数据
        
        Args:
            X_data: 输入数据
            is_training: 是否是训练过程
            
        Returns:
            预处理后的数据
        """
        try:
            # 填充缺失值
            if is_training:
                X_processed = self.preprocessor['imputer'].fit_transform(X_data)
            else:
                X_processed = self.preprocessor['imputer'].transform(X_data)
            
            # MSC 标准化
            X_processed = self._msc_correction(X_processed)
            
            # PCA 降维
            if is_training:
                X_processed = self.preprocessor['pca'].fit_transform(X_processed)
            else:
                X_processed = self.preprocessor['pca'].transform(X_processed)
            
            # 标准化
            if is_training:
                X_processed = self.preprocessor['scaler'].fit_transform(X_processed)
            else:
                X_processed = self.preprocessor['scaler'].transform(X_processed)
            
            return X_processed
        
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise
    
    def _msc_correction(self, sdata):
        """对光谱数据进行MSC（乘性散射校正）标准化
        
        Args:
            sdata: 原始光谱数据，行是样本，列是波长（特征）
            
        Returns:
            MSC 标准化后的数据
        """
        n = sdata.shape[0]  # 样本数量
        msc_corrected_data = np.zeros_like(sdata)
        
        for i in range(n):
            y = sdata[i, :]
            mean_y = np.mean(y)
            std_y = np.std(y)
            msc_corrected_data[i, :] = (y - mean_y) / std_y  # 标准化处理
        
        return msc_corrected_data
    
    def train_regression_model(self, file_path, test_size=0.2, random_state=42):
        """训练回归模型（掺伪量预测）
        
        Args:
            file_path: 训练数据文件路径
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练结果字典，包含评估指标
        """
        start_time = time.time()
        try:
            # 读取数据
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                data = pd.read_excel(file_path, header=None)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path, header=None)
            else:
                raise ValueError("不支持的文件格式，请使用.xlsx、.xls或.csv文件")
            
            # 提取特征
            X = data.values
            
            # 为每7行分配浓度标签
            n_samples = X.shape[0]
            y = np.repeat(np.linspace(0.0385, 0.5, 25), 7)[:n_samples]  # 3.85%到50%的浓度值
            
            # 选择光谱数据（去掉第一列）
            X = X[:, 1:]
            
            # 预处理数据
            X_processed = self.preprocess_data(X, is_training=True)
            
            # 保存预处理器
            joblib.dump(self.preprocessor, self.preprocessor_path)
            logger.info("预处理器保存成功")
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=random_state
            )
            
            # 创建并训练随机森林回归模型
            self.regression_model = RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_split=5, 
                min_samples_leaf=2, random_state=random_state, n_jobs=-1
            )
            self.regression_model.fit(X_train, y_train)
            
            # 保存模型
            joblib.dump(self.regression_model, self.regression_model_path)
            logger.info("回归模型保存成功")
            
            # 模型评估
            y_train_pred = self.regression_model.predict(X_train)
            y_test_pred = self.regression_model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # 计算特征重要性
            feature_importance = self.regression_model.feature_importances_
            
            # 计算训练时间
            training_time = time.time() - start_time
            
            # 返回结果
            result = {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "feature_importance": feature_importance,
                "training_time": training_time
            }
            
            logger.info(f"回归模型训练成功，测试集R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"回归模型训练失败: {str(e)}")
            raise
    
    def train_classification_model(self, file_path, test_size=0.2, random_state=42):
        """训练分类模型（真伪分类）
        
        Args:
            file_path: 训练数据文件路径
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练结果字典，包含评估指标
        """
        start_time = time.time()
        try:
            # 读取数据
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                data = pd.read_excel(file_path, header=None)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path, header=None)
            else:
                raise ValueError("不支持的文件格式，请使用.xlsx、.xls或.csv文件")
            
            # 提取特征和标签
            X = data.iloc[:, 1:].values  # 第2列到最后一列作为特征
            y = data.iloc[:, 0].values   # 第1列作为标签
            
            # 预处理数据
            X_processed = self.preprocess_data(X, is_training=True)
            
            # 保存预处理器（如果还没有保存）
            if not os.path.exists(self.preprocessor_path):
                joblib.dump(self.preprocessor, self.preprocessor_path)
                logger.info("预处理器保存成功")
            
            # 分割训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=random_state
            )
            
            # 创建并训练随机森林分类模型
            self.classification_model = RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5, 
                min_samples_leaf=2, random_state=random_state, n_jobs=-1
            )
            self.classification_model.fit(X_train, y_train)
            
            # 保存模型
            joblib.dump(self.classification_model, self.classification_model_path)
            logger.info("分类模型保存成功")
            
            # 模型评估
            y_train_pred = self.classification_model.predict(X_train)
            y_test_pred = self.classification_model.predict(X_test)
            
            train_accuracy = np.mean(y_train_pred == y_train)
            test_accuracy = np.mean(y_test_pred == y_test)
            
            # 计算特征重要性
            feature_importance = self.classification_model.feature_importances_
            
            # 计算训练时间
            training_time = time.time() - start_time
            
            # 返回结果
            result = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "feature_importance": feature_importance,
                "training_time": training_time
            }
            
            logger.info(f"分类模型训练成功，测试集准确率: {test_accuracy:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"分类模型训练失败: {str(e)}")
            raise
    
    def predict_adulteration(self, file_path):
        """预测掺伪量（回归预测）
        
        Args:
            file_path: 预测数据文件路径
            
        Returns:
            预测结果字典
        """
        try:
            # 检查模型是否已加载
            if self.regression_model is None:
                raise ValueError("回归模型未加载，请先训练模型")
            
            # 读取数据
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                data = pd.read_excel(file_path, header=None)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path, header=None)
            else:
                raise ValueError("不支持的文件格式，请使用.xlsx、.xls或.csv文件")
            
            # 提取特征
            if data.shape[1] == 1:  # 只有一列，可能是样本ID列
                raise ValueError("数据格式错误，至少需要包含光谱数据")
            elif data.shape[1] == 2:  # 两列，第一列是样本ID，第二列是光谱数据
                X = data.iloc[:, 1].values.reshape(-1, 1)  # 第二列作为特征
            else:  # 多列，第一列是样本ID，其余列是光谱数据
                X = data.iloc[:, 1:].values  # 第2列到最后一列作为特征
            
            # 预处理数据
            X_processed = self.preprocess_data(X, is_training=False)
            
            # 预测
            predictions = self.regression_model.predict(X_processed)
            
            # 获取样本ID
            sample_ids = data.iloc[:, 0].values
            
            # 整理结果
            results = []
            for i, sample_id in enumerate(sample_ids):
                results.append({
                    "sample_id": sample_id,
                    "predicted_adulteration": predictions[i],
                    "predicted_percentage": f"{predictions[i]*100:.2f}%"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"掺伪量预测失败: {str(e)}")
            raise
    
    def classify_baijiu(self, file_path):
        """分类白酒真伪（分类预测）
        
        Args:
            file_path: 预测数据文件路径
            
        Returns:
            分类结果字典
        """
        try:
            # 检查模型是否已加载
            if self.classification_model is None:
                raise ValueError("分类模型未加载，请先训练模型")
            
            # 读取数据
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                data = pd.read_excel(file_path, header=None)
            elif file_path.endswith('.csv'):
                data = pd.read_csv(file_path, header=None)
            else:
                raise ValueError("不支持的文件格式，请使用.xlsx、.xls或.csv文件")
            
            # 提取特征
            if data.shape[1] == 1:  # 只有一列，可能是样本ID列
                raise ValueError("数据格式错误，至少需要包含光谱数据")
            elif data.shape[1] == 2:  # 两列，第一列是样本ID，第二列是光谱数据
                X = data.iloc[:, 1].values.reshape(-1, 1)  # 第二列作为特征
            else:  # 多列，第一列是样本ID，其余列是光谱数据
                X = data.iloc[:, 1:].values  # 第2列到最后一列作为特征
            
            # 预处理数据
            X_processed = self.preprocess_data(X, is_training=False)
            
            # 预测
            predictions = self.classification_model.predict(X_processed)
            
            # 获取样本ID
            sample_ids = data.iloc[:, 0].values
            
            # 分类结果映射
            class_names = {
                0: "真酒",
                1: "掺水假酒",
                2: "掺工业酒精假酒"
            }
            
            # 整理结果
            results = []
            for i, sample_id in enumerate(sample_ids):
                results.append({
                    "sample_id": sample_id,
                    "predicted_class": predictions[i],
                    "predicted_label": class_names.get(predictions[i], "未知")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"白酒真伪分类失败: {str(e)}")
            raise

    def plot_regression_result(self, actual_values, predicted_values, save_path=None):
        """绘制回归结果图
        
        Args:
            actual_values: 实际值
            predicted_values: 预测值
            save_path: 图像保存路径(可选)
            
        Returns:
            图像对象
        """
        try:
            plt.figure(figsize=(8, 6))
            
            # 散点图
            plt.scatter(actual_values, predicted_values, color='#1f77b4', alpha=0.7, label='预测值')
            
            # 理想预测线
            min_val = min(min(actual_values), min(predicted_values))
            max_val = max(max(actual_values), max(predicted_values))
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='理想预测线')
            
            # 添加标签和标题
            plt.xlabel("实际掺伪量", fontsize=12)
            plt.ylabel("预测掺伪量", fontsize=12)
            plt.title("白酒掺伪量预测结果", fontsize=14)
            
            # 添加R²值
            r2 = r2_score(actual_values, predicted_values)
            rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
            plt.annotate(f"R² = {r2:.4f}\nRMSE = {rmse:.4f}", 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
            
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            # 保存图像
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"绘制回归结果图失败: {str(e)}")
            raise

    def plot_classification_result(self, actual_classes, predicted_classes, class_names=None, save_path=None):
        """绘制分类结果图
        
        Args:
            actual_classes: 实际类别
            predicted_classes: 预测类别
            class_names: 类别名称，默认为None
            save_path: 图像保存路径(可选)
            
        Returns:
            图像对象
        """
        try:
            from sklearn.metrics import confusion_matrix
            
            # 生成混淆矩阵
            cm = confusion_matrix(actual_classes, predicted_classes)
            
            # 设置类别名称
            if class_names is None:
                class_names = ["真酒", "掺水假酒", "掺工业酒精假酒"]
            
            plt.figure(figsize=(8, 6))
            
            # 绘制混淆矩阵热图
            sns_plot = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title("混淆矩阵", fontsize=14)
            plt.colorbar()
            
            # 设置坐标轴
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            
            # 添加文本注释
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('实际类别', fontsize=12)
            plt.xlabel('预测类别', fontsize=12)
            plt.tight_layout()
            
            # 计算准确率
            accuracy = np.trace(cm) / np.sum(cm)
            
            # 添加准确率注释
            plt.annotate(f"准确率 = {accuracy:.4f}", 
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=10, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))
            
            # 保存图像
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"绘制分类结果图失败: {str(e)}")
            raise

# 如果直接运行此脚本，执行示例训练
if __name__ == "__main__":
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='训练白酒掺伪量预测和分类模型')
    parser.add_argument('--data', type=str, help='训练数据文件路径')
    parser.add_argument('--type', type=str, choices=['regression', 'classification'], 
                       help='模型类型: regression(回归)或classification(分类)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    if args.data and args.type:
        model = BaijiuModel()
        
        if args.type == 'regression':
            print("训练回归模型(掺伪量预测)...")
            result = model.train_regression_model(args.data)
            print(f"训练完成，测试集R²: {result['test_r2']:.4f}, RMSE: {result['test_rmse']:.4f}")
        else:
            print("训练分类模型(真伪分类)...")
            result = model.train_classification_model(args.data)
            print(f"训练完成，测试集准确率: {result['test_accuracy']:.4f}")
    else:
        print("请提供数据文件路径和模型类型")
        print("示例: python train_model.py --data data/baijiu_data.xlsx --type regression") 