import os
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 设置日志
logger = logging.getLogger('baijiu_app')

def train_rf_classifier(data_path=None, save_path=None):
    """
    训练白酒分类的随机森林模型并保存
    """
    # 如果未指定路径，使用默认路径
    if data_path is None:
        # 尝试获取工作目录下的训练数据
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'classification_data.xlsx')
        
        # 如果找不到文件，则使用classify_tool.py中的默认路径
        if not os.path.exists(data_path):
            data_path = "C:\\Users\\19176\\Desktop\\MSC-PCA(三维).xlsx"
    
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'rf_model.pkl')
    
    try:
        # 加载数据
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_excel(data_path)
        
        # 检查数据格式
        logger.info(f"Data shape: {df.shape}")
        
        # 假设最后一列是标签，前面的列是特征
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # 数据处理
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 标准化
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 训练随机森林模型
        logger.info("Training RandomForest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Test Precision: {precision:.4f}")
        logger.info(f"Test Recall: {recall:.4f}")
        logger.info(f"Test F1-score: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # 保存模型
        logger.info(f"Saving model to: {save_path}")
        joblib.dump(model, save_path)
        logger.info("RandomForest model training and saving complete!")
        
        return True
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # 设置基本日志配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 训练并保存模型
    train_rf_classifier() 