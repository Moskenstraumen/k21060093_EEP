from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import joblib
import numpy as np
import warnings

class ANN:
    def __init__(self):
        self.base_config = {
            'hidden_layer_sizes': (64, 32),
            'activation': 'tanh',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 10,  # 每次只训练10个epoch
            'warm_start': True,  # 允许增量训练
            'n_iter_no_change': 15,
            'validation_fraction': 0.2,
            'random_state': 42
        }
        self.model = MLPRegressor(**self.base_config)  # 统一使用model属性
        self.best_weights = None
    
    def train(self, X, y, max_epochs=500):
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 初始化模型（避免首次fit警告）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.model.fit(X_train, y_train)
        
        '''# 克隆基础配置
        current_model = clone(self.model)
        current_model.fit(X_train, y_train)  # 初始化'''
        
        train_loss = []
        val_loss = []
        best_loss = np.inf
        
        for epoch in range(max_epochs):
            # 增量训练（实际执行1个epoch）
            self.model.partial_fit(X_train, y_train)
            
            # 记录训练损失
            train_loss.append(self.model.loss_)
            
            # 计算验证损失
            y_pred = self.model.predict(X_val)
            current_val_loss = mean_squared_error(y_val, y_pred)
            val_loss.append(current_val_loss)
            
            # 更新最佳权重
            if current_val_loss < best_loss * 0.999:
                best_loss = current_val_loss
                self.best_weights = [w.copy() for w in self.model.coefs_]
                
            # 动态早停检测
            if (epoch > 20 and 
                np.mean(val_loss[-10:]) >= np.mean(val_loss[-20:-10])):
                break
                
        # 恢复最佳权重
        self.model.coefs_ = self.best_weights
        return train_loss, val_loss


    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        # 维度处理
        y_test = np.asarray(y_test).reshape(-1, 2)
        y_pred = np.asarray(y_pred).reshape(-1, 2)

        # 计算MSE
        mse = mean_squared_error(y_test, y_pred)
        
        # 改进方差计算
        y_var = np.var(y_test, axis=0)
        valid_var = y_var[y_var > 1e-8]  # 过滤无效方差
        nmse = mse / np.mean(valid_var) if len(valid_var) > 0 else np.nan
        
        return float(mse), float(nmse)  # 修改返回值为元组
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path):
        model = cls()
        model.model = joblib.load(path)
        return model