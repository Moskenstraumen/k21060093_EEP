from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

class ANN:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='tanh',
                alpha=0.001,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=5000,
                early_stopping=True,
                n_iter_no_change=50,
                validation_fraction=0.2,
                random_state=42,
            ))
        ])
    
    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        # 使用独立验证集计算真实MSE
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        real_val_mse = mean_squared_error(y_val, y_pred)
    
        # 获取训练损失曲线
        mlp = self.model.named_steps['mlp']
        train_loss = mlp.loss_curve_
    
        # 生成模拟验证曲线
        val_loss = [real_val_mse] * len(train_loss)  # 简化为常量曲线
    
        return train_loss, val_loss

    
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)

        # 保持维度一致性
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # 计算整体MSE（标量）
        mse = mean_squared_error(y_test, y_pred)
        
        # 改进方差计算
        y_var = np.var(y_test, axis=0)
        valid_var = y_var[y_var > 1e-8]  # 过滤无效方差
        nmse = mse / np.mean(valid_var) if len(valid_var) > 0 else float('nan')
        
        return float(mse), float(nmse)  # 修改返回值为元组
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path):
        model = cls()
        model.model = joblib.load(path)
        return model