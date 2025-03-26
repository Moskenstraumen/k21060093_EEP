from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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
    
    def train(self, X, y, max_epochs=1000, verbose=True):
        # 新增参数类型校验
        if not isinstance(verbose, bool):
            raise ValueError("verbose参数必须为布尔类型")
            
        # 增加数据标准化
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 初始化模型（避免首次fit警告）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.model.fit(X_train, y_train)
        
        train_loss = []
        val_loss = []
        best_loss = np.inf
        
        best_epoch = 0
        early_stop_flag = False
        for epoch in range(max_epochs):
            if not early_stop_flag:
                # 继续正常训练流程
                self.model.partial_fit(X_train, y_train)
                
                # 记录损失和验证指标
                train_loss.append(self.model.loss_)
                current_val_loss = mean_squared_error(y_val, self.model.predict(X_val))
                val_loss.append(current_val_loss)
                
                # 更新最佳权重
                if current_val_loss < best_loss * 0.999:
                    best_loss = current_val_loss
                    self.best_weights = [w.copy() for w in self.model.coefs_]
                    best_epoch = epoch
                
                # 检测早停条件但不中断循环
                # 改为动态容忍度
                patience = max(20, int(0.1 * max_epochs))  # 至少20个epoch，最多10%总epoch数
                if (epoch - best_epoch) > patience:
                # if (epoch - best_epoch) > 2 * self.base_config['n_iter_no_change']:
                    early_stop_flag = True
                    if verbose:  # 添加条件判断
                        print(f"\nEarly stopping triggered at epoch {epoch}, continuing for visualization...")
            else:
                # 早停后保持记录最后的最佳值
                train_loss.append(train_loss[-1])
                val_loss.append(val_loss[-1])
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