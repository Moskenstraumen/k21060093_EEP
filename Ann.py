from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import warnings

class ANN:
    def __init__(self):
        self.base_config = {
            'hidden_layer_sizes': (8, 4), 
            'activation': 'tanh',  
            'solver': 'adam',
            'alpha': 0.5, 
            'batch_size': 32, 
            'learning_rate': 'adaptive', 
            'learning_rate_init': 0.009, 
            'max_iter': 1,
            'random_state': 42,
            'epsilon': 1e-8,
            'tol': 1e-8
        }
        self.max_epochs = 1000
        self.patience = 100
        self.validation_fraction = 0.2
        self.min_delta = 1e-5
        self.model = MLPRegressor(**self.base_config)
        self.scaler = StandardScaler()
        self.X_val = None
        self.y_val = None
        
    def train(self, X, y, verbose=True):
        '''Predicts end-effector position (y_train) from joint angles (X_train)'''
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.validation_fraction, 
            random_state=42
        )
        
        # Store validation data
        self.X_val = X_val
        self.y_val = y_val
        
        # Scale data with clipping for robustness
        X_train_scaled = np.clip(self.scaler.fit_transform(X_train), -3, 3)
        X_val_scaled = np.clip(self.scaler.transform(X_val), -3, 3)
        
        # Training variables
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        try:
            for epoch in range(self.max_epochs):
                # Smoother learning rate decay
                if epoch > 0:
                    if epoch < 500:
                        self.model.learning_rate_init = 0.002
                    elif epoch < 1000:
                        self.model.learning_rate_init = 0.001
                    elif epoch < 1500:
                        self.model.learning_rate_init = 0.0005
                    else:
                        self.model.learning_rate_init = 0.0001
                
                # Train one epoch
                self.model.partial_fit(X_train_scaled, y_train)
                
                # Compute losses
                train_pred = self.model.predict(X_train_scaled)
                val_pred = self.model.predict(X_val_scaled)
                
                train_loss = np.mean((train_pred - y_train) ** 2)
                val_loss = np.mean((val_pred - y_val) ** 2)
                
                # Smoothing
                if train_losses:
                    train_loss = 0.9 * train_losses[-1] + 0.1 * train_loss
                    val_loss = 0.9 * val_losses[-1] + 0.1 * val_loss
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Early stopping with overfitting detection
                if val_loss < best_val_loss: 
                    best_val_loss = val_loss
                    best_model = [w.copy() for w in self.model.coefs_]
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Modified stopping conditions
                # Only stop if either:
                # 1. No improvement for long time, or
                # 2. Clear sign of overfitting
                if any([
                    patience_counter >= self.patience,  # No improvement for 100 epochs
                    (epoch > 1000 and  # Only check overfitting after 1000 epochs
                     val_loss > 2 * train_loss and  # Clear overfitting
                     patience_counter > 20)  # Ensure it's not temporary
                ]):
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
                
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}: "
                          f"train={train_loss:.6f}, val={val_loss:.6f}, "
                          f"lr={self.model.learning_rate_init:.6f}")
            
            # Restore best model
            if best_model is not None:
                self.model.coefs_ = best_model
            
            return np.array(train_losses), np.array(val_losses)
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return np.array([]), np.array([])

    def evaluate(self, X_test, y_test):
        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path):
        model = cls()
        model.model = joblib.load(path)
        return model