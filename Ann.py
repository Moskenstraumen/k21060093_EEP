# Ann.py
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import warnings

# Disable convergence warning
warnings.filterwarnings('ignore', category=UserWarning)

class ExponentialLRScheduler:
    def __init__(self, initial_lr, decay_rate=0.95, decay_steps=50):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
    def get_lr(self, epoch):
        """Calculate learning rate for given epoch"""
        decay_factor = self.decay_rate ** (epoch / self.decay_steps)
        return self.initial_lr * decay_factor

class ANN:
    def __init__(self):
        self.base_config = {
            'hidden_layer_sizes': (8, 4),
            'activation': 'tanh',
            'solver': 'adam',
            'alpha': 0.6, 
            'batch_size': 32, 
            'learning_rate_init': 0.002, 
            'max_iter': 1000, 
            'random_state': 42, 
            'early_stopping': True,
            'validation_fraction': 0.2,
            'n_iter_no_change': 50, 
            'tol': 1e-5,
            'warm_start': True  # Enable warm start for lr scheduling
        }
        self.model = MLPRegressor(**self.base_config)
        self.scaler = StandardScaler()
        self.X_val = None
        self.y_val = None

        # Initialize learning rate scheduler
        self.lr_scheduler = ExponentialLRScheduler(
            initial_lr=self.base_config['learning_rate_init'],
            decay_rate=0.95,
            decay_steps=50
        )

    def train(self, X, y, verbose=True):
        '''Predicts end-effector position (y_train) from joint angles (X_train)'''
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.base_config['validation_fraction'], 
            random_state=42
        )
        # Store validation data
        self.X_val = X_val
        self.y_val = y_val
        
        # Normalizing data
        X_train_scaled = np.clip(self.scaler.fit_transform(X_train), -3, 3)
        X_val_scaled = np.clip(self.scaler.transform(X_val), -3, 3)
        
        try:
            config = self.base_config.copy()
            config.update({
                'early_stopping': False,
                'validation_fraction': 0.0,
                'max_iter': 1  # Train one iteration at a time
            })
            
            # Initialize model
            self.model = MLPRegressor(**config)
            
            train_losses = []
            val_losses = []
            best_val_loss = float('inf')
            early_stop_epoch = None
            patience_counter = 0
            current_lr = self.base_config['learning_rate_init']
            
            # Train for all epochs while tracking early stopping point
            for epoch in range(self.base_config['max_iter']):
                # Update learning rate
                current_lr = self.lr_scheduler.get_lr(epoch)
                self.model.learning_rate_init = current_lr
                    
                # Fit one epoch
                self.model.fit(X_train_scaled, y_train)
                    
                # Calculate losses
                train_pred = self.model.predict(X_train_scaled)
                val_pred = self.model.predict(X_val_scaled)
                    
                train_loss = np.mean((train_pred - y_train) ** 2)
                val_loss = np.mean((val_pred - y_val) ** 2)
                    
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                    
                # Track early stopping point
                if val_loss < best_val_loss - self.base_config['tol']:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                # Mark early stopping point but continue training
                if patience_counter >= self.base_config['n_iter_no_change'] and early_stop_epoch is None:
                    early_stop_epoch = epoch + 1 - self.base_config['n_iter_no_change']
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            return [], [], None
        
        if verbose:
            print(f"\nTraining Summary:")
            print(f"Total epochs: {len(train_losses)}")
            print(f"Early stopping epoch: {early_stop_epoch}")
            print(f"Initial LR: {self.base_config['learning_rate_init']:.6f}")
            print(f"Final LR: {current_lr:.6f}")
            print(f"Best validation MSE: {best_val_loss:.6f}")
        
        return np.array(train_losses), np.array(val_losses), early_stop_epoch

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