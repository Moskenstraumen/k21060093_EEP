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
            'hidden_layer_sizes': (15, 8),
            'activation': 'tanh',
            'solver': 'adam',
            'alpha': 0.001,  # Reduced regularization
            'batch_size': 16,  # More stable batch size
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.01,  # Lower initial rate
            'max_iter': 1,
            'random_state': 42,
            'tol': 1e-8
        }
        self.max_epochs = 1000  # More epochs
        self.patience = 100     # More patience
        self.min_improvement = 1e-7  # Stricter improvement threshold
        self.validation_fraction = 0.2
        self.model = MLPRegressor(**self.base_config)
        self.scaler = StandardScaler()
        self.X_val = None
        self.y_val = None
        
    def train(self, X, y, verbose=True):
        # Split and scale data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_fraction, random_state=42
        )
        
        # Store validation data
        self.X_val = X_val
        self.y_val = y_val
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Training loop with improved convergence checking
        best_val_loss = float('inf')
        best_model_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        try:
            for epoch in range(self.max_epochs):
                # Warmup learning rate for first 200 epochs
                if epoch < 200:
                    self.model.learning_rate_init = 0.005 * (1 + epoch/200)
                
                # Train one epoch
                self.model.partial_fit(X_train_scaled, y_train)
                
                # Compute losses
                train_pred = self.model.predict(X_train_scaled)
                val_pred = self.model.predict(X_val_scaled)
                
                train_loss = np.mean((train_pred - y_train) ** 2)
                val_loss = np.mean((val_pred - y_val) ** 2)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Compute moving average of losses
                if len(train_losses) >= 10:
                    avg_train_loss = np.mean(train_losses[-10:])
                    avg_val_loss = np.mean(val_losses[-10:])
                else:
                    avg_train_loss = train_loss
                    avg_val_loss = val_loss
                
                # Improved convergence checking
                improvement = (best_val_loss - val_loss) / best_val_loss if best_val_loss != float('inf') else 1
                
                if avg_val_loss < best_val_loss - self.min_improvement:
                    best_val_loss = avg_val_loss
                    best_model_loss = train_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping conditions
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                        print(f"Best validation loss: {best_val_loss:.6f}")
                        print(f"Final training loss: {train_loss:.6f}")
                    break
                
                # Check for convergence
                if len(train_losses) > 50:
                    recent_loss_std = np.std(train_losses[-20:])
                    if recent_loss_std < self.min_improvement and train_loss < 1e-4:
                        if verbose:
                            print(f"Converged at epoch {epoch}")
                        break
                
                if verbose and epoch % 100 == 0:
                    print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
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