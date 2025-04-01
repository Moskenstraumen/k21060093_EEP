import numpy as np
import os
from tqdm import tqdm
from Ann import ANN
from utils import generate_replacement, forward_kinematics

class IterativeMTSolver:
    def __init__(self, initial_data, optimal_data, dataset_num=1):
        self.initial_X, self.initial_y = initial_data
        self.optimal_X, self.optimal_y = optimal_data
        self.dataset_num = dataset_num
        
        # Ensure consistent y shape
        if len(self.initial_y.shape) == 1:
            self.initial_y = self.initial_y.reshape(-1, 1)
        if len(self.optimal_y.shape) == 1:
            self.optimal_y = self.optimal_y.reshape(-1, 1)
        
        # Initialize baseline model
        self.baseline_model = ANN()
        # Let ANN handle the train/validation split and normalization internally
        self.baseline_model.train(self.initial_X, self.initial_y, verbose=False)
        
        # Get baseline metrics
        baseline_metrics = self.baseline_model.evaluate(self.optimal_X, self.optimal_y)
        self.baseline_mse = baseline_metrics['mse']
        self.baseline_mae = baseline_metrics['mae']
        self.baseline_r2 = baseline_metrics['r2']
        
        # Initialize current best state
        self.current_X = self.initial_X.copy()
        self.current_y = self.initial_y.copy()
        self.best_mse = self.baseline_mse
        self.best_mae = self.baseline_mae
        self.best_r2 = self.baseline_r2
        self.best_X = self.current_X.copy()
        self.best_y = self.current_y.copy()

    def run_iteration(self, num_replace, trial_id):
        if num_replace == 0:
            return False, self.best_mse, self.best_mae, self.best_r2
        
        # Generate replacement samples
        np.random.seed(trial_id)
        replace_idx = np.random.choice(len(self.current_X), num_replace, False)
        new_samples = generate_replacement(self.current_X, num_replace, trial_id)
        
        # Create temporary dataset
        temp_X = self.current_X.copy()
        temp_X[replace_idx] = new_samples
        temp_y = forward_kinematics(temp_X)
        
        if len(temp_y.shape) == 1:
            temp_y = temp_y.reshape(-1, 1)
        
        # Train new model - let ANN handle normalization
        model = ANN()
        model.train(temp_X, temp_y, verbose=False)
        
        # Evaluate
        metrics = model.evaluate(self.optimal_X, self.optimal_y)
        
        # Update best results if improved
        if metrics['mse'] < self.best_mse:
            self.best_mse = metrics['mse']
            self.best_mae = metrics['mae']
            self.best_r2 = metrics['r2']
            self.best_X = temp_X.copy()
            self.best_y = temp_y.copy()
            self.current_X = temp_X  # Accept improvement
            return True, metrics['mse'], metrics['mae'], metrics['r2']
        
        return False, self.best_mse, self.best_mae, self.best_r2

    def full_optimization(self, num_replace, trials=50):
        # Ensure baseline uses exact same conditions
        self.baseline_model = ANN()
        self.baseline_model.train(self.initial_X, self.initial_y, verbose=False)
        baseline_metrics = self.baseline_model.evaluate(self.optimal_X, self.optimal_y)
        
        # Use same model for 0 replacement case
        if num_replace == 0:
            return {
                "initial_mse": baseline_metrics['mse'],
                "initial_mae": baseline_metrics['mae'],
                "initial_r2": baseline_metrics['r2'],
                "improvements": []  # Empty list for no improvements
            }
        
        progress = tqdm(total=trials, desc=f"Replacement {num_replace}")
        improvement_log = []
        
        for trial in range(trials):
            improved, mse, mae, r2 = self.run_iteration(num_replace, trial)
            improvement_log.append({
                "trial": trial + 1,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "improved": improved
            })
            progress.update()
            progress.set_postfix({
                "MSE": f"{mse:.4f}",
                "MAE": f"{mae:.4f}",
                "R²": f"{r2:.2%}"
            })

        # Save best dataset
        results_dir = f"results/dataset{self.dataset_num}"
        os.makedirs(results_dir, exist_ok=True)
        np.save(f"{results_dir}/best_{num_replace}_X.npy", self.best_X)
        np.save(f"{results_dir}/best_{num_replace}_y.npy", self.best_y)
        
        return {
            "initial_mse": self.baseline_mse,
            "initial_mae": self.baseline_mae,
            "initial_r2": self.baseline_r2,
            "final_mse": self.best_mse,
            "final_mae": self.best_mae,
            "final_r2": self.best_r2,
            "improvements": improvement_log
        }
