import numpy as np
import os
from tqdm import tqdm
from Ann import ANN
from utils import generate_replacement, forward_kinematics

class IterativeMTSolver:
    def __init__(self, initial_data, optimal_data):
        self.initial_X, self.initial_y = initial_data
        self.optimal_X, self.optimal_y = optimal_data
        
        # Initilize baseline model
        self.baseline_model = ANN()
        self.baseline_model.train(self.initial_X, self.initial_y)
        self.baseline_mse, self.baseline_nmse = self.baseline_model.evaluate(self.optimal_X, self.optimal_y)
        
        # Current best model
        self.current_X = self.initial_X.copy()
        self.current_y = self.initial_y.copy()
        self.best_mse = float(self.baseline_mse)
        self.best_nmse = float(self.baseline_nmse)
        self.best_X = self.current_X.copy()
        self.best_y = self.current_y.copy()
        
        # Train optimal model using optimal dataset
        self.optimal_model = ANN()
        self.optimal_model.train(self.optimal_X, self.optimal_y)
    
    def run_iteration(self, num_replace, trial_id):
        # Skip if no replacement is needed
        if num_replace == 0:
            return False, self.best_mse, self.best_nmse
        
        # Generate replacement
        np.random.seed(trial_id)
        replace_idx = np.random.choice(len(self.current_X), num_replace, False)
        new_samples = generate_replacement(self.current_X, num_replace, trial_id)
        
        # Create temp dataset
        temp_X = self.current_X.copy()
        temp_X[replace_idx] = new_samples
        temp_y = forward_kinematics(temp_X)
        
        # Evaluate a freshly trained ANN model using the modified dataset (temp_X, temp_y)
        model = ANN()
        model.train(temp_X, temp_y)
        current_mse, current_nmse = model.evaluate(self.optimal_X, self.optimal_y)
        
        # Compare and update better results
        if current_mse < self.best_mse:
            self.best_mse = current_mse
            self.best_nmse = current_nmse
            self.best_X = temp_X.copy()
            self.best_y = temp_y.copy()
            self.current_X = temp_X.copy()  # Accept better results
            return True, current_mse, current_nmse
        else:
            return False, self.best_mse, self.best_nmse  # Decline worse results
    
    def full_optimization(self, num_replace, trials=50):
        """Full run of the iterative optimization process"""
        progress = tqdm(total=trials, desc=f"Replacement {num_replace}")
        improvement_log = []
        
        for trial in range(trials):
            improved, mse, nmse = self.run_iteration(num_replace, trial)
            improvement_log.append({
                "trial": trial+1,
                "mse": mse,
                "nmse": nmse,
                "improved": improved
            })
            progress.update()
            progress.set_postfix({"Best MSE": f"{self.best_mse:.6f}"})
        
        # Save final results
        os.makedirs("results", exist_ok=True)
        np.save(f"results/best_{num_replace}_X.npy", self.best_X)
        np.save(f"results/best_{num_replace}_y.npy", self.best_y)
        
        return {
            "initial_mse": self.baseline_mse,
            "final_mse": self.best_mse,
            "initial_nmse": self.baseline_nmse,
            "final_nmse": self.best_nmse,
            "improvements": improvement_log
        }