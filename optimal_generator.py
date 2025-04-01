# optimal_generator.py
import numpy as np
from generator import forward_kinematics

def generate_optimal_data(n_samples=200):
    X = np.random.uniform(0, np.pi/2, (n_samples, 2))  # Input: joinst angles in radians
    y = forward_kinematics(X)  # Ouput: end-effector position
    
    # Add noise
    y += np.random.normal(0, 0.05, size=y.shape)
    
    np.save('saved_data/optimal_X.npy', X)
    np.save('saved_data/optimal_y.npy', y)
    return X, y

if __name__ == "__main__":
    # Generate and save the data when the script is run directly
    x, y = generate_optimal_data()
    print("Optimal dataset generated and saved successfully!")
    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")