import numpy as np
import os
from utils import forward_kinematics

def generate_initial_data():
    """Generate and save the initial dataset"""
    # Generate random joint angles and calculate the forward kinematics
    np.random.seed(42)
    X = np.random.uniform(
        low=[[0, -np.pi/2]],
        high=[[np.pi/2, np.pi/2]],
        size=(200, 2)
    )
    y = forward_kinematics(X)
    
    os.makedirs("saved_data", exist_ok=True)
    np.save("saved_data/initial_X.npy", X)
    np.save("saved_data/initial_y.npy", y)
    return X, y

if __name__ == "__main__":
    # Generate and save the data when the script is run directly
    x, y = generate_initial_data()
    print("Initial dataset generated and saved successfully!")
    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")