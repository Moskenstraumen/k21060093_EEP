import numpy as np
import os
from utils import forward_kinematics

def generate_initial_data(method=1):
    """Generate and save the initial dataset using different methods"""
    np.random.seed(42)
    
    if method == 1:
        # Original uniform random method
        X = np.random.uniform(
            low=[[0, -np.pi/2]],
            high=[[np.pi/2, np.pi/2]],
            size=(200, 2)
        )
    elif method == 2:
        # Grid-based sampling
        x1 = np.linspace(0, np.pi/2, 15)
        x2 = np.linspace(-np.pi/2, np.pi/2, 14)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.column_stack((X1.ravel(), X2.ravel()))
        # Add some noise
        X += np.random.normal(0, 0.05, X.shape)
        X = np.clip(X, [0, -np.pi/2], [np.pi/2, np.pi/2])
    else:
        # Gaussian mixture sampling
        centers = np.array([
            [np.pi/4, 0],
            [np.pi/6, np.pi/4],
            [np.pi/3, -np.pi/4]
        ])
        n_per_cluster = 200 // 3
        X = np.vstack([
            np.random.normal(center, [0.2, 0.3], (n_per_cluster, 2))
            for center in centers
        ])
        X = np.clip(X, [0, -np.pi/2], [np.pi/2, np.pi/2])
    
    y = forward_kinematics(X)
    
    # Save dataset
    os.makedirs("saved_data", exist_ok=True)
    np.save(f"saved_data/initial_X_method{method}.npy", X)
    np.save(f"saved_data/initial_y_method{method}.npy", y)
    return X, y

def generate_all_datasets():
    """Generate all three initial datasets"""
    for method in [1, 2, 3]:
        x, y = generate_initial_data(method)
        print(f"Dataset {method} generated and saved successfully!")
        print(f"X shape: {x.shape}")
        print(f"y shape: {y.shape}")

if __name__ == "__main__":
    generate_all_datasets()