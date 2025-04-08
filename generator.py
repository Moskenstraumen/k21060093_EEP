# generator.py
import numpy as np
import os
from utils import forward_kinematics

def generate_initial_data(method=1):
    """Generate and save the initial dataset using different methods"""
    np.random.seed(42)
    n_samples = 200  # Fixed dataset size for all methods
    
    if method == 1:
        # Uniform random sampling
        X = np.random.uniform(
            low=[0, -np.pi/2],
            high=[np.pi/2, np.pi/2],
            size=(n_samples, 2)
        )
    elif method == 2:
        # Grid-based sampling
        grid_points = int(np.sqrt(n_samples))  # ~14x14 grid
        x1 = np.linspace(0, np.pi/2, grid_points)
        x2 = np.linspace(-np.pi/2, np.pi/2, grid_points)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.column_stack((X1.ravel(), X2.ravel()))
        
        # Adjust to exact size
        if len(X) > n_samples:
            # Randomly select points
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
        elif len(X) < n_samples:
            # Add random points to reach n_samples
            n_extra = n_samples - len(X)
            X_extra = np.random.uniform(
                low=[0, -np.pi/2],
                high=[np.pi/2, np.pi/2],
                size=(n_extra, 2)
            )
            X = np.vstack([X, X_extra])
        
        # Add noise
        X += np.random.normal(0, 0.05, X.shape)
        X = np.clip(X, [0, -np.pi/2], [np.pi/2, np.pi/2])
    else:
        # Gaussian mixture
        centers = np.array([
            [np.pi/4, 0],
            [np.pi/6, np.pi/4],
            [np.pi/3, -np.pi/4]
        ])
        n_per_cluster = n_samples // 3
        remainder = n_samples % 3
        
        clusters = []
        for i, center in enumerate(centers):
            size = n_per_cluster + (1 if i < remainder else 0)
            cluster = np.random.normal(center, [0.2, 0.3], (size, 2))
            clusters.append(cluster)
        
        X = np.vstack(clusters)
        X = np.clip(X, [0, -np.pi/2], [np.pi/2, np.pi/2])
    
    # Verify size and shuffle
    assert X.shape == (n_samples, 2), f"Wrong shape: {X.shape}, expected ({n_samples}, 2)"
    np.random.shuffle(X)
    
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