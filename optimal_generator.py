# optimal_generator.py
import numpy as np
from generator import forward_kinematics

def generate_optimal_data(n_samples=100):
    X = np.random.uniform(0, np.pi/2, (n_samples, 2))  # 输入：两个关节角度
    y = forward_kinematics(X)  # 输出：二维坐标
    
    # 添加高斯噪声
    y += np.random.normal(0, 0.05, size=y.shape)
    
    # 验证数据维度
    print(f"生成数据维度验证: X.shape={X.shape}, y.shape={y.shape}")  # 应输出 (100,2) 和 (100,2)
    
    np.save('saved_data/optimal_X.npy', X)
    np.save('saved_data/optimal_y.npy', y)
    return X, y

if __name__ == "__main__":
    # Generate and save the data when the script is run directly
    x, y = generate_optimal_data()
    print("Optimal dataset generated and saved successfully!")
    print(f"X shape: {x.shape}")
    print(f"y shape: {y.shape}")