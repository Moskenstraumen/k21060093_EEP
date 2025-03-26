import numpy as np
from scipy.stats import gaussian_kde

def forward_kinematics(joint_angles):
    """正运动学计算"""
    l1, l2 = 1.0, 0.8
    theta1 = np.clip(joint_angles[:,0], 0, np.pi/2)
    theta2 = np.clip(joint_angles[:,1], -np.pi/2, np.pi/2)
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return np.column_stack((x, y)) + np.random.normal(0, 1e-6, size=(len(x),2))  # 防止零方差

def generate_replacement(base_data, num_replace, seed):
    """生成替换样本"""
    np.random.seed(seed)
    # Boundry case protection
    if num_replace <= 0:  
        return np.empty((0, 2))
    
    kde = gaussian_kde(base_data.T)
    return kde.resample(num_replace).T
    
    '''joint_limits = np.array([[0, np.pi/2], [-np.pi/2, np.pi/2]])
    return np.random.uniform(
        low=joint_limits[:,0],
        high=joint_limits[:,1],
        size=(num_replace, 2)
    )'''