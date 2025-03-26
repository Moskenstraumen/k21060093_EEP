import numpy as np
import matplotlib.pyplot as plt
import os
from Ann import ANN

def load_best_dataset(replacement_num):
    """加载指定替换配置的最佳数据集"""
    try:
        X = np.load(f"results/best_{replacement_num}_X.npy")
        y = np.load(f"results/best_{replacement_num}_y.npy")
        return X, y
    except FileNotFoundError:
        print(f"数据集未找到: replacement {replacement_num}")
        return None, None

def plot_learning_curve(replacement_num, train_loss, val_loss):
    """绘制并保存学习曲线图"""
    plt.figure(figsize=(10, 6))
    
    # 绘制双曲线
    plt.plot(train_loss, 'b-', linewidth=2, label='Training MSE')
    plt.plot(val_loss, 'r--', linewidth=2, label='Validation MSE')
    
    # 标注关键点
    min_train_idx = np.argmin(train_loss)
    min_val_idx = np.argmin(val_loss)
    plt.scatter(min_train_idx, train_loss[min_train_idx], 
                c='blue', s=100, edgecolor='k', label='Min Training MSE')
    plt.scatter(min_val_idx, val_loss[min_val_idx],
                c='red', s=100, edgecolor='k', label='Min Validation MSE')
    
    # 图表装饰
    plt.title(f"Learning Curve ({replacement_num} Replacements)\n"
              f"Final Train MSE: {train_loss[-1]:.4f}, Val MSE: {val_loss[-1]:.4f}",
              fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存输出
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(f"graphs/learning_curve_per_trial_{replacement_num}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_configs():
    """分析所有替换配置"""
    configs = [0, 1, 5, 10, 15, 20]
    
    for rep in configs:
        print(f"\nAnalyzing replacement configuration: {rep}")
        X, y = load_best_dataset(rep)
        
        if X is None or y is None:
            continue
            
        # 初始化并训练模型
        model = ANN()
        try:
            train_loss, val_loss = model.train(X, y)
            
            # 对齐数据长度
            min_length = min(len(train_loss), len(val_loss))
            plot_learning_curve(rep, 
                              train_loss[:min_length], 
                              val_loss[:min_length])
            
            print(f"训练完成: 最终训练MSE {train_loss[-1]:.4f}, "
                f"验证MSE {val_loss[-1]:.4f}")
        except Exception as e:
            print(f"训练失败: {str(e)}")

if __name__ == "__main__":
    analyze_all_configs()
