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
    plt.figure(figsize=(12, 8))
    
    # 绘制双轴曲线
    epochs = np.arange(len(train_loss)) + 1
    plt.semilogy(epochs, train_loss, 'b-', lw=1.5, label='Training MSE')
    plt.semilogy(epochs, val_loss, 'r--', lw=2, label='Validation MSE')
    
    # 标注关键点
    best_train = np.argmin(train_loss)
    best_val = np.argmin(val_loss)
    plt.scatter(best_train+1, train_loss[best_train], 
                c='blue', s=100, edgecolor='k', zorder=5,
                label=f'Best Train: {train_loss[best_train]:.2e}')
    plt.scatter(best_val+1, val_loss[best_val],
                c='red', s=100, edgecolor='k', zorder=5,
                label=f'Best Val: {val_loss[best_val]:.2e}')
    
    # 动态调整坐标范围
    max_loss = max(train_loss[0], val_loss[0]) * 1.2
    min_loss = min(np.min(train_loss), np.min(val_loss)) * 0.8
    plt.ylim(min_loss, max_loss)
    
    # 专业图表样式
    plt.title(f'Learning Dynamics ({replacement_num} Replacements)\n'
             f'Early Stop at Epoch {len(train_loss)}', fontsize=14, pad=15)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error (Log Scale)', fontsize=12)
    plt.legend(frameon=True, facecolor='ghostwhite', 
              loc='upper right', fontsize=10)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    
    # 保存高分辨率图片
    os.makedirs("graphs", exist_ok=True)
    plt.savefig(f"graphs/learning_curve_per_trial_{replacement_num}.png", 
               dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_configs():
    configs = [0, 1, 5, 10, 15, 20]
    reports = {}
    
    for rep in configs:
        X, y = load_best_dataset(rep)
        if X is None:
            continue
            
        print(f"\n分析配置 {rep} replacements:")
        model = ANN()
        try:
            train_loss, val_loss = model.train(X, y)
            
            # 生成可视化
            plot_learning_curve(rep, train_loss, val_loss)
            
            # 生成诊断报告
            reports[rep] = analyze_training_process(train_loss, val_loss)
            
        except Exception as e:
            print(f"分析失败: {str(e)}")
    
    # 生成对比报告
    print("\n全局对比报告:")
    compare = sorted(reports.items(), key=lambda x: x[1]['best_val'])
    for rep, data in compare:
        print(f"配置{rep:2d} replacements | 最佳验证MSE: {data['best_val']:.2e} "
             f"| 收敛epoch: {data['converge_epoch']:3d}")

def analyze_training_process(train_loss, val_loss):
    """训练过程诊断报告"""
    analysis = {
        'final_train': train_loss[-1],
        'final_val': val_loss[-1],
        'best_train': np.min(train_loss),
        'best_val': np.min(val_loss),
        'converge_epoch': len(train_loss),
        'overfit_gap': (val_loss[-1] - train_loss[-1]) / train_loss[-1]
    }
    
    print(f"训练诊断报告:")
    print(f"├── 最终训练MSE: {analysis['final_train']:.2e}")
    print(f"├── 最终验证MSE: {analysis['final_val']:.2e}")
    print(f"├── 最佳验证MSE: {analysis['best_val']:.2e}")
    print(f"├── 收敛epoch数: {analysis['converge_epoch']}")
    print(f"└── 过拟合程度: {analysis['overfit_gap']:.1%}")
    
    return analysis

if __name__ == "__main__":
    analyze_all_configs()
