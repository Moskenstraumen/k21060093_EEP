import numpy as np
import matplotlib.pyplot as plt
import os
from Ann import ANN
from sklearn.metrics import mean_absolute_error, r2_score

def load_best_dataset(replacement_num, dataset_num):
    # Load the best dataset for the given replacement configuration
    try:
        X = np.load(f"results/dataset{dataset_num}/best_{replacement_num}_X.npy")
        y = np.load(f"results/dataset{dataset_num}/best_{replacement_num}_y.npy")
        return X, y
    except FileNotFoundError:
        print(f"Dataset not found: dataset {dataset_num}, replacement {replacement_num}")
        return None, None

def plot_learning_curve(replacement_num, train_loss, val_loss, dataset_num):
    plt.figure(figsize=(12, 8))
    
    epochs = np.arange(len(train_loss)) + 1
    plt.semilogy(epochs, train_loss, 'b-', lw=1.5, label='Training MSE')
    plt.semilogy(epochs, val_loss, 'r--', lw=2, label='Validation MSE')
    
    best_train = np.argmin(train_loss)
    best_val = np.argmin(val_loss)
    plt.scatter(best_train+1, train_loss[best_train], 
                c='blue', s=100, edgecolor='k', zorder=5,
                label=f'Best Train: {train_loss[best_train]:.2e}')
    plt.scatter(best_val+1, val_loss[best_val],
                c='red', s=100, edgecolor='k', zorder=5,
                label=f'Best Val: {val_loss[best_val]:.2e}')
    
    max_loss = max(train_loss[0], val_loss[0]) * 1.2
    min_loss = min(np.min(train_loss), np.min(val_loss)) * 0.8
    plt.ylim(min_loss, max_loss)
    
    plt.title(f'Learning Curve for ({replacement_num} Replacements)\n'
             f'Early Stop at Epoch {len(train_loss)}', fontsize=14, pad=15)
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.legend(frameon=True, facecolor='ghostwhite', 
              loc='upper right', fontsize=10)
    plt.grid(True, which='both', ls='--', alpha=0.5)
    
    save_dir = f"graphs/dataset{dataset_num}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/learning_curve_per_trial_{replacement_num}.png", 
               dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_configs(dataset_num=1):
    configs = [0, 1, 5, 10, 15, 20]
    reports = {}
    
    print("="*50)
    print(f"\nAnalyzing Dataset {dataset_num}")
    
    for rep in configs:
        X, y = load_best_dataset(rep, dataset_num)
        if X is None:
            continue
            
        print(f"\nAnalyzing {rep} replacements:")
        model = ANN()
        try:
            # 训练模型并获取损失曲线
            train_loss, val_loss = model.train(X, y)
            
            # 获取训练集指标
            X_train_scaled = model.scaler.transform(X)  # 标准化训练数据
            train_pred = model.model.predict(X_train_scaled)
            train_mae = mean_absolute_error(y, train_pred)
            train_r2 = r2_score(y, train_pred)
            
            # 获取验证集指标
            X_val = model.scaler.transform(model.X_val)
            val_pred = model.model.predict(X_val)
            val_mae = mean_absolute_error(model.y_val, val_pred)
            val_r2 = r2_score(model.y_val, val_pred)
            
            # 生成可视化
            plot_learning_curve(rep, train_loss, val_loss, dataset_num)
            
            # 生成诊断报告（传递新增参数）
            reports[rep] = analyze_training_process(
                train_loss, val_loss,
                train_mae=train_mae,
                train_r2=train_r2,
                val_mae=val_mae,
                val_r2=val_r2
            )
            
        except Exception as e:
            print(f"Fail: {str(e)}")
    
    print(f"\nFinal report (Dataset {dataset_num}):")
    compare = sorted(reports.items(), key=lambda x: x[1]['best_val'])
    for rep, data in compare:
        print(f"{rep:2d} replacements | "
              f"Train MSE: {data['final_train']:.2e} | "
              f"Trian MAE: {data['train_mae']:.2e} | "
              f"Train R²: {data['train_r2']:.2f} | "
              f"Validation MSE: {data['final_val']:.2e} | "
              f"Validation MAE: {data['val_mae']:.2e} | "
              f"Validation R²: {data['val_r2']:.2f} | "
              f"Converge epoch: {data['converge_epoch']:3d}")

def analyze_training_process(train_loss, val_loss, 
                             train_mae, train_r2,
                             val_mae, val_r2):
    analysis = {
        'final_train': train_loss[-1],
        'final_val': val_loss[-1],
        'best_val': np.min(val_loss),
        'train_mae': train_mae, 
        'train_r2': train_r2, 
        'val_mae': val_mae, 
        'val_r2': val_r2,  
        'converge_epoch': len(train_loss),
        'overfit_gap': (val_loss[-1] - train_loss[-1]) / train_loss[-1]
    }
    
    print(f"Report:")
    print(f"├── Training set performance:")
    print(f"│   ├── MSE: {analysis['final_train']:.2e}")
    print(f"│   ├── MAE: {analysis['train_mae']:.2e}")
    print(f"│   └── R²: {analysis['train_r2']:.2f}")
    print(f"├── Validation set performance:")
    print(f"│   ├── MSE: {analysis['final_val']:.2e}")
    print(f"│   ├── MAE: {analysis['val_mae']:.2e}") 
    print(f"│   └── R²: {analysis['val_r2']:.2f}") 
    print(f"├── Best validation MSE: {analysis['best_val']:.2e}")
    print(f"├── Converge epoch: {analysis['converge_epoch']}")
    print(f"├── Overfitting gap: {analysis['overfit_gap']:.1%}")
    
    return analysis

if __name__ == "__main__":
    # Get dataset number from environment or default to 1
    dataset_num = int(os.environ.get('INITIAL_DATASET', 1))
    analyze_all_configs(dataset_num)
