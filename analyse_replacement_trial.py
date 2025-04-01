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
    plt.figure(figsize=(10, 6))
    
    # Style settings
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16
    })
    
    # Find best validation epoch
    best_epoch = np.argmin(val_loss) + 1
    
    # Plot full training history
    epochs = np.arange(len(train_loss)) + 1
    
    # Plot training curves with log scale
    plt.semilogy(epochs, train_loss, '-', 
                 color='#1f77b4', lw=2, 
                 label='Training Loss')
    plt.semilogy(epochs, val_loss, '-', 
                 color='#ff7f0e', lw=2, 
                 label='Validation Loss')
    
    # Add early stopping marker
    plt.axvline(x=best_epoch, 
                color='gray', 
                linestyle='--', 
                lw=1.5,
                label=f'Best Epoch {best_epoch}')
    
    # Set axis limits
    plt.xlim(0, len(train_loss) + 50)
    max_loss = max(max(train_loss), max(val_loss)) * 2
    min_loss = min(min(train_loss), min(val_loss)) * 0.5
    plt.ylim(min_loss, max_loss)
    
    # Add labels and title
    plt.title(f'Learning Curve ({replacement_num} Replacements)', pad=15)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (log scale)')
    
    # Style the legend
    plt.legend(frameon=True, 
              fancybox=True,
              edgecolor='black',
              facecolor='white',
              framealpha=1.0,
              loc='upper right')
    
    # Customize grid
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    
    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save with high quality
    save_dir = f"graphs/dataset{dataset_num}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/learning_curve_per_trial_{replacement_num}.png",
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.1)
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
            # Train model
            train_loss, val_loss = model.train(X, y)
            
            # Normalize training set
            X_train_scaled = model.scaler.transform(X)
            train_pred = model.model.predict(X_train_scaled)
            train_mae = mean_absolute_error(y, train_pred)
            train_r2 = r2_score(y, train_pred)
            
            # Normalize validation set
            X_val = model.scaler.transform(model.X_val)
            val_pred = model.model.predict(X_val)
            val_mae = mean_absolute_error(model.y_val, val_pred)
            val_r2 = r2_score(model.y_val, val_pred)
            
            plot_learning_curve(rep, train_loss, val_loss, dataset_num)
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
