# analyze_replacement_trial.py
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

def plot_learning_curve(replacement_num, train_loss, val_loss, early_stop_epoch, dataset_num):
    plt.figure(figsize=(10, 6))
    
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16
    })
    
    epochs = np.arange(1, len(train_loss) + 1)
    
    # Plot curves with log scale
    plt.semilogy(epochs, train_loss, '-', 
                color='#1f77b4', lw=2, 
                label='Training Loss')
    plt.semilogy(epochs, val_loss, '-', 
                color='#ff7f0e', lw=2, 
                label='Validation Loss')
    
    # Mark early stopping point
    if early_stop_epoch:
        plt.axvline(x=early_stop_epoch, 
                   color='gray', 
                   linestyle='--', 
                   lw=1.5,
                   label=f'Early Stopping (epoch {early_stop_epoch})')

    plt.xlim(0, len(train_loss))
    all_losses = np.concatenate([train_loss, val_loss])
    plt.ylim(np.min(all_losses) * 0.5, np.max(all_losses) * 2)
    
    plt.title(f'Learning Curve ({replacement_num} Replacements)', pad=15)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (log scale)')

    plt.legend(frameon=True, 
              fancybox=True,
              edgecolor='black',
              facecolor='white',
              framealpha=1.0,
              loc='upper right')
    
    plt.grid(True, which='major', linestyle='--', alpha=0.7)
    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    
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
            train_loss, val_loss, best_epoch = model.train(X, y)
            
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
            
            plot_learning_curve(rep, train_loss, val_loss, best_epoch, dataset_num)
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
              f"Validation R²: {data['val_r2']:.2f} | ")

def analyze_training_process(train_loss, val_loss, 
                           train_mae, train_r2,
                           val_mae, val_r2):
    """Simplified training analysis with focus on convergence and overfitting"""
    best_val_epoch = np.argmin(val_loss) + 1
    min_val_loss = val_loss[best_val_epoch - 1]
    overfit_ratio = (val_loss[-1] - min_val_loss) / min_val_loss * 100
    
    '''print(f"\nTraining Summary:")
    print(f"    Convergence:")
    print(f"    Total epochs: {len(train_loss)}")
    print(f"    Best epoch: {best_val_epoch}")
    print(f"    Overfitting: {overfit_ratio:+.1f}%")
    print(f"Final Performance:")
    print(f"    Train MSE: {train_loss[-1]:.2e} (R²: {train_r2:.2f})")
    print(f"    Val MSE: {val_loss[-1]:.2e} (R²: {val_r2:.2f})")'''
    
    return {
        'final_train': train_loss[-1],
        'final_val': val_loss[-1],
        'best_val': min_val_loss,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'val_mae': val_mae,
        'val_r2': val_r2,
        'converge_epoch': len(train_loss),
        'best_epoch': best_val_epoch,
        'overfit_ratio': overfit_ratio
    }

if __name__ == "__main__":
    # Get dataset number from environment or default to 1
    dataset_num = int(os.environ.get('INITIAL_DATASET', 1))
    analyze_all_configs(dataset_num)
