import numpy as np
import matplotlib.pyplot as plt
from generator import generate_initial_data
from optimal_generator import generate_optimal_data
from MT_solver import IterativeMTSolver
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import os

def analyze_results(num_replace, log_data, dataset_num):
    """Plot metrics comparison for a specific replacement configuration"""
    if not isinstance(log_data, dict):
        print(f"Invalid log_data format for replacement {num_replace}")
        return
        
    plt.figure(figsize=(18, 5))
    
    # Ensure all metrics start from baseline
    baseline_mse = log_data["initial_mse"]
    baseline_mae = log_data["initial_mae"]
    baseline_r2 = log_data["initial_r2"]
    
    # MSE graph
    plt.subplot(1, 3, 1)
    mse_values = [baseline_mse] + [entry["mse"] for entry in log_data["improvements"]]
    plt.plot(mse_values, 'b-', label='MSE')
    
    # Add improvement markers
    improvements = [False] + [entry["improved"] for entry in log_data["improvements"]]
    improved_indices = [i for i, imp in enumerate(improvements) if imp]
    improved_mses = [mse_values[i] for i in improved_indices]
    if num_replace > 0:  # Only show markers for non-zero replacements
        plt.scatter(improved_indices, improved_mses, c='b', edgecolor='k', zorder=5)
    
    plt.axhline(baseline_mse, color='r', linestyle='--', label='Initial MSE')
    plt.title("MSE Improvement")
    plt.grid(True)
    plt.legend()

    # MAE graph
    plt.subplot(1, 3, 2)
    mae_values = [baseline_mae] + [entry["mae"] for entry in log_data["improvements"]]
    plt.plot(mae_values, 'g-', label='MAE')
    
    # Add improvement markers for MAE
    improved_maes = [mae_values[i] for i in improved_indices]
    if num_replace > 0:
        plt.scatter(improved_indices, improved_maes, c='g', edgecolor='k', zorder=5)
    
    plt.axhline(baseline_mae, color='r', linestyle='--', label='Initial MAE')
    plt.title("MAE Improvement")
    plt.grid(True)
    plt.legend()

    # R² graph
    plt.subplot(1, 3, 3)
    r2_values = [baseline_r2] + [entry["r2"] for entry in log_data["improvements"]]
    plt.plot(r2_values, 'm-', label='R²')
    
    # Add improvement markers for R²
    improved_r2s = [r2_values[i] for i in improved_indices]
    if num_replace > 0:
        plt.scatter(improved_indices, improved_r2s, c='m', edgecolor='k', zorder=5)
    
    plt.axhline(baseline_r2, color='r', linestyle='--', label='Initial R²')
    plt.title("R² Improvement")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"graphs/dataset{dataset_num}/metrics_comparison_{num_replace}.png", dpi=300)
    plt.close()

def optimize_config(rep, initial_data, optimal_data, trials, dataset_num, baseline_mse):
    """Run optimization for a specific configuration"""
    print(f"\nProcessing replacement {rep} for dataset {dataset_num}...")
    try:
        solver = IterativeMTSolver(initial_data, optimal_data, dataset_num=dataset_num)
        # Override initial MSE with baseline
        solver.baseline_mse = baseline_mse
        result = solver.full_optimization(rep, trials)
        return rep, result
    except Exception as e:
        print(f"Error in optimization for replacement {rep}: {str(e)}")
        return rep, None

def run_comparison(trials=50, dataset_num=1):
    # Load dataset based on method number
    initial_data = (
        np.load(f"saved_data/initial_X_method{dataset_num}.npy"),
        np.load(f"saved_data/initial_y_method{dataset_num}.npy")
    )
    optimal_data = (
        np.load("saved_data/optimal_X.npy"),
        np.load("saved_data/optimal_y.npy")
    )
    
    # Create result directories
    results_dir = f"results/dataset{dataset_num}"
    graphs_dir = f"graphs/dataset{dataset_num}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Experiment setup
    replacements = [0, 1, 5, 10, 15, 20]
    final_results = {}
    
    # First, get baseline MSE
    print("Computing baseline MSE...")
    solver = IterativeMTSolver(initial_data, optimal_data, dataset_num=dataset_num)
    baseline_mse = solver.baseline_mse
    print(f"Baseline MSE: {baseline_mse:.6f}")

    # Run in parallel with shared baseline
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(
                optimize_config, 
                rep, 
                initial_data, 
                optimal_data, 
                trials,
                dataset_num,
                baseline_mse  # Pass baseline MSE
            ) for rep in replacements
        }
        for future in as_completed(futures):
            rep, result = future.result()
            if result is not None:
                final_results[rep] = result
                analyze_results(rep, result, dataset_num)

    # Plot with fixed marker handling
    plt.figure(figsize=(10, 6))
    
    # Sort replacements for ordered legend
    ordered_reps = sorted(final_results.keys())
    
    # First plot baseline as a horizontal line
    plt.axhline(y=baseline_mse, 
                color='red',
                linestyle='--',
                label='Baseline MSE',
                linewidth=1.5,
                alpha=0.8)
    
    # Then plot all replacements
    for rep in ordered_reps:
        # Get MSE values
        mse_values = [baseline_mse] + \
                    [entry['mse'] for entry in final_results[rep]['improvements']]
        
        # Plot line
        plt.plot(range(len(mse_values)), mse_values, 
                label=f'Replacement {rep}',
                linestyle='-',
                linewidth=1.5)
        
        # Add markers for improvements (skip baseline case)
        if rep > 0:
            for i in range(1, len(mse_values)):
                if mse_values[i] < mse_values[i-1]:  # Improvement detected
                    plt.scatter(i, mse_values[i],
                              marker='o',
                              s=30,  # Smaller marker size
                              c='blue',
                              edgecolor='black',
                              zorder=5)

    plt.xlabel("Trial")
    plt.ylabel("MSE")
    plt.title(f"MSE Comparison Across Configurations (Dataset {dataset_num})")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"graphs/dataset{dataset_num}/mse_comparison_all.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Get dataset number from environment or default to 1
    dataset_num = int(os.environ.get('INITIAL_DATASET', 1))
    run_comparison(trials=50, dataset_num=dataset_num)