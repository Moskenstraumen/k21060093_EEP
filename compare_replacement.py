import numpy as np
import matplotlib.pyplot as plt
from generator import generate_initial_data
from optimal_generator import generate_optimal_data
from MT_solver import IterativeMTSolver
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

def analyze_results(num_replace, log_data):
    """三指标分图可视化"""
    plt.figure(figsize=(18, 5))
    
    # MSE趋势图
    plt.subplot(1, 3, 1)
    mse_values = [entry["mse"] for entry in log_data["improvements"]]
    plt.plot(mse_values, 'b-', label='MSE')
    plt.scatter(
        [i for i, e in enumerate(log_data["improvements"]) if e["improved"]],
        [e["mse"] for e in log_data["improvements"] if e["improved"]],
        c='b', edgecolor='k', zorder=5
    )
    plt.axhline(log_data["initial_mse"], color='r', linestyle='--')
    plt.title("MSE Improvement")
    plt.grid(True)

    # MAE趋势图
    plt.subplot(1, 3, 2)
    mae_values = [entry["mae"] for entry in log_data["improvements"]] 
    plt.plot(mae_values, 'g-', label='MAE')
    plt.scatter(
        [i for i, e in enumerate(log_data["improvements"]) if e["improved"]],
        [e["mae"] for e in log_data["improvements"] if e["improved"]],
        c='g', edgecolor='k', zorder=5
    )
    plt.axhline(log_data["initial_mae"], color='r', linestyle='--')
    plt.title("MAE Improvement")
    plt.grid(True)

    # R²趋势图
    plt.subplot(1, 3, 3)
    r2_values = [entry["r2"] for entry in log_data["improvements"]]
    plt.plot(r2_values, 'r-', label='R²')
    plt.scatter(
        [i for i, e in enumerate(log_data["improvements"]) if e["improved"]],
        [e["r2"] for e in log_data["improvements"] if e["improved"]],
        c='r', edgecolor='k', zorder=5
    )
    plt.axhline(log_data["initial_r2"], color='b', linestyle='--')
    plt.title("R² Improvement")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"graphs/metrics_comparison_{num_replace}.png", dpi=300)
    plt.close()

def optimize_config(rep, initial_data, optimal_data, trials):
    print(f"\nProcessing replacement {rep}...")
    solver = IterativeMTSolver(initial_data, optimal_data)
    result = solver.full_optimization(rep, trials)
    return rep, result

def run_comparison(trials=50):
    # Generating initial and optimal data
    # generate_initial_data()
    # generate_optimal_data()
    
    # Load initial and optimal data
    initial_data = (np.load("saved_data/initial_X.npy"), np.load("saved_data/initial_y.npy"))
    optimal_data = (np.load("saved_data/optimal_X.npy"), np.load("saved_data/optimal_y.npy"))
    
    # Experiment setup
    replacements = [0, 1, 5, 10, 15, 20]
    final_results = {}
    
    '''for rep in replacements:
        print(f"\nCurrent replacement configuration = {rep}...")
        solver = IterativeMTSolver(initial_data, optimal_data)
        result = solver.full_optimization(rep, trials)
        final_results[rep] = result
        analyze_results(rep, result)'''
    
    # 并行优化配置
    final_results = {}
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(optimize_config, rep, initial_data, optimal_data, trials)
            for rep in replacements
        }
        for future in as_completed(futures):
            rep, result = future.result()
            final_results[rep] = result
            analyze_results(rep, result)

            # print(f"Initial MSE: {result['initial_mse']:.6f}")
            # print(f"Final MSE: {result['final_mse']:.6f}")
            # print(f"Improvement times: {sum(entry['improved'] for entry in result['improvements'])}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    for rep in replacements:
        mse_values = [final_results[rep]['initial_mse']] + \
                    [entry['mse'] for entry in final_results[rep]['improvements']]
        plt.plot(mse_values, label=f'Replacement {rep}')
    
    plt.xlabel("Trial")
    plt.ylabel("MSE")
    plt.title("MSE Comparison Across Configurations")
    plt.legend(loc = 'upper right')
    plt.grid(True)
    plt.savefig("graphs/mse_comparison_all.png")
    plt.close()

if __name__ == "__main__":
    run_comparison(trials=50)