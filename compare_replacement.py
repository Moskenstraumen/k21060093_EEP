import numpy as np
import matplotlib.pyplot as plt
from generator import generate_initial_data
from optimal_generator import generate_optimal_data
from MT_solver import IterativeMTSolver

def analyze_results(num_replace, log_data):
    """增强版结果分析函数"""
    # 创建双轴图表
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # 绘制MSE主曲线
    mse_values = [entry["mse"] for entry in log_data["improvements"]]
    line_mse, = ax1.plot(mse_values, 'b-', label="MSE")
    ax1.scatter(
        [i for i, entry in enumerate(log_data["improvements"]) if entry["improved"]],
        [entry["mse"] for entry in log_data["improvements"] if entry["improved"]],
        color='b', marker='o', s=60, edgecolor='k', label="Valid Improvement"
    )
    ax1.axhline(log_data["initial_mse"], color='r', linestyle='--', linewidth=2, label="Initial MSE")
    
    # 配置主坐标轴
    ax1.set_xlabel("Trial Number", fontsize=12)
    ax1.set_ylabel("Mean Squared Error (MSE)", color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 创建次坐标轴绘制NMSE
    ax2 = ax1.twinx()
    nmse_values = [entry["nmse"] for entry in log_data["improvements"]]
    line_nmse, = ax2.plot(nmse_values, 'g--', alpha=0.7, label="NMSE")
    ax2.set_ylabel("Normalized MSE (NMSE)", color='g', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='g')
    
    # 合并图例
    lines = [line_mse, line_nmse]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    
    plt.title(f"Learning Progress with {num_replace} Replacements", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(f"graphs/learning_curve_{num_replace}.png", dpi=300, bbox_inches='tight')
    plt.close()

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
    
    for rep in replacements:
        print(f"\nCurrent replacement configuration = {rep}...")
        solver = IterativeMTSolver(initial_data, optimal_data)
        result = solver.full_optimization(rep, trials)
        final_results[rep] = result
        analyze_results(rep, result)
        
        print(f"Initial MSE: {result['initial_mse']:.6f}")
        print(f"Final MSE: {result['final_mse']:.6f}")
        print(f"Initial NMSE: {result['initial_nmse']:.6f}")
        print(f"Final NMSE: {result['final_nmse']:.6f}")
        print(f"Improvement times: {sum(entry['improved'] for entry in result['improvements'])}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    for rep in replacements:
        mse_values = [final_results[rep]['initial_mse']] + \
                    [entry['mse'] for entry in final_results[rep]['improvements']]
        plt.plot(mse_values, label=f'Replacement {rep}')
    
    plt.xlabel("Trial")
    plt.ylabel("MSE")
    plt.title("MSE Comparison Across Configurations")
    plt.legend()
    plt.grid(True)
    plt.savefig("graphs/mse_comparison_all.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for rep in replacements:
        nmse_values = [final_results[rep]['improvements'][0]['nmse']] + \
                     [entry['nmse'] for entry in final_results[rep]['improvements']]
        plt.plot(nmse_values, linestyle='--', label=f'Replacement {rep}')
    
    plt.xlabel("Trial")
    plt.ylabel("Normalized MSE")
    plt.title("NMSE Comparison Across Configurations")
    plt.legend()
    plt.grid(True)
    plt.savefig("graphs/nmse_comparison_all.png")
    plt.close()

if __name__ == "__main__":
    run_comparison(trials=50)