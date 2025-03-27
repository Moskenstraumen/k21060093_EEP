import numpy as np
import matplotlib.pyplot as plt
import os
from Ann import ANN

def load_dataset_results(dataset_num):
    """Load all replacement results for a specific dataset"""
    replacements = [0, 1, 5, 10, 15, 20]
    results = {}
    
    for rep in replacements:
        try:
            X = np.load(f"results/dataset{dataset_num}/best_{rep}_X.npy")
            y = np.load(f"results/dataset{dataset_num}/best_{rep}_y.npy")
            results[rep] = (X, y)
        except FileNotFoundError:
            print(f"Warning: Missing data for dataset {dataset_num}, replacement {rep}")
            return None
    
    return results

def evaluate_against_optimal(X, y, X_opt, y_opt):
    """Evaluate dataset performance against optimal dataset"""
    model = ANN()
    model.train(X, y, verbose=False)
    metrics = model.evaluate(X_opt, y_opt)
    return metrics

def analyze_dataset_method(dataset_num, X_opt, y_opt):
    """Analyze performance of a specific dataset generation method"""
    results = load_dataset_results(dataset_num)
    if results is None:
        return None
        
    # Evaluate each replacement configuration
    metrics = {}
    for rep, (X, y) in results.items():
        metrics[rep] = evaluate_against_optimal(X, y, X_opt, y_opt)
    
    return metrics

def plot_method_comparison(all_metrics, save_dir='graphs/method_comparison'):
    """Plot comparison of different dataset generation methods"""
    os.makedirs(save_dir, exist_ok=True)
    replacements = [0, 1, 5, 10, 15, 20]
    methods = list(all_metrics.keys())
    
    # Plot MSE comparison
    plt.figure(figsize=(12, 6))
    for method in methods:
        mse_values = [all_metrics[method][rep]['mse'] for rep in replacements]
        plt.plot(replacements, mse_values, marker='o', label=f'Method {method}')
    
    plt.xlabel('Number of Replacements')
    plt.ylabel('MSE')
    plt.title('MSE Comparison Across Dataset Generation Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/method_comparison_mse.png')
    plt.close()
    
    # Plot improvement percentages
    plt.figure(figsize=(12, 6))
    for method in methods:
        base_mse = all_metrics[method][0]['mse']
        improvements = [(base_mse - all_metrics[method][rep]['mse'])/base_mse * 100 
                       for rep in replacements]
        plt.plot(replacements, improvements, marker='o', label=f'Method {method}')
    
    plt.xlabel('Number of Replacements')
    plt.ylabel('Improvement %')
    plt.title('Performance Improvement Across Dataset Generation Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/method_comparison_improvement.png')
    plt.close()

def main():
    # Load optimal dataset
    X_opt = np.load('saved_data/optimal_X.npy')
    y_opt = np.load('saved_data/optimal_y.npy')
    
    # Analyze each method
    all_metrics = {}
    method_names = {
        1: "Uniform Random",
        2: "Grid-based",
        3: "Gaussian Mixture"
    }
    
    print("="*50)
    print("\nAnalyzing Dataset Generation Methods...")
    
    for method in range(1, 4):
        print(f"\nAnalyzing {method_names[method]}...")
        metrics = analyze_dataset_method(method, X_opt, y_opt)
        if metrics:
            all_metrics[method_names[method]] = metrics
            
            # Print summary
            initial_mse = metrics[0]['mse']
            best_mse = min(m['mse'] for m in metrics.values())
            improvement = (initial_mse - best_mse) / initial_mse * 100
            
            print(f"Initial MSE: {initial_mse:.6f}")
            print(f"Best MSE: {best_mse:.6f}")
            print(f"Maximum Improvement: {improvement:.2f}%")
    
    # Generate comparison plots
    plot_method_comparison(all_metrics)
    
    # Determine best method
    best_method = min(all_metrics.keys(), 
                     key=lambda m: min(all_metrics[m][rep]['mse'] 
                                     for rep in [0,1,5,10,15,20]))
    
    print("="*50)
    print("\nOverall Best Method Analysis:")
    print(f"Best Generation Method: {best_method}")
    print("\nDetailed Performance:")
    for rep in [0,1,5,10,15,20]:
        mse = all_metrics[best_method][rep]['mse']
        print(f"Replacements {rep:2d}: MSE = {mse:.6f}")
    
    # Print comprehensive comparison
    print("="*50)
    print("\nMethod Comparison:")
    print(f"{'Method':<20} {'Initial MSE':>12} {'Best MSE':>12} {'Improvement':>12}")
    print("-"*50)
    
    for method in all_metrics:
        initial_mse = all_metrics[method][0]['mse']
        best_mse = min(all_metrics[method][rep]['mse'] for rep in [0,1,5,10,15,20])
        improvement = (initial_mse - best_mse) / initial_mse * 100
        
        print(f"{method:<20} {initial_mse:>12.6f} {best_mse:>12.6f} {improvement:>11.2f}%")
    
    # Determine best methods by different criteria
    best_absolute = min(all_metrics.keys(),
                       key=lambda m: min(all_metrics[m][rep]['mse'] 
                                       for rep in [0,1,5,10,15,20]))
    
    best_improvement = max(all_metrics.keys(),
                         key=lambda m: (all_metrics[m][0]['mse'] - 
                                      min(all_metrics[m][rep]['mse'] 
                                          for rep in [0,1,5,10,15,20])) /
                                      all_metrics[m][0]['mse'] * 100)
    
    print("="*50)
    print("\nBest Methods Analysis:")
    print(f"Best Absolute Performance: {best_absolute}")
    print(f"Best Improvement: {best_improvement}")

if __name__ == "__main__":
    main()