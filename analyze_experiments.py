import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Ann import ANN

def load_all_results():
    """Load results for all methods and replacement configurations"""
    methods = {
        1: "Uniform random",
        2: "Grid-based",
        3: "Gaussian mixture"
    }
    replacements = [0, 1, 5, 10, 15, 20]
    results = {}
    
    # Load optimal dataset for reference
    X_opt = np.load('saved_data/optimal_X.npy')
    y_opt = np.load('saved_data/optimal_y.npy')
    
    for method_id, method_name in methods.items():
        results[method_name] = {}
        for rep in replacements:
            try:
                X = np.load(f"results/dataset{method_id}/best_{rep}_X.npy")
                y = np.load(f"results/dataset{method_id}/best_{rep}_y.npy")
                
                # Train model and evaluate against optimal dataset
                model = ANN()
                model.train(X, y, verbose=False)
                X_scaled = model.scaler.transform(X_opt)
                y_pred = model.model.predict(X_scaled)
                
                metrics = {
                    'mse': mean_squared_error(y_opt, y_pred),
                    'mae': mean_absolute_error(y_opt, y_pred),
                    'r2': r2_score(y_opt, y_pred)
                }
                
                results[method_name][rep] = metrics
                
            except FileNotFoundError:
                print(f"Missing data for method {method_name}, replacement {rep}")
    
    return results

def create_comparison_plots(results):
    """Generate comprehensive comparison plots"""
    # Create output directory
    os.makedirs("experiment_analysis", exist_ok=True)
    
    # Prepare data for plotting
    methods = list(results.keys())
    replacements = list(results[methods[0]].keys())
    
    # 1. MSE Comparison Plot
    plt.figure(figsize=(12, 6))
    for method in methods:
        mse_values = [results[method][rep]['mse'] for rep in replacements]
        plt.semilogy(replacements, mse_values, 'o-', label=method)
    
    plt.xlabel('Number of replacements')
    plt.ylabel('MSE (log scale)')
    plt.title('Performance comparison across methods')
    plt.grid(True)
    plt.legend()
    plt.savefig('experiment_analysis/mse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Relative Improvement Heatmap
    improvement_data = np.zeros((len(methods), len(replacements)-1))
    for i, method in enumerate(methods):
        baseline = results[method][0]['mse']
        for j, rep in enumerate(replacements[1:]):
            improvement = (baseline - results[method][rep]['mse']) / baseline * 100
            improvement_data[i, j] = improvement
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(improvement_data, 
                xticklabels=replacements[1:],
                yticklabels=methods,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0)
    plt.xlabel('Number of replacements')
    plt.ylabel('Dataset synthesize method')
    plt.title('Relative improvement (%) from baseline')
    plt.savefig('experiment_analysis/improvement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_tables(results):
    """Generate summary tables for paper"""
    summary = {
        'method': [],
        'best_replacements': [],
        'initial_mse': [],
        'best_mse': [],
        'improvement': [],
        'final_r2': []
    }
    
    for method in results:
        initial_mse = results[method][0]['mse']
        best_mse = float('inf')
        best_rep = 0
        
        for rep in results[method]:
            if results[method][rep]['mse'] < best_mse:
                best_mse = results[method][rep]['mse']
                best_rep = rep
        
        improvement = (initial_mse - best_mse) / initial_mse * 100
        
        summary['method'].append(method)
        summary['best_replacements'].append(best_rep)
        summary['initial_mse'].append(initial_mse)
        summary['best_mse'].append(best_mse)
        summary['improvement'].append(improvement)
        summary['final_r2'].append(results[method][best_rep]['r2'])
    
    df = pd.DataFrame(summary)
    df.to_csv('experiment_analysis/method_comparison.csv', index=False)
    
    # Format for LaTeX table
    with open('experiment_analysis/latex_table.txt', 'w') as f:
        f.write(df.to_latex(index=False, float_format=lambda x: '%.3f' % x))
    
    return df

def analyze_initial_datasets():
    """Analyze performance of initial datasets from different generation methods"""
    methods = {
        1: "Uniform random",
        2: "Grid-based",
        3: "Gaussian mixture"
    }
    baseline_metrics = {
        'method': [],
        'mse': [],
        'mae': [],
        'r2': []
    }
    
    # Load optimal dataset
    X_opt = np.load('saved_data/optimal_X.npy')
    y_opt = np.load('saved_data/optimal_y.npy')
    
    print("\nBaseline Performance Analysis:")
    print("-" * 80)
    
    for method_id, method_name in methods.items():
        # Load initial dataset
        X = np.load(f"saved_data/initial_X_method{method_id}.npy")
        y = np.load(f"saved_data/initial_y_method{method_id}.npy")
        
        # Train model on initial dataset
        model = ANN()
        model.train(X, y, verbose=False)
        
        # Evaluate against optimal dataset
        X_scaled = model.scaler.transform(X_opt)
        y_pred = model.model.predict(X_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_opt, y_pred)
        mae = mean_absolute_error(y_opt, y_pred)
        r2 = r2_score(y_opt, y_pred)
        
        baseline_metrics['method'].append(method_name)
        baseline_metrics['mse'].append(mse)
        baseline_metrics['mae'].append(mae)
        baseline_metrics['r2'].append(r2)
    
    # Create DataFrame and save
    df_baseline = pd.DataFrame(baseline_metrics)
    df_baseline.to_csv('experiment_analysis/baseline_comparison.csv', index=False)
    
    # Format for LaTeX
    with open('experiment_analysis/baseline_latex_table.txt', 'w') as f:
        f.write(df_baseline.to_latex(index=False, float_format=lambda x: '%.3f' % x))
    
    print("-" * 80)
    print("\nBaseline Performance Summary:")
    print(df_baseline.to_string())
    
    return df_baseline

def main():
    print("Loading experimental results...")
    results = load_all_results()
    
    print("\nAnalyzing initial datasets...")
    baseline_results = analyze_initial_datasets()
    
    print("\nGenerating comparison plots...")
    create_comparison_plots(results)
    
    print("\nGenerating summary tables...")
    summary = generate_summary_tables(results)
    
    print("-" * 80)
    print("\nResults Summary (After Replacements):")
    print(summary.to_string())
    print("-" * 80)
    
    # Find best overall configuration
    best_method = summary.loc[summary['best_mse'].idxmin()]
    print(f"\nBest overall configuration:")
    print(f"Method: {best_method['method']}")
    print(f"Number of replacements: {best_method['best_replacements']}")
    print(f"MSE: {best_method['best_mse']:.6f}")
    print(f"Improvement: {best_method['improvement']:.2f}%")
    print(f"R² Score: {best_method['final_r2']:.4f}")

if __name__ == "__main__":
    main()