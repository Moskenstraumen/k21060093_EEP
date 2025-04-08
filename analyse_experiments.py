# analyse_experiments.py
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

def load_baseline_mse(method_id):
    """Load baseline MSE from solver's results"""
    try:
        # Load the initial solver results
        solver_results = np.load(f"results/dataset{method_id}/solver_baseline.npy")
        return float(solver_results)
    except FileNotFoundError:
        print(f"Baseline MSE not found for method {method_id}")
        return None

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
    
    baseline_results = {}
    
    # Load optimal dataset
    X_opt = np.load('saved_data/optimal_X.npy')
    y_opt = np.load('saved_data/optimal_y.npy')
    
    for method_id, method_name in methods.items():
        # Load baseline MSE from solver results
        baseline_mse = load_baseline_mse(method_id)
        
        # Calculate other metrics
        X = np.load(f"saved_data/initial_X_method{method_id}.npy")
        y = np.load(f"saved_data/initial_y_method{method_id}.npy")
        
        model = ANN()
        model.train(X, y, verbose=False)
        X_scaled = model.scaler.transform(X_opt)
        y_pred = model.model.predict(X_scaled)
        
        mae = mean_absolute_error(y_opt, y_pred)
        r2 = r2_score(y_opt, y_pred)
        
        # Store metrics
        baseline_metrics['method'].append(method_name)
        baseline_metrics['mse'].append(baseline_mse)
        baseline_metrics['mae'].append(mae)
        baseline_metrics['r2'].append(r2)
        
        baseline_results[method_name] = {
            'mse': baseline_mse,
            'mae': mae,
            'r2': r2
        }
    
    return pd.DataFrame(baseline_metrics), baseline_results

def generate_summary_tables(results, baseline_results):
    """Generate summary tables"""
    summary = {
        'method': [],
        'best_replacements': [],
        'initial_mse': [],
        'best_mse': [],
        'improvement': [],
        'final_r2': []
    }
    
    for method in results:
        initial_mse = baseline_results[method]['mse']
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
    return df

def analyse_learning_curves(dataset_num=None):
    """Analyze learning curves from the best replacement configurations"""
    methods = {
        1: "Uniform random",
        2: "Grid-based",
        3: "Gaussian mixture"
    }
    
    # Get best replacements for each method
    # These values are placeholders 
    # They can be adjusted based on actual results from experiment.py
    best_configs = {
        "Uniform random": 10, 
        "Grid-based": 5,
        "Gaussian mixture": 15
    }
    
    print("="*80)
    print("\nBest Configuration Learning Curves:")
    
    for method_id, method_name in methods.items():
        current_dataset = dataset_num if dataset_num else method_id
        best_rep = best_configs[method_name]
        
        try:
            X = np.load(f"results/dataset{current_dataset}/best_{best_rep}_X.npy")
            y = np.load(f"results/dataset{current_dataset}/best_{best_rep}_y.npy")
            
            # Train model to get learning curves
            model = ANN()
            train_loss, val_loss, early_stop = model.train(X, y, verbose=False)
            
            # Calculate metrics
            initial_train_loss = train_loss[0]
            final_train_loss = train_loss[-1]
            initial_val_loss = val_loss[0]
            final_val_loss = val_loss[-1]
            best_val_loss = np.min(val_loss)
            best_epoch = np.argmin(val_loss) + 1
            
            # Calculate improvements
            train_improvement = ((initial_train_loss - final_train_loss) / 
                               initial_train_loss * 100)
            val_improvement = ((initial_val_loss - best_val_loss) / 
                             initial_val_loss * 100)
            overfit_gap = ((final_val_loss - best_val_loss) / 
                          best_val_loss * 100)
            
            print(f"\n{method_name} (Best: {best_rep} replacements):")
            print(f"Training Loss: {initial_train_loss:.2e} → {final_train_loss:.2e} "
                  f"({train_improvement:+.1f}%)")
            print(f"Validation Loss: {initial_val_loss:.2e} → {final_val_loss:.2e} "
                  f"(best: {best_val_loss:.2e})")
            print(f"Best Epoch: {best_epoch}/{len(train_loss)}")
            print(f"Early Stopping: {early_stop if early_stop else 'None'}")
            print(f"Val Improvement: {val_improvement:+.1f}%")
            print(f"Overfit Gap: {overfit_gap:+.1f}%")
            
        except FileNotFoundError:
            print(f"Missing data for {method_name}")
            continue

def main():
    print("Loading experiment results...")
    results = load_all_results()
    
    print("\nAnalyzing initial datasets...")
    baseline_df, baseline_results = analyze_initial_datasets()
    
    print("\nGenerating comparison plots...")
    create_comparison_plots(results)
    
    print("\nGenerating summary tables...")
    summary = generate_summary_tables(results, baseline_results)
    
    print("\nAnalyzing learning curves...")
    analyse_learning_curves()
    
    print("\nBaseline Performance Summary:")
    print(baseline_df.to_string())
    
    print("\nResults Summary (After Replacements):")
    print(summary.to_string())
    
    # Save tables
    baseline_df.to_csv('experiment_analysis/baseline_comparison.csv', index=False)
    summary.to_csv('experiment_analysis/method_comparison.csv', index=False)
    
    with open('experiment_analysis/baseline_latex_table.txt', 'w') as f:
        f.write(baseline_df.to_latex(index=False, float_format=lambda x: '%.3f' % x))
    
    with open('experiment_analysis/latex_table.txt', 'w') as f:
        f.write(summary.to_latex(index=False, float_format=lambda x: '%.3f' % x))

if __name__ == "__main__":
    main()