"""
自动化实验流程控制器

功能特性：
1. 顺序执行数据优化与结果分析
2. 智能错误处理机制
3. 实验耗时统计
4. 跨平台兼容性支持
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_script(script_name, dataset_num=None):
    print(f"\n{'='*25} 启动 {script_name} (Dataset {dataset_num}) {'='*25}")
    
    start_time = time.time()
    try:
        env = os.environ.copy()
        if dataset_num is not None:
            env['INITIAL_DATASET'] = str(dataset_num)
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            env=env
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} 执行失败 (返回码: {e.returncode})")
        return False
    finally:
        elapsed = time.time() - start_time
        print(f"\n{'='*25} {script_name} 完成 [耗时: {elapsed:.1f}s] {'='*25}")

def validate_files(dataset_num):
    """结果文件完整性校验"""
    # Check data files
    data_files = [
        f"saved_data/initial_X_method{dataset_num}.npy",
        f"saved_data/initial_y_method{dataset_num}.npy",
        "saved_data/optimal_X.npy",
        "saved_data/optimal_y.npy"
    ]
    
    # Check results for each replacement
    replacements = [0, 1, 5, 10, 15, 20]
    result_files = []
    for rep in replacements:
        result_files.extend([
            f"results/dataset{dataset_num}/best_{rep}_X.npy",
            f"results/dataset{dataset_num}/best_{rep}_y.npy",
            f"graphs/dataset{dataset_num}/metrics_comparison_{rep}.png",
            f"graphs/dataset{dataset_num}/learning_curve_per_trial_{rep}.png"
        ])
    
    # Add overall comparison plot
    result_files.append(f"graphs/dataset{dataset_num}/mse_comparison_all.png")
    
    # Check all required files
    all_files = data_files + result_files
    missing = [f for f in all_files if not Path(f).exists()]
    
    if missing:
        print(f"\n⚠️ Dataset {dataset_num} missing required files:")
        for f in missing:
            print(f" - {f}")
        return False
    
    print(f"\n✅ Dataset {dataset_num} validation successful")
    return True

def run_experiment():
    """运行完整实验流程"""
    # Create all necessary directories
    for i in range(1, 4):
        os.makedirs(f"results/dataset{i}", exist_ok=True)
        os.makedirs(f"graphs/dataset{i}", exist_ok=True)
    os.makedirs("saved_data", exist_ok=True)
    
    # Generate all datasets first
    if not run_script("generator.py"):
        print("❌ Dataset generation failed")
        return
    
    # Run experiments for each dataset
    success = True
    for dataset_num in range(1, 4):
        print(f"\n{'#'*80}")
        print(f"Processing Dataset {dataset_num}")
        print(f"{'#'*80}")
        
        # Run optimization
        if not run_script("compare_replacement.py", dataset_num):
            print(f"❌ Optimization failed for dataset {dataset_num}")
            success = False
            continue
            
        # Run analysis
        if not run_script("analyse_replacement_trial.py", dataset_num):
            print(f"❌ Analysis failed for dataset {dataset_num}")
            success = False
            continue
        
        # Validate results
        if not validate_files(dataset_num):
            success = False
            continue
    
    if success:
        print("\n✅ All experiments completed successfully!")
    else:
        print("\n⚠️ Some experiments failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    run_experiment()
