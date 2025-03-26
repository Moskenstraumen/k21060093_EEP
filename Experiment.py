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

def run_script(script_name):
    """安全执行子脚本"""
    print(f"\n{'='*30} 启动 {script_name} {'='*30}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            stdout=sys.stdout,
            stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} 执行失败 (返回码: {e.returncode})")
        sys.exit(1)
    finally:
        elapsed = time.time() - start_time
        print(f"\n{'='*30} {script_name} 完成 [耗时: {elapsed:.1f}s] {'='*30}")

def validate_files():
    """结果文件完整性校验"""
    required_files = [
        "results/best_0_X.npy",
        "results/best_20_X.npy",
        "graphs/learning_curve_per_trial_0.png",
        "graphs/learning_curve_per_trial_20.png"
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print("\n⚠️ 缺失关键文件:")
        for f in missing:
            print(f" - {f}")
        sys.exit(2)

if __name__ == "__main__":
    # 实验流程编排
    experiment_steps = [
        "compare_replacement.py",
        "analyse_replacement_trial.py"
    ]
    
    # 创建必要目录
    os.makedirs("results", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)
    
    # 顺序执行实验步骤
    for step in experiment_steps:
        run_script(step)
    
    # 结果校验
    validate_files()
    print("\n✅ 实验成功完成! 结果保存在:"
          "\n- ./results/ 优化数据集"
          "\n- ./graphs/ 分析图表")
