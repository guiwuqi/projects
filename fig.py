# try_fixed.py - 修复后的画图代码
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from fas_environment import FASA2GEnv
import time
import random
from scipy.interpolate import griddata
import seaborn as sns
from collections import defaultdict
import os
import pandas as pd
import json

# 设置随机种子以确保可重复性
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# 你的最优配置（从训练中获得的）
OPTIMAL_CONFIG = {
    'W': 5.0,
    'Q_sub': 30,
    'delta_d': 5.0 / 29,  # W/(Q-1) = 5.0/(30-1)
    'psi_R': 5.759586334228516,
    'theta_R': 0.5235987901687622,
    'SNR': 10.0
}

# 定义配置列表
# PORT_CONFIGS = [
#     (np.pi / 2, np.pi / 3),  # (90°, 60°)
#     (np.pi / 4, np.pi / 4),  # (45°, 45°)
#     (3 * np.pi / 4, 2 * np.pi / 3),  # (135°, 120°)
#     (np.pi / 3, np.pi / 6),  # (60°, 30°)
#     (5 * np.pi / 6, np.pi / 2),  # (150°, 90°)
#     (np.pi / 6, np.pi / 3),  # (30°, 60°)
#     (np.pi / 3, np.pi / 2),  # (60°, 90°)
#     (np.pi / 2, np.pi / 2),  # (90°, 90°)
#     (2 * np.pi / 3, np.pi / 3),  # (120°, 60°)
#     (5 * np.pi / 6, np.pi / 4),  # (150°, 45°)
#     (np.pi, 5 * np.pi / 6),  # (180°, 150°)
#     (7 * np.pi / 6, np.pi / 3),  # (210°, 60°)
#     (4 * np.pi / 3, np.pi / 2),  # (240°, 90°)
#     (3 * np.pi / 2, np.pi / 3),  # (270°, 60°)
#     (11 * np.pi / 6, np.pi / 6),  # (330°, 30°)
# ]


def calculate_performance(env, config_dict):
    """
    计算给定配置的性能

    Args:
        env: FAS环境实例
        config_dict: 包含W, Q_sub, psi_R, theta_R的配置字典
    Returns:
        performance: 综合性能指标 (C - 0.5 * |delta_error|)
        capacity: 信道容量
        error: 模型误差
    """
    # 更新环境状态
    env.state = {
        'W': np.array([config_dict['W']], dtype=np.float32),
        'Q_sub': np.array([config_dict['Q_sub']], dtype=np.int32),
        'delta_d': np.array([config_dict['delta_d']], dtype=np.float32),
        'psi_R': np.array([config_dict['psi_R']], dtype=np.float32),
        'theta_R': np.array([config_dict['theta_R']], dtype=np.float32),
        'SNR': np.array([config_dict.get('SNR', 10.0)], dtype=np.float32)
    }

    # 计算信道容量和模型误差
    H_FAS = env._calculate_channel_matrix()
    C = env._calculate_channel_capacity(H_FAS)
    delta_error = env._calculate_modeling_error(H_FAS)

    # 计算综合性能指标
    performance = C - 0.5 * abs(delta_error)

    return performance, C, delta_error


def generate_random_configuration():
    """
    生成随机配置（W, Q_sub, psi_R, theta_R）

    注意：当Q_sub改变时，delta_d必须相应改变，因为delta_d = W/(Q-1)
    """
    # 随机选择W在允许范围内
    Q = 30  # 总端口数固定
    min_W = 0.01 * (Q - 1)  # 0.29
    max_W = 5.0
    W = random.uniform(min_W, max_W)

    # 随机选择Q_sub（1到30之间）
    Q_sub = random.randint(1, Q)

    # 计算delta_d
    delta_d = W / (Q - 1)

    # 随机选择角度
    psi_R = random.uniform(0, 2 * np.pi)
    theta_R = random.uniform(0, np.pi)

    return {
        'W': W,
        'Q_sub': Q_sub,
        'delta_d': delta_d,
        'psi_R': psi_R,
        'theta_R': theta_R,
        'SNR': 10.0
    }


def fair_comparison_analysis(num_random_configs=100, seed=42):
    """
    公平对比分析：比较最优配置与随机配置的性能

    Args:
        num_random_configs: 随机配置数量
        seed: 随机种子
    """
    set_random_seed(seed)

    # 创建环境实例
    env = FASA2GEnv()

    print("=" * 70)
    print("公平对比分析：最优RL配置 vs 随机配置")
    print("=" * 70)

    # 计算最优配置的性能
    print("\n1. 计算最优RL配置的性能...")
    optimal_perf, optimal_cap, optimal_err = calculate_performance(env, OPTIMAL_CONFIG)

    print(f"最优配置参数:")
    print(f"  W: {OPTIMAL_CONFIG['W']:.2f}λ")
    print(f"  Q_sub: {OPTIMAL_CONFIG['Q_sub']}")
    print(f"  delta_d: {OPTIMAL_CONFIG['delta_d']:.4f}λ")
    print(f"  psi_R: {OPTIMAL_CONFIG['psi_R']:.3f} rad ({np.degrees(OPTIMAL_CONFIG['psi_R']):.1f}°)")
    print(f"  theta_R: {OPTIMAL_CONFIG['theta_R']:.3f} rad ({np.degrees(OPTIMAL_CONFIG['theta_R']):.1f}°)")
    print(f"性能结果:")
    print(f"  信道容量: {optimal_cap:.2f} bps/Hz")
    print(f"  模型误差: {optimal_err:.2f} dB")
    print(f"  综合性能: {optimal_perf:.2f}")

    # 生成并计算随机配置的性能
    print(f"\n2. 生成并计算{num_random_configs}个随机配置的性能...")

    random_configs = []
    performances = []
    capacities = []
    errors = []

    for i in range(num_random_configs):
        config = generate_random_configuration()
        perf, cap, err = calculate_performance(env, config)

        random_configs.append(config)
        performances.append(perf)
        capacities.append(cap)
        errors.append(abs(err))

        if (i + 1) % 20 == 0:
            print(f"  已完成: {i + 1}/{num_random_configs}")

    # 统计分析
    print(f"\n3. 统计分析:")
    print(f"  随机配置平均性能: {np.mean(performances):.2f} ± {np.std(performances):.2f}")
    print(f"  随机配置中位数性能: {np.median(performances):.2f}")
    print(f"  随机配置最佳性能: {np.max(performances):.2f}")
    print(f"  最优RL配置性能: {optimal_perf:.2f}")
    print(f"  性能提升: {(optimal_perf - np.mean(performances)) / np.mean(performances) * 100:.1f}%")

    # 找出随机配置中的最佳配置
    best_random_idx = np.argmax(performances)
    best_random_config = random_configs[best_random_idx]
    best_random_perf = performances[best_random_idx]
    best_random_cap = capacities[best_random_idx]
    best_random_err = errors[best_random_idx]

    print(f"\n4. 随机配置中的最佳配置:")
    print(f"  W: {best_random_config['W']:.2f}λ")
    print(f"  Q_sub: {best_random_config['Q_sub']}")
    print(f"  delta_d: {best_random_config['delta_d']:.4f}λ")
    print(f"  psi_R: {best_random_config['psi_R']:.3f} rad ({np.degrees(best_random_config['psi_R']):.1f}°)")
    print(f"  theta_R: {best_random_config['theta_R']:.3f} rad ({np.degrees(best_random_config['theta_R']):.1f}°)")
    print(f"  信道容量: {best_random_cap:.2f} bps/Hz")
    print(f"  模型误差: {best_random_err:.2f} dB")
    print(f"  综合性能: {best_random_perf:.2f}")

    return {
        'optimal_config': OPTIMAL_CONFIG,
        'optimal_performance': optimal_perf,
        'optimal_capacity': optimal_cap,
        'optimal_error': optimal_err,
        'random_configs': random_configs,
        'random_performances': performances,
        'random_capacities': capacities,
        'random_errors': errors,
        'best_random_config': best_random_config,
        'best_random_performance': best_random_perf
    }


def plot_comparison_results(results, save_dir='./comparison_results/'):
    """
    绘制对比结果图
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    # 提取随机配置的参数
    W_vals = [c['W'] for c in results['random_configs']]
    Q_sub_vals = [c['Q_sub'] for c in results['random_configs']]
    psi_vals = [c['psi_R'] for c in results['random_configs']]
    theta_vals = [c['theta_R'] for c in results['random_configs']]
    perf_vals = results['random_performances']
    # 1. W vs Q_sub散点图（独立图表）
    fig1 = plt.figure(figsize=(10, 7))
    ax1 = fig1.add_subplot(111)
    scatter1 = ax1.scatter(W_vals, Q_sub_vals, c=perf_vals, cmap='viridis',
                           s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    # 首先在最优RL配置位置画一个点（使用相同的颜色映射）
    # 我们需要获取最优RL配置的性能值对应的颜色
    cmap1 = scatter1.get_cmap()
    norm1 = plt.Normalize(min(perf_vals), max(perf_vals))

    # 标记最优配置（先画点，再画圆圈框住）
    ax1.scatter(OPTIMAL_CONFIG['W'], OPTIMAL_CONFIG['Q_sub'],
                c=[results['optimal_performance']], cmap='viridis', norm=norm1,
                s=50, alpha=0.7, edgecolors='k', linewidth=0.5)

    # 然后用红色空心圆圈框住
    ax1.scatter(OPTIMAL_CONFIG['W'], OPTIMAL_CONFIG['Q_sub'],
                facecolors='none', edgecolors='red', s=300,
                linewidth=2, label='Optimal RL Configuration')
    # 找到最佳随机配置在列表中的索引
    best_random_idx = None
    for i, config in enumerate(results['random_configs']):
        if (abs(config['W'] - results['best_random_config']['W']) < 0.001 and
                config['Q_sub'] == results['best_random_config']['Q_sub'] and
                abs(config['psi_R'] - results['best_random_config']['psi_R']) < 0.001 and
                abs(config['theta_R'] - results['best_random_config']['theta_R']) < 0.001):
            best_random_idx = i
            break
    # 标记最佳随机配置（用绿色空心圆圈框住）
    if best_random_idx is not None:
        ax1.scatter(results['best_random_config']['W'],
                    results['best_random_config']['Q_sub'],
                    facecolors='none', edgecolors='green', s=300,
                    linewidth=2, label='Best Random Configuration')
    ax1.set_xlabel('FAS Length W (λ)', fontsize=12)
    ax1.set_ylabel('Number of Active Ports Q_sub', fontsize=12)
    ax1.set_title('Performance Distribution: W vs Q_sub', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    # 添加颜色条
    cbar1 = fig1.colorbar(scatter1, ax=ax1, shrink=0.8, aspect=20)
    cbar1.set_label('Comprehensive Performance', fontsize=11)
    plt.tight_layout()
    #plt.savefig(os.path.join(save_dir, 'W_vs_Qsub_scatter.svg'), dpi=300, bbox_inches='tight')
    plt.show()
    # 2. 角度空间散点图（独立图表）
    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111)
    scatter2 = ax2.scatter(psi_vals, theta_vals, c=perf_vals, cmap='plasma',
                           s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    # 首先在最优RL配置位置画一个点（使用相同的颜色映射）
    cmap2 = scatter2.get_cmap()
    norm2 = plt.Normalize(min(perf_vals), max(perf_vals))

    # 标记最优配置（先画点，再画圆圈框住）
    ax2.scatter(OPTIMAL_CONFIG['psi_R'], OPTIMAL_CONFIG['theta_R'],
                c=[results['optimal_performance']], cmap='plasma', norm=norm2,
                s=50, alpha=0.7, edgecolors='k', linewidth=0.5)

    # 然后用红色空心圆圈框住
    ax2.scatter(OPTIMAL_CONFIG['psi_R'], OPTIMAL_CONFIG['theta_R'],
                facecolors='none', edgecolors='red', s=300,
                linewidth=2, label='Optimal RL Configuration')
    # 标记最佳随机配置（用绿色空心圆圈框住）
    if best_random_idx is not None:
        ax2.scatter(results['best_random_config']['psi_R'],
                    results['best_random_config']['theta_R'],
                    facecolors='none', edgecolors='green', s=300,
                    linewidth=2, label='Best Random Configuration')
    ax2.set_xlabel('Azimuth Angle ψ_R (rad)', fontsize=12)
    ax2.set_ylabel('Elevation Angle θ_R (rad)', fontsize=12)
    ax2.set_title('Performance Distribution in Angular Space', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 2 * np.pi)
    ax2.set_ylim(0, np.pi)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    # 添加颜色条
    cbar2 = fig2.colorbar(scatter2, ax=ax2, shrink=0.8, aspect=20)
    cbar2.set_label('Comprehensive Performance', fontsize=11)
    plt.tight_layout()
    #plt.savefig(os.path.join(save_dir, 'angular_space_scatter.svg'), dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 3D可视化 - 第一个图：W, Q_sub, 性能
    fig3a = plt.figure(figsize=(10, 7))
    ax3a = fig3a.add_subplot(111, projection='3d')

    # 绘制随机配置
    scatter3da = ax3a.scatter(W_vals, Q_sub_vals, perf_vals,
                              c=perf_vals, cmap='viridis', s=30, alpha=0.6)

    # 标记最优配置
    ax3a.scatter(OPTIMAL_CONFIG['W'], OPTIMAL_CONFIG['Q_sub'], results['optimal_performance'],
                 c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                 label='Optimal RL Configuration')

    ax3a.set_xlabel('W (λ)', fontsize=12)
    ax3a.set_ylabel('Q_sub', fontsize=12)
    ax3a.set_zlabel('Performance', fontsize=12)
    ax3a.set_title('3D: W vs Q_sub vs Performance', fontsize=14, fontweight='bold')
    ax3a.legend()

    # 添加颜色条
    cbar3a = fig3a.colorbar(scatter3da, ax=ax3a, shrink=0.6, aspect=20)
    cbar3a.set_label('Comprehensive Performance', fontsize=11)

    plt.tight_layout()
    #plt.savefig(os.path.join(save_dir, '3d_w_qsub_performance.svg'), dpi=300, bbox_inches='tight')
    plt.show()

    # 4. 3D可视化 - 第二个图：psi_R, theta_R, 性能
    fig3b = plt.figure(figsize=(10, 7))
    ax3b = fig3b.add_subplot(111, projection='3d')

    scatter3db = ax3b.scatter(psi_vals, theta_vals, perf_vals,
                              c=perf_vals, cmap='plasma', s=30, alpha=0.6)

    # 标记最优配置
    ax3b.scatter(OPTIMAL_CONFIG['psi_R'], OPTIMAL_CONFIG['theta_R'], results['optimal_performance'],
                 c='red', s=200, marker='*', edgecolors='black', linewidth=2,
                 label='Optimal RL Configuration')

    ax3b.set_xlabel('ψ_R (rad)', fontsize=12)
    ax3b.set_ylabel('θ_R (rad)', fontsize=12)
    ax3b.set_zlabel('Performance', fontsize=12)
    ax3b.set_title('3D: Angle vs Performance', fontsize=14, fontweight='bold')
    ax3b.legend()

    # 添加颜色条
    cbar3b = fig3b.colorbar(scatter3db, ax=ax3b, shrink=0.6, aspect=20)
    cbar3b.set_label('Comprehensive Performance', fontsize=11)

    plt.tight_layout()
    #plt.savefig(os.path.join(save_dir, '3d_angle_performance.svg'), dpi=300, bbox_inches='tight')
    plt.show()

    # 5. 保存详细结果
    save_detailed_results(results, save_dir)

    print(f"\n所有图表已保存至: {save_dir}")


def save_detailed_results(results, save_dir):
    """保存详细结果到文件"""

    # 创建结果文件路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(save_dir, f'comparison_results_{timestamp}.json')
    csv_file = os.path.join(save_dir, f'random_configs_{timestamp}.csv')

    # 准备要保存的数据
    save_data = {
        'timestamp': timestamp,
        'optimal_config': results['optimal_config'],
        'optimal_performance': float(results['optimal_performance']),
        'optimal_capacity': float(results['optimal_capacity']),
        'optimal_error': float(results['optimal_error']),
        'best_random_config': results['best_random_config'],
        'best_random_performance': float(results['best_random_performance']),
        'summary': {
            'num_random_configs': len(results['random_configs']),
            'random_mean_performance': float(np.mean(results['random_performances'])),
            'random_std_performance': float(np.std(results['random_performances'])),
            'random_median_performance': float(np.median(results['random_performances'])),
            'performance_improvement_pct': float(
                (results['optimal_performance'] - np.mean(results['random_performances'])) / np.mean(
                    results['random_performances']) * 100)
        }
    }

    # 保存为JSON
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"详细结果已保存至: {results_file}")

    # 保存随机配置数据为CSV
    df_data = []
    for i, config in enumerate(results['random_configs']):
        row = {
            'config_id': i,
            'W': config['W'],
            'Q_sub': config['Q_sub'],
            'delta_d': config['delta_d'],
            'psi_R_rad': config['psi_R'],
            'theta_R_rad': config['theta_R'],
            'psi_R_deg': np.degrees(config['psi_R']),
            'theta_R_deg': np.degrees(config['theta_R']),
            'performance': results['random_performances'][i],
            'capacity': results['random_capacities'][i],
            'error': results['random_errors'][i]
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False, encoding='utf-8')

    print(f"随机配置数据已保存至: {csv_file}")

    # 打印总结报告
    print(f"\n{'=' * 60}")
    print("总结报告:")
    print(f"{'=' * 60}")
    print(f"最优RL配置性能: {results['optimal_performance']:.2f}")
    print(
        f"随机配置平均性能: {np.mean(results['random_performances']):.2f} ± {np.std(results['random_performances']):.2f}")
    print(f"性能提升: {save_data['summary']['performance_improvement_pct']:.1f}%")
    print(f"最佳随机配置性能: {results['best_random_performance']:.2f}")
    print(
        f"最优RL vs 最佳随机: {(results['optimal_performance'] - results['best_random_performance']) / results['best_random_performance'] * 100:.1f}%")


def main():
    """
    主函数：执行公平对比分析
    """
    print("FAS A2G信道优化 - 公平对比分析")
    print("=" * 60)
    print("说明:")
    print("1. 最优RL配置：训练得到的最优参数组合")
    print("2. 随机配置：W, Q_sub, psi_R, theta_R均随机选择")
    print("3. Q_sub改变时，delta_d = W/(Q-1) 自动调整")
    print("=" * 60)

    # 设置随机种子以确保可重复性
    set_random_seed(42)

    # 执行公平对比分析
    results = fair_comparison_analysis(
        num_random_configs=200,  # 可以调整随机配置数量
        seed=42
    )

    # 绘制对比结果
    plot_comparison_results(results, save_dir='./fig_comparison_results/')

    print("\n分析完成！")


if __name__ == "__main__":
    main()
