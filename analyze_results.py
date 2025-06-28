import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_prediction_results(result_file):
    """分析预测结果文件"""
    print(f"\n分析文件: {result_file}")
    
    # 加载结果
    results = torch.load(result_file)
    
    # 基本信息
    print(f"Combo: {results['combo']}")
    print(f"帧数: {results['frame_count']}")
    print(f"FPS: {results['fps']}")
    print(f"总时长: {results['frame_count'] / results['fps']:.2f} 秒")
    
    # 姿态数据分析
    pose = results['pose']  # [T, 24, 3, 3]
    translation = results['translation']  # [T, 3]
    joints = results['joints']  # [T, 72]
    contact = results['contact']  # [T, 2]
    
    print(f"\n=== 数据形状 ===")
    print(f"姿态: {pose.shape}")
    print(f"位移: {translation.shape}")
    print(f"关节: {joints.shape}")
    print(f"接触: {contact.shape}")
    
    # 位移统计
    print(f"\n=== 位移统计 ===")
    print(f"X轴范围: [{translation[:, 0].min():.3f}, {translation[:, 0].max():.3f}] m")
    print(f"Y轴范围: [{translation[:, 1].min():.3f}, {translation[:, 1].max():.3f}] m")
    print(f"Z轴范围: [{translation[:, 2].min():.3f}, {translation[:, 2].max():.3f}] m")
    
    # 速度分析（通过位移变化计算）
    velocity = torch.diff(translation, dim=0) * results['fps']  # 估算速度
    speed = torch.norm(velocity, dim=1)  # 速度大小
    
    print(f"\n=== 运动统计 ===")
    print(f"平均速度: {speed.mean():.3f} m/s")
    print(f"最大速度: {speed.max():.3f} m/s")
    print(f"最小速度: {speed.min():.3f} m/s")
    
    # 足地接触分析
    left_contact = contact[:, 0]  # 左脚接触概率
    right_contact = contact[:, 1]  # 右脚接触概率
    
    print(f"\n=== 足地接触统计 ===")
    print(f"左脚平均接触概率: {left_contact.mean():.3f}")
    print(f"右脚平均接触概率: {right_contact.mean():.3f}")
    print(f"双脚同时接触比例: {((left_contact > 0.5) & (right_contact > 0.5)).float().mean():.3f}")
    
    return results

def compare_combos(file1, file2):
    """比较两个combo的结果"""
    print(f"\n{'='*50}")
    print("比较不同combo的预测结果")
    print(f"{'='*50}")
    
    results1 = torch.load(file1)
    results2 = torch.load(file2)
    
    combo1 = results1['combo']
    combo2 = results2['combo']
    
    print(f"\nCombo {combo1} vs Combo {combo2}")
    print("-" * 40)
    
    # 比较位移差异
    trans1 = results1['translation']
    trans2 = results2['translation']
    
    trans_diff = torch.norm(trans1 - trans2, dim=1)
    print(f"位移差异 (平均): {trans_diff.mean():.4f} m")
    print(f"位移差异 (最大): {trans_diff.max():.4f} m")
    
    # 比较运动模式
    vel1 = torch.diff(trans1, dim=0) * results1['fps']
    vel2 = torch.diff(trans2, dim=0) * results2['fps']
    
    speed1 = torch.norm(vel1, dim=1)
    speed2 = torch.norm(vel2, dim=1)
    
    print(f"平均速度差异: {abs(speed1.mean() - speed2.mean()):.4f} m/s")
    
    # 比较足地接触
    contact1 = results1['contact']
    contact2 = results2['contact']
    
    contact_diff = torch.norm(contact1 - contact2, dim=1)
    print(f"接触预测差异: {contact_diff.mean():.4f}")

def create_visualization_plots(result_file, output_dir="plots"):
    """创建可视化图表"""
    results = torch.load(result_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    combo = results['combo']
    translation = results['translation'].numpy()
    contact = results['contact'].numpy()
    fps = results['fps']
    time_axis = np.arange(len(translation)) / fps
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'MobilePoser 预测结果分析 - Combo: {combo}', fontsize=16)
    
    # 1. 3D轨迹
    ax = axes[0, 0]
    ax.plot(translation[:, 0], translation[:, 2], 'b-', alpha=0.7, linewidth=1)
    ax.scatter(translation[0, 0], translation[0, 2], color='green', s=100, marker='o', label='起始点')
    ax.scatter(translation[-1, 0], translation[-1, 2], color='red', s=100, marker='s', label='结束点')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('运动轨迹 (俯视图)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 2. 位移vs时间
    ax = axes[0, 1]
    ax.plot(time_axis, translation[:, 0], 'r-', label='X', alpha=0.8)
    ax.plot(time_axis, translation[:, 1], 'g-', label='Y', alpha=0.8)
    ax.plot(time_axis, translation[:, 2], 'b-', label='Z', alpha=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('位移 (m)')
    ax.set_title('位移随时间变化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 速度分析
    velocity = np.diff(translation, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=1)
    
    ax = axes[1, 0]
    ax.plot(time_axis[:-1], speed, 'purple', alpha=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.set_title('运动速度')
    ax.grid(True, alpha=0.3)
    
    # 4. 足地接触
    ax = axes[1, 1]
    ax.plot(time_axis, contact[:, 0], 'blue', label='左脚', alpha=0.8)
    ax.plot(time_axis, contact[:, 1], 'red', label='右脚', alpha=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='阈值')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('接触概率')
    ax.set_title('足地接触预测')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = output_dir / f"analysis_{combo}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"分析图表已保存到: {plot_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析MobilePoser预测结果')
    parser.add_argument('--file1', type=str, help='第一个结果文件')
    parser.add_argument('--file2', type=str, help='第二个结果文件')
    parser.add_argument('--label1', type=str, default='结果1', help='第一个结果的标签')
    parser.add_argument('--label2', type=str, default='结果2', help='第二个结果的标签')
    
    args = parser.parse_args()
    
    if args.file1 and args.file2:
        # 分析两个指定文件
        print(f"分析对比：{args.label1} vs {args.label2}")
        
        if Path(args.file1).exists():
            print(f"\n=== {args.label1} ===")
            analyze_prediction_results(args.file1)
            create_visualization_plots(args.file1)
        
        if Path(args.file2).exists():
            print(f"\n=== {args.label2} ===")
            analyze_prediction_results(args.file2)
            create_visualization_plots(args.file2)
        
        if Path(args.file1).exists() and Path(args.file2).exists():
            print(f"\n=== 对比分析：{args.label1} vs {args.label2} ===")
            compare_combos(args.file1, args.file2)
    else:
        # 默认分析现有的结果文件
        result_files = ["results_lw_rp_corrected.pt", "results_rp_only_corrected.pt"]
        
        for file in result_files:
            if Path(file).exists():
                analyze_prediction_results(file)
                create_visualization_plots(file)
        
        # 比较不同combo
        if all(Path(f).exists() for f in result_files):
            compare_combos(result_files[0], result_files[1])
    
    print(f"\n{'='*50}")
    print("分析完成!")
    print(f"{'='*50}") 