import os
import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体
mpl.rcParams['font.family'] = 'WenQuanYi Zen Hei'
mpl.rcParams['axes.unicode_minus'] = False

from mobileposer.config import *
from mobileposer.utils.model_utils import *
from mobileposer.viewer import Viewer
from mobileposer.loader import DataLoader


def visualize_imu_data(dataset: str, seq_num: int, combo: str, device='cuda'):
    """
    可视化当前combo对应的IMU数据
    
    Args:
        dataset: 数据集名称
        seq_num: 序列编号
        combo: IMU组合方式
        device: 计算设备
    """
    try:
        print(f"\n=== IMU数据可视化 ===")
        print(f"数据集: {dataset}, 序列: {seq_num}, 组合: {combo}")
        
        # 加载数据
        dataloader = DataLoader(dataset, combo=combo, device=device)
        data = dataloader.load_data(seq_num)
        
        # 获取IMU数据
        imu_data = data['imu']  # [T, feature_dim]
        
        # 获取combo对应的IMU ID
        combo_ids = amass.combos[combo]
        print(f"使用的IMU传感器ID: {combo_ids}")
        
        # 从feature向量中提取IMU数据
        # feature格式：[acc_data(5*3=15), ori_data(5*3*3=45)] = 60维
        T = imu_data.shape[0]
        
        # 提取加速度数据 [T, 5, 3]
        acc_data = imu_data[:, :15].view(T, 5, 3)
        
        # 提取方向数据 [T, 5, 3, 3] 
        ori_data = imu_data[:, 15:].view(T, 5, 3, 3)
        
        # 计算时间轴（假设30fps）
        time_seconds = np.arange(T) / 30.0
        
        # 创建子图
        fig, axes = plt.subplots(len(combo_ids), 2, figsize=(15, 4 * len(combo_ids)))
        if len(combo_ids) == 1:
            axes = axes.reshape(1, -1)
        
        # IMU位置名称映射
        imu_names = {0: '左手腕', 1: '右手腕', 2: '左大腿', 3: '右大腿', 4: '头部'}
        
        # 为每个IMU传感器绘图
        for idx, imu_id in enumerate(combo_ids):
            imu_name = imu_names.get(imu_id, f'IMU_{imu_id}')
            
            # 提取当前IMU的数据
            acc_current = acc_data[:, imu_id, :].cpu().numpy()  # [T, 3]
            ori_current = ori_data[:, imu_id, :, :].cpu().numpy()  # [T, 3, 3]
            
            # 将旋转矩阵转换为欧拉角用于可视化
            euler_angles = rotation_matrices_to_euler_angles(ori_current)
            
            # 绘制加速度
            axes[idx, 0].plot(time_seconds, acc_current[:, 0], 'r-', alpha=0.8, label='X轴', linewidth=1.5)
            axes[idx, 0].plot(time_seconds, acc_current[:, 1], 'g-', alpha=0.8, label='Y轴', linewidth=1.5)
            axes[idx, 0].plot(time_seconds, acc_current[:, 2], 'b-', alpha=0.8, label='Z轴', linewidth=1.5)
            axes[idx, 0].set_title(f'{imu_name} - 线性加速度', fontsize=12)
            axes[idx, 0].set_xlabel('时间 (秒)')
            axes[idx, 0].set_ylabel('加速度 (m/s²)')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            
            # 绘制角度
            axes[idx, 1].plot(time_seconds, euler_angles[:, 0], 'r-', alpha=0.8, label='Roll', linewidth=1.5)
            axes[idx, 1].plot(time_seconds, euler_angles[:, 1], 'g-', alpha=0.8, label='Pitch', linewidth=1.5)
            axes[idx, 1].plot(time_seconds, euler_angles[:, 2], 'b-', alpha=0.8, label='Yaw', linewidth=1.5)
            axes[idx, 1].set_title(f'{imu_name} - 方向角度', fontsize=12)
            axes[idx, 1].set_xlabel('时间 (秒)')
            axes[idx, 1].set_ylabel('角度 (度)')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)
            
            # 添加统计信息
            acc_rms = np.sqrt(np.mean(acc_current**2, axis=0))
            ori_std = np.std(euler_angles, axis=0)
            
            # 在图上添加文本信息
            axes[idx, 0].text(0.02, 0.98, f'RMS: X={acc_rms[0]:.2f}, Y={acc_rms[1]:.2f}, Z={acc_rms[2]:.2f}',
                            transform=axes[idx, 0].transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            axes[idx, 1].text(0.02, 0.98, f'STD: R={ori_std[0]:.1f}°, P={ori_std[1]:.1f}°, Y={ori_std[2]:.1f}°',
                            transform=axes[idx, 1].transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 调整布局
        plt.tight_layout(pad=3.0)
        
        # 添加整体标题
        fig.suptitle(f'IMU数据可视化 - {dataset} 序列{seq_num} 组合{combo}', fontsize=16, y=0.98)
        
        # 保存图像
        output_filename = f'imu_visualization_{dataset}_seq{seq_num}_{combo}.png'
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"IMU可视化图已保存为: {output_filename}")
        
        # 显示图像
        plt.show()
        
        # 打印数据统计信息
        print(f"\n=== 数据统计信息 ===")
        print(f"数据长度: {T} 帧 ({T/30.0:.1f} 秒)")
        print(f"使用的IMU数量: {len(combo_ids)}")
        
        for idx, imu_id in enumerate(combo_ids):
            imu_name = imu_names.get(imu_id, f'IMU_{imu_id}')
            acc_current = acc_data[:, imu_id, :].cpu().numpy()
            ori_current = ori_data[:, imu_id, :, :].cpu().numpy()
            euler_angles = rotation_matrices_to_euler_angles(ori_current)
            
            acc_magnitude = np.sqrt(np.sum(acc_current**2, axis=1))
            print(f"\n{imu_name} (ID={imu_id}):")
            print(f"  加速度范围: {np.min(acc_magnitude):.2f} - {np.max(acc_magnitude):.2f} m/s²")
            print(f"  角度范围: Roll [{np.min(euler_angles[:, 0]):.1f}, {np.max(euler_angles[:, 0]):.1f}]°")
            print(f"           Pitch [{np.min(euler_angles[:, 1]):.1f}, {np.max(euler_angles[:, 1]):.1f}]°")
            print(f"           Yaw [{np.min(euler_angles[:, 2]):.1f}, {np.max(euler_angles[:, 2]):.1f}]°")
        
    except Exception as e:
        print(f"IMU数据可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def rotation_matrices_to_euler_angles(rotation_matrices):
    """
    将旋转矩阵转换为欧拉角（度数）
    
    Args:
        rotation_matrices: [T, 3, 3] 旋转矩阵
    
    Returns:
        euler_angles: [T, 3] 欧拉角（度数）格式为 [roll, pitch, yaw]
    """
    T = rotation_matrices.shape[0]
    euler_angles = np.zeros((T, 3))
    
    for i in range(T):
        R = rotation_matrices[i]
        
        # 提取欧拉角 (ZYX顺序)
        # Roll (x-axis rotation)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # Roll
            y = np.arctan2(-R[2, 0], sy)      # Pitch
            z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        # 转换为度数
        euler_angles[i] = [np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)]
    
    return euler_angles


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--model', type=str, default=paths.weights_file)
    args.add_argument('--dataset', type=str, default='CMU_mini')
    args.add_argument('--combo', type=str, default='lw_rp')
    args.add_argument('--with-tran', action='store_true')
    args.add_argument('--seq-num', type=int, default=7)
    args.add_argument('--force-aitviewer', action='store_true', help='强制使用aitviewer，即使可能出错')
    args.add_argument('--fallback-original', action='store_true', help='强制使用原始viewer，不尝试aitviewer')
    args.add_argument('--visualize-imu', action='store_true', help='生成IMU数据可视化图例')
    args = args.parse_args()

    # check for valid combo
    combos = amass.combos.keys()
    if args.combo not in combos:
        raise ValueError(f"Invalid combo: {args.combo}. Must be one of {combos}")

    print(f"=== MobilePoser 可视化 ===")
    print(f"模型权重: {args.model}")
    print(f"数据集: {args.dataset}")
    print(f"IMU组合: {args.combo}")
    print(f"序列编号: {args.seq_num}")
    print(f"包含位移: {args.with_tran}")
    print(f"强制使用aitviewer: {args.force_aitviewer}")
    print(f"强制使用原始viewer: {args.fallback_original}")
    print(f"生成IMU可视化: {args.visualize_imu}")
    print("=" * 30)

    # 根据参数选择viewer类型
    if args.fallback_original:
        # 强制使用原始viewer
        print("强制使用原始viewer...")
        from mobileposer.viewers import SMPLViewer
        import mobileposer.articulate as art
        
        # 创建基础viewer但不使用aitviewer功能
        class OriginalViewer:
            def __init__(self, dataset: str='imuposer', seq_num: int=0, combo: str='lw_rp'):
                from mobileposer.loader import DataLoader
                self.device = model_config.device
                self.model = load_model(args.model or paths.weights_file).to(self.device).eval()
                self.dataloader = DataLoader(dataset, combo=combo, device=self.device)
                self.data = self.dataloader.load_data(seq_num)
            
            def _evaluate_model(self):
                from tqdm import tqdm
                data = self.data['imu']
                if getenv('ONLINE'):
                    pose, joints, tran, contact = [torch.stack(_) for _ in zip(*[self.model.forward_online(f) for f in tqdm(data)])]
                else:
                    with torch.no_grad():
                        pose, joints, tran, contact = self.model.forward_offline(data.unsqueeze(0), [data.shape[0]]) 
                return pose, tran, joints, contact
            
            def view(self, with_tran: bool=False):
                pose_t, tran_t = self.data['pose'], self.data['tran']
                pose_p, tran_p, _, _ = self._evaluate_model()
                viewer = SMPLViewer()
                viewer.view(pose_p, tran_p, pose_t, tran_t, with_tran=with_tran)
        
        v = OriginalViewer(dataset=args.dataset, seq_num=args.seq_num, combo=args.combo)
        
    elif args.force_aitviewer:
        # 强制使用aitviewer
        print("强制使用aitviewer...")
        from mobileposer.viewer import AitViewer
        v = AitViewer(dataset=args.dataset, seq_num=args.seq_num, combo=args.combo)
        
    else:
        # 默认行为：优先尝试aitviewer，失败后回退到原始viewer
        print("自动选择最佳viewer...")
        v = Viewer(dataset=args.dataset, seq_num=args.seq_num, combo=args.combo)

    # 检查是否需要生成IMU可视化
    if args.visualize_imu:
        print("生成IMU数据可视化...")
        device = model_config.device
        visualize_imu_data(args.dataset, args.seq_num, args.combo, device)
    
    # 执行3D姿态可视化
    print("开始3D姿态可视化...")
    v.view(with_tran=args.with_tran)

