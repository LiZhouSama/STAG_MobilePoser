import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import os

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角(度)转换为旋转矩阵
    使用ZYX顺序 (yaw-pitch-roll)
    """
    # 转换为弧度
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # 计算三角函数
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    
    # ZYX欧拉角转旋转矩阵
    R = np.array([
        [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
        [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
        [-sin_p, cos_p * sin_r, cos_p * cos_r]
    ])
    
    return R

def plot_coordinate_frame(ax, position, rotation_matrix, scale=0.3, labels=None):
    """
    在3D图中绘制坐标系
    
    Args:
        ax: 3D坐标轴
        position: 坐标系原点位置 [x, y, z]
        rotation_matrix: 3x3旋转矩阵
        scale: 坐标轴长度
        labels: 坐标轴标签 ['X', 'Y', 'Z']
    """
    if labels is None:
        labels = ['X', 'Y', 'Z']
    
    colors = ['red', 'green', 'blue']
    
    # 绘制三个坐标轴
    for i in range(3):
        # 计算轴端点
        axis_end = position + scale * rotation_matrix[:, i]
        
        # 绘制箭头
        ax.quiver(position[0], position[1], position[2],
                 rotation_matrix[0, i], rotation_matrix[1, i], rotation_matrix[2, i],
                 length=scale, color=colors[i], arrow_length_ratio=0.2, linewidth=2)
        
        # 添加标签
        label_pos = position + (scale + 0.1) * rotation_matrix[:, i]
        ax.text(label_pos[0], label_pos[1], label_pos[2], labels[i], 
                color=colors[i], fontsize=10, fontweight='bold')

def load_original_data():
    """
    加载原始的IMU数据文件
    
    Returns:
        phone_ori_original, watch_ori_original: 原始方向数据
    """
    phone_file = "mobileposer/stag_raw_data/IMUPoser_Phone4.csv"
    watch_file = "mobileposer/stag_raw_data/IMUPoser_Watch4.csv"
    
    if not os.path.exists(phone_file) or not os.path.exists(watch_file):
        print("警告：找不到原始数据文件，将跳过原始坐标系显示")
        return None, None
    
    import pandas as pd
    
    phone_data = pd.read_csv(phone_file)
    watch_data = pd.read_csv(watch_file)
    
    # 提取原始方向数据
    phone_ori_original = phone_data[['ori_roll_deg', 'ori_pitch_deg', 'ori_yaw_deg']].values
    watch_ori_original = watch_data[['ori_roll_deg', 'ori_pitch_deg', 'ori_yaw_deg']].values
    
    return phone_ori_original, watch_ori_original

def visualize_frame_coordinates(data_file, frame_idx=0, show_animation=False):
    """
    可视化指定帧的IMU坐标系朝向
    
    Args:
        data_file: 处理后的数据文件路径
        frame_idx: 要显示的帧索引
        show_animation: 是否显示动画
    """
    # 加载处理后的数据
    if not os.path.exists(data_file):
        print(f"错误：找不到数据文件 {data_file}")
        return
    
    data = torch.load(data_file)
    frame_count = data['frame_count']
    
    print(f"数据总帧数: {frame_count}")
    print(f"IMU位置: {data['imu_positions']}")
    
    # 提取变换后的方向数据
    phone_ori_transformed = torch.stack([
        data['imu_data']['rp']['ori_roll_deg'],
        data['imu_data']['rp']['ori_pitch_deg'], 
        data['imu_data']['rp']['ori_yaw_deg']
    ], dim=1).numpy()  # [frame_count, 3]
    
    watch_ori_transformed = torch.stack([
        data['imu_data']['lw']['ori_roll_deg'],
        data['imu_data']['lw']['ori_pitch_deg'],
        data['imu_data']['lw']['ori_yaw_deg']
    ], dim=1).numpy()  # [frame_count, 3]
    
    # 加载原始数据
    phone_ori_original, watch_ori_original = load_original_data()
    
    if show_animation:
        create_animation(phone_ori_transformed, watch_ori_transformed, data, 
                        phone_ori_original, watch_ori_original)
    else:
        create_static_visualization(phone_ori_transformed, watch_ori_transformed, frame_idx, data,
                                  phone_ori_original, watch_ori_original)

def create_static_visualization(phone_ori_transformed, watch_ori_transformed, frame_idx, data,
                               phone_ori_original=None, watch_ori_original=None):
    """
    创建静态的坐标系可视化，包含原始和变换后的坐标系对比
    """
    frame_count = len(phone_ori_transformed)
    frame_idx = max(0, min(frame_idx, frame_count - 1))
    
    # 计算原始数据对应的帧索引
    metadata = data['metadata']
    phone_start = metadata['phone_valid_range'][0]
    watch_start = metadata['watch_valid_range'][0]
    alignment_shift = metadata['alignment_shift']
    
    # 计算原始数据中的对应帧
    phone_original_idx = phone_start + frame_idx
    if alignment_shift > 0:
        watch_original_idx = watch_start + frame_idx + alignment_shift
    else:
        phone_original_idx += abs(alignment_shift)
        watch_original_idx = watch_start + frame_idx
    
    # 创建图形
    fig = plt.figure(figsize=(18, 12))
    
    # 主3D图 - 变换后的坐标系
    ax_3d_transformed = fig.add_subplot(231, projection='3d')
    
    # 获取当前帧的变换后方向数据
    phone_roll_t, phone_pitch_t, phone_yaw_t = phone_ori_transformed[frame_idx]
    watch_roll_t, watch_pitch_t, watch_yaw_t = watch_ori_transformed[frame_idx]
    
    print(f"帧 {frame_idx}:")
    print(f"手机变换后方向: Roll={phone_roll_t:.2f}°, Pitch={phone_pitch_t:.2f}°, Yaw={phone_yaw_t:.2f}°")
    print(f"手表变换后方向: Roll={watch_roll_t:.2f}°, Pitch={watch_pitch_t:.2f}°, Yaw={watch_yaw_t:.2f}°")
    
    # 计算变换后的旋转矩阵
    phone_rot_matrix_t = euler_to_rotation_matrix(phone_roll_t, phone_pitch_t, phone_yaw_t)
    watch_rot_matrix_t = euler_to_rotation_matrix(watch_roll_t, watch_pitch_t, watch_yaw_t)
    
    # 设置IMU位置
    phone_pos = np.array([0, 0, 0])
    watch_pos = np.array([1.5, 0, 0])
    
    # 绘制变换后的坐标系
    plot_coordinate_frame(ax_3d_transformed, phone_pos, phone_rot_matrix_t, scale=0.5, 
                         labels=['P_X', 'P_Y', 'P_Z'])
    plot_coordinate_frame(ax_3d_transformed, watch_pos, watch_rot_matrix_t, scale=0.5,
                         labels=['W_X', 'W_Y', 'W_Z'])
    
    # 绘制设备表示
    ax_3d_transformed.scatter(*phone_pos, color='blue', s=100, label='Phone (右口袋)')
    ax_3d_transformed.scatter(*watch_pos, color='orange', s=100, label='Watch (左手腕)')
    
    # 添加SMPL参考坐标系
    ref_pos = np.array([-0.8, 0, 0])
    ref_rot = np.eye(3)
    plot_coordinate_frame(ax_3d_transformed, ref_pos, ref_rot, scale=0.4,
                         labels=['SMPL_X', 'SMPL_Y', 'SMPL_Z'])
    ax_3d_transformed.scatter(*ref_pos, color='black', s=80, marker='s', label='SMPL参考')
    
    # 设置变换后3D图属性
    ax_3d_transformed.set_xlim([-1.2, 2.2])
    ax_3d_transformed.set_ylim([-0.8, 0.8])
    ax_3d_transformed.set_zlim([-0.8, 0.8])
    ax_3d_transformed.set_xlabel('X (Left/Right)')
    ax_3d_transformed.set_ylabel('Y (Up/Down)')
    ax_3d_transformed.set_zlabel('Z (Forward/Backward)')
    ax_3d_transformed.set_title(f'变换后坐标系 (帧 {frame_idx})\nSMPL坐标系: X=Left, Y=Up, Z=Forward')
    ax_3d_transformed.legend()
    
    # 原始坐标系3D图
    if phone_ori_original is not None and watch_ori_original is not None:
        ax_3d_original = fig.add_subplot(232, projection='3d')
        
        # 确保索引在有效范围内
        if (phone_original_idx < len(phone_ori_original) and 
            watch_original_idx < len(watch_ori_original)):
            
            # 获取原始方向数据
            phone_roll_o, phone_pitch_o, phone_yaw_o = phone_ori_original[phone_original_idx]
            watch_roll_o, watch_pitch_o, watch_yaw_o = watch_ori_original[watch_original_idx]
            
            print(f"手机原始方向: Roll={phone_roll_o:.2f}°, Pitch={phone_pitch_o:.2f}°, Yaw={phone_yaw_o:.2f}°")
            print(f"手表原始方向: Roll={watch_roll_o:.2f}°, Pitch={watch_pitch_o:.2f}°, Yaw={watch_yaw_o:.2f}°")
            
            # 计算原始旋转矩阵
            phone_rot_matrix_o = euler_to_rotation_matrix(phone_roll_o, phone_pitch_o, phone_yaw_o)
            watch_rot_matrix_o = euler_to_rotation_matrix(watch_roll_o, watch_pitch_o, watch_yaw_o)
            
            # 绘制原始坐标系
            plot_coordinate_frame(ax_3d_original, phone_pos, phone_rot_matrix_o, scale=0.5,
                                labels=['P_X_orig', 'P_Y_orig', 'P_Z_orig'])
            plot_coordinate_frame(ax_3d_original, watch_pos, watch_rot_matrix_o, scale=0.5,
                                labels=['W_X_orig', 'W_Y_orig', 'W_Z_orig'])
            
            # 绘制设备表示
            ax_3d_original.scatter(*phone_pos, color='blue', s=100, label='Phone (原始)')
            ax_3d_original.scatter(*watch_pos, color='orange', s=100, label='Watch (原始)')
            
            # 设置原始3D图属性
            ax_3d_original.set_xlim([-1.2, 2.2])
            ax_3d_original.set_ylim([-0.8, 0.8])
            ax_3d_original.set_zlim([-0.8, 0.8])
            ax_3d_original.set_xlabel('X')
            ax_3d_original.set_ylabel('Y')
            ax_3d_original.set_zlabel('Z')
            ax_3d_original.set_title(f'原始坐标系 (帧 {frame_idx})\n手机: X=Right, Y=Down, Z=Forward\n手表: X=Left, Y=Backward, Z=Up')
            ax_3d_original.legend()
        else:
            ax_3d_original.text(0.5, 0.5, 0.5, '原始数据索引超出范围', 
                               transform=ax_3d_original.transAxes, ha='center', va='center')
            ax_3d_original.set_title('原始坐标系 (数据不可用)')
    
    # 对比图 - 显示坐标系定义
    ax_comparison = fig.add_subplot(233, projection='3d')
    
    # 绘制坐标系定义对比
    phone_pos_comp = np.array([-1, 0, 0])
    watch_pos_comp = np.array([0, 0, 0])
    smpl_pos_comp = np.array([1, 0, 0])
    
    # 手机原始坐标系: X=Right, Y=Down, Z=Forward
    phone_original_def = np.array([
        [1, 0, 0],   # X=Right
        [0, -1, 0],  # Y=Down  
        [0, 0, 1]    # Z=Forward
    ])
    
    # 手表原始坐标系: X=Left, Y=Backward, Z=Up
    watch_original_def = np.array([
        [-1, 0, 0],  # X=Left
        [0, -1, 0],  # Y=Backward
        [0, 0, 1]    # Z=Up
    ])
    
    # SMPL坐标系: X=Left, Y=Up, Z=Forward
    smpl_def = np.eye(3)
    
    plot_coordinate_frame(ax_comparison, phone_pos_comp, phone_original_def, scale=0.3,
                         labels=['Phone_R', 'Phone_D', 'Phone_F'])
    plot_coordinate_frame(ax_comparison, watch_pos_comp, watch_original_def, scale=0.3,
                         labels=['Watch_L', 'Watch_B', 'Watch_U'])
    plot_coordinate_frame(ax_comparison, smpl_pos_comp, smpl_def, scale=0.3,
                         labels=['SMPL_L', 'SMPL_U', 'SMPL_F'])
    
    ax_comparison.scatter(*phone_pos_comp, color='blue', s=80, label='Phone原始定义')
    ax_comparison.scatter(*watch_pos_comp, color='orange', s=80, label='Watch原始定义')
    ax_comparison.scatter(*smpl_pos_comp, color='black', s=80, marker='s', label='SMPL定义')
    
    ax_comparison.set_xlim([-1.5, 1.5])
    ax_comparison.set_ylim([-0.5, 0.5])
    ax_comparison.set_zlim([-0.5, 0.5])
    ax_comparison.set_title('坐标系定义对比')
    ax_comparison.legend()
    
    # 绘制方向角时间序列
    frames = np.arange(len(phone_ori_transformed))
    
    # 手机方向角对比 (变换后)
    ax_phone = fig.add_subplot(234)
    ax_phone.plot(frames, phone_ori_transformed[:, 0], 'r-', label='Roll (变换后)', alpha=0.7)
    ax_phone.plot(frames, phone_ori_transformed[:, 1], 'g-', label='Pitch (变换后)', alpha=0.7)
    ax_phone.plot(frames, phone_ori_transformed[:, 2], 'b-', label='Yaw (变换后)', alpha=0.7)
    ax_phone.axvline(frame_idx, color='black', linestyle='--', alpha=0.8, label=f'当前帧({frame_idx})')
    ax_phone.set_title('手机方向角时间序列 (变换后)')
    ax_phone.set_xlabel('帧数')
    ax_phone.set_ylabel('角度 (度)')
    ax_phone.legend()
    ax_phone.grid(True, alpha=0.3)
    
    # 手表方向角对比 (变换后)
    ax_watch = fig.add_subplot(235)
    ax_watch.plot(frames, watch_ori_transformed[:, 0], 'r-', label='Roll (变换后)', alpha=0.7)
    ax_watch.plot(frames, watch_ori_transformed[:, 1], 'g-', label='Pitch (变换后)', alpha=0.7)
    ax_watch.plot(frames, watch_ori_transformed[:, 2], 'b-', label='Yaw (变换后)', alpha=0.7)
    ax_watch.axvline(frame_idx, color='black', linestyle='--', alpha=0.8, label=f'当前帧({frame_idx})')
    ax_watch.set_title('手表方向角时间序列 (变换后)')
    ax_watch.set_xlabel('帧数')
    ax_watch.set_ylabel('角度 (度)')
    ax_watch.legend()
    ax_watch.grid(True, alpha=0.3)
    
    # 加速度对比
    ax_acc = fig.add_subplot(236)
    phone_acc_x = data['imu_data']['rp']['lin_acc_x'].numpy()
    watch_acc_x = data['imu_data']['lw']['lin_acc_x'].numpy()
    ax_acc.plot(frames, phone_acc_x, label='手机 X轴加速度 (变换后)', alpha=0.7)
    ax_acc.plot(frames, watch_acc_x, label='手表 X轴加速度 (变换后)', alpha=0.7)
    ax_acc.axvline(frame_idx, color='black', linestyle='--', alpha=0.8, label=f'当前帧({frame_idx})')
    ax_acc.set_title('X轴加速度对比 (变换后)')
    ax_acc.set_xlabel('帧数')
    ax_acc.set_ylabel('加速度 (m/s²)')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'mobileposer/stag_raw_data/coordinate_visualization_comparison_frame_{frame_idx}.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

def create_interactive_visualization(data_file):
    """
    创建交互式可视化，带有帧数滑块
    """
    if not os.path.exists(data_file):
        print(f"错误：找不到数据文件 {data_file}")
        return
    
    data = torch.load(data_file)
    frame_count = data['frame_count']
    
    # 提取变换后的方向数据
    phone_ori_transformed = torch.stack([
        data['imu_data']['rp']['ori_roll_deg'],
        data['imu_data']['rp']['ori_pitch_deg'], 
        data['imu_data']['rp']['ori_yaw_deg']
    ], dim=1).numpy()
    
    watch_ori_transformed = torch.stack([
        data['imu_data']['lw']['ori_roll_deg'],
        data['imu_data']['lw']['ori_pitch_deg'],
        data['imu_data']['lw']['ori_yaw_deg']
    ], dim=1).numpy()
    
    # 加载原始数据
    phone_ori_original, watch_ori_original = load_original_data()
    
    # 创建图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 初始化空的图形元素
    phone_quivers = []
    watch_quivers = []
    phone_texts = []
    watch_texts = []
    
    def update_frame(frame_idx):
        """更新显示的帧"""
        frame_idx = int(frame_idx)
        ax.clear()
        
        # 获取当前帧的变换后方向数据
        phone_roll_t, phone_pitch_t, phone_yaw_t = phone_ori_transformed[frame_idx]
        watch_roll_t, watch_pitch_t, watch_yaw_t = watch_ori_transformed[frame_idx]
        
        # 计算变换后的旋转矩阵
        phone_rot_matrix_t = euler_to_rotation_matrix(phone_roll_t, phone_pitch_t, phone_yaw_t)
        watch_rot_matrix_t = euler_to_rotation_matrix(watch_roll_t, watch_pitch_t, watch_yaw_t)
        
        # 设置IMU位置 - 变换后
        phone_pos_t = np.array([0, 0, 0])
        watch_pos_t = np.array([1.5, 0, 0])
        
        # 绘制变换后的坐标系
        plot_coordinate_frame(ax, phone_pos_t, phone_rot_matrix_t, scale=0.5, 
                             labels=['P_X_T', 'P_Y_T', 'P_Z_T'])
        plot_coordinate_frame(ax, watch_pos_t, watch_rot_matrix_t, scale=0.5,
                             labels=['W_X_T', 'W_Y_T', 'W_Z_T'])
        
        # 绘制变换后的设备
        ax.scatter(*phone_pos_t, color='blue', s=150, marker='o', label='Phone (变换后)')
        ax.scatter(*watch_pos_t, color='orange', s=150, marker='o', label='Watch (变换后)')
        
        # 绘制原始坐标系（如果有数据）
        if phone_ori_original is not None and watch_ori_original is not None:
            # 计算原始数据的对应帧索引
            metadata = data['metadata']
            phone_start = metadata['phone_valid_range'][0]
            watch_start = metadata['watch_valid_range'][0]
            alignment_shift = metadata['alignment_shift']
            
            phone_original_idx = phone_start + frame_idx
            if alignment_shift > 0:
                watch_original_idx = watch_start + frame_idx + alignment_shift
            else:
                phone_original_idx += abs(alignment_shift)
                watch_original_idx = watch_start + frame_idx
            
            if (phone_original_idx < len(phone_ori_original) and 
                watch_original_idx < len(watch_ori_original)):
                
                # 获取原始方向数据
                phone_roll_o, phone_pitch_o, phone_yaw_o = phone_ori_original[phone_original_idx]
                watch_roll_o, watch_pitch_o, watch_yaw_o = watch_ori_original[watch_original_idx]
                
                # 计算原始旋转矩阵
                phone_rot_matrix_o = euler_to_rotation_matrix(phone_roll_o, phone_pitch_o, phone_yaw_o)
                watch_rot_matrix_o = euler_to_rotation_matrix(watch_roll_o, watch_pitch_o, watch_yaw_o)
                
                # 设置原始IMU位置（稍微偏移以避免重叠）
                phone_pos_o = np.array([0, 0, -1])
                watch_pos_o = np.array([1.5, 0, -1])
                
                # 绘制原始坐标系（用更淡的颜色和虚线效果）
                # 为了区分，使用不同的标记和透明度
                for i in range(3):
                    colors = ['red', 'green', 'blue']
                    # 原始坐标系用半透明
                    ax.quiver(phone_pos_o[0], phone_pos_o[1], phone_pos_o[2],
                             phone_rot_matrix_o[0, i], phone_rot_matrix_o[1, i], phone_rot_matrix_o[2, i],
                             length=0.4, color=colors[i], arrow_length_ratio=0.2, linewidth=1.5, alpha=0.6)
                    ax.quiver(watch_pos_o[0], watch_pos_o[1], watch_pos_o[2],
                             watch_rot_matrix_o[0, i], watch_rot_matrix_o[1, i], watch_rot_matrix_o[2, i],
                             length=0.4, color=colors[i], arrow_length_ratio=0.2, linewidth=1.5, alpha=0.6)
                
                # 绘制原始设备位置
                ax.scatter(*phone_pos_o, color='blue', s=100, marker='^', alpha=0.6, label='Phone (原始)')
                ax.scatter(*watch_pos_o, color='orange', s=100, marker='^', alpha=0.6, label='Watch (原始)')
                
                # 添加连接线显示对应关系
                ax.plot([phone_pos_t[0], phone_pos_o[0]], [phone_pos_t[1], phone_pos_o[1]], 
                       [phone_pos_t[2], phone_pos_o[2]], 'b--', alpha=0.3, linewidth=1)
                ax.plot([watch_pos_t[0], watch_pos_o[0]], [watch_pos_t[1], watch_pos_o[1]], 
                       [watch_pos_t[2], watch_pos_o[2]], 'orange', linestyle='--', alpha=0.3, linewidth=1)
        
        # 添加SMPL参考坐标系
        ref_pos = np.array([-1.0, 0, 0])
        ref_rot = np.eye(3)
        plot_coordinate_frame(ax, ref_pos, ref_rot, scale=0.4,
                             labels=['SMPL_X', 'SMPL_Y', 'SMPL_Z'])
        ax.scatter(*ref_pos, color='black', s=100, marker='s', label='SMPL参考')
        
        # 设置图形属性
        ax.set_xlim([-1.5, 2.5])
        ax.set_ylim([-0.8, 0.8])
        ax.set_zlim([-1.5, 0.8])
        ax.set_xlabel('X (Left/Right)')
        ax.set_ylabel('Y (Up/Down)')
        ax.set_zlabel('Z (Forward/Backward)')
        ax.set_title(f'IMU坐标系对比 (帧 {frame_idx}/{frame_count-1})\n上层=变换后, 下层=原始')
        ax.legend()
        
        # 显示数值
        info_text = f"变换后 - Phone: R={phone_roll_t:.1f}° P={phone_pitch_t:.1f}° Y={phone_yaw_t:.1f}°\n"
        info_text += f"         Watch: R={watch_roll_t:.1f}° P={watch_pitch_t:.1f}° Y={watch_yaw_t:.1f}°"
        
        if phone_ori_original is not None and watch_ori_original is not None:
            if (phone_original_idx < len(phone_ori_original) and 
                watch_original_idx < len(watch_ori_original)):
                info_text += f"\n原始   - Phone: R={phone_roll_o:.1f}° P={phone_pitch_o:.1f}° Y={phone_yaw_o:.1f}°\n"
                info_text += f"         Watch: R={watch_roll_o:.1f}° P={watch_pitch_o:.1f}° Y={watch_yaw_o:.1f}°"
        
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.draw()
    
    # 创建滑块
    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])
    slider = Slider(ax_slider, '帧数', 0, frame_count-1, valinit=0, valfmt='%d')
    slider.on_changed(update_frame)
    
    # 初始显示
    update_frame(0)
    plt.show()

def create_animation(phone_ori_transformed, watch_ori_transformed, data,
                    phone_ori_original=None, watch_ori_original=None):
    """
    创建动画显示坐标系变化
    """
    frame_count = len(phone_ori_transformed)
    
    # 创建图形
    fig = plt.figure(figsize=(15, 6))
    ax_transformed = fig.add_subplot(121, projection='3d')
    ax_original = fig.add_subplot(122, projection='3d')
    
    def animate(frame_idx):
        ax_transformed.clear()
        ax_original.clear()
        
        # 获取当前帧的变换后方向数据
        phone_roll_t, phone_pitch_t, phone_yaw_t = phone_ori_transformed[frame_idx]
        watch_roll_t, watch_pitch_t, watch_yaw_t = watch_ori_transformed[frame_idx]
        
        # 计算变换后的旋转矩阵
        phone_rot_matrix_t = euler_to_rotation_matrix(phone_roll_t, phone_pitch_t, phone_yaw_t)
        watch_rot_matrix_t = euler_to_rotation_matrix(watch_roll_t, watch_pitch_t, watch_yaw_t)
        
        # 设置IMU位置
        phone_pos = np.array([0, 0, 0])
        watch_pos = np.array([1.5, 0, 0])
        
        # 绘制变换后的坐标系
        plot_coordinate_frame(ax_transformed, phone_pos, phone_rot_matrix_t, scale=0.5)
        plot_coordinate_frame(ax_transformed, watch_pos, watch_rot_matrix_t, scale=0.5)
        
        # 绘制设备
        ax_transformed.scatter(*phone_pos, color='blue', s=150, label='Phone')
        ax_transformed.scatter(*watch_pos, color='orange', s=150, label='Watch')
        
        # 设置变换后图形属性
        ax_transformed.set_xlim([-1, 2.5])
        ax_transformed.set_ylim([-1, 1])
        ax_transformed.set_zlim([-1, 1])
        ax_transformed.set_xlabel('X')
        ax_transformed.set_ylabel('Y') 
        ax_transformed.set_zlabel('Z')
        ax_transformed.set_title(f'变换后坐标系 (帧 {frame_idx}/{frame_count-1})')
        ax_transformed.legend()
        
        # 绘制原始坐标系（如果有数据）
        if phone_ori_original is not None and watch_ori_original is not None:
            # 计算原始数据的对应帧索引
            metadata = data['metadata']
            phone_start = metadata['phone_valid_range'][0]
            watch_start = metadata['watch_valid_range'][0]
            alignment_shift = metadata['alignment_shift']
            
            phone_original_idx = phone_start + frame_idx
            if alignment_shift > 0:
                watch_original_idx = watch_start + frame_idx + alignment_shift
            else:
                phone_original_idx += abs(alignment_shift)
                watch_original_idx = watch_start + frame_idx
            
            if (phone_original_idx < len(phone_ori_original) and 
                watch_original_idx < len(watch_ori_original)):
                
                phone_roll_o, phone_pitch_o, phone_yaw_o = phone_ori_original[phone_original_idx]
                watch_roll_o, watch_pitch_o, watch_yaw_o = watch_ori_original[watch_original_idx]
                
                phone_rot_matrix_o = euler_to_rotation_matrix(phone_roll_o, phone_pitch_o, phone_yaw_o)
                watch_rot_matrix_o = euler_to_rotation_matrix(watch_roll_o, watch_pitch_o, watch_yaw_o)
                
                plot_coordinate_frame(ax_original, phone_pos, phone_rot_matrix_o, scale=0.5)
                plot_coordinate_frame(ax_original, watch_pos, watch_rot_matrix_o, scale=0.5)
                
                ax_original.scatter(*phone_pos, color='blue', s=150, label='Phone')
                ax_original.scatter(*watch_pos, color='orange', s=150, label='Watch')
        
        # 设置原始图形属性
        ax_original.set_xlim([-1, 2.5])
        ax_original.set_ylim([-1, 1])
        ax_original.set_zlim([-1, 1])
        ax_original.set_xlabel('X')
        ax_original.set_ylabel('Y') 
        ax_original.set_zlabel('Z')
        ax_original.set_title(f'原始坐标系 (帧 {frame_idx}/{frame_count-1})')
        ax_original.legend()
        
        return []
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=min(frame_count, 500), 
                                 interval=50, blit=False)
    
    # 保存动画（可选）
    # anim.save('coordinate_animation.mp4', writer='ffmpeg', fps=20)
    
    plt.show()

if __name__ == "__main__":
    data_file = "mobileposer/stag_raw_data/aligned_imu_data_normalized.pt"
    
    print("IMU坐标系可视化工具")
    print("1. 静态显示指定帧")
    print("2. 交互式显示（带滑块）")
    print("3. 动画显示")
    
    choice = input("请选择显示方式 (1/2/3): ").strip()
    
    if choice == '1':
        frame_idx = int(input("请输入要显示的帧数 (默认0): ") or "0")
        visualize_frame_coordinates(data_file, frame_idx, show_animation=False)
    elif choice == '2':
        create_interactive_visualization(data_file)
    elif choice == '3':
        print("创建动画中...")
        data = torch.load(data_file)
        phone_ori_transformed = torch.stack([
            data['imu_data']['rp']['ori_roll_deg'],
            data['imu_data']['rp']['ori_pitch_deg'], 
            data['imu_data']['rp']['ori_yaw_deg']
        ], dim=1).numpy()
        watch_ori_transformed = torch.stack([
            data['imu_data']['lw']['ori_roll_deg'],
            data['imu_data']['lw']['ori_pitch_deg'],
            data['imu_data']['lw']['ori_yaw_deg']
        ], dim=1).numpy()
        phone_ori_original, watch_ori_original = load_original_data()
        create_animation(phone_ori_transformed, watch_ori_transformed, data,
                        phone_ori_original, watch_ori_original)
    else:
        print("默认显示第0帧")
        visualize_frame_coordinates(data_file, 0, show_animation=False) 