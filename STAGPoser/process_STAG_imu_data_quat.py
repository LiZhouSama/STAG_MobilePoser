import pandas as pd
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'WenQuanYi Zen Hei'      # 使用文泉驿正黑
mpl.rcParams['axes.unicode_minus'] = False       # 解决负号 '-' 显示为方块的问题
from scipy.signal import find_peaks
import pytorch3d.transforms as transforms
import os
acc_scale = 1

def smooth_imu_acceleration(acc_data, smooth_n=4, frame_rate=100):
    """
    使用优化的张量运算对IMU加速度数据进行平滑
    
    参数:
        acc_data: 加速度数据 [T, 3] 或 [T, N, 3] 
        smooth_n: 平滑窗口大小
        frame_rate: 数据帧率
    
    返回:
        smoothed_acc: 平滑后的加速度数据，与输入相同形状
    """
    # 转换为torch tensor
    if isinstance(acc_data, np.ndarray):
        acc_tensor = torch.tensor(acc_data, dtype=torch.float32)
        return_numpy = True
    else:
        acc_tensor = acc_data
        return_numpy = False
    
    # 获取维度信息
    orig_shape = acc_tensor.shape
    if len(orig_shape) == 3:
        # 如果是 [T, N, 3]，展平为 [T, N*3]
        T, N, D = orig_shape
        acc_flat = acc_tensor.reshape(T, -1)
    else:
        # 如果是 [T, 3]
        acc_flat = acc_tensor
    
    T = acc_flat.shape[0]
    smoothed_acc = acc_flat.clone()
    
    # 应用平滑滤波
    mid = smooth_n // 2
    if mid != 0 and T > smooth_n * 2:
        # 对中间部分应用更强的平滑
        smooth_range = slice(smooth_n, -smooth_n)
        
        # 使用加权平均进行平滑
        for i in range(smooth_n, T - smooth_n):
            # 取前后smooth_n帧的加权平均
            weights = torch.ones(2 * smooth_n + 1, device=acc_flat.device)
            weights = weights / weights.sum()  # 归一化权重
            
            window_data = acc_flat[i - smooth_n:i + smooth_n + 1]
            smoothed_acc[i] = torch.sum(window_data * weights.unsqueeze(-1), dim=0)
    
    # 对边界进行简单平滑
    if T > 2:
        # 前边界
        smoothed_acc[0] = (2 * acc_flat[0] + acc_flat[1]) / 3
        # 后边界  
        smoothed_acc[-1] = (2 * acc_flat[-1] + acc_flat[-2]) / 3
    
    # 恢复原始形状
    smoothed_acc = smoothed_acc.reshape(orig_shape)
    
    if return_numpy:
        return smoothed_acc.numpy()
    else:
        return smoothed_acc

def detect_jump_sequences(lin_acc_data, magnitude_threshold=20.0, min_peak_distance=30, 
                         min_jump_interval=50, max_jump_interval=150, 
                         tpose_start=None, tpose_end=None):
    """
    检测连续的跳跃序列（开始3次，结束3次）
    基于加速度模长的峰值检测，确保结束跳跃在T-pose之后
    
    Args:
        lin_acc_data: 线性加速度数据 (x, y, z)
        magnitude_threshold: 加速度模长阈值（m/s²）
        min_peak_distance: 两个峰值之间的最小距离（帧数）
        min_jump_interval: 跳跃之间的最小间隔（帧数）
        max_jump_interval: 跳跃之间的最大间隔（帧数）
        tpose_start, tpose_end: T-pose段的开始和结束位置（如果已知）
    
    Returns:
        start_sequence, end_sequence: 开始和结束跳跃序列的帧索引
    """
    # 计算加速度模长
    acc_magnitude = np.sqrt(np.sum(lin_acc_data**2, axis=1))
    
    # 找到所有超过阈值的峰值
    peaks, properties = find_peaks(acc_magnitude, 
                                 height=magnitude_threshold,
                                 distance=min_peak_distance)
    
    if len(peaks) < 6:
        print(f"警告：只检测到 {len(peaks)} 个峰值，少于预期的6个")
        return np.array([]), np.array([])
    
    print(f"检测到 {len(peaks)} 个加速度峰值: {peaks}")
    print(f"峰值强度: {acc_magnitude[peaks]}")
    
    # 如果已知T-pose位置，则分别在T-pose前后寻找开始和结束跳跃
    if tpose_start is not None and tpose_end is not None and tpose_start >= 0:
        print(f"使用T-pose信息进行跳跃检测: T-pose在帧{tpose_start}-{tpose_end}")
        
        # 分离T-pose前后的峰值
        peaks_before_tpose = peaks[peaks < tpose_start]
        peaks_after_tpose = peaks[peaks > tpose_end]
        
        print(f"T-pose前的峰值: {len(peaks_before_tpose)} 个")
        print(f"T-pose后的峰值: {len(peaks_after_tpose)} 个")
        
        # 在T-pose前寻找开始跳跃序列
        start_sequence = find_jump_sequence_in_range(peaks_before_tpose, acc_magnitude, 
                                                   min_jump_interval, max_jump_interval, 
                                                   sequence_type="开始")
        
        # 在T-pose后寻找结束跳跃序列
        end_sequence = find_jump_sequence_in_range(peaks_after_tpose, acc_magnitude, 
                                                 min_jump_interval, max_jump_interval, 
                                                 sequence_type="结束")
        
    else:
        print("未提供T-pose信息，使用传统方法检测跳跃序列")
        # 寻找连续的跳跃序列（3次跳跃为一组）
        jump_sequences = []
        i = 0
        
        while i < len(peaks) - 2:  # 至少需要3个峰值
            # 检查是否能组成一个跳跃序列（连续3次跳跃）
            current_sequence = [peaks[i]]
            j = i + 1
            
            # 寻找后续的跳跃
            while j < len(peaks) and len(current_sequence) < 3:
                time_gap = peaks[j] - current_sequence[-1]
                
                # 检查间隔是否合理（不能太近也不能太远）
                if min_jump_interval <= time_gap <= max_jump_interval:
                    current_sequence.append(peaks[j])
                    j += 1
                elif time_gap < min_jump_interval:
                    # 太近了，跳过这个峰值
                    j += 1
                else:
                    # 太远了，停止当前序列
                    break
            
            # 如果找到了3次跳跃，保存序列
            if len(current_sequence) == 3:
                jump_sequences.append(current_sequence)
                print(f"找到跳跃序列: {current_sequence}, 间隔: {[current_sequence[k+1]-current_sequence[k] for k in range(2)]}")
                i = j  # 跳到下一个可能的序列起点
            else:
                i += 1  # 从下一个峰值开始尝试
        
        print(f"检测到 {len(jump_sequences)} 个跳跃序列:")
        for i, seq in enumerate(jump_sequences):
            duration = seq[-1] - seq[0]
            intervals = [seq[k+1] - seq[k] for k in range(len(seq)-1)]
            print(f"  序列{i+1}: 位置{seq}, 总时长{duration}帧, 间隔{intervals}")
        
        if len(jump_sequences) < 2:
            print("错误：未检测到足够的跳跃序列（需要至少2个序列：开始和结束）")
            return np.array([]), np.array([])
        
        # 选择第一个和最后一个序列作为开始和结束
        start_sequence = np.array(jump_sequences[0])
        end_sequence = np.array(jump_sequences[-1])
    
    print(f"最终检测结果:")
    print(f"  开始跳跃序列: {start_sequence}")
    print(f"  结束跳跃序列: {end_sequence}")
    
    return start_sequence, end_sequence


def find_jump_sequence_in_range(peaks_in_range, acc_magnitude, min_jump_interval, 
                               max_jump_interval, sequence_type=""):
    """
    在指定的峰值范围内寻找跳跃序列
    
    Args:
        peaks_in_range: 指定范围内的峰值
        acc_magnitude: 加速度模长数组
        min_jump_interval, max_jump_interval: 跳跃间隔限制
        sequence_type: 序列类型描述（用于日志）
    
    Returns:
        jump_sequence: 找到的跳跃序列，如果未找到则返回空数组
    """
    if len(peaks_in_range) < 3:
        print(f"警告：{sequence_type}跳跃范围内只有 {len(peaks_in_range)} 个峰值，少于3个")
        return np.array([])
    
    # 寻找连续的3次跳跃
    jump_sequences = []
    i = 0
    
    while i < len(peaks_in_range) - 2:
        current_sequence = [peaks_in_range[i]]
        j = i + 1
        
        # 寻找后续的跳跃
        while j < len(peaks_in_range) and len(current_sequence) < 3:
            time_gap = peaks_in_range[j] - current_sequence[-1]
            
            if min_jump_interval <= time_gap <= max_jump_interval:
                current_sequence.append(peaks_in_range[j])
                j += 1
            elif time_gap < min_jump_interval:
                j += 1
            else:
                break
        
        if len(current_sequence) == 3:
            jump_sequences.append(current_sequence)
            intervals = [current_sequence[k+1] - current_sequence[k] for k in range(2)]
            print(f"找到{sequence_type}跳跃序列: {current_sequence}, 间隔: {intervals}")
            i = j
        else:
            i += 1
    
    if not jump_sequences:
        print(f"警告：未找到有效的{sequence_type}跳跃序列")
        return np.array([])
    
    # 对于开始跳跃，选择第一个序列；对于结束跳跃，选择最后一个序列
    if sequence_type == "开始":
        selected_sequence = jump_sequences[0]
    else:  # 结束跳跃
        selected_sequence = jump_sequences[-1]
    
    print(f"选择{sequence_type}跳跃序列: {selected_sequence}")
    return np.array(selected_sequence)

def detect_tpose_segment(lin_acc_data, quat_data, start_jump_frames=None, fps=100, min_duration=4.0, 
                        acc_stability_threshold=0.5, ori_stability_threshold=0.1):
    """
    检测T-pose段：IMU数据变化范围小且持续至少4秒的时间段，且必须在开始跳跃之后
    
    Args:
        lin_acc_data: 线性加速度数据 [N, 3]
        quat_data: 四元数方向数据 [N, 4] (w, x, y, z)
        start_jump_frames: 开始跳跃序列的帧索引，T-pose必须在此之后
        fps: 数据帧率
        min_duration: T-pose的最小持续时间（秒）
        acc_stability_threshold: 加速度稳定性阈值（m/s²）
        ori_stability_threshold: 方向稳定性阈值（四元数角速度的标准差）
    
    Returns:
        tpose_start, tpose_end: T-pose段的开始和结束帧索引
    """
    min_frames = int(min_duration * fps)  # 最小帧数
    window_size = int(0.5 * fps)  # 0.5秒窗口用于计算稳定性
    
    # 确定搜索起始位置：如果有开始跳跃，则从最后一次开始跳跃后开始搜索
    search_start = 0
    if start_jump_frames is not None and len(start_jump_frames) > 0:
        search_start = int(start_jump_frames[-1]) + 100  # 最后一次开始跳跃后100帧开始搜索
        print(f"从开始跳跃后的帧 {search_start} 开始搜索T-pose")
    else:
        print("警告：未提供开始跳跃信息，从数据开始处搜索T-pose")
    
    search_start = max(0, min(search_start, len(lin_acc_data) - min_frames))
    
    # 计算加速度模长
    acc_magnitude = np.sqrt(np.sum(lin_acc_data**2, axis=1))
    
    # 计算四元数变化率（作为方向稳定性的指标）
    # 相邻帧之间的四元数差值作为角速度的近似
    quat_diff = np.diff(quat_data, axis=0)
    quat_change_magnitude = np.sqrt(np.sum(quat_diff**2, axis=1))
    # 在开头补充一个0，使长度与原数据一致
    quat_change_magnitude = np.concatenate([[quat_change_magnitude[0]], quat_change_magnitude])
    
    # 滑动窗口检测稳定段（从搜索起始位置开始）
    stable_segments = []
    
    for i in range(search_start, len(acc_magnitude) - window_size + 1):
        # 计算窗口内的标准差
        acc_std = np.std(acc_magnitude[i:i+window_size])
        quat_std = np.std(quat_change_magnitude[i:i+window_size])
        
        # 判断是否稳定
        if acc_std < acc_stability_threshold and quat_std < ori_stability_threshold:
            stable_segments.append(i)
    
    if not stable_segments:
        print("警告：未检测到稳定的T-pose段")
        return -1, -1
    
    # 寻找连续的稳定段
    tpose_segments = []
    current_segment = []
    
    for frame in stable_segments:
        if len(current_segment) == 0:
            current_segment = [frame, frame + window_size]
        elif frame <= current_segment[1] + 10:  # 允许小间隔
            current_segment[1] = frame + window_size
        else:
            # 检查当前段是否足够长
            if current_segment[1] - current_segment[0] >= min_frames:
                tpose_segments.append(current_segment)
            current_segment = [frame, frame + window_size]
    
    # 检查最后一段
    if len(current_segment) > 0 and current_segment[1] - current_segment[0] >= min_frames:
        tpose_segments.append(current_segment)
    
    print(f"检测到 {len(tpose_segments)} 个T-pose候选段:")
    for i, seg in enumerate(tpose_segments):
        duration = (seg[1] - seg[0]) / fps
        print(f"  段{i+1}: 帧{seg[0]}-{seg[1]} ({duration:.1f}s)")
    
    if not tpose_segments:
        print("错误：未检测到足够长的T-pose段")
        return -1, -1
    
    # 选择第一个（通常是开始跳跃后的T-pose）
    tpose_start, tpose_end = tpose_segments[0]
    print(f"选择T-pose段: 帧{tpose_start}-{tpose_end} ({(tpose_end-tpose_start)/fps:.1f}s)")
    
    return tpose_start, tpose_end

def apply_coordinate_transform(acc_data, quat_data, device_type='watch'):
    """
    应用坐标系变换，从IMU坐标系转换到SMPL坐标系
    
    SMPL坐标系: x=Left, y=Up, z=Forward
    IMU坐标系: x=Right, y=Forward, z=Up
    
    Args:
        acc_data: 加速度数据 [N, 3]
        quat_data: 四元数方向数据 [N, 4] (w, x, y, z)
        device_type: 'watch' 或 'phone'
    
    Returns:
        transformed_acc, transformed_ori: 变换后的数据 (加速度[N,3], 旋转矩阵[N,3,3])
    """
    # IMU坐标系到SMPL坐标系的变换矩阵
    # IMU: x=Right, y=Forward, z=Up
    # SMPL: x=Left, y=Up, z=Forward
    transform_matrix = torch.tensor([
        [-1,  0,  0],  # x_smpl = -x_imu (Right -> Left)
        [ 0,  0,  1],  # y_smpl = z_imu (Up -> Up)
        [ 0,  1,  0]   # z_smpl = y_imu (Forward -> Forward)
    ], dtype=torch.float32)
    
    # 转换加速度数据
    acc_tensor = torch.tensor(acc_data, dtype=torch.float32)
    transformed_acc = torch.matmul(acc_tensor, transform_matrix.T)
    
    # 转换方向数据 - 从四元数到旋转矩阵，然后应用坐标变换
    quat_tensor = torch.tensor(quat_data, dtype=torch.float32)
    
    # 将四元数转换为旋转矩阵（IMU坐标系）
    R_imu = transforms.quaternion_to_matrix(quat_tensor)  # [N, 3, 3]
    
    # 应用坐标系变换: R_smpl = T * R_imu * T^T
    N = R_imu.shape[0]
    transformed_ori = torch.zeros(N, 3, 3, dtype=torch.float32)
    
    for i in range(N):
        # 应用坐标系变换
        R_smpl = transform_matrix @ R_imu[i] @ transform_matrix.T
        transformed_ori[i] = R_smpl
    
    return transformed_acc.numpy(), transformed_ori.numpy()

def normalize_orientation_data(ori_data, tpose_start, tpose_end, start_idx):
    """
    对方向数据进行归一化，计算device2bone矩阵
    
    Args:
        ori_data: 方向数据 [N, 3, 3] (rotation matrices)
        tpose_start, tpose_end: T-pose段在原始数据中的位置
        start_idx: 有效数据在原始数据中的开始位置
    
    Returns:
        normalized_ori: 归一化后的方向数据 [M, 3, 3]
        device2bone: device到bone的变换矩阵 [3, 3]
    """
    if tpose_start == -1 or tpose_end == -1:
        print("警告：无效的T-pose段，跳过方向归一化")
        # 返回原始数据和单位矩阵
        return ori_data[start_idx:], np.eye(3)
    
    # 计算T-pose时的参考旋转矩阵（取T-pose段的平均）
    tpose_ref_start = max(0, tpose_start + 10)  # T-pose开始后10帧
    tpose_ref_end = min(tpose_end - 10, len(ori_data))  # T-pose结束前10帧
    
    # 计算参考旋转矩阵（T-pose时的平均旋转）
    ref_rot_matrix = ori_data[tpose_ref_start:tpose_ref_end].mean(axis=0)  # [3, 3]
    
    # 计算device2bone矩阵（MobilePoser的关键）
    # device2bone = ref_rot^T，使得T-pose时的旋转变为单位矩阵
    device2bone = ref_rot_matrix.T
    
    print(f"T-pose参考旋转矩阵计算完成")
    print(f"Device2Bone矩阵形状: {device2bone.shape}")
    
    # 提取有效数据段
    valid_ori_data = ori_data[start_idx:]
    normalized_ori = np.zeros_like(valid_ori_data)
    
    # 对每一帧进行归一化
    for i, ori_frame in enumerate(valid_ori_data):
        # 归一化：当前矩阵 * device2bone
        # 这样T-pose时的旋转会变成单位矩阵
        normalized_ori[i] = ori_frame @ device2bone
    
    # 验证归一化效果：检查T-pose开始处的值
    tpose_rel_start = max(0, tpose_start - start_idx)
    if tpose_rel_start < len(normalized_ori):
        tpose_normalized = normalized_ori[tpose_rel_start]
        # 检查是否接近单位矩阵
        identity_error = np.linalg.norm(tpose_normalized - np.eye(3))
        print(f"T-pose开始处归一化后与单位矩阵的误差: {identity_error:.6f}")
    
    return normalized_ori, device2bone


def extract_valid_data_range_with_tpose(start_sequence, end_sequence, tpose_start, tpose_end, 
                                       total_frames, buffer_frames=100):
    """
    根据跳跃序列和T-pose段提取有效数据范围
    
    Args:
        start_sequence: 开始跳跃序列
        end_sequence: 结束跳跃序列
        tpose_start, tpose_end: T-pose段的开始和结束
        total_frames: 总帧数
        buffer_frames: 缓冲帧数
    
    Returns:
        start_idx, end_idx: 有效数据的开始和结束索引
        tpose_relative_start, tpose_relative_end: T-pose在有效数据中的相对位置
    """
    if len(start_sequence) == 0 or len(end_sequence) == 0 or tpose_start == -1:
        print("警告：检测失败，使用整个数据范围")
        return 0, total_frames, -1, -1
    
    # 有效数据从T-pose结束前3秒开始，到结束跳跃前结束
    start_idx = max(0, tpose_end - 300)  # T-pose结束前3秒(300帧)
    end_idx = end_sequence[0] - buffer_frames
    
    # 确保索引有效
    start_idx = max(0, min(start_idx, total_frames))
    end_idx = max(start_idx, min(end_idx, total_frames))
    
    if start_idx >= end_idx:
        print("警告：有效数据范围无效，调整参数")
        start_idx = max(0, tpose_end - 100)
        end_idx = end_sequence[0] - 50
        start_idx = max(0, min(start_idx, total_frames))
        end_idx = max(start_idx, min(end_idx, total_frames))
    
    # 计算T-pose在有效数据中的相对位置
    tpose_relative_start = max(0, tpose_start - start_idx)
    tpose_relative_end = min(end_idx - start_idx, tpose_end - start_idx)
    
    return start_idx, end_idx, tpose_relative_start, tpose_relative_end

def align_imu_data(data1, data2, max_shift=100):
    """
    通过交叉相关对齐两个IMU数据
    """
    # 使用加速度模长进行对齐
    mag1 = np.sqrt(np.sum(data1**2, axis=1))
    mag2 = np.sqrt(np.sum(data2**2, axis=1))
    
    # 计算交叉相关
    min_len = min(len(mag1), len(mag2))
    mag1 = mag1[:min_len]
    mag2 = mag2[:min_len]
    
    best_corr = -1
    best_shift = 0
    
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            if shift >= min_len:
                continue
            corr = np.corrcoef(mag1[:-shift if shift > 0 else None], 
                              mag2[shift:])[0, 1]
        else:
            if abs(shift) >= min_len:
                continue
            corr = np.corrcoef(mag1[abs(shift):], 
                              mag2[:shift if shift < 0 else None])[0, 1]
        
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_shift = shift
    
    print(f"最佳对齐偏移量: {best_shift}, 相关系数: {best_corr:.4f}")
    return best_shift

def process_imu_files(phone_file, watch_file, output_file):
    """
    处理两个IMU文件并生成对齐的数据，包含T-pose检测和坐标系变换
    """
    # 读取数据
    print("读取数据文件...")
    phone_data = pd.read_csv(phone_file)
    watch_data = pd.read_csv(watch_file)
    
    print(f"手机数据: {len(phone_data)} 帧")
    print(f"手表数据: {len(watch_data)} 帧")
    
    # 提取线性加速度数据
    phone_lin_acc = phone_data[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
    watch_lin_acc = watch_data[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
    
    # 提取四元数方向数据
    phone_quat = phone_data[['quat_w', 'quat_x', 'quat_y', 'quat_z']].values
    watch_quat = watch_data[['quat_w', 'quat_x', 'quat_y', 'quat_z']].values
    
    # 第一阶段：初步检测开始跳跃序列（无T-pose信息）
    print("\n=== 第一阶段：初步跳跃检测 ===")
    print("--- 手机IMU初步跳跃检测 ---")
    phone_start_seq_temp, phone_end_seq_temp = detect_jump_sequences(phone_lin_acc, magnitude_threshold=5.0)
    
    print("\n--- 手表IMU初步跳跃检测 ---")  
    watch_start_seq_temp, watch_end_seq_temp = detect_jump_sequences(watch_lin_acc, magnitude_threshold=10.0)
    
    # 第二阶段：基于开始跳跃检测T-pose段
    print("\n=== 第二阶段：T-pose检测 ===")
    print("手机T-pose检测:")
    phone_tpose_start, phone_tpose_end = detect_tpose_segment(phone_lin_acc, phone_quat, phone_start_seq_temp, acc_stability_threshold=0.4, ori_stability_threshold=0.05)
    
    print("手表T-pose检测:")
    watch_tpose_start, watch_tpose_end = detect_tpose_segment(watch_lin_acc, watch_quat, watch_start_seq_temp, acc_stability_threshold=0.5, ori_stability_threshold=0.08)
    
    # 第三阶段：基于T-pose信息重新精确检测跳跃序列
    print("\n=== 第三阶段：基于T-pose重新检测跳跃序列 ===")
    print("--- 手机IMU精确跳跃检测 ---")
    phone_start_seq, phone_end_seq = detect_jump_sequences(phone_lin_acc, magnitude_threshold=5.0, 
                                                          tpose_start=phone_tpose_start, tpose_end=phone_tpose_end)
    
    print("\n--- 手表IMU精确跳跃检测 ---")  
    watch_start_seq, watch_end_seq = detect_jump_sequences(watch_lin_acc, magnitude_threshold=10.0,
                                                          tpose_start=watch_tpose_start, tpose_end=watch_tpose_end)
    
    # 提取有效数据范围
    phone_start, phone_end, phone_tpose_rel_start, phone_tpose_rel_end = extract_valid_data_range_with_tpose(
        phone_start_seq, phone_end_seq, phone_tpose_start, phone_tpose_end, len(phone_data))
    
    watch_start, watch_end, watch_tpose_rel_start, watch_tpose_rel_end = extract_valid_data_range_with_tpose(
        watch_start_seq, watch_end_seq, watch_tpose_start, watch_tpose_end, len(watch_data))
    
    print(f"\n手机有效数据范围: {phone_start} - {phone_end} (共{phone_end-phone_start}帧)")
    print(f"手表有效数据范围: {watch_start} - {watch_end} (共{watch_end-watch_start}帧)")
    
    # 提取有效数据段
    phone_valid = phone_data.iloc[phone_start:phone_end].copy().reset_index(drop=True)
    watch_valid = watch_data.iloc[watch_start:watch_end].copy().reset_index(drop=True)
    
    # 对齐数据
    print("\n对齐IMU数据...")
    phone_lin_acc_valid = phone_valid[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
    watch_lin_acc_valid = watch_valid[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
    
    shift = align_imu_data(phone_lin_acc_valid, watch_lin_acc_valid)
    
    # 根据偏移量调整数据
    if shift > 0:
        watch_aligned = watch_valid.iloc[shift:].copy()
        phone_aligned = phone_valid.iloc[:len(watch_aligned)].copy()
        # 调整T-pose相对位置
        watch_tpose_rel_start = max(0, watch_tpose_rel_start - shift)
        watch_tpose_rel_end = max(0, watch_tpose_rel_end - shift)
    elif shift < 0:
        phone_aligned = phone_valid.iloc[abs(shift):].copy()
        watch_aligned = watch_valid.iloc[:len(phone_aligned)].copy()
        # 调整T-pose相对位置
        phone_tpose_rel_start = max(0, phone_tpose_rel_start - abs(shift))
        phone_tpose_rel_end = max(0, phone_tpose_rel_end - abs(shift))
    else:
        min_len = min(len(phone_valid), len(watch_valid))
        phone_aligned = phone_valid.iloc[:min_len].copy()
        watch_aligned = watch_valid.iloc[:min_len].copy()
    
    # 确保长度一致
    aligned_frames = min(len(phone_aligned), len(watch_aligned))
    phone_aligned = phone_aligned.iloc[:aligned_frames].copy().reset_index(drop=True)
    watch_aligned = watch_aligned.iloc[:aligned_frames].copy().reset_index(drop=True)
    
    print(f"对齐后数据长度: {aligned_frames} 帧")
    
    # 提取特征数据并应用坐标系变换
    print("\n应用坐标系变换...")
    
    # 手机数据变换
    phone_acc_raw = phone_aligned[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values / acc_scale
    phone_quat_raw = phone_aligned[['quat_w', 'quat_x', 'quat_y', 'quat_z']].values
    phone_acc_transformed, phone_ori_transformed = apply_coordinate_transform(
        phone_acc_raw, phone_quat_raw, device_type='phone')
    
    # 手表数据变换
    watch_acc_raw = watch_aligned[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values / acc_scale
    watch_quat_raw = watch_aligned[['quat_w', 'quat_x', 'quat_y', 'quat_z']].values
    watch_acc_transformed, watch_ori_transformed = apply_coordinate_transform(
        watch_acc_raw, watch_quat_raw, device_type='watch')
    
    print("坐标系变换完成")
    
    # 应用加速度平滑
    print("应用加速度平滑...")
    phone_acc_smoothed = smooth_imu_acceleration(phone_acc_transformed, smooth_n=4, frame_rate=100)
    watch_acc_smoothed = smooth_imu_acceleration(watch_acc_transformed, smooth_n=4, frame_rate=100)
    print("加速度平滑完成")
    
    # 对方向数据进行归一化（使T-pose开始时方向为0）
    print("\n应用方向归一化...")
    
    # 计算归一化需要的参数（基于对齐前的原始索引）
    phone_tpose_original = (phone_tpose_start, phone_tpose_end)
    watch_tpose_original = (watch_tpose_start, watch_tpose_end)
    
    # 对手机方向数据归一化
    if phone_tpose_start >= 0:
        # 调整T-pose位置到对齐后的数据中
        phone_tpose_aligned = (max(0, phone_tpose_start - phone_start), 
                              min(aligned_frames, phone_tpose_end - phone_start))
        if shift > 0:  # 手表延迟，手机数据不变
            phone_tpose_in_aligned = phone_tpose_aligned
        elif shift < 0:  # 手机延迟，需要调整
            phone_tpose_in_aligned = (max(0, phone_tpose_aligned[0] - abs(shift)),
                                    max(0, phone_tpose_aligned[1] - abs(shift)))
        else:
            phone_tpose_in_aligned = phone_tpose_aligned
        
        if phone_tpose_in_aligned[0] >= 0 and phone_tpose_in_aligned[1] > phone_tpose_in_aligned[0]:
            phone_ori_normalized, phone_device2bone = normalize_orientation_data(
                phone_ori_transformed, phone_tpose_in_aligned[0], phone_tpose_in_aligned[1], 0)
            print(f"手机方向数据已归一化，T-pose位置: {phone_tpose_in_aligned}")
        else:
            phone_ori_normalized = phone_ori_transformed
            phone_device2bone = np.eye(3)
            print("手机T-pose位置无效，跳过归一化")
    else:
        phone_ori_normalized = phone_ori_transformed
        phone_device2bone = np.eye(3)
        print("未检测到手机T-pose，跳过归一化")
    
    # 对手表方向数据归一化
    if watch_tpose_start >= 0:
        # 调整T-pose位置到对齐后的数据中
        watch_tpose_aligned = (max(0, watch_tpose_start - watch_start),
                              min(aligned_frames, watch_tpose_end - watch_start))
        if shift > 0:  # 手表延迟，需要调整
            watch_tpose_in_aligned = (max(0, watch_tpose_aligned[0] - shift),
                                    max(0, watch_tpose_aligned[1] - shift))
        elif shift < 0:  # 手机延迟，手表数据不变
            watch_tpose_in_aligned = watch_tpose_aligned
        else:
            watch_tpose_in_aligned = watch_tpose_aligned
        
        if watch_tpose_in_aligned[0] >= 0 and watch_tpose_in_aligned[1] > watch_tpose_in_aligned[0]:
            watch_ori_normalized, watch_device2bone = normalize_orientation_data(
                watch_ori_transformed, watch_tpose_in_aligned[0], watch_tpose_in_aligned[1], 0)
            print(f"手表方向数据已归一化，T-pose位置: {watch_tpose_in_aligned}")
        else:
            watch_ori_normalized = watch_ori_transformed
            watch_device2bone = np.eye(3)
            print("手表T-pose位置无效，跳过归一化")
    else:
        watch_ori_normalized = watch_ori_transformed
        watch_device2bone = np.eye(3)
        print("未检测到手表T-pose，跳过归一化")
    
    print("方向归一化完成")
    
    # 创建输出数据结构
    processed_data = {
        'frame_count': aligned_frames,
        'imu_data': {
            'rp': {  # 右口袋(手机) - 变换和归一化后的数据
                'lin_acc_x': torch.tensor(phone_acc_smoothed[:, 0], dtype=torch.float32),
                'lin_acc_y': torch.tensor(phone_acc_smoothed[:, 1], dtype=torch.float32),
                'lin_acc_z': torch.tensor(phone_acc_smoothed[:, 2], dtype=torch.float32),
                'ori': torch.tensor(phone_ori_normalized, dtype=torch.float32)  # [N, 3, 3] 旋转矩阵
            },
            'lw': {  # 左手腕(手表) - 变换和归一化后的数据
                'lin_acc_x': torch.tensor(watch_acc_smoothed[:, 0], dtype=torch.float32),
                'lin_acc_y': torch.tensor(watch_acc_smoothed[:, 1], dtype=torch.float32),
                'lin_acc_z': torch.tensor(watch_acc_smoothed[:, 2], dtype=torch.float32),
                'ori': torch.tensor(watch_ori_normalized, dtype=torch.float32)  # [N, 3, 3] 旋转矩阵
            }
        },
        'calibration': {
            'rp': {
                'device2bone': torch.tensor(phone_device2bone, dtype=torch.float32)
            },
            'lw': {
                'device2bone': torch.tensor(watch_device2bone, dtype=torch.float32)
            }
        },
        'imu_positions': ['rp', 'lw'],
        'tpose_info': {
            'phone_tpose_range': (phone_tpose_rel_start, phone_tpose_rel_end),
            'watch_tpose_range': (watch_tpose_rel_start, watch_tpose_rel_end),
            'has_valid_tpose': phone_tpose_rel_start >= 0 and watch_tpose_rel_start >= 0
        },
        'metadata': {
            'original_phone_frames': len(phone_data),
            'original_watch_frames': len(watch_data),
            'phone_valid_range': (phone_start, phone_end),
            'watch_valid_range': (watch_start, watch_end),
            'alignment_shift': shift,
            'phone_start_jumps': phone_start_seq.tolist() if len(phone_start_seq) > 0 else [],
            'phone_end_jumps': phone_end_seq.tolist() if len(phone_end_seq) > 0 else [],
            'watch_start_jumps': watch_start_seq.tolist() if len(watch_start_seq) > 0 else [],
            'watch_end_jumps': watch_end_seq.tolist() if len(watch_end_seq) > 0 else [],
            'phone_tpose_original': (phone_tpose_start, phone_tpose_end),
            'watch_tpose_original': (watch_tpose_start, watch_tpose_end),
            'coordinate_transformed': True,
            'orientation_normalized': True,
            'rotation_format': 'matrix'  # 标记使用旋转矩阵格式
        }
    }
    
    # 保存数据
    print(f"\n保存处理后的数据到 {output_file}")
    torch.save(processed_data, output_file)
    
    # 简单的可视化
    create_enhanced_visualization(phone_data, watch_data, processed_data, 
                                phone_acc_smoothed, watch_acc_smoothed,
                                phone_ori_normalized, watch_ori_normalized)
    
    # 新增：坐标变换和归一化对比可视化
    create_transformation_comparison_visualization(
        # 原始数据
        phone_acc_raw, watch_acc_raw,
        phone_quat_raw, watch_quat_raw,
        # 坐标变换后数据
        phone_acc_transformed, watch_acc_transformed,
        phone_ori_transformed, watch_ori_transformed,
        # 归一化后数据
        phone_ori_normalized, watch_ori_normalized,
        # 设备到骨骼的变换矩阵
        phone_device2bone, watch_device2bone
    )
    
    return processed_data

def create_enhanced_visualization(phone_data, watch_data, processed_data, 
                                phone_acc_smoothed, watch_acc_smoothed,
                                phone_ori_normalized, watch_ori_normalized):
    """
    创建增强的可视化图，包含T-pose检测、坐标变换、加速度平滑和方向归一化对比
    """
    try:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 计算原始数据的加速度模长
        phone_lin_acc = phone_data[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
        watch_lin_acc = watch_data[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
        
        phone_mag = np.sqrt(np.sum(phone_lin_acc**2, axis=1))
        watch_mag = np.sqrt(np.sum(watch_lin_acc**2, axis=1))
        
        # 获取检测信息
        phone_start_jumps = processed_data['metadata']['phone_start_jumps']
        phone_end_jumps = processed_data['metadata']['phone_end_jumps']
        watch_start_jumps = processed_data['metadata']['watch_start_jumps']  
        watch_end_jumps = processed_data['metadata']['watch_end_jumps']
        
        phone_range = processed_data['metadata']['phone_valid_range']
        watch_range = processed_data['metadata']['watch_valid_range']
        
        phone_tpose_orig = processed_data['metadata']['phone_tpose_original']
        watch_tpose_orig = processed_data['metadata']['watch_tpose_original']
        
        # 第一行：跳跃和T-pose检测
        axes[0, 0].plot(phone_mag, alpha=0.7, color='blue', linewidth=0.8)
        if phone_start_jumps:
            axes[0, 0].scatter(phone_start_jumps, phone_mag[phone_start_jumps], 
                              color='green', s=60, marker='v', label='开始跳跃', zorder=5)
        if phone_end_jumps:
            axes[0, 0].scatter(phone_end_jumps, phone_mag[phone_end_jumps], 
                              color='red', s=60, marker='^', label='结束跳跃', zorder=5)
        if phone_tpose_orig[0] >= 0:
            axes[0, 0].axvspan(phone_tpose_orig[0], phone_tpose_orig[1], alpha=0.3, color='yellow', label='T-pose')
        axes[0, 0].axvline(phone_range[0], color='orange', linestyle='--', alpha=0.8, label='有效范围')
        axes[0, 0].axvline(phone_range[1], color='orange', linestyle='--', alpha=0.8)
        axes[0, 0].set_title('Phone IMU - Jump & T-pose Detection')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(watch_mag, alpha=0.7, color='orange', linewidth=0.8)
        if watch_start_jumps:
            axes[0, 1].scatter(watch_start_jumps, watch_mag[watch_start_jumps], 
                              color='green', s=60, marker='v', label='开始跳跃', zorder=5)
        if watch_end_jumps:
            axes[0, 1].scatter(watch_end_jumps, watch_mag[watch_end_jumps], 
                              color='red', s=60, marker='^', label='结束跳跃', zorder=5)
        if watch_tpose_orig[0] >= 0:
            axes[0, 1].axvspan(watch_tpose_orig[0], watch_tpose_orig[1], alpha=0.3, color='yellow', label='T-pose')
        axes[0, 1].axvline(watch_range[0], color='blue', linestyle='--', alpha=0.8, label='有效范围')
        axes[0, 1].axvline(watch_range[1], color='blue', linestyle='--', alpha=0.8)
        axes[0, 1].set_title('Watch IMU - Jump & T-pose Detection')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 第二行：原始数据对比
        rp_acc_mag_orig = torch.sqrt(
            processed_data['imu_data']['rp']['lin_acc_x']**2 + 
            processed_data['imu_data']['rp']['lin_acc_y']**2 + 
            processed_data['imu_data']['rp']['lin_acc_z']**2
        ).numpy()
        lw_acc_mag_orig = torch.sqrt(
            processed_data['imu_data']['lw']['lin_acc_x']**2 + 
            processed_data['imu_data']['lw']['lin_acc_y']**2 + 
            processed_data['imu_data']['lw']['lin_acc_z']**2
        ).numpy()
        
        frames = np.arange(len(rp_acc_mag_orig))
        
        axes[1, 0].plot(frames, rp_acc_mag_orig, label='右口袋 (手机) - 平滑后', alpha=0.8)
        axes[1, 0].plot(frames, lw_acc_mag_orig, label='左手腕 (手表) - 平滑后', alpha=0.8) 
        axes[1, 0].set_title('Smoothed Acceleration Magnitude (SMPL Coordinate)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Z轴加速度对比
        rp_acc_z = processed_data['imu_data']['rp']['lin_acc_z'].numpy()
        lw_acc_z = processed_data['imu_data']['lw']['lin_acc_z'].numpy()
        
        axes[1, 1].plot(frames, rp_acc_z, label='右口袋 (手机)', alpha=0.8)
        axes[1, 1].plot(frames, lw_acc_z, label='左手腕 (手表)', alpha=0.8)
        axes[1, 1].set_title('Z-axis Acceleration (SMPL Coordinate)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 第三行：方向归一化效果展示
        rp_ori_normalized = processed_data['imu_data']['rp']['ori'].numpy()
        lw_ori_normalized = processed_data['imu_data']['lw']['ori'].numpy()
        
        # 添加旋转矩阵到欧拉角的转换函数
        def rotation_matrix_to_euler_degrees(rotation_matrices):
            """将旋转矩阵转换为欧拉角（度数）用于可视化"""
            n_frames = rotation_matrices.shape[0]
            euler_angles = np.zeros((n_frames, 3))
            
            for i in range(n_frames):
                R = rotation_matrices[i]
                
                # 提取欧拉角 (ZYX顺序)
                sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                singular = sy < 1e-6
                
                if not singular:
                    x = np.arctan2(R[2, 1], R[2, 2])
                    y = np.arctan2(-R[2, 0], sy)
                    z = np.arctan2(R[1, 0], R[0, 0])
                else:
                    x = np.arctan2(-R[1, 2], R[1, 1])
                    y = np.arctan2(-R[2, 0], sy)
                    z = 0
                
                euler_angles[i] = [np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)]
            
            return euler_angles
        
        # 转换为欧拉角
        rp_euler_normalized = rotation_matrix_to_euler_degrees(rp_ori_normalized)
        lw_euler_normalized = rotation_matrix_to_euler_degrees(lw_ori_normalized)
        
        axes[2, 0].plot(frames, rp_euler_normalized[:, 0], label='手机 Roll (归一化)', alpha=0.8)
        axes[2, 0].plot(frames, rp_euler_normalized[:, 1], label='手机 Pitch (归一化)', alpha=0.8)
        axes[2, 0].plot(frames, rp_euler_normalized[:, 2], label='手机 Yaw (归一化)', alpha=0.8)
        
        # 标记T-pose段
        if processed_data['tpose_info']['has_valid_tpose']:
            phone_tpose_range = processed_data['tpose_info']['phone_tpose_range']
            if phone_tpose_range[1] > phone_tpose_range[0]:
                axes[2, 0].axvspan(phone_tpose_range[0], phone_tpose_range[1], 
                                  alpha=0.3, color='yellow', label='T-pose段')
        
        axes[2, 0].set_title('手机方向角度 (归一化后)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_ylabel('角度 (度)')
        
        axes[2, 1].plot(frames, lw_euler_normalized[:, 0], label='手表 Roll (归一化)', alpha=0.8)
        axes[2, 1].plot(frames, lw_euler_normalized[:, 1], label='手表 Pitch (归一化)', alpha=0.8)
        axes[2, 1].plot(frames, lw_euler_normalized[:, 2], label='手表 Yaw (归一化)', alpha=0.8)
        
        # 标记T-pose段
        if processed_data['tpose_info']['has_valid_tpose']:
            watch_tpose_range = processed_data['tpose_info']['watch_tpose_range']
            if watch_tpose_range[1] > watch_tpose_range[0]:
                axes[2, 1].axvspan(watch_tpose_range[0], watch_tpose_range[1], 
                                  alpha=0.3, color='yellow', label='T-pose段')
        
        axes[2, 1].set_title('手表方向角度 (归一化后)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylabel('角度 (度)')
        
        plt.tight_layout()
        plt.savefig('STAGPoser/STAG_data/enhanced_detection_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("增强检测结果图已保存为 enhanced_detection_results.png")
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")

def create_transformation_comparison_visualization(
    phone_acc_raw, watch_acc_raw, phone_quat_raw, watch_quat_raw,
    phone_acc_transformed, watch_acc_transformed, phone_ori_transformed, watch_ori_transformed,
    phone_ori_normalized, watch_ori_normalized, phone_device2bone, watch_device2bone):
    """
    创建坐标变换和方向归一化的前后对比可视化
    
    Args:
        phone_acc_raw, watch_acc_raw: 原始加速度数据
        phone_quat_raw, watch_quat_raw: 原始四元数数据
        *_transformed: 坐标变换后的数据
        *_normalized: 方向归一化后的数据
        *_device2bone: 设备到骨骼的变换矩阵
    """
    try:
        # 创建一个大图，包含多个子图
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        
        # 计算时间轴（假设100fps）
        frames = np.arange(len(phone_acc_raw))
        time_seconds = frames / 100.0
        
        # 将原始四元数转换为欧拉角用于可视化
        phone_ori_raw_matrices = transforms.quaternion_to_matrix(phone_quat_raw).numpy()
        watch_ori_raw_matrices = transforms.quaternion_to_matrix(watch_quat_raw).numpy()
        
        phone_ori_raw = transforms.matrix_to_euler_angles(phone_ori_raw_matrices, convention='XYZ')
        watch_ori_raw = transforms.matrix_to_euler_angles(watch_ori_raw_matrices, convention='XYZ')
        
        # === 第一行：手机加速度坐标变换前后对比 ===
        axes[0, 0].plot(time_seconds, phone_acc_raw[:, 0], 'r-', alpha=0.7, label='X轴 (原始)')
        axes[0, 0].plot(time_seconds, phone_acc_raw[:, 1], 'g-', alpha=0.7, label='Y轴 (原始)')
        axes[0, 0].plot(time_seconds, phone_acc_raw[:, 2], 'b-', alpha=0.7, label='Z轴 (原始)')
        axes[0, 0].set_title('手机加速度 - 坐标变换前 (IMU坐标系)', fontsize=12)
        axes[0, 0].set_xlabel('时间 (秒)')
        axes[0, 0].set_ylabel('加速度 (m/s²)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(time_seconds, phone_acc_transformed[:, 0], 'r--', alpha=0.8, label='X轴 (变换后)')
        axes[0, 1].plot(time_seconds, phone_acc_transformed[:, 1], 'g--', alpha=0.8, label='Y轴 (变换后)')
        axes[0, 1].plot(time_seconds, phone_acc_transformed[:, 2], 'b--', alpha=0.8, label='Z轴 (变换后)')
        axes[0, 1].set_title('手机加速度 - 坐标变换后 (SMPL坐标系)', fontsize=12)
        axes[0, 1].set_xlabel('时间 (秒)')
        axes[0, 1].set_ylabel('加速度 (m/s²)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # === 第二行：手表加速度坐标变换前后对比 ===
        axes[1, 0].plot(time_seconds, watch_acc_raw[:, 0], 'r-', alpha=0.7, label='X轴 (原始)')
        axes[1, 0].plot(time_seconds, watch_acc_raw[:, 1], 'g-', alpha=0.7, label='Y轴 (原始)')
        axes[1, 0].plot(time_seconds, watch_acc_raw[:, 2], 'b-', alpha=0.7, label='Z轴 (原始)')
        axes[1, 0].set_title('手表加速度 - 坐标变换前 (IMU坐标系)', fontsize=12)
        axes[1, 0].set_xlabel('时间 (秒)')
        axes[1, 0].set_ylabel('加速度 (m/s²)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(time_seconds, watch_acc_transformed[:, 0], 'r--', alpha=0.8, label='X轴 (变换后)')
        axes[1, 1].plot(time_seconds, watch_acc_transformed[:, 1], 'g--', alpha=0.8, label='Y轴 (变换后)')
        axes[1, 1].plot(time_seconds, watch_acc_transformed[:, 2], 'b--', alpha=0.8, label='Z轴 (变换后)')
        axes[1, 1].set_title('手表加速度 - 坐标变换后 (SMPL坐标系)', fontsize=12)
        axes[1, 1].set_xlabel('时间 (秒)')
        axes[1, 1].set_ylabel('加速度 (m/s²)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # === 第三行：手机方向归一化前后对比 ===
        # 从旋转矩阵提取欧拉角用于可视化
        phone_euler_transformed = transforms.matrix_to_euler_angles(phone_ori_transformed, convention='XYZ')
        phone_euler_normalized = transforms.matrix_to_euler_angles(phone_ori_normalized, convention='XYZ')
        
        axes[2, 0].plot(time_seconds, phone_euler_transformed[:, 0], 'r-', alpha=0.7, label='Roll (变换后)')
        axes[2, 0].plot(time_seconds, phone_euler_transformed[:, 1], 'g-', alpha=0.7, label='Pitch (变换后)')
        axes[2, 0].plot(time_seconds, phone_euler_transformed[:, 2], 'b-', alpha=0.7, label='Yaw (变换后)')
        axes[2, 0].set_title('手机方向 - 坐标变换后 (SMPL坐标系)', fontsize=12)
        axes[2, 0].set_xlabel('时间 (秒)')
        axes[2, 0].set_ylabel('角度 (度)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(time_seconds, phone_euler_normalized[:, 0], 'r--', alpha=0.8, label='Roll (归一化)')
        axes[2, 1].plot(time_seconds, phone_euler_normalized[:, 1], 'g--', alpha=0.8, label='Pitch (归一化)')
        axes[2, 1].plot(time_seconds, phone_euler_normalized[:, 2], 'b--', alpha=0.8, label='Yaw (归一化)')
        axes[2, 1].set_title('手机方向 - 归一化后 (T-pose=0)', fontsize=12)
        axes[2, 1].set_xlabel('时间 (秒)')
        axes[2, 1].set_ylabel('角度 (度)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # === 第四行：手表方向归一化前后对比 ===
        watch_euler_transformed = transforms.matrix_to_euler_angles(watch_ori_transformed, convention='XYZ')
        watch_euler_normalized = transforms.matrix_to_euler_angles(watch_ori_normalized, convention='XYZ')
        
        axes[3, 0].plot(time_seconds, watch_euler_transformed[:, 0], 'r-', alpha=0.7, label='Roll (变换后)')
        axes[3, 0].plot(time_seconds, watch_euler_transformed[:, 1], 'g-', alpha=0.7, label='Pitch (变换后)')
        axes[3, 0].plot(time_seconds, watch_euler_transformed[:, 2], 'b-', alpha=0.7, label='Yaw (变换后)')
        axes[3, 0].set_title('手表方向 - 坐标变换后 (SMPL坐标系)', fontsize=12)
        axes[3, 0].set_xlabel('时间 (秒)')
        axes[3, 0].set_ylabel('角度 (度)')
        axes[3, 0].legend()
        axes[3, 0].grid(True, alpha=0.3)
        
        axes[3, 1].plot(time_seconds, watch_euler_normalized[:, 0], 'r--', alpha=0.8, label='Roll (归一化)')
        axes[3, 1].plot(time_seconds, watch_euler_normalized[:, 1], 'g--', alpha=0.8, label='Pitch (归一化)')
        axes[3, 1].plot(time_seconds, watch_euler_normalized[:, 2], 'b--', alpha=0.8, label='Yaw (归一化)')
        axes[3, 1].set_title('手表方向 - 归一化后 (T-pose=0)', fontsize=12)
        axes[3, 1].set_xlabel('时间 (秒)')
        axes[3, 1].set_ylabel('角度 (度)')
        axes[3, 1].legend()
        axes[3, 1].grid(True, alpha=0.3)
        
        # 调整布局
        plt.tight_layout(pad=3.0)
        
        # 添加整体标题
        fig.suptitle('坐标变换和方向归一化前后对比', fontsize=16, y=0.98)
        
        # 保存图像
        plt.savefig('STAGPoser/transformation_comparison.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print("坐标变换和归一化对比图已保存为 transformation_comparison.png")
        
        # 打印变换矩阵信息
        print("\n=== 变换矩阵信息 ===")
        print("手机 Device2Bone 矩阵:")
        print(phone_device2bone)
        print("\n手表 Device2Bone 矩阵:")
        print(watch_device2bone)
        
        # 计算和显示变换效果统计
        print("\n=== 变换效果统计 ===")
        
        # 加速度变换效果
        phone_acc_change = np.std(phone_acc_transformed, axis=0) / np.std(phone_acc_raw, axis=0)
        watch_acc_change = np.std(watch_acc_transformed, axis=0) / np.std(watch_acc_raw, axis=0)
        
        print(f"手机加速度标准差比值 (变换后/变换前): X={phone_acc_change[0]:.3f}, Y={phone_acc_change[1]:.3f}, Z={phone_acc_change[2]:.3f}")
        print(f"手表加速度标准差比值 (变换后/变换前): X={watch_acc_change[0]:.3f}, Y={watch_acc_change[1]:.3f}, Z={watch_acc_change[2]:.3f}")
        
        # 方向归一化效果 - T-pose开始时的角度
        phone_start_euler_before = phone_euler_transformed[0]
        phone_start_euler_after = phone_euler_normalized[0]
        watch_start_euler_before = watch_euler_transformed[0]
        watch_start_euler_after = watch_euler_normalized[0]
        
        print(f"\n手机数据开始时的欧拉角:")
        print(f"  归一化前: Roll={phone_start_euler_before[0]:.1f}°, Pitch={phone_start_euler_before[1]:.1f}°, Yaw={phone_start_euler_before[2]:.1f}°")
        print(f"  归一化后: Roll={phone_start_euler_after[0]:.1f}°, Pitch={phone_start_euler_after[1]:.1f}°, Yaw={phone_start_euler_after[2]:.1f}°")
        
        print(f"\n手表数据开始时的欧拉角:")
        print(f"  归一化前: Roll={watch_start_euler_before[0]:.1f}°, Pitch={watch_start_euler_before[1]:.1f}°, Yaw={watch_start_euler_before[2]:.1f}°")
        print(f"  归一化后: Roll={watch_start_euler_after[0]:.1f}°, Pitch={watch_start_euler_after[1]:.1f}°, Yaw={watch_start_euler_after[2]:.1f}°")
        
    except Exception as e:
        print(f"坐标变换对比可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 文件路径
    phone_file = "STAGPoser/STAG_data/STAG_0731_phone.csv"
    watch_file = "STAGPoser/STAG_data/STAG_0731_watch.csv"
    output_file = "STAGPoser/STAG_data/aligned_imu_data_normalized.pt"
    
    # 检查文件是否存在
    if not os.path.exists(phone_file):
        print(f"错误：找不到手机数据文件 {phone_file}")
        exit(1)
    
    if not os.path.exists(watch_file):
        print(f"错误：找不到手表数据文件 {watch_file}")
        exit(1)
    
    # 处理数据
    try:
        result = process_imu_files(phone_file, watch_file, output_file)
        print("\n=== 处理完成 ===")
        print(f"输出文件: {output_file}")
        print(f"对齐后帧数: {result['frame_count']}")
        print(f"IMU位置: {result['imu_positions']}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 