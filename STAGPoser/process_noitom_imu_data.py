import pandas as pd
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'WenQuanYi Zen Hei'
mpl.rcParams['axes.unicode_minus'] = False
from scipy.signal import find_peaks
import os

def quaternion_to_rotation_matrix(quat):
    """
    将四元数转换为旋转矩阵
    Args:
        quat: 四元数 [x, y, z, w] 或 [N, 4]
    Returns:
        旋转矩阵 [3, 3] 或 [N, 3, 3]
    """
    if len(quat.shape) == 1:
        # 单个四元数
        x, y, z, w = quat
        
        # 归一化四元数
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # 构建旋转矩阵
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        return R
    else:
        # 多个四元数
        N = quat.shape[0]
        R = np.zeros((N, 3, 3))
        
        for i in range(N):
            x, y, z, w = quat[i]
            
            # 归一化四元数
            norm = np.sqrt(x*x + y*y + z*z + w*w)
            if norm > 0:
                x, y, z, w = x/norm, y/norm, z/norm, w/norm
            
            # 构建旋转矩阵
            R[i] = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ])
        
        return R

def smooth_imu_acceleration(acc_data, smooth_n=4, frame_rate=100):
    """
    使用优化的张量运算对IMU加速度数据进行平滑
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

def detect_jump_sequences(lin_acc_data, magnitude_threshold=15.0, min_peak_distance=30, 
                         min_jump_interval=50, max_jump_interval=150, 
                         tpose_start=None, tpose_end=None):
    """
    检测连续的跳跃序列（开始3次，结束3次）
    基于y轴加速度的峰值检测，确保结束跳跃在T-pose之后
    """
    # 计算y轴加速度的绝对值（跳跃主要在垂直方向）
    acc_magnitude = np.abs(lin_acc_data[:, 1])
    
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

def detect_tpose_segment(lin_acc_data, ori_data, start_jump_frames=None, fps=60, min_duration=3.0, 
                        acc_stability_threshold=0.5, ori_stability_threshold=0.3):
    """
    检测T-pose段：IMU数据变化范围小且持续至少4秒的时间段，且必须在开始跳跃之后
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
    
    # 计算方向变化（四元数的变化）
    ori_magnitude = np.sqrt(np.sum(ori_data**2, axis=1))
    
    # 滑动窗口检测稳定段（从搜索起始位置开始）
    stable_segments = []
    
    for i in range(search_start, len(acc_magnitude) - window_size + 1):
        # 计算窗口内的标准差
        acc_std = np.std(acc_magnitude[i:i+window_size])
        ori_std = np.std(ori_magnitude[i:i+window_size])
        
        # 判断是否稳定
        if acc_std < acc_stability_threshold and ori_std < ori_stability_threshold:
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
    # 使用y轴加速度绝对值进行对齐（与跳跃检测保持一致）
    mag1 = np.abs(data1[:, 1])
    mag2 = np.abs(data2[:, 1])
    
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

def process_noitom_imu_file(input_file, output_file):
    """
    处理Noitom IMU文件并生成对齐的数据，包含T-pose检测和归一化
    """
    # 读取数据
    print("读取Noitom数据文件...")
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 第一行是总帧数
    total_frames = int(lines[0].strip())
    print(f"总帧数: {total_frames}")
    
    # 第二行是表头
    headers = lines[1].strip().split(',')
    
    # 找到需要的列索引
    leftforearm_indices = {}
    rightupleg_indices = {}
    
    for i, header in enumerate(headers):
        # LeftForeArm 传感器数据
        if 'LeftForeArm-Sensor-Acce-x' in header:
            leftforearm_indices['acce_x'] = i
        elif 'LeftForeArm-Sensor-Acce-y' in header:
            leftforearm_indices['acce_y'] = i
        elif 'LeftForeArm-Sensor-Acce-z' in header:
            leftforearm_indices['acce_z'] = i
        elif 'LeftForeArm-Sensor-Quat-x' in header:
            leftforearm_indices['quat_x'] = i
        elif 'LeftForeArm-Sensor-Quat-y' in header:
            leftforearm_indices['quat_y'] = i
        elif 'LeftForeArm-Sensor-Quat-z' in header:
            leftforearm_indices['quat_z'] = i
        elif 'LeftForeArm-Sensor-Quat-w' in header:
            leftforearm_indices['quat_w'] = i
            
        # RightUpLeg 传感器数据
        elif 'RightUpLeg-Sensor-Acce-x' in header:
            rightupleg_indices['acce_x'] = i
        elif 'RightUpLeg-Sensor-Acce-y' in header:
            rightupleg_indices['acce_y'] = i
        elif 'RightUpLeg-Sensor-Acce-z' in header:
            rightupleg_indices['acce_z'] = i
        elif 'RightUpLeg-Sensor-Quat-x' in header:
            rightupleg_indices['quat_x'] = i
        elif 'RightUpLeg-Sensor-Quat-y' in header:
            rightupleg_indices['quat_y'] = i
        elif 'RightUpLeg-Sensor-Quat-z' in header:
            rightupleg_indices['quat_z'] = i
        elif 'RightUpLeg-Sensor-Quat-w' in header:
            rightupleg_indices['quat_w'] = i
    
    print(f"找到LeftForeArm列索引: {leftforearm_indices}")
    print(f"找到RightUpLeg列索引: {rightupleg_indices}")
    
    # 检查是否找到所有需要的列
    required_keys = ['acce_x', 'acce_y', 'acce_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    for key in required_keys:
        if key not in leftforearm_indices:
            raise ValueError(f"未找到LeftForeArm的{key}列")
        if key not in rightupleg_indices:
            raise ValueError(f"未找到RightUpLeg的{key}列")
    
    # 读取数据
    leftforearm_acc = []
    leftforearm_quat = []
    rightupleg_acc = []
    rightupleg_quat = []
    
    for i in range(2, len(lines)):  # 从第三行开始读取数据
        if lines[i].strip():  # 跳过空行
            data = lines[i].strip().split(',')
            if len(data) >= max(max(leftforearm_indices.values()), max(rightupleg_indices.values())) + 1:
                # LeftForeArm数据
                lf_acc = [float(data[leftforearm_indices['acce_x']]), 
                         float(data[leftforearm_indices['acce_y']]), 
                         float(data[leftforearm_indices['acce_z']])]
                lf_quat = [float(data[leftforearm_indices['quat_x']]), 
                          float(data[leftforearm_indices['quat_y']]), 
                          float(data[leftforearm_indices['quat_z']]), 
                          float(data[leftforearm_indices['quat_w']])]
                
                # RightUpLeg数据
                rul_acc = [float(data[rightupleg_indices['acce_x']]), 
                          float(data[rightupleg_indices['acce_y']]), 
                          float(data[rightupleg_indices['acce_z']])]
                rul_quat = [float(data[rightupleg_indices['quat_x']]), 
                           float(data[rightupleg_indices['quat_y']]), 
                           float(data[rightupleg_indices['quat_z']]), 
                           float(data[rightupleg_indices['quat_w']])]
                
                leftforearm_acc.append(lf_acc)
                leftforearm_quat.append(lf_quat)
                rightupleg_acc.append(rul_acc)
                rightupleg_quat.append(rul_quat)
    
    # 转换为numpy数组
    leftforearm_acc = np.array(leftforearm_acc)
    leftforearm_quat = np.array(leftforearm_quat)
    rightupleg_acc = np.array(rightupleg_acc)
    rightupleg_quat = np.array(rightupleg_quat)
    
    print(f"LeftForeArm数据: {len(leftforearm_acc)} 帧")
    print(f"RightUpLeg数据: {len(rightupleg_acc)} 帧")
    
    # 分析加速度数据范围，帮助调整跳跃检测阈值
    lf_acc_mag = np.abs(leftforearm_acc[:, 1])  # y轴加速度绝对值
    rul_acc_mag = np.abs(rightupleg_acc[:, 1])  # y轴加速度绝对值
    
    print(f"\nLeftForeArm加速度分析 (y轴绝对值):")
    print(f"  y轴范围: {lf_acc_mag.min():.2f} - {lf_acc_mag.max():.2f}")
    print(f"  y轴均值: {lf_acc_mag.mean():.2f} ± {lf_acc_mag.std():.2f}")
    print(f"  95%分位数: {np.percentile(lf_acc_mag, 95):.2f}")
    
    print(f"\nRightUpLeg加速度分析 (y轴绝对值):")
    print(f"  y轴范围: {rul_acc_mag.min():.2f} - {rul_acc_mag.max():.2f}")
    print(f"  y轴均值: {rul_acc_mag.mean():.2f} ± {rul_acc_mag.std():.2f}")
    print(f"  95%分位数: {np.percentile(rul_acc_mag, 95):.2f}")
    
    # 将四元数转换为旋转矩阵
    print("将四元数转换为旋转矩阵...")
    leftforearm_ori = quaternion_to_rotation_matrix(leftforearm_quat)  # [N, 3, 3]
    rightupleg_ori = quaternion_to_rotation_matrix(rightupleg_quat)    # [N, 3, 3]
    
    # 第一阶段：初步检测开始跳跃序列（无T-pose信息）
    print("\n=== 第一阶段：初步跳跃检测 ===")
    print("--- LeftForeArm IMU初步跳跃检测 ---")
    # 根据数据分析调整阈值
    lf_threshold = max(1.5, np.percentile(lf_acc_mag, 85))  # 使用85%分位数或最小3.0
    lf_start_seq_temp, lf_end_seq_temp = detect_jump_sequences(leftforearm_acc, magnitude_threshold=lf_threshold)
    
    print("\n--- RightUpLeg IMU初步跳跃检测 ---")  
    rul_threshold = max(2.0, np.percentile(rul_acc_mag, 85))  # 使用85%分位数或最小4.0
    rul_start_seq_temp, rul_end_seq_temp = detect_jump_sequences(rightupleg_acc, magnitude_threshold=rul_threshold)
    
    # 第二阶段：基于开始跳跃检测T-pose段
    print("\n=== 第二阶段：T-pose检测 ===")
    print("LeftForeArm T-pose检测:")
    lf_tpose_start, lf_tpose_end = detect_tpose_segment(leftforearm_acc, leftforearm_quat, lf_start_seq_temp, 
                                                       acc_stability_threshold=0.05, ori_stability_threshold=0.05)
    
    print("RightUpLeg T-pose检测:")
    rul_tpose_start, rul_tpose_end = detect_tpose_segment(rightupleg_acc, rightupleg_quat, rul_start_seq_temp, 
                                                         acc_stability_threshold=0.05, ori_stability_threshold=0.05)
    
    # 第三阶段：基于T-pose信息重新精确检测跳跃序列
    print("\n=== 第三阶段：基于T-pose重新检测跳跃序列 ===")
    print("--- LeftForeArm IMU精确跳跃检测 ---")
    lf_start_seq, lf_end_seq = detect_jump_sequences(leftforearm_acc, magnitude_threshold=lf_threshold, 
                                                    tpose_start=lf_tpose_start, tpose_end=lf_tpose_end)
    
    print("\n--- RightUpLeg IMU精确跳跃检测 ---")  
    rul_start_seq, rul_end_seq = detect_jump_sequences(rightupleg_acc, magnitude_threshold=rul_threshold,
                                                      tpose_start=rul_tpose_start, tpose_end=rul_tpose_end)
    
    # 提取有效数据范围
    lf_start, lf_end, lf_tpose_rel_start, lf_tpose_rel_end = extract_valid_data_range_with_tpose(
        lf_start_seq, lf_end_seq, lf_tpose_start, lf_tpose_end, len(leftforearm_acc))
    
    rul_start, rul_end, rul_tpose_rel_start, rul_tpose_rel_end = extract_valid_data_range_with_tpose(
        rul_start_seq, rul_end_seq, rul_tpose_start, rul_tpose_end, len(rightupleg_acc))
    
    print(f"\nLeftForeArm有效数据范围: {lf_start} - {lf_end} (共{lf_end-lf_start}帧)")
    print(f"RightUpLeg有效数据范围: {rul_start} - {rul_end} (共{rul_end-rul_start}帧)")
    
    # 提取有效数据段
    lf_acc_valid = leftforearm_acc[lf_start:lf_end]
    lf_ori_valid = leftforearm_ori[lf_start:lf_end]
    rul_acc_valid = rightupleg_acc[rul_start:rul_end]
    rul_ori_valid = rightupleg_ori[rul_start:rul_end]
    
    # 对齐数据
    print("\n对齐IMU数据...")
    shift = align_imu_data(lf_acc_valid, rul_acc_valid)
    
    # 根据偏移量调整数据
    if shift > 0:
        rul_aligned_acc = rul_acc_valid[shift:]
        rul_aligned_ori = rul_ori_valid[shift:]
        lf_aligned_acc = lf_acc_valid[:len(rul_aligned_acc)]
        lf_aligned_ori = lf_ori_valid[:len(rul_aligned_acc)]
        # 调整T-pose相对位置
        rul_tpose_rel_start = max(0, rul_tpose_rel_start - shift)
        rul_tpose_rel_end = max(0, rul_tpose_rel_end - shift)
    elif shift < 0:
        lf_aligned_acc = lf_acc_valid[abs(shift):]
        lf_aligned_ori = lf_ori_valid[abs(shift):]
        rul_aligned_acc = rul_acc_valid[:len(lf_aligned_acc)]
        rul_aligned_ori = rul_ori_valid[:len(lf_aligned_acc)]
        # 调整T-pose相对位置
        lf_tpose_rel_start = max(0, lf_tpose_rel_start - abs(shift))
        lf_tpose_rel_end = max(0, lf_tpose_rel_end - abs(shift))
    else:
        min_len = min(len(lf_acc_valid), len(rul_acc_valid))
        lf_aligned_acc = lf_acc_valid[:min_len]
        lf_aligned_ori = lf_ori_valid[:min_len]
        rul_aligned_acc = rul_acc_valid[:min_len]
        rul_aligned_ori = rul_ori_valid[:min_len]
    
    # 确保长度一致
    aligned_frames = min(len(lf_aligned_acc), len(rul_aligned_acc))
    lf_aligned_acc = lf_aligned_acc[:aligned_frames]
    lf_aligned_ori = lf_aligned_ori[:aligned_frames]
    rul_aligned_acc = rul_aligned_acc[:aligned_frames]
    rul_aligned_ori = rul_aligned_ori[:aligned_frames]
    
    print(f"对齐后数据长度: {aligned_frames} 帧")
    
    # 应用加速度平滑
    print("应用加速度平滑...")
    lf_acc_smoothed = smooth_imu_acceleration(lf_aligned_acc, smooth_n=4, frame_rate=100)
    rul_acc_smoothed = smooth_imu_acceleration(rul_aligned_acc, smooth_n=4, frame_rate=100)
    print("加速度平滑完成")
    
    # 对方向数据进行归一化（使T-pose开始时方向为0）
    print("\n应用方向归一化...")
    
    # 对LeftForeArm方向数据归一化
    if lf_tpose_start >= 0:
        # 调整T-pose位置到对齐后的数据中
        lf_tpose_aligned = (max(0, lf_tpose_start - lf_start), 
                           min(aligned_frames, lf_tpose_end - lf_start))
        if shift > 0:  # RightUpLeg延迟，LeftForeArm数据不变
            lf_tpose_in_aligned = lf_tpose_aligned
        elif shift < 0:  # LeftForeArm延迟，需要调整
            lf_tpose_in_aligned = (max(0, lf_tpose_aligned[0] - abs(shift)),
                                  max(0, lf_tpose_aligned[1] - abs(shift)))
        else:
            lf_tpose_in_aligned = lf_tpose_aligned
        
        if lf_tpose_in_aligned[0] >= 0 and lf_tpose_in_aligned[1] > lf_tpose_in_aligned[0]:
            lf_ori_normalized, lf_device2bone = normalize_orientation_data(
                lf_aligned_ori, lf_tpose_in_aligned[0], lf_tpose_in_aligned[1], 0)
            print(f"LeftForeArm方向数据已归一化，T-pose位置: {lf_tpose_in_aligned}")
        else:
            lf_ori_normalized = lf_aligned_ori
            lf_device2bone = np.eye(3)
            print("LeftForeArm T-pose位置无效，跳过归一化")
    else:
        lf_ori_normalized = lf_aligned_ori
        lf_device2bone = np.eye(3)
        print("未检测到LeftForeArm T-pose，跳过归一化")
    
    # 对RightUpLeg方向数据归一化
    if rul_tpose_start >= 0:
        # 调整T-pose位置到对齐后的数据中
        rul_tpose_aligned = (max(0, rul_tpose_start - rul_start),
                            min(aligned_frames, rul_tpose_end - rul_start))
        if shift > 0:  # RightUpLeg延迟，需要调整
            rul_tpose_in_aligned = (max(0, rul_tpose_aligned[0] - shift),
                                   max(0, rul_tpose_aligned[1] - shift))
        elif shift < 0:  # LeftForeArm延迟，RightUpLeg数据不变
            rul_tpose_in_aligned = rul_tpose_aligned
        else:
            rul_tpose_in_aligned = rul_tpose_aligned
        
        if rul_tpose_in_aligned[0] >= 0 and rul_tpose_in_aligned[1] > rul_tpose_in_aligned[0]:
            rul_ori_normalized, rul_device2bone = normalize_orientation_data(
                rul_aligned_ori, rul_tpose_in_aligned[0], rul_tpose_in_aligned[1], 0)
            print(f"RightUpLeg方向数据已归一化，T-pose位置: {rul_tpose_in_aligned}")
        else:
            rul_ori_normalized = rul_aligned_ori
            rul_device2bone = np.eye(3)
            print("RightUpLeg T-pose位置无效，跳过归一化")
    else:
        rul_ori_normalized = rul_aligned_ori
        rul_device2bone = np.eye(3)
        print("未检测到RightUpLeg T-pose，跳过归一化")
    
    print("方向归一化完成")
    
    # 创建输出数据结构（与原代码相同的格式）
    processed_data = {
        'frame_count': aligned_frames,
        'imu_data': {
            'rp': {  # 右口袋(RightUpLeg) - 变换和归一化后的数据
                'lin_acc_x': torch.tensor(rul_acc_smoothed[:, 0], dtype=torch.float32),
                'lin_acc_y': torch.tensor(rul_acc_smoothed[:, 1], dtype=torch.float32),
                'lin_acc_z': torch.tensor(rul_acc_smoothed[:, 2], dtype=torch.float32),
                'ori': torch.tensor(rul_ori_normalized, dtype=torch.float32)  # [N, 3, 3] 旋转矩阵
            },
            'lw': {  # 左手腕(LeftForeArm) - 变换和归一化后的数据
                'lin_acc_x': torch.tensor(lf_acc_smoothed[:, 0], dtype=torch.float32),
                'lin_acc_y': torch.tensor(lf_acc_smoothed[:, 1], dtype=torch.float32),
                'lin_acc_z': torch.tensor(lf_acc_smoothed[:, 2], dtype=torch.float32),
                'ori': torch.tensor(lf_ori_normalized, dtype=torch.float32)  # [N, 3, 3] 旋转矩阵
            }
        },
        'calibration': {
            'rp': {
                'device2bone': torch.tensor(rul_device2bone, dtype=torch.float32)
            },
            'lw': {
                'device2bone': torch.tensor(lf_device2bone, dtype=torch.float32)
            }
        },
        'imu_positions': ['rp', 'lw'],
        'tpose_info': {
            'rightupleg_tpose_range': (rul_tpose_rel_start, rul_tpose_rel_end),
            'leftforearm_tpose_range': (lf_tpose_rel_start, lf_tpose_rel_end),
            'has_valid_tpose': lf_tpose_rel_start >= 0 and rul_tpose_rel_start >= 0
        },
        'metadata': {
            'original_leftforearm_frames': len(leftforearm_acc),
            'original_rightupleg_frames': len(rightupleg_acc),
            'leftforearm_valid_range': (lf_start, lf_end),
            'rightupleg_valid_range': (rul_start, rul_end),
            'alignment_shift': shift,
            'leftforearm_start_jumps': lf_start_seq.tolist() if len(lf_start_seq) > 0 else [],
            'leftforearm_end_jumps': lf_end_seq.tolist() if len(lf_end_seq) > 0 else [],
            'rightupleg_start_jumps': rul_start_seq.tolist() if len(rul_start_seq) > 0 else [],
            'rightupleg_end_jumps': rul_end_seq.tolist() if len(rul_end_seq) > 0 else [],
            'leftforearm_tpose_original': (lf_tpose_start, lf_tpose_end),
            'rightupleg_tpose_original': (rul_tpose_start, rul_tpose_end),
            'coordinate_transformed': False,  # Noitom数据不需要坐标变换
            'orientation_normalized': True,
            'rotation_format': 'matrix'  # 标记使用旋转矩阵格式
        }
    }
    
    # 保存数据
    print(f"\n保存处理后的数据到 {output_file}")
    torch.save(processed_data, output_file)
    
    # 简单的可视化
    create_noitom_visualization(processed_data, leftforearm_acc, rightupleg_acc, 
                               lf_acc_smoothed, rul_acc_smoothed,
                               lf_ori_normalized, rul_ori_normalized)
    
    return processed_data

def create_noitom_visualization(processed_data, lf_acc_raw, rul_acc_raw,
                               lf_acc_smoothed, rul_acc_smoothed,
                               lf_ori_normalized, rul_ori_normalized):
    """
    创建Noitom数据的可视化图
    """
    try:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 计算原始数据的y轴加速度绝对值（用于跳跃检测）
        lf_mag = np.abs(lf_acc_raw[:, 1])
        rul_mag = np.abs(rul_acc_raw[:, 1])
        
        # 获取检测信息
        lf_start_jumps = processed_data['metadata']['leftforearm_start_jumps']
        lf_end_jumps = processed_data['metadata']['leftforearm_end_jumps']
        rul_start_jumps = processed_data['metadata']['rightupleg_start_jumps']  
        rul_end_jumps = processed_data['metadata']['rightupleg_end_jumps']
        
        lf_range = processed_data['metadata']['leftforearm_valid_range']
        rul_range = processed_data['metadata']['rightupleg_valid_range']
        
        lf_tpose_orig = processed_data['metadata']['leftforearm_tpose_original']
        rul_tpose_orig = processed_data['metadata']['rightupleg_tpose_original']
        
        # 第一行：跳跃和T-pose检测
        axes[0, 0].plot(lf_mag, alpha=0.7, color='blue', linewidth=0.8)
        if lf_start_jumps:
            axes[0, 0].scatter(lf_start_jumps, lf_mag[lf_start_jumps], 
                              color='green', s=60, marker='v', label='开始跳跃', zorder=5)
        if lf_end_jumps:
            axes[0, 0].scatter(lf_end_jumps, lf_mag[lf_end_jumps], 
                              color='red', s=60, marker='^', label='结束跳跃', zorder=5)
        if lf_tpose_orig[0] >= 0:
            axes[0, 0].axvspan(lf_tpose_orig[0], lf_tpose_orig[1], alpha=0.3, color='yellow', label='T-pose')
        axes[0, 0].axvline(lf_range[0], color='orange', linestyle='--', alpha=0.8, label='有效范围')
        axes[0, 0].axvline(lf_range[1], color='orange', linestyle='--', alpha=0.8)
        axes[0, 0].set_title('LeftForeArm IMU - Y轴加速度跳跃检测')
        axes[0, 0].set_ylabel('Y轴加速度绝对值 (m/s²)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(rul_mag, alpha=0.7, color='orange', linewidth=0.8)
        if rul_start_jumps:
            axes[0, 1].scatter(rul_start_jumps, rul_mag[rul_start_jumps], 
                              color='green', s=60, marker='v', label='开始跳跃', zorder=5)
        if rul_end_jumps:
            axes[0, 1].scatter(rul_end_jumps, rul_mag[rul_end_jumps], 
                              color='red', s=60, marker='^', label='结束跳跃', zorder=5)
        if rul_tpose_orig[0] >= 0:
            axes[0, 1].axvspan(rul_tpose_orig[0], rul_tpose_orig[1], alpha=0.3, color='yellow', label='T-pose')
        axes[0, 1].axvline(rul_range[0], color='blue', linestyle='--', alpha=0.8, label='有效范围')
        axes[0, 1].axvline(rul_range[1], color='blue', linestyle='--', alpha=0.8)
        axes[0, 1].set_title('RightUpLeg IMU - Y轴加速度跳跃检测')
        axes[0, 1].set_ylabel('Y轴加速度绝对值 (m/s²)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 第二行：平滑后加速度对比
        lw_acc_mag_smoothed = torch.sqrt(
            processed_data['imu_data']['lw']['lin_acc_x']**2 + 
            processed_data['imu_data']['lw']['lin_acc_y']**2 + 
            processed_data['imu_data']['lw']['lin_acc_z']**2
        ).numpy()
        rp_acc_mag_smoothed = torch.sqrt(
            processed_data['imu_data']['rp']['lin_acc_x']**2 + 
            processed_data['imu_data']['rp']['lin_acc_y']**2 + 
            processed_data['imu_data']['rp']['lin_acc_z']**2
        ).numpy()
        
        frames = np.arange(len(lw_acc_mag_smoothed))
        
        axes[1, 0].plot(frames, lw_acc_mag_smoothed, label='左手腕 (LeftForeArm) - 平滑后', alpha=0.8)
        axes[1, 0].plot(frames, rp_acc_mag_smoothed, label='右口袋 (RightUpLeg) - 平滑后', alpha=0.8) 
        axes[1, 0].set_title('Smoothed Acceleration Magnitude (Noitom Data)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Z轴加速度对比
        lw_acc_z = processed_data['imu_data']['lw']['lin_acc_z'].numpy()
        rp_acc_z = processed_data['imu_data']['rp']['lin_acc_z'].numpy()
        
        axes[1, 1].plot(frames, lw_acc_z, label='左手腕 (LeftForeArm)', alpha=0.8)
        axes[1, 1].plot(frames, rp_acc_z, label='右口袋 (RightUpLeg)', alpha=0.8)
        axes[1, 1].set_title('Z-axis Acceleration (Noitom Data)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 第三行：方向归一化效果展示
        lw_ori_normalized = processed_data['imu_data']['lw']['ori'].numpy()
        rp_ori_normalized = processed_data['imu_data']['rp']['ori'].numpy()
        
        # 旋转矩阵到欧拉角的转换函数
        def rotation_matrix_to_euler_degrees(rotation_matrices):
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
        lw_euler_normalized = rotation_matrix_to_euler_degrees(lw_ori_normalized)
        rp_euler_normalized = rotation_matrix_to_euler_degrees(rp_ori_normalized)
        
        axes[2, 0].plot(frames, lw_euler_normalized[:, 0], label='LeftForeArm Roll (归一化)', alpha=0.8)
        axes[2, 0].plot(frames, lw_euler_normalized[:, 1], label='LeftForeArm Pitch (归一化)', alpha=0.8)
        axes[2, 0].plot(frames, lw_euler_normalized[:, 2], label='LeftForeArm Yaw (归一化)', alpha=0.8)
        
        # 标记T-pose段
        if processed_data['tpose_info']['has_valid_tpose']:
            lf_tpose_range = processed_data['tpose_info']['leftforearm_tpose_range']
            if lf_tpose_range[1] > lf_tpose_range[0]:
                axes[2, 0].axvspan(lf_tpose_range[0], lf_tpose_range[1], 
                                  alpha=0.3, color='yellow', label='T-pose段')
        
        axes[2, 0].set_title('LeftForeArm方向角度 (归一化后)')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_ylabel('角度 (度)')
        
        axes[2, 1].plot(frames, rp_euler_normalized[:, 0], label='RightUpLeg Roll (归一化)', alpha=0.8)
        axes[2, 1].plot(frames, rp_euler_normalized[:, 1], label='RightUpLeg Pitch (归一化)', alpha=0.8)
        axes[2, 1].plot(frames, rp_euler_normalized[:, 2], label='RightUpLeg Yaw (归一化)', alpha=0.8)
        
        # 标记T-pose段
        if processed_data['tpose_info']['has_valid_tpose']:
            rul_tpose_range = processed_data['tpose_info']['rightupleg_tpose_range']
            if rul_tpose_range[1] > rul_tpose_range[0]:
                axes[2, 1].axvspan(rul_tpose_range[0], rul_tpose_range[1], 
                                  alpha=0.3, color='yellow', label='T-pose段')
        
        axes[2, 1].set_title('RightUpLeg方向角度 (归一化后)')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylabel('角度 (度)')
        
        plt.tight_layout()
        plt.savefig('STAGPoser/noitom_detection_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Noitom检测结果图已保存为 noitom_detection_results.png")
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")

if __name__ == "__main__":
    # 文件路径
    input_file = "STAGPoser/noitom_data/625_sample_0_jump_T_shortwalk_jump.csv"
    output_file = "STAGPoser/noitom_data/aligned_imu_data_normalized.pt"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到数据文件 {input_file}")
        exit(1)
    
    # 处理数据
    try:
        result = process_noitom_imu_file(input_file, output_file)
        print("\n=== 处理完成 ===")
        print(f"输出文件: {output_file}")
        print(f"对齐后帧数: {result['frame_count']}")
        print(f"IMU位置: {result['imu_positions']}")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 