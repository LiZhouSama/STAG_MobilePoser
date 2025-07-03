import os
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
from scipy.interpolate import interp1d
import yaml
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
mpl.rcParams['font.family'] = 'WenQuanYi Zen Hei'      # 使用文泉驿正黑:contentReference[oaicite:5]{index=5}
mpl.rcParams['axes.unicode_minus'] = False       # 解决负号 '-' 显示为方块的问题

import sys
sys.path.insert(0, "/mnt/d/a_WORK/Projects/PhD/tasks/MobilePoser/human_body_prior/src")
# aitviewer imports
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.spheres import Spheres
from aitviewer.viewer import Viewer
from aitviewer.scene.camera import Camera
from moderngl_window.context.base import KeyModifiers
from human_body_prior.body_model.body_model import BodyModel
import imgui
import pytorch3d.transforms as transforms

from mobileposer.config import *
from mobileposer.utils.model_utils import *
import mobileposer.articulate as art
from mobileposer.articulate.math import *


# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)


def matrix_to_axis_angle(matrix):
    """将旋转矩阵转换为轴角表示"""
    # 使用articulate库的现有函数
    return rotation_matrix_to_axis_angle(matrix)


def axis_angle_to_matrix(axis_angle):
    """将轴角表示转换为旋转矩阵"""
    # 使用articulate库的现有函数
    return axis_angle_to_rotation_matrix(axis_angle)


def convert_orientation_to_rotation_matrix(ori_roll, ori_pitch, ori_yaw):
    """
    将欧拉角（度数）转换为旋转矩阵
    
    Args:
        ori_roll, ori_pitch, ori_yaw: 旋转角度（度数）
    
    Returns:
        rotation matrices [N, 3, 3]
    """
    # 转换为弧度
    roll = torch.deg2rad(ori_roll)
    pitch = torch.deg2rad(ori_pitch) 
    yaw = torch.deg2rad(ori_yaw)
    
    # 计算旋转矩阵分量
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    
    # 构建旋转矩阵 (ZYX欧拉角顺序)
    N = len(roll)
    R = torch.zeros(N, 3, 3)
    
    R[:, 0, 0] = cos_y * cos_p
    R[:, 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
    R[:, 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
    R[:, 1, 0] = sin_y * cos_p
    R[:, 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
    R[:, 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
    R[:, 2, 0] = -sin_p
    R[:, 2, 1] = cos_p * sin_r
    R[:, 2, 2] = cos_p * cos_r
    
    return R


def downsample_data(data, original_fps=100, target_fps=30):
    """
    将数据从原始帧率降采样到目标帧率
    
    Args:
        data: 输入数据 [T, ...]
        original_fps: 原始帧率
        target_fps: 目标帧率
    
    Returns:
        downsampled data
    """
    if original_fps == target_fps:
        return data
    
    original_length = len(data)
    target_length = int(original_length * target_fps / original_fps)
    
    # 创建插值索引
    original_indices = np.linspace(0, original_length - 1, original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)
    
    # 对每个维度进行插值
    if data.dim() == 1:
        # 1D数据
        interp_func = interp1d(original_indices, data.numpy(), kind='linear')
        downsampled = torch.tensor(interp_func(target_indices), dtype=data.dtype)
    elif data.dim() == 2:
        # 2D数据
        data_np = data.numpy()
        downsampled_list = []
        
        for i in range(data.shape[1]):
            interp_func = interp1d(original_indices, data_np[:, i], kind='linear')
            downsampled_list.append(interp_func(target_indices))
        
        downsampled = torch.tensor(np.stack(downsampled_list, axis=1), dtype=data.dtype)
    elif data.dim() == 3:
        # 3D数据 [T, N, D]
        T, N, D = data.shape
        data_2d = data.view(T, -1)
        downsampled_2d = downsample_data(data_2d, original_fps, target_fps)
        T_new = downsampled_2d.shape[0]
        downsampled = downsampled_2d.view(T_new, N, D)
    elif data.dim() == 4:
        # 4D数据 [T, N, D1, D2] - 特殊处理旋转矩阵
        T, N, D1, D2 = data.shape
        if D1 == 3 and D2 == 3:
            # 这是旋转矩阵数据 [T, N, 3, 3]
            # 使用SLERP插值而不是线性插值
            downsampled = torch.zeros(target_length, N, D1, D2, dtype=data.dtype)
            
            for n in range(N):
                # 对每个IMU位置单独处理
                rotation_matrices = data[:, n]  # [T, 3, 3]
                
                # 简单的最近邻降采样（避免插值问题）
                # 或者可以转换为四元数进行SLERP插值
                indices = np.round(target_indices).astype(int)
                indices = np.clip(indices, 0, original_length - 1)
                downsampled[:, n] = rotation_matrices[indices]
        else:
            # 非旋转矩阵的4D数据，使用原来的方法
            data_2d = data.view(T, -1)
            downsampled_2d = downsample_data(data_2d, original_fps, target_fps)
            T_new = downsampled_2d.shape[0]
            downsampled = downsampled_2d.view(T_new, N, D1, D2)
    else:
        raise ValueError(f"不支持的数据维度: {data.dim()}")
    
    return downsampled

def prepare_model_input_with_calibration(data_dict, combo='lw_rp'):
    """
    准备符合模型输入格式的数据（数据已预处理和归一化）
    
    Args:
        data_dict: 从pt文件加载的数据字典（已归一化）
        combo: IMU组合方式
        
    Returns:
        model_input: 符合模型输入格式的tensor [T, feature_dim]
        tpose_pose: T-pose时的SMPL pose参数（全零）
    """
    imu_data = data_dict['imu_data']
    
    # 检查combo是否有效
    if combo not in amass.combos:
        raise ValueError(f"无效的combo: {combo}. 支持的combo: {list(amass.combos.keys())}")
    
    combo_ids = amass.combos[combo]
    print(f"使用combo: {combo}, 对应的IMU IDs: {combo_ids}")
    
    # 从我们的数据映射到amass的IMU位置
    # 我们的数据: 'lw'(左手腕/手表) -> amass ID 0 (left wrist), 'rp'(右口袋/手机) -> amass ID 3 (right thigh)
    our_to_amass = {'lw': 0, 'rp': 3}
    
    # 提取数据长度
    T = len(imu_data['lw']['lin_acc_x'])
    print(f"原始数据长度: {T} 帧 (100fps)")
    
    # 准备全部5个IMU位置的数据 (加速度和方向)
    all_acc = torch.zeros(T, 5, 3)  # [T, 5, 3]
    all_ori = torch.zeros(T, 5, 3, 3)  # [T, 5, 3, 3]
    
    # 填入我们实际有的数据
    for our_pos, imu_id in our_to_amass.items():
        if our_pos in imu_data:
            # 提取加速度数据
            acc_x = imu_data[our_pos]['lin_acc_x']
            acc_y = imu_data[our_pos]['lin_acc_y'] 
            acc_z = imu_data[our_pos]['lin_acc_z']
            acc_data = torch.stack([acc_x, acc_y, acc_z], dim=1)  # [T, 3]
            
            # 提取方向数据 - 现在直接就是旋转矩阵格式
            if 'ori' in imu_data[our_pos]:
                # 新格式：直接是旋转矩阵
                ori_data = imu_data[our_pos]['ori']  # [T, 3, 3]
            else:
                # 旧格式：需要从欧拉角转换（向后兼容）
                ori_roll = imu_data[our_pos]['ori_roll_deg']
                ori_pitch = imu_data[our_pos]['ori_pitch_deg']
                ori_yaw = imu_data[our_pos]['ori_yaw_deg']
                ori_data = convert_orientation_to_rotation_matrix(ori_roll, ori_pitch, ori_yaw)
            
            all_acc[:, imu_id] = acc_data
            all_ori[:, imu_id] = ori_data
            print(f"位置 {our_pos} (amass ID {imu_id}): 数据已加载")
    
    # 其他位置填充单位矩阵
    for i in range(5):
        if torch.all(all_ori[:, i] == 0):
            all_ori[:, i] = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
    
    print(f"合并后数据形状 - 加速度: {all_acc.shape}, 方向: {all_ori.shape}")
    
    # 降采样到30fps
    print("降采样数据从100fps到30fps...")
    all_acc_30fps = downsample_data(all_acc, 100, 30)
    all_ori_30fps = downsample_data(all_ori, 100, 30)
    
    T_30fps = all_acc_30fps.shape[0]
    print(f"降采样后数据长度: {T_30fps} 帧 (30fps)")
    
    # 按照DataLoader的_get_imu函数的方式处理数据
    # 1. 创建完整的acc和ori数据
    acc = torch.zeros(T_30fps, 5, 3)
    ori = torch.zeros(T_30fps, 5, 3, 3)
    
    # 2. 只填入指定combo的数据，并进行缩放
    acc[:, combo_ids] = all_acc_30fps[:, combo_ids] / amass.acc_scale
    ori[:, combo_ids] = all_ori_30fps[:, combo_ids]
    
    # 3. 提取amass.all_imu_ids对应的数据（保持与原始逻辑一致）
    acc = acc[:, amass.all_imu_ids]  # [T, 5, 3]
    ori = ori[:, amass.all_imu_ids]  # [T, 5, 3, 3]
    
    # 4. 平滑处理
    acc = smooth_avg(acc)
    
    # 5. 拼接数据
    model_input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)  # [T, 60]
    
    # 6. 创建T-pose的SMPL pose参数（全零，表示T-pose）
    tpose_pose = torch.zeros(24, 3, 3)  # 24个关节的单位旋转矩阵
    for i in range(24):
        tpose_pose[i] = torch.eye(3)
    
    print(f"最终模型输入数据形状: {model_input.shape}")
    print(f"T-pose SMPL pose形状: {tpose_pose.shape}")
    
    # 检查校准信息
    if 'calibration' in data_dict:
        print("找到校准信息(device2bone矩阵)")
        for pos in ['lw', 'rp']:
            if pos in data_dict['calibration']:
                device2bone = data_dict['calibration'][pos]['device2bone']
                print(f"  {pos} device2bone形状: {device2bone.shape}")
    
    return model_input, tpose_pose


def load_smpl_model(smpl_model_path, device):
    """加载 SMPL 模型 using human_body_prior"""
    print(f"Loading SMPL model from: {smpl_model_path}")
    if not os.path.exists(smpl_model_path):
        print(f"Error: SMPL model path not found: {smpl_model_path}")
        raise FileNotFoundError(f"SMPL model not found at {smpl_model_path}")
    smpl_model = BodyModel(
        bm_fname=str(smpl_model_path),
        num_betas=16,
        model_type='smplh'  # 明确使用 smplh
    ).to(device)
    return smpl_model


class AITViewerCustom(Viewer):
    """自定义aitviewer用于可视化预测结果"""
    
    def __init__(self, pose_pred, tran_pred, smpl_model, device, **kwargs):
        super().__init__(**kwargs)
        self.pose_pred = pose_pred.to(device)  # [T, 24, 3, 3]
        self.tran_pred = tran_pred.to(device)  # [T, 3]
        self.smpl_model = smpl_model
        self.device = device
        
        # 设置初始相机位置
        self.scene.camera.position = np.array([0.0, 1.0, 3.0])
        self.scene.camera.target = np.array([0.0, 0.8, 0.0])
        
        # 初始化可视化
        self._setup_visualization()
    
    def gui_scene(self):
        """重载GUI场景方法，添加根关节位置显示"""
        # 调用父类的GUI方法
        super().gui_scene()
        
        # 添加根关节位置信息窗口
        imgui.begin("根关节位置信息", True)
        
        # 获取当前帧ID
        current_frame = self.scene.current_frame_id
        if current_frame < self.tran_pred.shape[0]:
            # 获取当前帧的根关节位置
            root_pos = self.tran_pred[current_frame].cpu().numpy()  # [3]
            
            # 显示当前帧号
            imgui.text(f"Frame: {current_frame}")
            imgui.separator()
            
            # 显示根关节三维位置，保留两位小数
            imgui.text("Root Position (m):")
            imgui.text(f"X: {root_pos[0]:.2f}")
            imgui.text(f"Y: {root_pos[1]:.2f}")
            imgui.text(f"Z: {root_pos[2]:.2f}")
            
            imgui.separator()
            
            # 添加一些额外信息
            imgui.text(f"Total Frames: {self.tran_pred.shape[0]}")
            imgui.text(f"Progress: {current_frame + 1}/{self.tran_pred.shape[0]}")
            
            # 显示根关节移动距离（相对于第一帧）
            if current_frame > 0:
                initial_pos = self.tran_pred[0].cpu().numpy()
                distance = np.linalg.norm(root_pos - initial_pos)
                imgui.text(f"Total Distance: {distance:.2f} m")
        else:
            imgui.text("Invalid Frame")
        
        imgui.end()
    
    def _generate_all_frames_mesh(self):
        """生成所有帧的网格数据"""
        total_frames = self.pose_pred.shape[0]
        print(f"正在生成 {total_frames} 帧的网格数据...")
        
        all_verts = []
        
        with torch.no_grad():
            for frame_idx in range(total_frames):
                if frame_idx % 50 == 0:
                    print(f"  处理第 {frame_idx+1}/{total_frames} 帧...")
                    
                pose_frame = self.pose_pred[frame_idx]  # [24, 3, 3] 旋转矩阵
                trans_frame = self.tran_pred[frame_idx]  # [3]
                
                try:
                    # 转换旋转矩阵为轴角表示
                    pose_axis_angle = transforms.matrix_to_axis_angle(pose_frame)  # [24, 3]
                    
                    # 分离root orientation和body pose
                    root_orient = pose_axis_angle[0]  # [3] - root orientation
                    pose_body = pose_axis_angle[1:22].reshape(-1)  # [21*3 = 63] - body pose
                    
                    # 构建SMPL输入
                    smpl_input = {
                        'root_orient': root_orient.unsqueeze(0),  # [1, 3]
                        'pose_body': pose_body.unsqueeze(0),      # [1, 63]
                        'trans': trans_frame.unsqueeze(0)         # [1, 3]
                    }
                    
                    # 通过SMPL模型获取mesh
                    body_pose = self.smpl_model(**smpl_input)
                    verts = body_pose.v[0]  # [Nv, 3]
                    
                    # 应用坐标系转换
                    global R_yup
                    verts_yup = torch.matmul(verts, R_yup.T.to(self.device))
                    all_verts.append(verts_yup.cpu())
                    
                except Exception as e:
                    print(f"生成第 {frame_idx} 帧SMPL mesh时出错: {e}")
                    # 使用前一帧的数据或零数据
                    if len(all_verts) > 0:
                        all_verts.append(all_verts[-1])
                    else:
                        all_verts.append(torch.zeros(6890, 3))  # SMPL模型默认顶点数
        
        # 堆叠所有帧的顶点数据
        all_verts_tensor = torch.stack(all_verts, dim=0)  # [T, Nv, 3]
        faces = self.smpl_model.f.cpu().numpy() if isinstance(self.smpl_model.f, torch.Tensor) else self.smpl_model.f
        
        print(f"网格数据生成完成，形状: {all_verts_tensor.shape}")
        return all_verts_tensor.numpy(), faces
    
    def _setup_visualization(self):
        """设置可视化内容"""
        # 清除现有场景
        # self.scene.clear()
        
        try:
            # 生成所有帧的网格数据
            all_verts, faces = self._generate_all_frames_mesh()
            
            # 创建包含所有帧的网格对象
            pred_mesh = Meshes(
                all_verts,  # [T, Nv, 3]
                faces,      # [Nf, 3]
                name="MobilePoser-Human",
                color=(0.2, 0.6, 0.8, 0.9),  # 蓝色
                gui_affine=False,
                is_selectable=False
            )
            
            # 添加到场景
            self.scene.add(pred_mesh)
            
            print(f"成功添加预测人体网格，顶点形状: {all_verts.shape}")
            print("控制说明:")
            print("  空格: 播放/暂停")
            print("  左/右箭头: 前/后一帧")
            print("  鼠标拖拽: 旋转视角")
            print("  鼠标滚轮: 缩放")
            
        except Exception as e:
            print(f"设置可视化时出错: {e}")
            import traceback
            traceback.print_exc()


class CustomViewer:
    def __init__(self, model_path: str=None, combo: str='lw_rp'):
        """自定义可视化器"""
        # 加载模型
        self.device = model_config.device
        model_path = model_path or paths.weights_file
        self.model = load_model(model_path).to(self.device).eval()
        self.combo = combo
        
        print(f"模型已加载: {model_path}")
        print(f"使用combo: {combo}")
        print(f"设备: {self.device}")
        
        # 加载SMPL模型
        body_model_path = paths.body_model_file
        self.smpl_model = load_smpl_model(body_model_path, self.device)
    
    def load_custom_data(self, data_file: str):
        """加载自定义处理的IMU数据"""
        print(f"加载数据文件: {data_file}")
        data_dict = torch.load(data_file)
        
        print(f"数据帧数: {data_dict['frame_count']}")
        print(f"IMU位置: {data_dict['imu_positions']}")
        
        # 检查是否有坐标系变换
        if data_dict['metadata'].get('coordinate_transformed', False):
            print("数据已进行坐标系变换")
        else:
            print("警告：数据未进行坐标系变换，可能影响结果精度")
        
        # 检查方向归一化
        if data_dict['metadata'].get('orientation_normalized', False):
            print("数据已进行方向归一化（T-pose时方向为0）")
        else:
            print("警告：数据未进行方向归一化，可能影响结果精度")
        
        # 检查T-pose信息
        if 'tpose_info' in data_dict and data_dict['tpose_info']['has_valid_tpose']:
            print("检测到有效的T-pose段")
        else:
            print("警告：未检测到有效的T-pose段")
        
        # 准备模型输入（数据已归一化）
        self.model_input, self.tpose_pose = prepare_model_input_with_calibration(data_dict, self.combo)
        self.data_dict = data_dict
        return self.model_input
    
    def _evaluate_model(self):
        """评估模型"""
        data = self.model_input.to(self.device)
        
        print("开始模型推理...")
        print("注意：输入数据已归一化，T-pose时方向为0，与AMASS训练数据一致")
        with torch.no_grad():
            # 使用离线模式进行评估
            pose, joints, tran, contact = self.model.forward_offline(
                data.unsqueeze(0), [data.shape[0]]
            )
        
        pose = pose.squeeze(0)  # [T, 24, 3, 3]
        tran = tran.squeeze(0)  # [T, 3]
        joints = joints.squeeze(0)  # [T, num_joints, 3]
        contact = contact.squeeze(0)  # [T, 4]
        
        print(f"推理完成. 姿态形状: {pose.shape}, 位移形状: {tran.shape}")
        return pose, tran, joints, contact
    
    def save_results(self, output_file: str="prediction_results.pt"):
        """保存预测结果到文件"""
        pose_pred, tran_pred, joints_pred, contact_pred = self._evaluate_model()
        
        # 保存预测结果
        results = {
            'pose': pose_pred.cpu(),
            'translation': tran_pred.cpu(), 
            'joints': joints_pred.cpu(),
            'contact': contact_pred.cpu(),
            'combo': self.combo,
            'fps': 30,
            'frame_count': pose_pred.shape[0],
            'coordinate_transformed': self.data_dict['metadata'].get('coordinate_transformed', False),
            'orientation_normalized': self.data_dict['metadata'].get('orientation_normalized', False)
        }
        
        torch.save(results, output_file)
        print(f"预测结果已保存到: {output_file}")
        
        # 打印一些统计信息
        print(f"\n=== 预测结果统计 ===")
        print(f"总帧数: {results['frame_count']}")
        print(f"姿态参数: {pose_pred.shape} (24个关节的旋转矩阵)")
        print(f"根节点位移: {tran_pred.shape}")
        print(f"关节位置: {joints_pred.shape}")
        print(f"足地接触: {contact_pred.shape}")
        print(f"坐标系已变换: {results['coordinate_transformed']}")
        print(f"方向已归一化: {results['orientation_normalized']}")
        
        return results
    
    def visualize_with_aitviewer(self, with_tran: bool=False):
        """使用aitviewer可视化预测结果"""
        try:
            pose_pred, tran_pred, joints_pred, contact_pred = self._evaluate_model()
            
            print("使用aitviewer开始可视化...")
            print(f"总帧数: {pose_pred.shape[0]}")
            print(f"姿态数据形状: {pose_pred.shape}")
            print(f"平移数据形状: {tran_pred.shape}")
            
            # 创建自定义viewer，设置fps为30
            viewer = AITViewerCustom(
                pose_pred=pose_pred,
                tran_pred=tran_pred,
                smpl_model=self.smpl_model,
                device=self.device,
                fps=30,
                window_size=(1920, 1080)
            )
            
            print("启动aitviewer界面...")
            print("可视化准备完成！")
            viewer.run()
            
        except Exception as e:
            print(f"aitviewer可视化失败: {e}")
            import traceback
            traceback.print_exc()
            print("建议使用 --save-only 选项来保存结果")
    
    def visualize(self, with_tran: bool=False):
        """可视化预测结果（使用aitviewer）"""
        return self.visualize_with_aitviewer(with_tran)
    
    def visualize_imu_data_processed(self, output_file: str="processed_imu_data_visualization.png"):
        """可视化处理后的IMU数据（降采样和除以acc_scale后）"""
        if not hasattr(self, 'data_dict'):
            print("错误：需要先调用 load_custom_data() 加载数据")
            return
        
        print("正在可视化处理后的IMU数据...")
        visualize_processed_imu_data(self.data_dict, self.combo, output_file)
        print(f"IMU数据可视化完成，保存到: {output_file}")


def visualize_processed_imu_data(data_dict, combo='lw_rp', output_file='processed_imu_data_visualization.png'):
    """
    可视化降采样和除以acc_scale后的IMU数据（输入模型前的数据）
    
    Args:
        data_dict: 从pt文件加载的数据字典
        combo: IMU组合方式
        output_file: 输出图片文件名
    """
    print(f"开始可视化处理后的IMU数据 (combo: {combo})")
    
    imu_data = data_dict['imu_data']
    
    # 检查combo是否有效
    if combo not in amass.combos:
        raise ValueError(f"无效的combo: {combo}. 支持的combo: {list(amass.combos.keys())}")
    
    combo_ids = amass.combos[combo]
    our_to_amass = {'lw': 0, 'rp': 3}
    
    # 提取数据长度
    T = len(imu_data['lw']['lin_acc_x'])
    
    # 准备全部5个IMU位置的数据
    all_acc = torch.zeros(T, 5, 3)
    all_ori = torch.zeros(T, 5, 3, 3)
    
    # 填入我们实际有的数据
    valid_positions = []
    for our_pos, imu_id in our_to_amass.items():
        if our_pos in imu_data and imu_id in combo_ids:  # 只处理combo中包含的IMU
            # 提取加速度数据
            acc_x = imu_data[our_pos]['lin_acc_x']
            acc_y = imu_data[our_pos]['lin_acc_y'] 
            acc_z = imu_data[our_pos]['lin_acc_z']
            acc_data = torch.stack([acc_x, acc_y, acc_z], dim=1)  # [T, 3]
            
            # 提取方向数据
            if 'ori' in imu_data[our_pos]:
                ori_data = imu_data[our_pos]['ori']  # [T, 3, 3]
            else:
                ori_roll = imu_data[our_pos]['ori_roll_deg']
                ori_pitch = imu_data[our_pos]['ori_pitch_deg']
                ori_yaw = imu_data[our_pos]['ori_yaw_deg']
                ori_data = convert_orientation_to_rotation_matrix(ori_roll, ori_pitch, ori_yaw)
            
            all_acc[:, imu_id] = acc_data
            all_ori[:, imu_id] = ori_data
            valid_positions.append((our_pos, imu_id))
    
    # 降采样到30fps
    all_acc_30fps = downsample_data(all_acc, 100, 30)
    all_ori_30fps = downsample_data(all_ori, 100, 30)
    
    T_30fps = all_acc_30fps.shape[0]
    
    # 按照模型输入的处理方式
    acc = torch.zeros(T_30fps, 5, 3)
    ori = torch.zeros(T_30fps, 5, 3, 3)
    
    # 只填入指定combo的数据，并进行缩放
    acc[:, combo_ids] = all_acc_30fps[:, combo_ids] / amass.acc_scale  # 除以acc_scale
    ori[:, combo_ids] = all_ori_30fps[:, combo_ids]
    
    # 提取combo对应的数据用于可视化
    acc_combo = acc[:, combo_ids]  # [T, len(combo_ids), 3]
    ori_combo = ori[:, combo_ids]  # [T, len(combo_ids), 3, 3]
    
    # 将旋转矩阵转换为欧拉角以便可视化
    ori_euler_combo = []
    
    def rotation_matrix_to_euler_degrees(rotation_matrices):
        """将旋转矩阵转换为欧拉角（度数）用于可视化 - 与process_imu_data.py保持一致"""
        n_frames = rotation_matrices.shape[0]
        euler_angles = np.zeros((n_frames, 3))
        
        for i in range(n_frames):
            R = rotation_matrices[i].numpy() if isinstance(rotation_matrices[i], torch.Tensor) else rotation_matrices[i]
            
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
            
            # 返回弧度值（稍后转换为度数）
            euler_angles[i] = [x, y, z]
        
        return euler_angles
    
    for i in range(len(combo_ids)):
        # 为每个IMU位置提取旋转矩阵并转换为欧拉角
        R = ori_combo[:, i]  # [T, 3, 3]
        euler_angles = rotation_matrix_to_euler_degrees(R)  # [T, 3] in radians
        ori_euler_combo.append(torch.tensor(euler_angles, dtype=torch.float32))  # [T, 3]
    
    # 创建时间轴 (30fps)
    time_axis = np.arange(T_30fps) / 30.0  # 秒
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形：每个IMU位置一行，每行2个子图（1个加速度 + 1个角度）
    num_imus = len(valid_positions)
    fig, axes = plt.subplots(num_imus, 2, figsize=(15, 4 * num_imus))
    if num_imus == 1:
        axes = axes.reshape(1, -1)
    
    # IMU位置名称映射
    imu_names = {0: '左手腕', 1: '右手腕', 2: '左大腿', 3: '右大腿', 4: '头部'}
    
    for imu_idx, (our_pos, imu_id) in enumerate(valid_positions):
        combo_idx = combo_ids.index(imu_id)
        imu_name = imu_names.get(imu_id, f'IMU_{imu_id}')
        
        # 获取该IMU的数据
        acc_data = acc_combo[:, combo_idx].numpy()  # [T, 3]
        ori_data = ori_euler_combo[combo_idx].numpy()  # [T, 3]
        
        # 计算统计信息
        acc_rms = np.sqrt(np.mean(acc_data ** 2, axis=0))
        acc_std = np.std(acc_data, axis=0)
        ori_rms = np.sqrt(np.mean(ori_data ** 2, axis=0))
        ori_std = np.std(ori_data, axis=0)
        
        # 绘制加速度（三轴在同一张图）
        axes[imu_idx, 0].plot(time_axis, acc_data[:, 0], 'r-', alpha=0.8, label='X轴', linewidth=1.5)
        axes[imu_idx, 0].plot(time_axis, acc_data[:, 1], 'g-', alpha=0.8, label='Y轴', linewidth=1.5)
        axes[imu_idx, 0].plot(time_axis, acc_data[:, 2], 'b-', alpha=0.8, label='Z轴', linewidth=1.5)
        axes[imu_idx, 0].set_title(f'{imu_name} - 线性加速度 (处理后)', fontsize=12)
        axes[imu_idx, 0].set_xlabel('时间 (秒)')
        axes[imu_idx, 0].set_ylabel('加速度 (除以acc_scale后)')
        axes[imu_idx, 0].legend()
        axes[imu_idx, 0].grid(True, alpha=0.3)
        
        # 绘制角度（三轴在同一张图，转换为度数）
        ori_data_deg = np.rad2deg(ori_data)
        ori_rms_deg = np.rad2deg(ori_rms)
        ori_std_deg = np.rad2deg(ori_std)
        
        axes[imu_idx, 1].plot(time_axis, ori_data_deg[:, 0], 'r-', alpha=0.8, label='Roll', linewidth=1.5)
        axes[imu_idx, 1].plot(time_axis, ori_data_deg[:, 1], 'g-', alpha=0.8, label='Pitch', linewidth=1.5)
        axes[imu_idx, 1].plot(time_axis, ori_data_deg[:, 2], 'b-', alpha=0.8, label='Yaw', linewidth=1.5)
        axes[imu_idx, 1].set_title(f'{imu_name} - 方向角度 (处理后)', fontsize=12)
        axes[imu_idx, 1].set_xlabel('时间 (秒)')
        axes[imu_idx, 1].set_ylabel('角度 (度)')
        axes[imu_idx, 1].legend()
        axes[imu_idx, 1].grid(True, alpha=0.3)
        
        # 在图上添加统计信息
        axes[imu_idx, 0].text(0.02, 0.98, f'RMS: X={acc_rms[0]:.3f}, Y={acc_rms[1]:.3f}, Z={acc_rms[2]:.3f}',
                            transform=axes[imu_idx, 0].transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[imu_idx, 1].text(0.02, 0.98, f'STD: R={ori_std_deg[0]:.1f}°, P={ori_std_deg[1]:.1f}°, Y={ori_std_deg[2]:.1f}°',
                            transform=axes[imu_idx, 1].transAxes, fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    
    # 添加整体标题
    fig.suptitle(f'处理后IMU数据可视化 - combo: {combo} (降采样30fps + 除以acc_scale)', fontsize=16, y=0.98)
    
    # 保存图片
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"处理后的IMU数据可视化已保存到: {output_file}")
    print(f"数据信息:")
    print(f"  - 原始帧率: 100fps -> 降采样后: 30fps")
    print(f"  - 总帧数: {T} -> {T_30fps}")
    print(f"  - 持续时间: {T_30fps/30.0:.2f} 秒")
    print(f"  - 加速度已除以acc_scale: {amass.acc_scale}")
    print(f"  - 使用的combo: {combo} (IMU IDs: {combo_ids})")
    print(f"  - 有效IMU位置: {[pos for pos, _ in valid_positions]}")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--model', type=str, default=None, help='模型权重文件路径')
    args.add_argument('--data', type=str, default='mobileposer/stag_raw_data/aligned_imu_data_normalized.pt', 
                     help='处理好的IMU数据文件路径')
    args.add_argument('--combo', type=str, default='lw_rp', 
                     help='IMU组合方式')
    args.add_argument('--with-tran', action='store_true', help='是否包含位移可视化')
    args.add_argument('--save-only', action='store_true', help='只保存结果，不进行实时可视化')
    args.add_argument('--output', type=str, default='prediction_results.pt', help='输出文件名')
    args.add_argument('--visualize-processed-imu', action='store_true', help='可视化处理后的IMU数据（降采样和除以acc_scale后）')
    args.add_argument('--imu-output', type=str, default='mobileposer/stag_raw_data/processed_imu_data_visualization.png', help='IMU数据可视化输出文件名')
    args = args.parse_args()

    # 检查combo是否有效
    if args.combo not in amass.combos:
        raise ValueError(f"无效的combo: {args.combo}. 支持的combo: {list(amass.combos.keys())}")

    # 检查数据文件是否存在
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"数据文件不存在: {args.data}")

    try:
        # 创建可视化器
        viewer = CustomViewer(model_path=args.model, combo=args.combo)
        
        # 加载数据
        viewer.load_custom_data(args.data)
        
        # 根据选项执行操作
        if args.visualize_processed_imu:
            # 可视化处理后的IMU数据
            viewer.visualize_imu_data_processed(args.imu_output)
        
        if args.save_only:
            # 只保存结果
            viewer.save_results(args.output)
        elif not args.visualize_processed_imu:
            # 使用aitviewer可视化（如果没有指定只可视化IMU数据）
            viewer.visualize(with_tran=args.with_tran)
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 