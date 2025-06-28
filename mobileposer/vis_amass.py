import os
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser

import sys
sys.path.insert(0, "/mnt/d/a_WORK/Projects/PhD/tasks/MobilePoser/human_body_prior/src")

# aitviewer imports
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.spheres import Spheres
from aitviewer.viewer import Viewer as AITViewer
from aitviewer.scene.camera import Camera
from moderngl_window.context.base import KeyModifiers
from human_body_prior.body_model.body_model import BodyModel
import imgui
import pytorch3d.transforms as transforms

# MobilePoser imports
from mobileposer.config import *
from mobileposer.utils.model_utils import *
from mobileposer.loader import DataLoader
import mobileposer.articulate as art


# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)


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


def get_dataset_info(dataset):
    """获取数据集信息"""
    try:
        dataloader = DataLoader(dataset, combo='lw_rp', device='cpu')
        seq_num = 0
        while True:
            try:
                dataloader.load_data(seq_num)
                seq_num += 1
                if seq_num > 1000:  # 防止无限循环
                    break
            except:
                break
        
        max_seq = max(0, seq_num - 1)
        return max_seq
        
    except Exception as e:
        print(f"获取数据集信息时出错: {e}")
        return 0


def generate_all_frames_mesh(pose_matrices, trans_vec, smpl_model, device):
    """
    生成所有帧的网格数据 (参考vis.py的方法)
    
    Args:
        pose_matrices: [T, 24, 3, 3] 旋转矩阵
        trans_vec: [T, 3] 平移向量
        smpl_model: SMPL模型
        device: 计算设备
    
    Returns:
        all_verts: [T, Nv, 3] 顶点数据 (numpy)
        faces: [Nf, 3] 面数据 (numpy)
    """
    total_frames = pose_matrices.shape[0]
    print(f"正在生成 {total_frames} 帧的网格数据...")
    
    all_verts = []
    
    with torch.no_grad():
        for frame_idx in range(total_frames):
            if frame_idx % 50 == 0:
                print(f"  处理第 {frame_idx+1}/{total_frames} 帧...")
                
            pose_frame = pose_matrices[frame_idx].to(device)  # [24, 3, 3] 旋转矩阵
            trans_frame = trans_vec[frame_idx].to(device)  # [3]
            
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
                body_pose = smpl_model(**smpl_input)
                verts = body_pose.v[0]  # [Nv, 3]
                
                # 应用坐标系转换
                global R_yup
                verts_yup = torch.matmul(verts, R_yup.T.to(device))
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
    faces = smpl_model.f.cpu().numpy() if isinstance(smpl_model.f, torch.Tensor) else smpl_model.f
    
    print(f"网格数据生成完成，形状: {all_verts_tensor.shape}")
    return all_verts_tensor.numpy(), faces


def visualize_sequence_comparison(viewer, gt_pose, gt_trans, pred_pose, pred_trans, smpl_model, device):
    """在aitviewer场景中可视化单个序列的数据（真值和预测）"""
    
    # 清除现有场景中的网格 (参考vis2.py方法)
    try:
        nodes_to_remove = [
            node for node in viewer.scene.collect_nodes()
            if hasattr(node, 'name') and node.name is not None and 
               (node.name.startswith("GT-") or node.name.startswith("Pred-"))
        ]
        
        for node_to_remove in nodes_to_remove:
            try:
                viewer.scene.remove(node_to_remove)
            except Exception as e:
                print(f"Error removing node '{node_to_remove.name}': {e}")
    except Exception as e:
        print(f"Error during scene clearing: {e}")
    
    # 生成真值网格 (绿色, 不偏移)
    try:
        gt_verts, faces = generate_all_frames_mesh(gt_pose, gt_trans, smpl_model, device)
        
        # 创建真值网格对象
        gt_mesh = Meshes(
            gt_verts,  # [T, Nv, 3]
            faces,     # [Nf, 3]
            name="GT-Human",
            color=(0.1, 0.8, 0.3, 0.8),  # 绿色
            gui_affine=False,
            is_selectable=False
        )
        viewer.scene.add(gt_mesh)
        print(f"添加真值人体网格，顶点形状: {gt_verts.shape}")
        
    except Exception as e:
        print(f"生成真值网格时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成预测网格 (红色, x轴偏移)
    try:
        pred_verts, _ = generate_all_frames_mesh(pred_pose, pred_trans, smpl_model, device)
        
        # 添加偏移以区分预测和真值
        pred_offset = np.array([0.0, 0.0, 0.0])  # x轴偏移1.5米
        pred_verts_offset = pred_verts + pred_offset
        
        # 创建预测网格对象
        pred_mesh = Meshes(
            pred_verts_offset,  # [T, Nv, 3]
            faces,              # [Nf, 3]
            name="Pred-Human",
            color=(0.9, 0.2, 0.2, 0.8),  # 红色
            gui_affine=False,
            is_selectable=False
        )
        viewer.scene.add(pred_mesh)
        print(f"添加预测人体网格，顶点形状: {pred_verts.shape}")
        
    except Exception as e:
        print(f"生成预测网格时出错: {e}")
        import traceback
        traceback.print_exc()


class InteractiveMobilePoserViewer(AITViewer):
    """交互式MobilePoser数据可视化器 (参考vis2.py的结构)"""
    
    def __init__(self, dataset='CMU_mini', combo='lw_rp', **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.combo = combo
        self.current_seq = 0
        
        # 设置设备和模型 (参考example.py)
        self.device = model_config.device
        self.model = load_model(paths.weights_file).to(self.device).eval()
        
        # 设置数据加载器
        self.dataloader = DataLoader(dataset, combo=combo, device=self.device)
        
        # 加载SMPL模型
        body_model_path = paths.body_model_file
        self.smpl_model = load_smpl_model(body_model_path, self.device)
        
        # 获取数据集信息
        self.max_seq = get_dataset_info(dataset)
        
        print(f"数据集: {dataset}, 序列数量: {self.max_seq + 1}, 当前序列: {self.current_seq}")
        
        # 设置初始相机位置
        self.scene.camera.position = np.array([0.0, 1.0, 3.0])
        self.scene.camera.target = np.array([0.75, 0.8, 0.0])  # 对准偏移后的中间区域
        
        # 初始化可视化
        self.visualize_current_sequence()
    
    def _evaluate_model(self, data):
        """评估模型 (参考example.py的方法)"""
        imu_data = data['imu']
        with torch.no_grad():
            pose, joints, tran, contact = self.model.forward_offline(
                imu_data.unsqueeze(0), [imu_data.shape[0]]
            )
        return pose.squeeze(0), tran.squeeze(0), joints.squeeze(0), contact.squeeze(0)
    
    def visualize_current_sequence(self):
        """可视化当前序列"""
        if not (0 <= self.current_seq <= self.max_seq):
            print("序列索引超出范围")
            return
        
        try:
            # 加载当前序列数据 (使用DataLoader)
            data = self.dataloader.load_data(self.current_seq)
            
            # 获取真值数据
            gt_pose = data['pose']  # [T, 24, 3, 3]
            gt_trans = data['tran'] # [T, 3]
            
            # 获取预测数据 (参考example.py)
            pred_pose, pred_trans, pred_joints, pred_contact = self._evaluate_model(data)
            
            print(f"可视化序列 {self.current_seq}: GT帧数={gt_pose.shape[0]}, 预测帧数={pred_pose.shape[0]}")
            
            # 可视化对比
            visualize_sequence_comparison(
                self, gt_pose, gt_trans, pred_pose, pred_trans, 
                self.smpl_model, self.device
            )
            
            # 更新标题
            self.title = f"MobilePoser Sequence {self.current_seq}/{self.max_seq} | 帧数: {gt_pose.shape[0]} | q/e:±1, Ctrl+q/e:±10"
            
        except Exception as e:
            print(f"可视化序列 {self.current_seq} 时出错: {e}")
            import traceback
            traceback.print_exc()
            self.title = f"Error visualizing sequence {self.current_seq}"
    
    def key_event(self, key, action, modifiers):
        """处理键盘事件 (参考vis2.py的方法)"""
        # 调用父类方法
        super().key_event(key, action, modifiers)
        
        # 检查ImGui是否需要键盘输入
        io = imgui.get_io()
        if self.render_gui and (io.want_capture_keyboard or io.want_text_input):
            return
        
        # 检查按键按下事件
        is_press = action == self.wnd.keys.ACTION_PRESS
        
        if is_press:
            ctrl_pressed = modifiers.ctrl
            
            if key == self.wnd.keys.Q:
                if ctrl_pressed:
                    # Ctrl + Q: 后退10个序列
                    step = 10
                    new_seq = max(0, self.current_seq - step)
                    if new_seq != self.current_seq:
                        self.current_seq = new_seq
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0
                        print(f"后退10个序列到: {self.current_seq}")
                    else:
                        print("已经在最前面的序列。")
                else:
                    # Q: 后退1个序列
                    if self.current_seq > 0:
                        self.current_seq -= 1
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0
                        print(f"后退到序列: {self.current_seq}")
                    else:
                        print("已经在第一个序列。")
                        
            elif key == self.wnd.keys.E:
                if ctrl_pressed:
                    # Ctrl + E: 前进10个序列
                    step = 10
                    new_seq = min(self.max_seq, self.current_seq + step)
                    if new_seq != self.current_seq:
                        self.current_seq = new_seq
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0
                        print(f"前进10个序列到: {self.current_seq}")
                    else:
                        print("已经在最后面的序列。")
                else:
                    # E: 前进1个序列
                    if self.current_seq < self.max_seq:
                        self.current_seq += 1
                        self.visualize_current_sequence()
                        self.scene.current_frame_id = 0
                        print(f"前进到序列: {self.current_seq}")
                    else:
                        print("已经在最后一个序列。")


class MobilePoserVisualizer:
    """MobilePoser可视化器主类 (简化版，直接使用DataLoader)"""
    
    def __init__(self, dataset='CMU_mini', combo='lw_rp', seq_num=0):
        """初始化可视化器"""
        self.dataset = dataset
        self.combo = combo
        self.seq_num = seq_num
        
        # 检查combo是否有效 (参考example.py)
        combos = amass.combos.keys()
        if combo not in combos:
            raise ValueError(f"Invalid combo: {combo}. Must be one of {combos}")
        
        print(f"数据集: {dataset}")
        print(f"使用combo: {combo}")
        print(f"初始序列: {seq_num}")
    
    def view_interactive(self):
        """启动交互式可视化"""
        try:
            print("启动交互式可视化...")
            print("控制说明:")
            print("  q/e: 前进/后退 1个序列")
            print("  Ctrl+q/e: 前进/后退 10个序列")
            print("  空格: 播放/暂停动画")
            print("  左/右箭头: 前/后一帧")
            print("  鼠标拖拽: 旋转视角")
            print("  鼠标滚轮: 缩放")
            
            # 创建交互式查看器
            viewer = InteractiveMobilePoserViewer(
                dataset=self.dataset,
                combo=self.combo,
                fps=30,
                window_size=(1920, 1080)
            )
            
            # 设置初始序列
            viewer.current_seq = self.seq_num
            viewer.visualize_current_sequence()
            
            # 运行可视化
            viewer.run()
            
        except Exception as e:
            print(f"交互式可视化失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    parser = ArgumentParser(description='MobilePoser AMASS数据可视化工具')
    parser.add_argument('--dataset', type=str, default='RUN', help='数据集名称')
    parser.add_argument('--combo', type=str, default='lw_rp', help='IMU组合方式')
    parser.add_argument('--seq-num', type=int, default=0, help='初始序列编号')
    args = parser.parse_args()

    try:
        # 创建可视化器 (参考example.py的参数)
        visualizer = MobilePoserVisualizer(
            dataset=args.dataset, 
            combo=args.combo, 
            seq_num=args.seq_num
        )
        
        # 启动交互式可视化
        visualizer.view_interactive()
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc() 