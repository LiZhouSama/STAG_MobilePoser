import os
import torch
import numpy as np
import sys

# 添加必要的路径
sys.path.insert(0, "/mnt/d/a_WORK/Projects/PhD/tasks/MobilePoser")

# aitviewer imports
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.spheres import Spheres  
from aitviewer.viewer import Viewer

# MobilePoser imports
import mobileposer.articulate as art


# 定义坐标系转换矩阵（Z-up to Y-up）
R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], 
                      [0.0, 0.0, 1.0]], dtype=torch.float32)


def load_smpl_model(smpl_model_path, device):
    """加载SMPL模型"""
    print(f"正在加载SMPL模型: {smpl_model_path}")
    if not os.path.exists(smpl_model_path):
        print(f"错误: SMPL模型路径未找到: {smpl_model_path}")
        raise FileNotFoundError(f"SMPL模型未找到: {smpl_model_path}")
    
    # 使用MobilePoser的ParametricModel
    smpl_model = art.model.ParametricModel(smpl_model_path, device=device)
    return smpl_model


def create_sample_pose(pose_type='tpose', device='cpu'):
    """
    创建示例姿态
    
    Args:
        pose_type: 姿态类型 ('tpose', 'apose', 'standing')
        device: 计算设备
    
    Returns:
        pose: SMPL姿态参数 [24, 3, 3] 旋转矩阵格式
    """
    
    # 创建单位旋转矩阵 [24, 3, 3]
    pose = torch.eye(3, device=device).unsqueeze(0).repeat(24, 1, 1)  # [24, 3, 3]
    
    if pose_type == 'tpose':
        # T-pose: 所有关节角度为0（默认单位矩阵）
        pass
        
    elif pose_type == 'apose':
        # A-pose: 手臂向下放
        # 左肩关节（joint 16）：向下旋转30度
        angle = torch.tensor(0.5, device=device)  # 约30度
        pose[16] = art.math.axis_angle_to_rotation_matrix(torch.tensor([0.0, 0.0, angle], device=device))
        # 右肩关节（joint 17）：向下旋转30度  
        pose[17] = art.math.axis_angle_to_rotation_matrix(torch.tensor([0.0, 0.0, -angle], device=device))
        
    elif pose_type == 'standing':
        # 自然站立姿态
        # 左肩关节稍微向前
        pose[16] = art.math.axis_angle_to_rotation_matrix(torch.tensor([0.2, 0.0, 0.3], device=device))
        # 右肩关节稍微向前
        pose[17] = art.math.axis_angle_to_rotation_matrix(torch.tensor([0.2, 0.0, -0.3], device=device))
        # 左肘关节稍微弯曲
        pose[18] = art.math.axis_angle_to_rotation_matrix(torch.tensor([0.0, 0.0, 0.3], device=device))
        # 右肘关节稍微弯曲
        pose[19] = art.math.axis_angle_to_rotation_matrix(torch.tensor([0.0, 0.0, -0.3], device=device))
    
    return pose


def generate_smpl_mesh(smpl_model, pose=None, trans=None, shape=None, device='cpu'):
    """
    生成SMPL mesh
    
    Args:
        smpl_model: 加载的SMPL模型（ParametricModel）
        pose: 姿态参数 [24, 3, 3] 旋转矩阵格式，默认为T-pose
        trans: 平移参数 [3]，默认为原点
        shape: 形状参数 [10]，默认为平均体型
        device: 计算设备
    
    Returns:
        verts: 顶点坐标 [Nv, 3]
        faces: 面信息 [Nf, 3]
    """
    
    with torch.no_grad():
        # 设置默认参数
        if pose is None:
            # T-pose: 单位矩阵
            pose = torch.eye(3, device=device).unsqueeze(0).repeat(24, 1, 1)  # [24, 3, 3]
        else:
            pose = pose.to(device)
            
        if trans is None:
            trans = torch.zeros(3, device=device)  # [3]
        else:
            trans = trans.to(device)
            
        if shape is None:
            shape = torch.zeros(10, device=device)  # [10] 
        else:
            shape = shape.to(device)
        
        # 使用ParametricModel的forward_kinematics方法
        _, _, verts = smpl_model.forward_kinematics(
            pose.unsqueeze(0),      # 添加batch维度 [1, 24, 3, 3]
            shape.unsqueeze(0),     # 添加batch维度 [1, 10]
            trans.unsqueeze(0),     # 添加batch维度 [1, 3]
            calc_mesh=True
        )
        
        verts = verts[0]  # 移除batch维度 [Nv, 3]
        faces = smpl_model.face  # [Nf, 3]
        
        # 应用坐标系转换
        verts_yup = torch.matmul(verts, R_yup.T.to(device))
        
        return verts_yup.cpu().numpy(), faces


def visualize_smpl_with_highlighted_vertices(smpl_model_path, vi_mask, pose_type='standing', device='cpu'):
    """
    可视化SMPL模型并突出显示特定顶点
    
    Args:
        smpl_model_path: SMPL模型文件路径
        vi_mask: 要突出显示的顶点索引 [N]
        pose_type: 姿态类型 ('tpose', 'apose', 'standing')
        device: 计算设备
    """
    
    # 加载SMPL模型
    smpl_model = load_smpl_model(smpl_model_path, device)
    
    # 创建姿态
    pose = create_sample_pose(pose_type, device)
    
    # 生成人体mesh
    verts, faces = generate_smpl_mesh(smpl_model, pose=pose, device=device)
    
    print(f"SMPL模型顶点数: {verts.shape[0]}")
    print(f"使用姿态类型: {pose_type}")
    print(f"要突出显示的顶点索引: {vi_mask.tolist()}")
    
    # 创建aitviewer实例
    viewer = Viewer()
    
    # 添加人体mesh
    human_mesh = Meshes(
        verts[None, ...],  # 添加时间维度 [1, Nv, 3]
        faces,             # [Nf, 3]
        name="SMPL-Human",
        color=(0.8, 0.8, 0.8, 0.8),  # 灰色半透明
        gui_affine=False,
        is_selectable=False
    )
    viewer.scene.add(human_mesh)
    
    # 提取要突出显示的顶点位置
    highlighted_positions = verts[vi_mask]  # [N, 3]
    
    # 定义不同颜色用于区分不同的顶点
    colors = [
        (1.0, 0.0, 0.0, 1.0),  # 红色
        (0.0, 1.0, 0.0, 1.0),  # 绿色
        (0.0, 0.0, 1.0, 1.0),  # 蓝色
        (1.0, 1.0, 0.0, 1.0),  # 黄色
        (1.0, 0.0, 1.0, 1.0),  # 紫色
        (0.0, 1.0, 1.0, 1.0),  # 青色
    ]
    
    # 为每个顶点创建球体来突出显示
    for i, (vertex_idx, pos) in enumerate(zip(vi_mask, highlighted_positions)):
        # 选择颜色（循环使用）
        color = colors[i % len(colors)]
        
        # 创建球体
        sphere = Spheres(
            pos[None, None, :],  # [1, 1, 3] - 单帧，单个球体
            name=f"Vertex-{vertex_idx}",
            color=color,
            radius=0.02,  # 球体半径
            gui_affine=False,
            is_selectable=True
        )
        viewer.scene.add(sphere)
        print(f"添加顶点 {vertex_idx} 的高亮球体，位置: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), 颜色: {color[:3]}")
    
    # 设置相机位置
    viewer.scene.camera.position = np.array([0.0, 1.0, 3.0])
    viewer.scene.camera.target = np.array([0.0, 0.8, 0.0])
    
    print("\n可视化准备完成！")
    print("控制说明:")
    print("  鼠标拖拽: 旋转视角")
    print("  鼠标滚轮: 缩放")
    print("  按住Shift+鼠标拖拽: 平移视角")
    print("  按住Ctrl+鼠标拖拽: 平移相机")
    print(f"  彩色球体标记了以下顶点: {vi_mask.tolist()}")
    print("  可以在aitviewer界面中选择和查看每个球体的详细信息")
    
    # 运行可视化
    viewer.run()


def main():
    """主函数"""
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # SMPL模型路径（根据您的项目结构调整）
    smpl_model_path = "mobileposer/smpl/basicmodel_m.pkl"
    
    # 检查模型文件是否存在
    if not os.path.exists(smpl_model_path):
        print(f"SMPL模型文件不存在: {smpl_model_path}")
        print("请确保SMPL模型文件路径正确")
        return
    
    # 要突出显示的顶点索引
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    
    # 验证顶点索引有效性（SMPL模型通常有6890个顶点）
    max_vertex_idx = 6889  # SMPL顶点索引从0开始
    if torch.any(vi_mask > max_vertex_idx) or torch.any(vi_mask < 0):
        print(f"警告: 部分顶点索引超出范围 [0, {max_vertex_idx}]")
        print(f"给定的顶点索引: {vi_mask.tolist()}")
        # 过滤有效的顶点索引
        vi_mask = vi_mask[(vi_mask >= 0) & (vi_mask <= max_vertex_idx)]
        print(f"过滤后的有效顶点索引: {vi_mask.tolist()}")
    
    if len(vi_mask) == 0:
        print("错误: 没有有效的顶点索引")
        return
    
    # 可选择不同的姿态类型：'tpose', 'apose', 'standing'
    pose_type = 'standing'  # 您可以修改为 'tpose' 或 'apose'
    
    try:
        # 开始可视化
        print(f"\n=== 开始可视化SMPL模型 ===")
        print(f"顶点索引: {vi_mask.tolist()}")
        print(f"姿态类型: {pose_type}")
        
        visualize_smpl_with_highlighted_vertices(smpl_model_path, vi_mask, pose_type, device)
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 