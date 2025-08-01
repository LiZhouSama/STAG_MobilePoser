from aitviewer.configuration import CONFIG as C

# Windows路径配置
C.update_conf({
    "run_animations": True,
    "smplx_models": r"/mnt/d/a_WORK/Projects/PhD/datasets/smpl_models",  # 根据实际路径修改
})

import numpy as np
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.plane import ChessboardPlane
import torch
import trimesh
from aitviewer.renderables.meshes import Meshes
from scipy.spatial.transform import Rotation as R
import os
import sys
sys.path.insert(0, "/mnt/d/a_WORK/Projects/PhD/tasks/MobilePoser/human_body_prior/src")
from human_body_prior.body_model.body_model import BodyModel
import pytorch3d.transforms as transforms

def ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    else:
        raise TypeError(f"不支持的类型: {type(x)}")

def add_marker(viewer, positions, color, name):
    """在viewer中添加一个动画球体轨迹"""
    positions = np.asarray(positions)  # 保证是numpy数组
    base_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.03)
    base_vertices = np.array(base_sphere.vertices, dtype=np.float32)
    faces = np.array(base_sphere.faces, dtype=np.int32)
    n_frames = positions.shape[0]
    n_verts = base_vertices.shape[0]
    animated_vertices = np.zeros((n_frames, n_verts, 3), dtype=np.float32)
    for i in range(n_frames):
        animated_vertices[i] = base_vertices + positions[i]  # 直接加即可
    marker = Meshes(
        name=name,
        vertices=animated_vertices,
        faces=faces,
        color=np.array(color)
    )
    viewer.scene.add(marker)

def visualize_all_markers(viewer, data, smpl_layer, gt_poses_body, gt_poses_root, gt_trans,
                           root_trans, body_root_trans):
    # 1. SMPL人体
    sequence_gt = SMPLSequence(
        smpl_layer=smpl_layer,
        poses_body=gt_poses_body,
        poses_root=gt_poses_root,
        trans=gt_trans,
        name='Ground Truth',
        color=(0.2, 0.2, 1.0, 0.8),
        device='cpu'
    )
    viewer.scene.add(sequence_gt)

    # 2. head_trans 绿色
    # add_marker(viewer, head_trans, [0,1,0,1], "HeadTrans")
    # 3. root_trans 红色
    add_marker(viewer, root_trans, [1,0,0,1], "RootTrans")
    # 4. body_root_trans 蓝色
    add_marker(viewer, body_root_trans, [0,0,1,1], "BodyRootTrans")
    # 5. body_head_trans 黄色
    # add_marker(viewer, body_head_trans, [1,1,0,1], "BodyHeadTrans")

    # 6. 棋盘格地面
    chessboard_plane = ChessboardPlane(
        side_length=20.0,
        n_tiles=20,
        color1=(0.5, 0.5, 0.5, 1.0),
        color2=(1.0, 1.0, 1.0, 1.0),
        plane="xz",
        position=(0, 0, 0),
        tiling=True,
        name="ChessboardPlane"
    )
    viewer.scene.floor.enabled = False
    viewer.scene.add(chessboard_plane)

    # 7. 可视化全局坐标系原点和坐标轴
    # 原点小球
    origin_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
    origin_marker = Meshes(
        name="Origin",
        vertices=np.expand_dims(origin_sphere.vertices, axis=0),
        faces=origin_sphere.faces,
        color=np.array([1, 1, 1, 1])
    )
    viewer.scene.add(origin_marker)
    # 坐标轴箭头
    def add_axis_arrow(start, end, color, name):
        # 用细长圆柱体+锥体表示箭头
        # 圆柱体
        vec = np.array(end) - np.array(start)
        length = np.linalg.norm(vec)
        if length < 1e-6:
            return
        direction = vec / length
        # 创建圆柱体
        cylinder = trimesh.creation.cylinder(radius=0.015, height=length-0.08, sections=12)
        # 沿z轴，需旋转到direction
        # 先平移到start
        cylinder.apply_translation([0, 0, (length-0.08)/2])
        # 旋转
        z_axis = np.array([0, 0, 1])
        rot, _ = trimesh.geometry.align_vectors(z_axis, direction, return_angle=True)
        cylinder.apply_transform(rot)
        cylinder.apply_translation(start)
        # 锥体箭头
        cone = trimesh.creation.cone(radius=0.03, height=0.08, sections=12)
        cone.apply_translation([0, 0, 0.08/2])
        cone.apply_translation([0, 0, length-0.08])
        cone.apply_transform(rot)
        cone.apply_translation(start)
        # 合并
        arrow_mesh = trimesh.util.concatenate([cylinder, cone])
        arrow_marker = Meshes(
            name=name,
            vertices=np.expand_dims(arrow_mesh.vertices, axis=0),
            faces=arrow_mesh.faces,
            color=np.array(color)
        )
        viewer.scene.add(arrow_marker)
    # X轴(红)
    add_axis_arrow([0,0,0], [0.5,0,0], [1,0,0,1], "X_Axis")
    # Y轴(绿)
    add_axis_arrow([0,0,0], [0,0.5,0], [0,1,0,1], "Y_Axis")
    # Z轴(蓝)
    add_axis_arrow([0,0,0], [0,0,0.5], [0,0,1,1], "Z_Axis")

    # 刷新视角
    if hasattr(viewer, 'reset_camera'):
        viewer.reset_camera()
    if hasattr(viewer, 'show') and callable(viewer.show):
        viewer.show()


if __name__ == "__main__":
    file_path = r'LAN_TEST/CMU_mini.pt'
    data = torch.load(file_path, map_location='cpu', weights_only=False)
    
    data_seq_0 = {k: v[0] for k, v in data.items()}
    root_ori = transforms.matrix_to_axis_angle(data_seq_0['pose'][:,0,:,:])
    pose = transforms.matrix_to_axis_angle(data_seq_0['pose'][:,1:22,:,:]).reshape(-1, 63)
    trans = data_seq_0['tran']
    shape = data_seq_0['shape']
    body_parms = {}
    body_parms['root_orient'] = torch.zeros_like(root_ori)
    body_parms['pose_body'] = torch.zeros_like(pose)
    body_parms['trans'] = torch.zeros_like(trans)
    body_parms['betas'] = shape.expand(trans.shape[0], 10)
    # body_parms = normalize_smpl_params(body_parms, offset_floor_height)
    
    bm_fname_male = "/mnt/d/a_WORK/Projects/PhD/datasets/smpl_models/smplx/SMPLX_MALE.npz"

    num_betas = 10  # number of body parameters
    bm_male = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
        model_type='smplx'
    ).cpu()

    # 计算SMPL关节点
    with torch.no_grad():
        body_pose_world = bm_male(
            **{
                k: v if isinstance(v, torch.Tensor) else torch.tensor(v)
                for k, v in body_parms.items()
            }
        )
        position_global_full_gt_world = body_pose_world.Jtr[:, :22, :].cpu().numpy()
    root_trans = body_parms['trans']
    if isinstance(root_trans, torch.Tensor):
        root_trans = root_trans.cpu().numpy()
    body_root_trans = position_global_full_gt_world[:, 0, :]
    body_head_trans = position_global_full_gt_world[:, 15, :]


    if isinstance(root_trans, torch.Tensor):
        root_trans = root_trans.cpu().numpy()
    if isinstance(body_root_trans, torch.Tensor):
        body_root_trans = body_root_trans.cpu().numpy()


    # SMPL参数也要旋转
    gt_poses_body = ensure_tensor(body_parms['pose_body'])
    gt_poses_root = ensure_tensor(body_parms['root_orient'])
    gt_trans = ensure_tensor(body_parms['trans'])
    if gt_poses_body.shape[1] > 63:
        gt_poses_body = gt_poses_body[:, :63]
    if gt_poses_root.shape[1] > 3:
        gt_poses_root = gt_poses_root[:, :3]

    v = Viewer()
    v.run_animations = True
    visualize_all_markers(v, data, SMPLLayer(model_type='smplx', gender='male', num_betas=10, device='cpu'),
                         gt_poses_body, gt_poses_root, gt_trans, root_trans, body_root_trans)
    v.run()