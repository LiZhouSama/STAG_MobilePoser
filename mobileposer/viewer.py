import os
import numpy as np
import torch
from tqdm import tqdm 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mobileposer.models import *
from mobileposer.config import *
from mobileposer.utils.model_utils import *
from mobileposer.viewers import SMPLViewer
from mobileposer.loader import DataLoader
import mobileposer.articulate as art

# aitviewer imports for new implementation
try:
    import sys
    sys.path.insert(0, "/mnt/d/a_WORK/Projects/PhD/tasks/MobilePoser/human_body_prior/src")
    from aitviewer.renderables.meshes import Meshes
    from aitviewer.renderables.spheres import Spheres
    from aitviewer.viewer import Viewer as AitViewerBase
    from aitviewer.scene.camera import Camera
    from moderngl_window.context.base import KeyModifiers
    from human_body_prior.body_model.body_model import BodyModel
    import imgui
    import pytorch3d.transforms as transforms
    HAS_AITVIEWER = True
except ImportError:
    HAS_AITVIEWER = False
    print("警告: aitviewer 相关库未安装，将回退到原始viewer")

# --- 定义 Z-up 到 Y-up 的旋转矩阵 ---
R_yup = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)


class AITViewerCustom(AitViewerBase):
    """自定义aitviewer用于可视化预测结果"""
    
    def __init__(self, pose_pred, tran_pred, pose_true, tran_true, smpl_model, device, with_tran=False, **kwargs):
        super().__init__(**kwargs)
        self.pose_pred = pose_pred.to(device)  # [T, 24, 3, 3]
        self.tran_pred = tran_pred.to(device) if with_tran else torch.zeros_like(tran_pred.to(device))  # [T, 3]
        self.pose_true = pose_true.to(device)  # [T, 24, 3, 3]
        self.tran_true = tran_true.to(device) if with_tran else torch.zeros_like(tran_true.to(device))  # [T, 3]
        self.smpl_model = smpl_model
        self.device = device
        self.with_tran = with_tran
        
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
            root_pos_pred = self.tran_pred[current_frame].cpu().numpy()  # [3]
            root_pos_true = self.tran_true[current_frame].cpu().numpy()  # [3]
            
            # 显示当前帧号
            imgui.text(f"Frame: {current_frame}")
            imgui.separator()
            
            # 显示预测的根关节三维位置
            imgui.text("Predicted Root Position (m):")
            imgui.text(f"X: {root_pos_pred[0]:.2f}")
            imgui.text(f"Y: {root_pos_pred[1]:.2f}")
            imgui.text(f"Z: {root_pos_pred[2]:.2f}")
            
            imgui.separator()
            
            # 显示真实的根关节三维位置（原始位置，未偏移）
            imgui.text("Ground Truth Root Position (m):")
            imgui.text(f"X: {root_pos_true[0]:.2f}")
            imgui.text(f"Y: {root_pos_true[1]:.2f}")
            imgui.text(f"Z: {root_pos_true[2]:.2f}")
            
            # 显示可视化中的实际位置（包含偏移）
            imgui.text("GT Visualization Position (m):")
            imgui.text(f"X: {root_pos_true[0] + 1.2:.2f}")
            imgui.text(f"Y: {root_pos_true[1]:.2f}")
            imgui.text(f"Z: {root_pos_true[2]:.2f}")
            
            imgui.separator()
            
            # 添加一些额外信息
            imgui.text(f"Total Frames: {self.tran_pred.shape[0]}")
            imgui.text(f"Progress: {current_frame + 1}/{self.tran_pred.shape[0]}")
            imgui.text(f"With Translation: {self.with_tran}")
            
            # 显示根关节移动距离（相对于第一帧）
            if current_frame > 0:
                initial_pos_pred = self.tran_pred[0].cpu().numpy()
                distance_pred = np.linalg.norm(root_pos_pred - initial_pos_pred)
                initial_pos_true = self.tran_true[0].cpu().numpy()
                distance_true = np.linalg.norm(root_pos_true - initial_pos_true)
                imgui.text(f"Pred Distance: {distance_pred:.2f} m")
                imgui.text(f"True Distance: {distance_true:.2f} m")
        else:
            imgui.text("Invalid Frame")
        
        imgui.end()
    
    def _generate_all_frames_mesh(self, pose_data, tran_data, color):
        """生成所有帧的网格数据"""
        total_frames = pose_data.shape[0]
        print(f"正在生成 {total_frames} 帧的网格数据...")
        
        all_verts = []
        
        with torch.no_grad():
            for frame_idx in range(total_frames):
                if frame_idx % 50 == 0:
                    print(f"  处理第 {frame_idx+1}/{total_frames} 帧...")
                    
                pose_frame = pose_data[frame_idx]  # [24, 3, 3] 旋转矩阵
                trans_frame = tran_data[frame_idx]  # [3]
                
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
        try:
            # 生成预测结果的网格数据
            all_verts_pred, faces = self._generate_all_frames_mesh(
                self.pose_pred, self.tran_pred, (0.2, 0.6, 0.8, 0.9)  # 蓝色
            )
            
            # 生成真值的网格数据，添加位移偏移避免重叠
            # 为真值添加固定的X轴偏移，但保持相对位移变化
            tran_true_offset = self.tran_true.clone()
            tran_true_offset[:, 0] += 1.2  # 在X轴方向偏移1.2米，避免重叠
            
            all_verts_true, _ = self._generate_all_frames_mesh(
                self.pose_true, tran_true_offset, (0.8, 0.2, 0.2, 0.9)  # 红色
            )
            
            # 创建预测结果网格对象
            pred_mesh = Meshes(
                all_verts_pred,  # [T, Nv, 3]
                faces,          # [Nf, 3]
                name="Prediction",
                color=(0.2, 0.6, 0.8, 0.9),  # 蓝色
                gui_affine=False,
                is_selectable=False
            )
            
            # 创建真值网格对象
            true_mesh = Meshes(
                all_verts_true,  # [T, Nv, 3]
                faces,          # [Nf, 3]
                name="Ground Truth",
                color=(0.8, 0.2, 0.2, 0.9),  # 红色
                gui_affine=False,
                is_selectable=False
            )
            
            # 添加到场景
            self.scene.add(pred_mesh)
            self.scene.add(true_mesh)
            
            print(f"成功添加预测人体网格和真值网格，顶点形状: {all_verts_pred.shape}")
            print("控制说明:")
            print("  空格: 播放/暂停")
            print("  左/右箭头: 前/后一帧")
            print("  鼠标拖拽: 旋转视角")
            print("  鼠标滚轮: 缩放")
            print("  蓝色: 预测结果")
            print("  红色: 真值")
            
        except Exception as e:
            print(f"设置可视化时出错: {e}")
            import traceback
            traceback.print_exc()


class Viewer:
    def __init__(self, dataset: str='imuposer', seq_num: int=0, combo: str='lw_rp'):
        """Viewer class for visualizing pose."""
        # load models 
        self.device = model_config.device
        self.model = load_model(paths.weights_file).to(self.device).eval()

        # setup dataloader
        self.dataloader = DataLoader(dataset, combo=combo, device=self.device)
        self.data = self.dataloader.load_data(seq_num)
        
        # 加载SMPL模型（用于aitviewer）
        if HAS_AITVIEWER:
            try:
                body_model_path = paths.body_model_file
                self.smpl_model = BodyModel(
                    bm_fname=str(body_model_path),
                    num_betas=16,
                    model_type='smplh'
                ).to(self.device)
                print(f"成功加载SMPL模型: {body_model_path}")
            except Exception as e:
                print(f"加载SMPL模型失败: {e}")
                self.smpl_model = None
        else:
            self.smpl_model = None
    
    def _evaluate_model(self):
        """Evaluate the model."""
        data = self.data['imu']
        if getenv('ONLINE'):
            # online model evaluation (slower)
            pose, joints, tran, contact = [torch.stack(_) for _ in zip(*[self.model.forward_online(f) for f in tqdm(data)])]
        else:
            # offline model evaluation  
            with torch.no_grad():
                pose, joints, tran, contact = self.model.forward_offline(data.unsqueeze(0), [data.shape[0]]) 
        return pose, tran, joints, contact

    def view(self, with_tran: bool=False):
        """View the pose using aitviewer if available, otherwise fallback to original viewer."""
        pose_t, tran_t = self.data['pose'], self.data['tran']
        pose_p, tran_p, _, _ = self._evaluate_model()
        
        # 尝试使用aitviewer
        if HAS_AITVIEWER and self.smpl_model is not None:
            try:
                print("使用aitviewer进行可视化...")
                print(f"总帧数: {pose_p.shape[0]}")
                print(f"预测姿态数据形状: {pose_p.shape}")
                print(f"真值姿态数据形状: {pose_t.shape}")
                print(f"预测平移数据形状: {tran_p.shape}")
                print(f"真值平移数据形状: {tran_t.shape}")
                
                # 展平并重塑数据
                pose_p = pose_p.squeeze(0) if pose_p.dim() == 4 else pose_p  # [T, 24, 3, 3]
                pose_t = pose_t.squeeze(0) if pose_t.dim() == 4 else pose_t  # [T, 24, 3, 3]
                tran_p = tran_p.squeeze(0) if tran_p.dim() == 3 else tran_p  # [T, 3]
                tran_t = tran_t.squeeze(0) if tran_t.dim() == 3 else tran_t  # [T, 3]
                
                # 创建自定义viewer，设置fps为30
                viewer = AITViewerCustom(
                    pose_pred=pose_p,
                    tran_pred=tran_p,
                    pose_true=pose_t,
                    tran_true=tran_t,
                    smpl_model=self.smpl_model,
                    device=self.device,
                    with_tran=with_tran,
                    fps=30,
                    window_size=(1920, 1080)
                )
                
                print("启动aitviewer界面...")
                viewer.run()
                return
                
            except Exception as e:
                print(f"aitviewer可视化失败: {e}")
                import traceback
                traceback.print_exc()
                print("回退到原始viewer...")
        
        # 回退到原始viewer
        print("使用原始viewer进行可视化...")
        viewer = SMPLViewer()
        viewer.view(pose_p, tran_p, pose_t, tran_t, with_tran=with_tran)


class AitViewer(Viewer):
    """专门使用aitviewer的Viewer类"""
    
    def __init__(self, dataset: str='imuposer', seq_num: int=0, combo: str='lw_rp'):
        """强制使用aitviewer的Viewer类"""
        if not HAS_AITVIEWER:
            raise ImportError("aitviewer库未安装，无法使用AitViewer")
        
        super().__init__(dataset, seq_num, combo)
        
        if self.smpl_model is None:
            raise RuntimeError("SMPL模型加载失败，无法使用aitviewer")
    
    def view(self, with_tran: bool=False):
        """强制使用aitviewer进行可视化"""
        pose_t, tran_t = self.data['pose'], self.data['tran']
        pose_p, tran_p, _, _ = self._evaluate_model()
        
        print("使用aitviewer进行可视化...")
        print(f"总帧数: {pose_p.shape[0]}")
        print(f"预测姿态数据形状: {pose_p.shape}")
        print(f"真值姿态数据形状: {pose_t.shape}")
        
        # 展平并重塑数据
        pose_p = pose_p.squeeze(0) if pose_p.dim() == 4 else pose_p  # [T, 24, 3, 3]
        pose_t = pose_t.squeeze(0) if pose_t.dim() == 4 else pose_t  # [T, 24, 3, 3]
        tran_p = tran_p.squeeze(0) if tran_p.dim() == 3 else tran_p  # [T, 3]
        tran_t = tran_t.squeeze(0) if tran_t.dim() == 3 else tran_t  # [T, 3]
        
        # 创建自定义viewer
        viewer = AITViewerCustom(
            pose_pred=pose_p,
            tran_pred=tran_p,
            pose_true=pose_t,
            tran_true=tran_t,
            smpl_model=self.smpl_model,
            device=self.device,
            with_tran=with_tran,
            fps=30,
            window_size=(1920, 1080)
        )
        
        print("启动aitviewer界面...")
        viewer.run()
