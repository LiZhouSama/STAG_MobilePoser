# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import argparse
import os
import numpy as np
import torch

from mojito.human_body_prior.body_model.body_model import BodyModel
from mojito.human_body_prior.tools.rotation_tools import aa2matrot, local2global_pose
from tqdm import tqdm
from mojito.utils import utils_transform
import glob
import shutil
from sklearn.cluster import DBSCAN
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11, 
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]

DISCARD_TERRAIN_SEQUENCES = True # throw away sequences where the person steps onto objects (determined by a heuristic)
DISCARD_SHORTER_THAN = 1.0 # seconds

# for determining floor height
FLOOR_VEL_THRESH = 0.005
FLOOR_HEIGHT_OFFSET = 0.01
# for determining contacts
CONTACT_VEL_THRESH = 0.005 #0.015
CONTACT_TOE_HEIGHT_THRESH = 0.04
CONTACT_ANKLE_HEIGHT_THRESH = 0.08
# for determining terrain interaction
TERRAIN_HEIGHT_THRESH = 0.04 # if static toe is above this height
ROOT_HEIGHT_THRESH = 0.04 # if maximum "static" root height is more than this + root_floor_height
CLUSTER_SIZE_THRESH = 0.25 # if cluster has more than this faction of fps (30 for 120 fps)

def detect_joint_contact(body_joint_seq, joint_name, floor_height, vel_thresh, height_thresh):
    # calc velocity
    joint_seq = body_joint_seq[:, SMPL_JOINTS[joint_name], :]
    joint_vel = np.linalg.norm(joint_seq[1:] - joint_seq[:-1], axis=1)
    joint_vel = np.append(joint_vel, joint_vel[-1])
    # determine contact by velocity
    joint_contact = joint_vel < vel_thresh
    # compute heights
    joint_heights = joint_seq[:, 2] - floor_height
    # compute contact by vel + height
    joint_contact = np.logical_and(joint_contact, joint_heights < height_thresh)

    return joint_contact

def determine_floor_height_and_contacts(body_joint_seq, fps):
    '''
    Input: body_joint_seq N x 21 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    '''
    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, SMPL_JOINTS['hips'], :]
    left_toe_seq = body_joint_seq[:, SMPL_JOINTS['leftToeBase'], :]
    right_toe_seq = body_joint_seq[:, SMPL_JOINTS['rightToeBase'], :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]


    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < FLOOR_VEL_THRESH]
    left_static_inds = all_inds[left_toe_vel < FLOOR_VEL_THRESH]
    right_static_foot_heights = right_toe_heights[right_toe_vel < FLOOR_VEL_THRESH]
    right_static_inds = all_inds[right_toe_vel < FLOOR_VEL_THRESH]

    all_static_foot_heights = np.append(left_static_foot_heights, right_static_foot_heights)
    all_static_inds = np.append(left_static_inds, right_static_inds)


    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(all_static_foot_heights.reshape(-1, 1))
        all_labels = np.unique(clustering.labels_)
        # print(all_labels)
        min_median = min_root_median = float('inf')
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(all_static_inds[clustering.labels_ == cur_label]) # inds in the original sequence that correspond to this cluster
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        floor_height = min_median 
        offset_floor_height = floor_height - FLOOR_HEIGHT_OFFSET # toe joint is actually inside foot mesh a bit

        # if DISCARD_TERRAIN_SEQUENCES:
        #     # print(min_median + TERRAIN_HEIGHT_THRESH)
        #     # print(min_root_median + ROOT_HEIGHT_THRESH)
        #     for cluster_root_height, cluster_height, cluster_size in zip (cluster_root_heights, cluster_heights, cluster_sizes):
        #         root_above_thresh = cluster_root_height > (min_root_median + ROOT_HEIGHT_THRESH)
        #         toe_above_thresh = cluster_height > (min_median + TERRAIN_HEIGHT_THRESH)
        #         cluster_size_above_thresh = cluster_size > int(CLUSTER_SIZE_THRESH*fps)
        #         if root_above_thresh and toe_above_thresh and cluster_size_above_thresh:
        #             discard_seq = True
        #             print('DISCARDING sequence based on terrain interaction!')
        #             break
    else:
        floor_height = offset_floor_height = 0.0

    # now find contacts (feet are below certain velocity and within certain range of floor)
    # compute heel velocities
    left_heel_seq = body_joint_seq[:, SMPL_JOINTS['leftFoot'], :]
    right_heel_seq = body_joint_seq[:, SMPL_JOINTS['rightFoot'], :]
    left_heel_vel = np.linalg.norm(left_heel_seq[1:] - left_heel_seq[:-1], axis=1)
    left_heel_vel = np.append(left_heel_vel, left_heel_vel[-1])
    right_heel_vel = np.linalg.norm(right_heel_seq[1:] - right_heel_seq[:-1], axis=1)
    right_heel_vel = np.append(right_heel_vel, right_heel_vel[-1])

    left_heel_contact = left_heel_vel < CONTACT_VEL_THRESH
    right_heel_contact = right_heel_vel < CONTACT_VEL_THRESH
    left_toe_contact = left_toe_vel < CONTACT_VEL_THRESH
    right_toe_contact = right_toe_vel < CONTACT_VEL_THRESH

    # compute heel heights
    left_heel_heights = left_heel_seq[:, 2] - floor_height
    right_heel_heights = right_heel_seq[:, 2] - floor_height
    left_toe_heights =  left_toe_heights - floor_height
    right_toe_heights =  right_toe_heights - floor_height

    left_heel_contact = np.logical_and(left_heel_contact, left_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    right_heel_contact = np.logical_and(right_heel_contact, right_heel_heights < CONTACT_ANKLE_HEIGHT_THRESH)
    left_toe_contact = np.logical_and(left_toe_contact, left_toe_heights < CONTACT_TOE_HEIGHT_THRESH)
    right_toe_contact = np.logical_and(right_toe_contact, right_toe_heights < CONTACT_TOE_HEIGHT_THRESH)

    contacts = np.zeros((num_frames, len(SMPL_JOINTS)))
    contacts[:,SMPL_JOINTS['leftFoot']] = left_heel_contact
    contacts[:,SMPL_JOINTS['leftToeBase']] = left_toe_contact
    contacts[:,SMPL_JOINTS['rightFoot']] = right_heel_contact
    contacts[:,SMPL_JOINTS['rightToeBase']] = right_toe_contact

    # hand contacts
    left_hand_contact = detect_joint_contact(body_joint_seq, 'leftHand', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_hand_contact = detect_joint_contact(body_joint_seq, 'rightHand', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:,SMPL_JOINTS['leftHand']] = left_hand_contact
    contacts[:,SMPL_JOINTS['rightHand']] = right_hand_contact

    # knee contacts
    left_knee_contact = detect_joint_contact(body_joint_seq, 'leftLeg', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    right_knee_contact = detect_joint_contact(body_joint_seq, 'rightLeg', floor_height, CONTACT_VEL_THRESH, CONTACT_ANKLE_HEIGHT_THRESH)
    contacts[:,SMPL_JOINTS['leftLeg']] = left_knee_contact
    contacts[:,SMPL_JOINTS['rightLeg']] = right_knee_contact

    return offset_floor_height, contacts, discard_seq


def syn_acc(v, smooth_n=4):
    """
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0 and v.shape[0] >= 8:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc

def main(args, bm):
    for dataroot_subset in ['MPI_HDM05_test', 'MPI_Limits', 'MPI_mosh', 'SFU', 'TotalCapture', 'Transitions_mocap']:
        # ['ACCAD', 'BioMotionLab_NTroje_train', 'BioMotionLab_NTroje_test', 'BMLmovi', 'CMU_train',
        #                     'CMU_test', 'EKUT', 'Eyes_Japan_Dataset', 'HumanEva', 'KIT', 'MPI_HDM05_train',
        #                     'MPI_HDM05_test', 'MPI_Limits', 'MPI_mosh', 'SFU', 'TotalCapture', 'Transitions_mocap']:
        print(f"\n=== Processing {dataroot_subset} ===")

        savedir = os.path.join(args.save_dir, dataroot_subset)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        
        # 统计文件数量
        if ("train" in dataroot_subset) or ("test" in dataroot_subset):
            split_file = os.path.join("/mlx_devbox/users/sunlan/repo/20782/VR-based-text2motion-new/SAGE-main/prepare_data/data_split", dataroot_subset + ".txt")
            if os.path.exists(split_file):
                with open(split_file, "r") as f:
                    filepaths = [line.strip() for line in f]
                print(f"  - Split file: {os.path.abspath(split_file)}")
                print(f"  - Lines in txt file: {len(filepaths)}")
            else:
                print(f"  - Split file not found: {os.path.abspath(split_file)}")
                filepaths = []
        else:
            # 统计目录中的.npz文件数量
            npz_files = glob.glob(os.path.join(args.root_dir, dataroot_subset, '**', '*.npz'), recursive=True)
            print(f"  - Directory: {os.path.abspath(os.path.join(args.root_dir, dataroot_subset))}")
            print(f"  - .npz files in directory: {len(npz_files)}")
            filepaths = npz_files
        
        print(f"  - Total filepaths to process: {len(filepaths)}")

        rotation_local_full_gt_list = []
        hmd_position_global_full_gt_list = []
        body_parms_list = []
        head_global_trans_list = []

        idx = 0
        for filepath in tqdm(filepaths):
            data = {}
            bdata = np.load(
                os.path.join(args.root_dir, filepath), allow_pickle=True
            )

            if "mocap_framerate" in bdata:
                framerate = bdata["mocap_framerate"]
            else:
                continue
            idx += 1

            if framerate == 120:
                stride = 2
            elif framerate == 60:
                stride = 1
            else:
                # raise AssertionError(
                #     "Please check your AMASS data, should only have 2 types of framerate, either 120 or 60!!!"
                # )
                stride = round(framerate / 60)

            bdata_poses = bdata["poses"][::stride, ...]
            bdata_trans = bdata["trans"][::stride, ...]
            subject_gender = bdata["gender"]

            body_parms = {
                "root_orient": torch.Tensor(
                    bdata_poses[:, :3]
                ),  # .to(comp_device), # controls the global root orientation
                "pose_body": torch.Tensor(
                    bdata_poses[:, 3:66]
                ),  # .to(comp_device), # controls the body
                "trans": torch.Tensor(
                    bdata_trans
                ),  # .to(comp_device), # controls the global body position
            }

            body_parms_list = body_parms

            body_pose_world = bm(
                **{
                    k: v.cuda()
                    for k, v in body_parms.items()
                    if k in ["pose_body", "root_orient", "trans"]
                }
            )
            if bdata_poses.shape[0] < 5:
                continue
            

            output_aa = torch.Tensor(bdata_poses[:, :66]).reshape(-1,3)
            output_6d = utils_transform.aa2sixd(output_aa).reshape(bdata_poses.shape[0],-1)
            rotation_local_full_gt_list = output_6d[1:]
            rotation_local_matrot = aa2matrot(torch.tensor(bdata_poses).reshape(-1,3)).reshape(bdata_poses.shape[0],-1,9)
            rotation_global_matrot = local2global_pose(rotation_local_matrot, bm.kintree_table[0].long()) # rotation of joints relative to the origin

            # pass very short sequence
            if body_pose_world.v.shape[0] <= 10:
                continue

            # -------------------------------- get synthetic IMU data ----------------------------------
            ji_mask = [4, 5, 0]
            vi_mask = [1176, 4662, 3021]
            imu_rot = rotation_global_matrot[:, ji_mask]
            imu_acc = syn_acc(body_pose_world.v[:, vi_mask])
            # ------------------------------------------------------------------------------------------

            head_rotation_global_matrot = rotation_global_matrot[:,[15],:,:]
            rotation_global_6d = utils_transform.matrot2sixd(rotation_global_matrot.reshape(-1,3,3)).reshape(rotation_global_matrot.shape[0],-1,6)
            input_rotation_global_6d = rotation_global_6d[1:,:22,:]
            rotation_velocity_global_matrot = torch.matmul(torch.inverse(rotation_global_matrot[:-1]),rotation_global_matrot[1:])
            rotation_velocity_global_6d = utils_transform.matrot2sixd(rotation_velocity_global_matrot.reshape(-1,3,3)).reshape(rotation_velocity_global_matrot.shape[0],-1,6)
            input_rotation_velocity_global_6d = rotation_velocity_global_6d[:,:22,:]
            position_global_full_gt_world = body_pose_world.Jtr[:,:22,:] # position of joints relative to the world origin

            offset_floor_height, contacts, discard_seq = determine_floor_height_and_contacts(position_global_full_gt_world.cpu().numpy(), 60)

            position_head_world = position_global_full_gt_world[:,15,:] # world position of head
            head_global_trans = torch.eye(4).repeat(position_head_world.shape[0],1,1)
            head_global_trans[:,:3,:3] = head_rotation_global_matrot.squeeze()
            head_global_trans[:,:3,3] = position_global_full_gt_world[:,15,:]

            head_global_trans_list = head_global_trans[1:]

            num_frames = position_global_full_gt_world.shape[0] - 1
            # provide the hand representations in the head space
            hands_rotation_mat_in_head_space = rotation_global_matrot[
                :, 15:16, :, :].transpose(2, 3).matmul(
                    rotation_global_matrot[:, [20, 21], :, :])
            hands_rotation_in_head_space_r6d = utils_transform.matrot2sixd(
                hands_rotation_mat_in_head_space.reshape(-1, 3, 3)
            ).reshape(hands_rotation_mat_in_head_space.shape[0], -1, 6)[1:]
            
            rotation_velocity_handsinheadspace = torch.matmul(
                torch.inverse(hands_rotation_mat_in_head_space[:-1]),
                hands_rotation_mat_in_head_space[1:],
            )
            rotation_velocity_handsinheadspace_r6d = utils_transform.matrot2sixd(
                rotation_velocity_handsinheadspace.reshape(-1, 3, 3)
            ).reshape(rotation_velocity_handsinheadspace.shape[0], -1, 6)

            hands_position_in_head_space = (position_global_full_gt_world[:, [20, 21], :] - position_global_full_gt_world[:, 15:16, :]).double().bmm(rotation_global_matrot[:, 15].to(position_global_full_gt_world.device))
        
            device = position_global_full_gt_world.device
            print(device)
            hmd_position_global_full_gt_list = torch.cat([
                input_rotation_global_6d.reshape(num_frames,-1).to(device),
                input_rotation_velocity_global_6d.reshape(num_frames,-1).to(device),
                position_global_full_gt_world[1:, :22, :].reshape(num_frames,-1).to(device), 
                (position_global_full_gt_world[1:, :22, :].reshape(num_frames,-1) - position_global_full_gt_world[:-1, :22, :].reshape(num_frames,-1)).to(device)
            ], dim=-1)


            body_parms_list = {k: v[1:].cpu() for k, v in body_parms_list.items()}

            hmd_input = torch.cat(
                [
                    input_rotation_global_6d[:, [15, 20, 21, 4, 5, 0], :].reshape(num_frames, -1).cpu(),
                    input_rotation_velocity_global_6d[:, [15, 20, 21, 4, 5, 0], :].reshape(num_frames, -1).cpu(),
                    position_global_full_gt_world[1:, [15, 20, 21], :].reshape(num_frames, -1).cpu(),
                    (position_global_full_gt_world[1:, [15, 20, 21], :] - position_global_full_gt_world[:-1, [15, 20, 21], :]).reshape(num_frames, -1).cpu(),
                    hands_rotation_in_head_space_r6d.reshape(num_frames, -1).cpu(),
                    rotation_velocity_handsinheadspace_r6d.reshape(num_frames, -1).cpu(),
                    hands_position_in_head_space[1:].reshape(num_frames, -1).cpu(),
                    (hands_position_in_head_space[1:] - hands_position_in_head_space[:-1]).reshape(num_frames, -1).cpu(),
                    imu_acc[1:].reshape(num_frames, -1).cpu(),
                    imu_rot[1:].reshape(num_frames, -1).cpu(),
                ],
                dim=-1,
            )
            data['rotation_local_full_gt_list'] = rotation_local_full_gt_list.cpu()
            data['hmd_position_global_full_gt_list'] = hmd_position_global_full_gt_list.cpu()
            data['hmd_input'] = hmd_input
            data['body_parms_list'] = body_parms_list
            data['head_global_trans_list'] = head_global_trans_list.cpu()
            data['framerate'] = 60
            data['gender'] = subject_gender
            data['filepath'] = filepath
            data['IMU_global_rotation'] = imu_rot.cpu()[1:]
            data['IMU_global_acceleration'] = imu_acc.cpu()[1:]
            data['shape'] = bdata["betas"]
            data['offset_floor_height'] = offset_floor_height
            data['contacts'] = contacts[1:]

            torch.save(data, os.path.join(savedir, "{}.pt".format(idx)))
        
        # 输出处理统计信息
        print(f"  - Successfully processed: {idx} files")
        print(f"  - Saved .pt files: {idx}")
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        default="/mlx/users/sunlan/repo/20782/VR-based-text2motion-new/body_model",
        help="=dir where you put your smplh and dmpls dirs",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/mnt/bn/robotics-training-data-lf/sunlan/dataset/bytedance/JLM_data",
        help="=dir where you want to save your generated data",
    )
    parser.add_argument(
        "--root_dir", type=str, default="/mnt/bn/robotics-training-data-lf/sunlan/dataset/bytedance/humanml3d/amass_data",
        help="=dir where you put your AMASS data"
    )
    args = parser.parse_args()

    # Here we follow the AvatarPoser paper and use male model for all sequences
    bm_fname_male = os.path.join(args.support_dir, "smplh/{}/model.npz".format("male"))
    dmpl_fname_male = os.path.join(
        args.support_dir, "dmpls/{}/model.npz".format("male")
    )

    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters
    bm_male = BodyModel(
        bm_fname=bm_fname_male,
        num_betas=num_betas,
        num_dmpls=num_dmpls,
        dmpl_fname=dmpl_fname_male,
    ).cuda()

    main(args, bm_male)
