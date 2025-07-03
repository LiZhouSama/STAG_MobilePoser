import torch

# 加载数据
data = torch.load('mobileposer/stag_raw_data/aligned_imu_data.pt', weights_only=False)

print('=== 数据结构验证 ===')
print(f'帧数: {data["frame_count"]}')
print(f'IMU位置: {data["imu_positions"]}')

print('\n=== 元数据 ===')
for k, v in data['metadata'].items():
    print(f'  {k}: {v}')

print('\n=== IMU数据形状 ===')
for pos in data['imu_positions']:
    print(f'  {pos}:')
    for feature in ['lin_acc_x', 'lin_acc_y', 'lin_acc_z', 'ori_roll_deg', 'ori_pitch_deg', 'ori_yaw_deg']:
        shape = data["imu_data"][pos][feature].shape
        print(f'    {feature}: {shape}')

print('\n=== 跳跃检测结果 ===')
print(f'手机开始跳跃: {data["metadata"]["phone_start_jumps"]}')
print(f'手机结束跳跃: {data["metadata"]["phone_end_jumps"]}')
print(f'手表开始跳跃: {data["metadata"]["watch_start_jumps"]}')
print(f'手表结束跳跃: {data["metadata"]["watch_end_jumps"]}')

print('\n=== 数据质量 ===')
for pos in data['imu_positions']:
    print(f'  {pos} ({'右手腕/手机' if pos == 'rp' else '左手腕/手表'}):')
    
    # 线性加速度统计
    lin_acc_x = data['imu_data'][pos]['lin_acc_x']
    lin_acc_y = data['imu_data'][pos]['lin_acc_y']
    lin_acc_z = data['imu_data'][pos]['lin_acc_z']
    
    acc_magnitude = torch.sqrt(lin_acc_x**2 + lin_acc_y**2 + lin_acc_z**2)
    
    print(f'    加速度模长 - 均值: {acc_magnitude.mean():.4f}, 标准差: {acc_magnitude.std():.4f}')
    print(f'    加速度范围: [{acc_magnitude.min():.4f}, {acc_magnitude.max():.4f}] m/s²')

print('\n=== 验证完成 ===') 