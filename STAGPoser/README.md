# IMU数据处理说明

这个目录包含了IMU数据的处理脚本和结果文件。

## 原始数据文件

- `IMUPoser_Phone.csv`: 手机IMU原始数据 (8167帧)
- `IMUPoser_Watch.csv`: 手表IMU原始数据 (8290帧)

每个CSV文件包含以下列：
- `frame_index`: 帧索引
- `raw_acc_x/y/z`: 原始加速度
- `raw_gyro_x/y/z`: 原始陀螺仪
- `raw_mag_x/y/z`: 原始磁力计
- `lin_acc_x/y/z`: 线性加速度 (去除重力)
- `ori_roll_deg/pitch_deg/yaw_deg`: 方向角度

## 处理脚本

### `process_imu_data.py`
主要的数据处理脚本，完成以下功能：

1. **跳变检测**: 通过分析线性加速度的变化检测开始和结束的跳动
2. **数据提取**: 提取开始三次跳动和结束三次跳动之间的有效数据
3. **时间对齐**: 使用交叉相关算法对齐两个IMU的时间序列
4. **数据保存**: 将处理后的数据保存为PyTorch的pt格式

运行方法：
```bash
python mobileposer/stag_raw_data/process_imu_data.py
```

### 处理结果

**检测统计:**
- 手机IMU检测到 99 次跳变
- 手表IMU检测到 35 次跳变
- 手机有效数据范围: 第249帧 - 第7593帧
- 手表有效数据范围: 第592帧 - 第7972帧
- 对齐偏移量: -55帧
- 最终对齐数据: 7289帧

## 输出文件

### `aligned_imu_data.pt`
处理后的对齐IMU数据，包含：

```python
{
    'frame_count': 7289,  # 对齐后的总帧数
    'imu_data': {
        'rp': {  # 右手腕 (手机)
            'lin_acc_x': torch.tensor[7289],
            'lin_acc_y': torch.tensor[7289], 
            'lin_acc_z': torch.tensor[7289],
            'ori_roll_deg': torch.tensor[7289],
            'ori_pitch_deg': torch.tensor[7289],
            'ori_yaw_deg': torch.tensor[7289]
        },
        'lw': {  # 左手腕 (手表)
            # 同样的结构
        }
    },
    'imu_positions': ['rp', 'lw'],  # IMU位置标识
    'metadata': {
        # 原始数据信息、处理参数等
    }
}
```

### 数据特征

**线性加速度范围 (m/s²):**
- 右手腕: X[-24.98, 17.28], Y[-12.45, 23.73], Z[-27.76, 17.52]
- 左手腕: X[-17.06, 12.73], Y[-11.20, 11.21], Z[-9.70, 9.29]

**方向角度范围 (度):**
- 右手腕: Roll[96.15, 143.84], Pitch[-40.34, -6.05], Yaw[-179.98, 179.98]
- 左手腕: Roll[-180.00, 179.98], Pitch[34.51, 61.75], Yaw[-179.99, 179.95]

## 使用示例

### `usage_example.py`
演示如何使用处理后的数据：

```python
import torch

# 加载数据
data = torch.load('aligned_imu_data.pt', weights_only=False)

# 获取特定IMU位置的特征
def get_imu_features(data, position, frame_range=None):
    # 返回 (N, 6) 的tensor: [lin_acc_x, lin_acc_y, lin_acc_z, roll, pitch, yaw]
    pass

# 批处理示例
batch_size = 64
sequence_length = 100
# 可以创建 112 个批次的训练数据
```

运行示例：
```bash
python mobileposer/stag_raw_data/usage_example.py
```

### `verify_data.py`
验证处理后数据的完整性和统计信息。

运行验证：
```bash
python mobileposer/stag_raw_data/verify_data.py
```

## 生成的可视化文件

- `imu_comparison.png`: 两个IMU的数据对比图
- `motion_intensity.png`: 运动强度时间序列图

## IMU位置说明

- **rp (Right Wrist/Phone)**: 右手腕位置的手机IMU
- **lw (Left Wrist/Watch)**: 左手腕位置的手表IMU

## 数据质量

- 对齐相关系数: 0.4717 (中等强度的相关性)
- 右手腕平均运动强度: 1.6205 m/s²
- 左手腕平均运动强度: 0.4302 m/s²

## 注意事项

1. 使用 `torch.load()` 时需要设置 `weights_only=False`
2. 角度数据可能存在±180°的跳变，使用时需要注意
3. 数据已经去除了开始和结束的跳动信号，为干净的运动数据
4. 两个IMU的数据已经时间对齐，可以直接用于双IMU的运动分析

## 下一步

处理后的数据可以用于：
- 人体姿态估计训练
- 运动识别算法开发  
- 双IMU传感器融合研究
- 运动模式分析 