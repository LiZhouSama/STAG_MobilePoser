import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

def analyze_jump_detection():
    """
    Analyze jump detection results and data quality
    """
    print("=== Jump Detection Analysis ===")
    
    # Load processed data
    data = torch.load('mobileposer/stag_raw_data/aligned_imu_data.pt', weights_only=False)
    
    print(f"Processed data frames: {data['frame_count']}")
    print(f"Original phone data frames: {data['metadata']['original_phone_frames']}")
    print(f"Original watch data frames: {data['metadata']['original_watch_frames']}")
    
    # Analyze jump detection results
    phone_start_jumps = data['metadata']['phone_start_jumps']
    phone_end_jumps = data['metadata']['phone_end_jumps']
    watch_start_jumps = data['metadata']['watch_start_jumps']
    watch_end_jumps = data['metadata']['watch_end_jumps']
    
    print(f"\n=== Jump Detection Results ===")
    print(f"Phone start jumps: {len(phone_start_jumps)} times - positions: {phone_start_jumps}")
    print(f"Phone end jumps: {len(phone_end_jumps)} times - positions: {phone_end_jumps}")
    print(f"Watch start jumps: {len(watch_start_jumps)} times - positions: {watch_start_jumps}")
    print(f"Watch end jumps: {len(watch_end_jumps)} times - positions: {watch_end_jumps}")
    
    # Analyze jump intervals
    if len(phone_start_jumps) > 1:
        phone_start_intervals = np.diff(phone_start_jumps)
        print(f"\nPhone start jump intervals (frames): {phone_start_intervals}")
        print(f"Phone start jump intervals (seconds): {[f'{x/100:.2f}' for x in phone_start_intervals]}")
        print(f"Phone start jump average interval: {np.mean(phone_start_intervals):.0f} frames ({np.mean(phone_start_intervals)/100:.2f}s)")
    
    if len(phone_end_jumps) > 1:
        phone_end_intervals = np.diff(phone_end_jumps)
        print(f"Phone end jump intervals (frames): {phone_end_intervals}")
        print(f"Phone end jump intervals (seconds): {[f'{x/100:.2f}' for x in phone_end_intervals]}")
        print(f"Phone end jump average interval: {np.mean(phone_end_intervals):.0f} frames ({np.mean(phone_end_intervals)/100:.2f}s)")
    
    if len(watch_start_jumps) > 1:
        watch_start_intervals = np.diff(watch_start_jumps)
        print(f"\nWatch start jump intervals (frames): {watch_start_intervals}")
        print(f"Watch start jump intervals (seconds): {[f'{x/100:.2f}' for x in watch_start_intervals]}")
        print(f"Watch start jump average interval: {np.mean(watch_start_intervals):.0f} frames ({np.mean(watch_start_intervals)/100:.2f}s)")
    
    if len(watch_end_jumps) > 1:
        watch_end_intervals = np.diff(watch_end_jumps)
        print(f"Watch end jump intervals (frames): {watch_end_intervals}")
        print(f"Watch end jump intervals (seconds): {[f'{x/100:.2f}' for x in watch_end_intervals]}")
        print(f"Watch end jump average interval: {np.mean(watch_end_intervals):.0f} frames ({np.mean(watch_end_intervals)/100:.2f}s)")
    
    # Analyze valid data range
    phone_range = data['metadata']['phone_valid_range']
    watch_range = data['metadata']['watch_valid_range']
    
    print(f"\n=== Valid Data Range ===")
    print(f"Phone valid range: {phone_range[0]} - {phone_range[1]} (total {phone_range[1]-phone_range[0]} frames)")
    print(f"Watch valid range: {watch_range[0]} - {watch_range[1]} (total {watch_range[1]-watch_range[0]} frames)")
    
    phone_usage = (phone_range[1]-phone_range[0]) / data['metadata']['original_phone_frames'] * 100
    watch_usage = (watch_range[1]-watch_range[0]) / data['metadata']['original_watch_frames'] * 100
    
    print(f"Phone data utilization: {phone_usage:.1f}%")
    print(f"Watch data utilization: {watch_usage:.1f}%")
    
    # Analyze alignment quality
    alignment_shift = data['metadata']['alignment_shift']
    print(f"\n=== Alignment Quality ===")
    print(f"Time offset: {alignment_shift} frames ({alignment_shift/100:.2f}s)")
    print(f"Final aligned data: {data['frame_count']} frames ({data['frame_count']/100:.1f}s)")
    
    # Calculate data quality metrics
    print(f"\n=== Data Quality Metrics ===")
    for pos in data['imu_positions']:
        print(f"\n{pos} ({'Right wrist/Phone' if pos == 'rp' else 'Left wrist/Watch'}):")
        
        # Linear acceleration statistics
        lin_acc_x = data['imu_data'][pos]['lin_acc_x']
        lin_acc_y = data['imu_data'][pos]['lin_acc_y']
        lin_acc_z = data['imu_data'][pos]['lin_acc_z']
        
        acc_magnitude = torch.sqrt(lin_acc_x**2 + lin_acc_y**2 + lin_acc_z**2)
        
        print(f"  Linear acceleration magnitude - mean: {acc_magnitude.mean():.4f}, std: {acc_magnitude.std():.4f}")
        print(f"  Max acceleration: {acc_magnitude.max():.4f} m/s²")
        print(f"  Min acceleration: {acc_magnitude.min():.4f} m/s²")
        
        # Angle ranges
        roll = data['imu_data'][pos]['ori_roll_deg']
        pitch = data['imu_data'][pos]['ori_pitch_deg']
        yaw = data['imu_data'][pos]['ori_yaw_deg']
        
        print(f"  Roll range: [{roll.min():.1f}, {roll.max():.1f}]°")
        print(f"  Pitch range: [{pitch.min():.1f}, {pitch.max():.1f}]°")
        print(f"  Yaw range: [{yaw.min():.1f}, {yaw.max():.1f}]°")
    
    # Create improved visualization
    create_improved_visualization(data)

def create_improved_visualization(data):
    """
    Create improved visualization showing jump detection and data alignment results
    """
    # Reload original data for comparison
    phone_data = pd.read_csv('mobileposer/stag_raw_data/IMUPoser_Phone.csv')
    watch_data = pd.read_csv('mobileposer/stag_raw_data/IMUPoser_Watch.csv')
    
    # Extract metadata
    phone_start_jumps = data['metadata']['phone_start_jumps']
    phone_end_jumps = data['metadata']['phone_end_jumps'] 
    watch_start_jumps = data['metadata']['watch_start_jumps']
    watch_end_jumps = data['metadata']['watch_end_jumps']
    
    phone_range = data['metadata']['phone_valid_range']
    watch_range = data['metadata']['watch_valid_range']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate acceleration magnitude from original data
    phone_lin_acc = phone_data[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
    watch_lin_acc = watch_data[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
    
    phone_mag = np.sqrt(np.sum(phone_lin_acc**2, axis=1))
    watch_mag = np.sqrt(np.sum(watch_lin_acc**2, axis=1))
    
    # Top left: Phone jump detection
    axes[0, 0].plot(phone_mag, alpha=0.7, color='blue', linewidth=0.8)
    
    # Mark start and end jumps
    if phone_start_jumps:
        axes[0, 0].scatter(phone_start_jumps, phone_mag[phone_start_jumps], 
                          color='green', s=100, marker='v', label=f'Start jumps ({len(phone_start_jumps)})', 
                          zorder=5, edgecolors='darkgreen', linewidth=2)
    if phone_end_jumps:
        axes[0, 0].scatter(phone_end_jumps, phone_mag[phone_end_jumps], 
                          color='red', s=100, marker='^', label=f'End jumps ({len(phone_end_jumps)})', 
                          zorder=5, edgecolors='darkred', linewidth=2)
    
    # Mark valid data range
    axes[0, 0].axvline(phone_range[0], color='orange', linestyle='--', linewidth=2, 
                      label='Valid data start', alpha=0.8)
    axes[0, 0].axvline(phone_range[1], color='orange', linestyle='--', linewidth=2, 
                      label='Valid data end', alpha=0.8)
    axes[0, 0].fill_betweenx([0, phone_mag.max()], phone_range[0], phone_range[1], 
                            alpha=0.15, color='orange', label='Valid data region')
    
    axes[0, 0].set_title('Phone IMU - Jump Detection Results', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Linear Acceleration Magnitude (m/s²)')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top right: Watch jump detection
    axes[0, 1].plot(watch_mag, alpha=0.7, color='orange', linewidth=0.8)
    
    if watch_start_jumps:
        axes[0, 1].scatter(watch_start_jumps, watch_mag[watch_start_jumps], 
                          color='green', s=100, marker='v', label=f'Start jumps ({len(watch_start_jumps)})', 
                          zorder=5, edgecolors='darkgreen', linewidth=2)
    if watch_end_jumps:
        axes[0, 1].scatter(watch_end_jumps, watch_mag[watch_end_jumps], 
                          color='red', s=100, marker='^', label=f'End jumps ({len(watch_end_jumps)})', 
                          zorder=5, edgecolors='darkred', linewidth=2)
    
    axes[0, 1].axvline(watch_range[0], color='blue', linestyle='--', linewidth=2, 
                      label='Valid data start', alpha=0.8)
    axes[0, 1].axvline(watch_range[1], color='blue', linestyle='--', linewidth=2, 
                      label='Valid data end', alpha=0.8)
    axes[0, 1].fill_betweenx([0, watch_mag.max()], watch_range[0], watch_range[1], 
                            alpha=0.15, color='blue', label='Valid data region')
    
    axes[0, 1].set_title('Watch IMU - Jump Detection Results', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Linear Acceleration Magnitude (m/s²)')
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom: Aligned data comparison
    rp_acc_x = data['imu_data']['rp']['lin_acc_x'].numpy()
    lw_acc_x = data['imu_data']['lw']['lin_acc_x'].numpy()
    rp_acc_mag = torch.sqrt(
        data['imu_data']['rp']['lin_acc_x']**2 + 
        data['imu_data']['rp']['lin_acc_y']**2 + 
        data['imu_data']['rp']['lin_acc_z']**2
    ).numpy()
    lw_acc_mag = torch.sqrt(
        data['imu_data']['lw']['lin_acc_x']**2 + 
        data['imu_data']['lw']['lin_acc_y']**2 + 
        data['imu_data']['lw']['lin_acc_z']**2
    ).numpy()
    
    frames = np.arange(len(rp_acc_x))
    
    # Bottom left: X-axis acceleration comparison
    axes[1, 0].plot(frames, rp_acc_x, label='Right wrist (Phone)', alpha=0.8, linewidth=1.2, color='blue')
    axes[1, 0].plot(frames, lw_acc_x, label='Left wrist (Watch)', alpha=0.8, linewidth=1.2, color='orange')
    axes[1, 0].set_title('Aligned X-axis Linear Acceleration Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Frame (after alignment)')
    axes[1, 0].set_ylabel('Linear Acceleration (m/s²)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom right: Acceleration magnitude comparison
    axes[1, 1].plot(frames, rp_acc_mag, label='Right wrist (Phone)', alpha=0.8, linewidth=1.2, color='blue')
    axes[1, 1].plot(frames, lw_acc_mag, label='Left wrist (Watch)', alpha=0.8, linewidth=1.2, color='orange')
    axes[1, 1].set_title('Aligned Acceleration Magnitude Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Frame (after alignment)')
    axes[1, 1].set_ylabel('Acceleration Magnitude (m/s²)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mobileposer/stag_raw_data/improved_jump_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nImproved jump detection analysis plot saved as improved_jump_analysis.png")

def plot_jump_sequence_details(data):
    """
    Plot detailed view of jump sequences
    """
    # Reload original data
    phone_data = pd.read_csv('mobileposer/stag_raw_data/IMUPoser_Phone.csv')
    watch_data = pd.read_csv('mobileposer/stag_raw_data/IMUPoser_Watch.csv')
    
    phone_lin_acc = phone_data[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
    watch_lin_acc = watch_data[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
    
    phone_mag = np.sqrt(np.sum(phone_lin_acc**2, axis=1))
    watch_mag = np.sqrt(np.sum(watch_lin_acc**2, axis=1))
    
    phone_start_jumps = data['metadata']['phone_start_jumps']
    phone_end_jumps = data['metadata']['phone_end_jumps']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Phone start jump detailed view
    if phone_start_jumps:
        start_window = 200
        start_center = phone_start_jumps[1] if len(phone_start_jumps) > 1 else phone_start_jumps[0]
        start_range = (max(0, start_center - start_window), 
                      min(len(phone_mag), start_center + start_window))
        
        x_range = np.arange(start_range[0], start_range[1])
        axes[0, 0].plot(x_range, phone_mag[start_range[0]:start_range[1]], 
                       color='blue', linewidth=1.5, alpha=0.8)
        axes[0, 0].scatter(phone_start_jumps, phone_mag[phone_start_jumps], 
                          color='green', s=120, marker='v', zorder=5, 
                          edgecolors='darkgreen', linewidth=2)
        axes[0, 0].set_title('Phone Start Jump Sequence (Detailed View)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Acceleration Magnitude (m/s²)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Annotate jump intervals
        for i, jump in enumerate(phone_start_jumps):
            axes[0, 0].annotate(f'Jump{i+1}', (jump, phone_mag[jump]), 
                               xytext=(5, 10), textcoords='offset points', 
                               fontsize=10, ha='left')
    
    # Phone end jump detailed view
    if phone_end_jumps:
        end_window = 200
        end_center = phone_end_jumps[1] if len(phone_end_jumps) > 1 else phone_end_jumps[0]
        end_range = (max(0, end_center - end_window), 
                    min(len(phone_mag), end_center + end_window))
        
        x_range = np.arange(end_range[0], end_range[1])
        axes[0, 1].plot(x_range, phone_mag[end_range[0]:end_range[1]], 
                       color='blue', linewidth=1.5, alpha=0.8)
        axes[0, 1].scatter(phone_end_jumps, phone_mag[phone_end_jumps], 
                          color='red', s=120, marker='^', zorder=5, 
                          edgecolors='darkred', linewidth=2)
        axes[0, 1].set_title('Phone End Jump Sequence (Detailed View)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Acceleration Magnitude (m/s²)')
        axes[0, 1].grid(True, alpha=0.3)
        
        for i, jump in enumerate(phone_end_jumps):
            axes[0, 1].annotate(f'Jump{i+1}', (jump, phone_mag[jump]), 
                               xytext=(5, 10), textcoords='offset points', 
                               fontsize=10, ha='left')
    
    # Aligned data correlation analysis
    rp_acc_mag = torch.sqrt(
        data['imu_data']['rp']['lin_acc_x']**2 + 
        data['imu_data']['rp']['lin_acc_y']**2 + 
        data['imu_data']['rp']['lin_acc_z']**2
    ).numpy()
    lw_acc_mag = torch.sqrt(
        data['imu_data']['lw']['lin_acc_x']**2 + 
        data['imu_data']['lw']['lin_acc_y']**2 + 
        data['imu_data']['lw']['lin_acc_z']**2
    ).numpy()
    
    # Show first 1000 frames comparison
    sample_frames = min(1000, len(rp_acc_mag))
    frames = np.arange(sample_frames)
    
    axes[1, 0].plot(frames, rp_acc_mag[:sample_frames], 
                   label='Right wrist (Phone)', alpha=0.8, linewidth=1.5, color='blue')
    axes[1, 0].plot(frames, lw_acc_mag[:sample_frames], 
                   label='Left wrist (Watch)', alpha=0.8, linewidth=1.5, color='orange')
    axes[1, 0].set_title(f'Aligned Acceleration Comparison (First {sample_frames} frames)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Acceleration Magnitude (m/s²)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation scatter plot
    sample_indices = np.linspace(0, len(rp_acc_mag)-1, min(2000, len(rp_acc_mag)), dtype=int)
    axes[1, 1].scatter(rp_acc_mag[sample_indices], lw_acc_mag[sample_indices], 
                      alpha=0.6, s=20, color='purple')
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(rp_acc_mag, lw_acc_mag)[0, 1]
    axes[1, 1].set_title(f'IMU Correlation Analysis (r={correlation:.3f})', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Right wrist Acceleration Magnitude (m/s²)')
    axes[1, 1].set_ylabel('Left wrist Acceleration Magnitude (m/s²)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add fit line
    z = np.polyfit(rp_acc_mag[sample_indices], lw_acc_mag[sample_indices], 1)
    p = np.poly1d(z)
    x_line = np.linspace(rp_acc_mag.min(), rp_acc_mag.max(), 100)
    axes[1, 1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('mobileposer/stag_raw_data/jump_sequence_details.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Jump sequence detailed analysis plot saved as jump_sequence_details.png")

if __name__ == "__main__":
    analyze_jump_detection()
    
    # Load data for detailed analysis
    data = torch.load('mobileposer/stag_raw_data/aligned_imu_data.pt', weights_only=False)
    plot_jump_sequence_details(data) 