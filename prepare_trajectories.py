import os
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm
import numpy as np

# 1. 路径配置
DATAROOT = "./nuScenes"
OUTPUT_ROOT = "./training_data"
TRAJ_DIR = os.path.join(OUTPUT_ROOT, "trajectories")
MAP_DIR = os.path.join(OUTPUT_ROOT, "hdmaps")

def main():
    if not os.path.exists(DATAROOT):
        print("❌ 找不到 nuScenes 数据！")
        return
    
    os.makedirs(TRAJ_DIR, exist_ok=True)
    
    print("🚀 正在加载 nuScenes...")
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    
    # 获取已经存在的地图文件名，只生成对应的轨迹，确保一一对应
    if not os.path.exists(MAP_DIR):
        print("❌ 请先运行之前的 prepare_data.py 生成 hdmaps！")
        return
    
    valid_tokens = [f.split('.')[0] for f in os.listdir(MAP_DIR) if f.endswith('.png')]
    print(f"🎯 目标生成 {len(valid_tokens)} 张轨迹图...")

    for token in tqdm(valid_tokens):
        try:
            sample = nusc.get('sample', token)
            
            # --- 绘制轨迹 (简化版: 只画 Ego 车未来轨迹) ---
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_facecolor('black')
            
            # 获取当前帧和未来帧的 Ego 位置
            current_token = token
            positions = []
            for _ in range(6): # 取未来 3秒 (每秒2帧)
                sd_token = nusc.get('sample', current_token)['data']['CAM_FRONT']
                ego_pose = nusc.get('ego_pose', nusc.get('sample_data', sd_token)['ego_pose_token'])
                positions.append(ego_pose['translation'][:2])
                
                next_token = nusc.get('sample', current_token)['next']
                if not next_token: break
                current_token = next_token
                
            positions = np.array(positions)
            
            # 将绝对坐标转换为相对坐标 (以第一帧为中心)
            if len(positions) > 1:
                # 简单平移，不旋转 (简化处理)
                base_x, base_y = positions[0]
                rel_x = positions[:, 0] - base_x
                rel_y = positions[:, 1] - base_y
                
                # 画轨迹线 (黄色)
                ax.plot(rel_x, rel_y, color='yellow', linewidth=5)
            
            # 设置范围 (保持和地图一致)
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.axis('off')
            
            # 保存
            save_path = os.path.join(TRAJ_DIR, f"{token}.png")
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100, facecolor='black')
            plt.close(fig)
            
        except Exception as e:
            continue

    print(f"🎉 轨迹数据生成完毕！保存在: {TRAJ_DIR}")

if __name__ == "__main__":
    main()