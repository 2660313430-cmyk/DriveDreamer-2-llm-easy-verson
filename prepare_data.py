import os
import sys
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from tqdm import tqdm

# ================= 🔧 配置区域 =================
DATAROOT = "./nuScenes"
OUTPUT_DIR = "./training_data/hdmaps"
# ==========================================

def main():
    if not os.path.exists(DATAROOT):
        print(f"❌ 错误: 找不到 {DATAROOT}，请检查路径！")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🚀 [1/2] 正在加载 nuScenes 数据库...")
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    samples = nusc.sample
    print(f"🚀 [2/2] 正在生成 {len(samples)} 张高清地图...")
    
    nusc_maps = {}
    
    for i, sample in enumerate(tqdm(samples)):
        # 获取场景信息
        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        location = log['location']
        
        if location not in nusc_maps:
            nusc_maps[location] = NuScenesMap(dataroot=DATAROOT, map_name=location)
        
        nusc_map = nusc_maps[location]
        
        # 获取车辆位置
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        x, y = ego_pose['translation'][0], ego_pose['translation'][1]
        
        # 🔧 修复点：计算正方形边界 (x_min, y_min, x_max, y_max)
        # 生成 100x100 米的地图，所以半径是 50
        radius = 50
        patch_box = (x - radius, y - radius, x + radius, y + radius)
        
        # 🔧 修复点：移除了报错的 patch_angle 参数
        try:
            fig, ax = nusc_map.render_map_patch(
                patch_box, 
                layer_names=['lane', 'road_segment', 'ped_crossing'], 
                figsize=(4, 4), 
                alpha=0.5, 
                render_egoposes_range=False
            )
        except TypeError:
            # 兼容旧版本 API
            fig, ax = nusc_map.render_map_patch(
                patch_box, 
                layer_names=['lane', 'road_segment', 'ped_crossing'], 
                figsize=(4, 4), 
                alpha=0.5
            )
        
        # 去除边框
        ax.axis('off')
        fig.patch.set_visible(False)
        ax.axis('tight')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 保存
        save_path = os.path.join(OUTPUT_DIR, f"{sample['token']}.png")
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100, facecolor='black')
        
        # 释放内存（重要！否则会爆内存）
        plt.close(fig)

    print("-" * 50)
    print(f"🎉 成功生成 {len(samples)} 张地图！")
    print(f"📂 保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()