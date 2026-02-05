import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import (
    ControlNetModel,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from PIL import Image
from tqdm.auto import tqdm
from nuscenes.nuscenes import NuScenes
import bitsandbytes as bnb 

# ================= 🚀 路径配置 (根据你的 find 结果修正) =================

# 1. 原始 nuScenes 数据 (v1.0-mini, samples) 在这里
NUSC_ROOT = "./nuScenes"

# 2. 生成的地图 (hdmaps) 在这里
# 根据你的反馈: ./training_data/hdmaps
DATA_ROOT = "./training_data"

OUTPUT_DIR = "./models/unimvm_5090_paper"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. 性能参数 (RTX 5090 满血版)
HEIGHT = 256
WIDTH = 448
BATCH_SIZE = 8           # 5090 显存大，直接拉满
MAX_TRAIN_STEPS = 5000   # 约1小时
LEARNING_RATE = 1e-5

# 4. 网络加速
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_NAME = "runwayml/stable-diffusion-v1-5" 

# =================================================================

class UniMVMDataset(Dataset):
    def __init__(self, processed_root, nusc_root):
        # 这里会拼接成 ./training_data/hdmaps
        self.map_dir = os.path.join(processed_root, "hdmaps")
        print(f"🚀 [1/3] 正在加载 nuScenes... (Root: {nusc_root})")
        
        # --- 路径防御性检查 ---
        if not os.path.exists(nusc_root):
            print(f"❌ 严重错误: 找不到 nuScenes 文件夹: {nusc_root}")
            sys.exit(1)
            
        version_path = os.path.join(nusc_root, "v1.0-mini")
        if not os.path.exists(version_path):
            print(f"❌ 严重错误: 在 {nusc_root} 里没找到 v1.0-mini！")
            sys.exit(1)

        if not os.path.exists(self.map_dir):
             print(f"❌ 严重错误: 找不到地图文件夹: {self.map_dir}")
             print("请确认你是否有 ./training_data/hdmaps 这个目录")
             sys.exit(1)
        # ---------------------

        try:
            self.nusc = NuScenes(version='v1.0-mini', dataroot=nusc_root, verbose=False)
        except Exception as e:
            print(f"❌ nuScenes 初始化报错: {e}")
            sys.exit(1)

        self.data_pairs = []
        map_files = [f for f in os.listdir(self.map_dir) if f.endswith('.png')]
        
        print(f"🔍 [2/3] 正在匹配数据 (找到 {len(map_files)} 张地图)...")
        # 这里的逻辑是：地图文件名 = sample_token.png
        # 我们要通过 sample_token 找到对应的真实照片
        for f in tqdm(map_files):
            sample_token = f.split('.')[0]
            try:
                sample = self.nusc.get('sample', sample_token)
                cam_token = sample['data']['CAM_FRONT']
                cam_data = self.nusc.get('sample_data', cam_token)
                
                # 真实照片路径
                cam_path = os.path.join(nusc_root, cam_data['filename'])
                # 地图路径
                map_path = os.path.join(self.map_dir, f)
                
                if os.path.exists(cam_path) and os.path.exists(map_path):
                    self.data_pairs.append((map_path, cam_path))
            except:
                continue
                
        print(f"✅ [3/3] 数据匹配完毕! 有效训练样本: {len(self.data_pairs)} 对")
        
        if len(self.data_pairs) == 0:
            print("❌ 警告: 匹配数量为 0！")
            print("可能原因：")
            print("1. training_data 里的地图文件名不对（不是 token.png）")
            print("2. nuScenes/samples 文件夹里没有对应的 jpg 图片")
            sys.exit(1)

        self.transform = transforms.Compose([
            transforms.Resize((HEIGHT, WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        map_path, cam_path = self.data_pairs[idx]
        control_image = Image.open(map_path).convert("RGB")
        target_image = Image.open(cam_path).convert("RGB")
        
        return {
            "pixel_values": self.transform(target_image),
            "conditioning_pixel_values": self.transform(control_image)
        }

def train():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    
    print(f"🔥 5090 引擎启动 | Res: {HEIGHT}x{WIDTH} | Steps: {MAX_TRAIN_STEPS}")
    
    # 加载模型 (会自动断点续传下载)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    controlnet = ControlNetModel.from_unet(unet).to(device)
    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()
    
    # 优化器
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(controlnet.parameters(), lr=LEARNING_RATE)
        print("✨ 启用 bitsandbytes 8-bit 优化")
    except ImportError:
        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LEARNING_RATE)
    
    # 数据加载
    dataset = UniMVMDataset(DATA_ROOT, NUSC_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 训练循环
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), desc="Training")
    global_step = 0
    
    while global_step < MAX_TRAIN_STEPS:
        for batch in dataloader:
            with torch.cuda.amp.autocast():
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                control_image = batch["conditioning_pixel_values"].to(device)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                empty_text_embeds = torch.zeros((latents.shape[0], 77, 768), device=device)
                
                down_res, mid_res = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=empty_text_embeds,
                    controlnet_cond=control_image,
                    return_dict=False,
                )
                
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=empty_text_embeds,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                ).sample
                
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1
            
            if global_step >= MAX_TRAIN_STEPS:
                break

    controlnet.save_pretrained(OUTPUT_DIR)
    print(f"\n🎉 训练完成！模型已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()