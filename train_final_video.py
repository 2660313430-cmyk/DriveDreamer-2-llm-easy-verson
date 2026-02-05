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
import warnings

# ================= 🚀 5090 最终战场配置 =================
# 1. 路径配置 (绝对正确版)
NUSC_ROOT = "./nuScenes"            # 原始照片在这里
DATA_ROOT = "./training_data"       # 地图(hdmaps)在这里
OUTPUT_DIR = "./models/unimvm_video_model"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 5090 性能全开
# 这一步模型很大，Batch Size 设为 4 或 8 (如果显存不够会自动报错，5090 应该能抗 8)
BATCH_SIZE = 8
MAX_TRAIN_STEPS = 5000   # 训练 5000 步 (约 20-40 分钟)
LEARNING_RATE = 1e-5
IMG_SIZE = [256, 448]    # [高, 宽] 论文标准分辨率

# 3. 基础模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_NAME = "runwayml/stable-diffusion-v1-5" 

warnings.filterwarnings("ignore")
# =======================================================

class UniMVMDataset(Dataset):
    def __init__(self, processed_root, nusc_root):
        self.map_dir = os.path.join(processed_root, "hdmaps")
        print(f"🚀 [1/3] 正在加载 nuScenes... (Root: {nusc_root})")
        
        from nuscenes.nuscenes import NuScenes
        try:
            self.nusc = NuScenes(version='v1.0-mini', dataroot=nusc_root, verbose=False)
        except Exception as e:
            print(f"❌ nuScenes 报错: {e}")
            sys.exit(1)

        self.data_pairs = []
        # 寻找对应的地图文件
        if not os.path.exists(self.map_dir):
             print(f"❌ 找不到地图文件夹: {self.map_dir}")
             sys.exit(1)

        map_files = [f for f in os.listdir(self.map_dir) if f.endswith('.png')]
        
        print(f"🔍 [2/3] 正在匹配数据 (地图 -> 真实照片)...")
        for f in tqdm(map_files):
            sample_token = f.split('.')[0]
            try:
                sample = self.nusc.get('sample', sample_token)
                cam_token = sample['data']['CAM_FRONT']
                cam_data = self.nusc.get('sample_data', cam_token)
                
                # 真实照片 (Target)
                cam_path = os.path.join(nusc_root, cam_data['filename'])
                # 地图 (Condition)
                map_path = os.path.join(self.map_dir, f)
                
                if os.path.exists(cam_path) and os.path.exists(map_path):
                    self.data_pairs.append((map_path, cam_path))
            except:
                continue
                
        print(f"✅ [3/3] 配对成功! 有效训练样本: {len(self.data_pairs)} 对")

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE[0], IMG_SIZE[1])), # 256x448
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        map_path, cam_path = self.data_pairs[idx]
        
        # 条件图: 地图
        control_image = Image.open(map_path).convert("RGB")
        # 目标图: 真实街景
        target_image = Image.open(cam_path).convert("RGB")
        
        return {
            "pixel_values": self.transform(target_image),
            "conditioning_pixel_values": self.transform(control_image)
        }

def train():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    
    print(f"🔥 Step 3 训练启动 | Res: {IMG_SIZE} | Steps: {MAX_TRAIN_STEPS}")
    
    # 加载组件
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    controlnet = ControlNetModel.from_unet(unet).to(device)
    
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()
    
    # 8-bit 优化
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(controlnet.parameters(), lr=LEARNING_RATE)
        print("✨ 已启用 bitsandbytes 8-bit 优化")
    except:
        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LEARNING_RATE)
    
    dataset = UniMVMDataset(DATA_ROOT, NUSC_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), desc="Video Model Training")
    global_step = 0
    
    while global_step < MAX_TRAIN_STEPS:
        for batch in dataloader:
            with torch.cuda.amp.autocast():
                # 编码真实图片 -> Latents
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 读取地图条件
                control_image = batch["conditioning_pixel_values"].to(device)
                
                # 加噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                empty_text = torch.zeros((latents.shape[0], 77, 768), device=device)
                
                # ControlNet 前向
                down_res, mid_res = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=empty_text,
                    controlnet_cond=control_image,
                    return_dict=False,
                )
                
                # UNet 前向 (接受 ControlNet 的指导)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=empty_text,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                ).sample
                
                loss = F.mse_loss(model_pred.float(), noise.float())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1
            
            if global_step >= MAX_TRAIN_STEPS:
                break
        
        # 每 1000 步保存一次 checkpoints
        if global_step % 1000 == 0:
             controlnet.save_pretrained(OUTPUT_DIR)
             print(f"\n💾 中途存档已保存 (Step {global_step})")

    controlnet.save_pretrained(OUTPUT_DIR)
    print(f"\n🎉🎉🎉 所有训练全部完成！模型保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()