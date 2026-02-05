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

# ================= 🚀 5090 训练配置 =================
# 1. 路径 (自动读取刚才生成的 ./training_data)
DATA_ROOT = "./training_data"
OUTPUT_DIR = "./models/hdmap_controlnet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 训练参数
# 5090 显存极大，Batch Size 开到 8 没问题，跑得快
BATCH_SIZE = 8
# 图像尺寸 (512x512 是标准 SD 分辨率)
IMG_SIZE = 512 
# 训练轮数 (15轮足够让它学会画地图了)
NUM_EPOCHS = 15 
LEARNING_RATE = 1e-5

# 3. 这里的模型是通用的 SD 1.5
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_NAME = "runwayml/stable-diffusion-v1-5" 

# 忽略 5090 的兼容性警告
warnings.filterwarnings("ignore")
# ===================================================

class DriveDreamerDataset(Dataset):
    def __init__(self, root_dir):
        self.traj_dir = os.path.join(root_dir, "trajectories")
        self.map_dir = os.path.join(root_dir, "hdmaps")
        
        self.filenames = []
        # 确保两个文件夹都存在
        if os.path.exists(self.traj_dir) and os.path.exists(self.map_dir):
            traj_files = set(f for f in os.listdir(self.traj_dir) if f.endswith('.png'))
            map_files = set(f for f in os.listdir(self.map_dir) if f.endswith('.png'))
            # 取交集 (确保每一对数据都完整)
            self.filenames = list(traj_files & map_files)
        
        print(f"🔍 数据集就绪: 找到 {len(self.filenames)} 组训练数据 (Trajectory -> Map)")
        if len(self.filenames) == 0:
            print("❌ 错误: 没有找到配对数据！请确认 Step 1 和 Step 2 都跑成功了。")
            sys.exit(1)

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        # 输入条件: 轨迹图 (Condition)
        traj_path = os.path.join(self.traj_dir, filename)
        control_image = Image.open(traj_path).convert("RGB")
        
        # 训练目标: 高精地图 (Ground Truth)
        map_path = os.path.join(self.map_dir, filename)
        target_image = Image.open(map_path).convert("RGB")
        
        return {
            "pixel_values": self.transform(target_image),
            "conditioning_pixel_values": self.transform(control_image)
        }

def train():
    device = torch.device("cuda")
    # 5090 开启 TF32 加速
    torch.backends.cuda.matmul.allow_tf32 = True
    
    print(f"🔥 RTX 5090 训练引擎启动 | Batch: {BATCH_SIZE} | Epochs: {NUM_EPOCHS}")
    
    # 加载模型组件
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(device)
    # 初始化 ControlNet (这就是我们要训练的核心)
    controlnet = ControlNetModel.from_unet(unet).to(device)
    
    # 冻结其他部分，只训练 ControlNet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()
    
    # 尝试使用 8-bit 优化器 (省显存神器)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(controlnet.parameters(), lr=LEARNING_RATE)
        print("✨ 已启用 bitsandbytes 8-bit 优化")
    except ImportError:
        print("⚠️ 未找到 bitsandbytes，使用原生优化器")
        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=LEARNING_RATE)
    
    dataset = DriveDreamerDataset(DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 训练循环
    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        total_loss = 0
        
        for batch in dataloader:
            with torch.cuda.amp.autocast(): # 混合精度训练
                # 1. 图像编码
                latents = vae.encode(batch["pixel_values"].to(device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 2. 读取条件 (轨迹)
                control_image = batch["conditioning_pixel_values"].to(device)
                
                # 3. 加噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                empty_text = torch.zeros((latents.shape[0], 77, 768), device=device)
                
                # 4. 前向传播
                down_res, mid_res = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=empty_text,
                    controlnet_cond=control_image,
                    return_dict=False,
                )
                
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=empty_text,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                ).sample
                
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            # 5. 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            progress_bar.update(1)
        
        # 每轮保存一次，防止意外
        save_path = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
        controlnet.save_pretrained(save_path)
        print(f"💾 存档已保存: {save_path}")

    # 最终保存
    controlnet.save_pretrained(OUTPUT_DIR)
    print(f"\n🎉 训练大功告成！模型保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    train()