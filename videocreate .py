import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import imageio
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from openai import OpenAI

# ================= 🔧 配置区域 =================

# ⚠️ 安全警告：这是你的私钥，请勿上传此文件到 GitHub！
# 如果上传代码，请务必将此处改回 os.getenv("LLM_API_KEY")
API_KEY = "在这里填入你的DeepSeek_Key" 

BASE_URL = "https://api.deepseek.com"
MODEL_PATH = "/root/autodl-tmp/models/unimvm_video_model"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# --- 🎞️ 视频流畅度核心参数 ---
TARGET_FPS = 12        # 目标帧率：12 FPS (人眼流畅标准)
VIDEO_DURATION = 10    # 视频时长：10秒
TOTAL_FRAMES = TARGET_FPS * VIDEO_DURATION # 总帧数：120帧
# ==============================================

pipe = None

def load_model():
    global pipe
    if pipe is None:
        print("⏳ 正在加载模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            controlnet = ControlNetModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                BASE_MODEL, controlnet=controlnet, torch_dtype=torch.float16
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.to(device)
            # 开启显存优化
            pipe.enable_model_cpu_offload()
            print("✅ 模型加载完毕！")
        except Exception as e:
            print(f"⚠️ 模型加载警告: {e}")
            print(f"请检查路径是否存在: {MODEL_PATH}")
    return pipe

def get_trajectory_from_llm(prompt):
    """使用 DeepSeek 获取原始轨迹"""
    if not API_KEY or "你的" in API_KEY:
        print("❌ 错误：请先在代码第 16 行填入正确的 API Key！")
        return np.linspace(0, 0, 30)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        print(f"🤖 DeepSeek 正在规划路径...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个自动驾驶规划师。请输出未来10秒的30个横向坐标点，逗号分隔。范围-4到4（负数为左，正数为右）。仅输出数字。"},
                {"role": "user", "content": f"指令: {prompt}"},
            ],
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        # 数据清洗
        content = content.replace('\n', ',').replace(' ', '')
        traj = np.fromstring(content, sep=',')
        
        # 兜底补全
        if len(traj) < 30:
            traj = np.pad(traj, (0, 30 - len(traj)), 'edge')
        return traj[:30]
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        # 失败时返回直线
        return np.linspace(0, 0, 30)

def interpolate_trajectory(original_traj, target_length):
    """
    🧮 插值算法：将 30 个点平滑扩展到 120 个点
    """
    old_indices = np.linspace(0, 10, len(original_traj))
    new_indices = np.linspace(0, 10, target_length)
    new_traj = np.interp(new_indices, old_indices, original_traj)
    return new_traj

def draw_smooth_map(trajectory, frame_idx, window_size=40):
    """
    绘图函数 (适配 120 帧的大窗口)
    """
    plt.figure(figsize=(4, 2.5), dpi=100)
    plt.style.use('dark_background')
    
    # 防止数组越界
    padded_traj = np.pad(trajectory, (0, window_size), 'edge')
    
    start_y = frame_idx
    end_y = frame_idx + window_size
    
    # 绘制车道线
    y_bg = np.arange(window_size)
    plt.plot(np.zeros_like(y_bg) - 2.0, y_bg, color='white', linestyle='--', alpha=0.3)
    plt.plot(np.zeros_like(y_bg) + 2.0, y_bg, color='white', linestyle='--', alpha=0.3)
    
    # 绘制红色轨迹
    current_traj_segment = padded_traj[start_y:end_y]
    plt.plot(current_traj_segment, np.arange(len(current_traj_segment)), color='red', linewidth=4)
    
    plt.xlim(-5, 5); plt.ylim(0, window_size); plt.axis('off'); plt.tight_layout(pad=0)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return Image.open(buf).convert("RGB").resize((448, 256))

def generate_smooth_video(user_prompt):
    pipeline = load_model()
    if pipeline is None:
        return None, "模型加载失败，请检查路径"

    # 1. 获取 LLM 规划
    raw_traj = get_trajectory_from_llm(user_prompt)
    
    # 2. 插值变平滑 (30 -> 120)
    smooth_traj = interpolate_trajectory(raw_traj, TOTAL_FRAMES)
    
    print(f"🎬 开始渲染... (FPS: {TARGET_FPS} | 总帧数: {TOTAL_FRAMES})")
    
    frames = []
    generator = torch.Generator(device="cuda").manual_seed(42)
    # 动态调整视野：总是看未来约 3 秒的路
    window_size = int(TOTAL_FRAMES / 3) 

    for i in range(TOTAL_FRAMES):
        if i % 10 == 0:
            print(f"🚀 进度: {i}/{TOTAL_FRAMES} 帧")
            
        map_img = draw_smooth_map(smooth_traj, i, window_size=window_size)
        
        # 3. 生成每一帧
        frame = pipeline(
            prompt=f"first person view driving video, {user_prompt}, realistic highway, 4k, motion blur",
            negative_prompt="blurry, distorted, text, low quality, cartoon",
            image=map_img,
            num_inference_steps=15, # 步数调低至15以加快速度
            generator=generator
        ).images[0]
        
        frames.append(np.array(frame))

    video_path = "smooth_driving_12fps.mp4"
    imageio.mimsave(video_path, frames, fps=TARGET_FPS)
    print(f"🎉 视频生成完成: {video_path}")
    
    return draw_smooth_map(smooth_traj, 0, window_size), video_path

# ================= 🎨 界面启动 =================
with gr.Blocks(title="UniMVM Pro (Port 6008)") as demo:
    gr.Markdown(f"# 🚗 UniMVM 自动驾驶视频生成 (Pro版)")
    gr.Markdown(f"**状态**: 端口 6008 | {TARGET_FPS} FPS | 10秒时长")
    
    with gr.Row():
        txt_input = gr.Textbox(label="输入指令", value="向左平稳变道", placeholder="例如：向右急转弯")
        btn_submit = gr.Button("🎬 开始渲染", variant="primary")
        
    with gr.Row():
        img_pre = gr.Image(label="轨迹预览", type="pil")
        vid_out = gr.Video(label="最终视频")
        
    btn_submit.click(fn=generate_smooth_video, inputs=txt_input, outputs=[img_pre, vid_out])

if __name__ == "__main__":
    # 【关键修改】这里改成了 6008 端口，避开冲突
    try:
        demo.queue().launch(server_name="0.0.0.0", server_port=6008)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 建议：尝试修改代码最后一行，换成 server_port=6009 试试")
