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
import os
# 优先读取环境变量，读不到就用空字符串 (让用户自己填)
API_KEY = os.getenv("LLM_API_KEY", "") 
# 或者直接留空，写个注释提醒用户填
# API_KEY = "填入你的DeepSeek_Key"
BASE_URL = "https://api.deepseek.com"

MODEL_PATH = "/root/autodl-tmp/models/unimvm_video_model"
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
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
            # 使用更通用的显存优化，避免 xformers 报错
            pipe.enable_model_cpu_offload()
            print("✅ 模型加载完毕！")
        except Exception as e:
            print(f"⚠️ 模型加载警告: {e}")
    return pipe

def get_trajectory_30pts(prompt):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        print(f"🤖 LLM 正在规划路径...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个自动驾驶规划师。请输出未来10秒的30个横向坐标点，逗号分隔。范围-4到4。"},
                {"role": "user", "content": f"指令: {prompt}"},
            ],
            temperature=0.1
        )
        content = response.choices[0].message.content.strip()
        traj = np.fromstring(content, sep=',')
        # 补齐到 30 个点
        if len(traj) < 30:
            traj = np.pad(traj, (0, 30 - len(traj)), 'edge')
        return traj[:30]
    except Exception as e:
        print(f"LLM 故障: {e}")
        return np.linspace(0, 0, 30)

def draw_slow_map(trajectory, frame_idx, total_frames=30):
    """修复越界问题的绘图函数"""
    plt.figure(figsize=(4, 2.5), dpi=100)
    plt.style.use('dark_background')
    
    window_size = 10
    
    # 【修复重点】：对轨迹进行末端填充，防止切片越界
    # 这样当 frame_idx 增加时，后面总是有数据可以画
    padded_traj = np.pad(trajectory, (0, window_size), 'edge')
    
    start_y = frame_idx
    end_y = frame_idx + window_size
    
    # 绘制背景车道线
    y_bg = np.arange(window_size)
    plt.plot(np.zeros_like(y_bg) - 2.0, y_bg, color='white', linestyle='--', alpha=0.3)
    plt.plot(np.zeros_like(y_bg) + 2.0, y_bg, color='white', linestyle='--', alpha=0.3)
    
    # 绘制当前窗口内的轨迹
    current_traj_segment = padded_traj[start_y:end_y]
    plt.plot(current_traj_segment, np.arange(len(current_traj_segment)), color='red', linewidth=4)
    
    plt.xlim(-5, 5); plt.ylim(0, 10); plt.axis('off'); plt.tight_layout(pad=0)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return Image.open(buf).convert("RGB").resize((448, 256))

def generate_slow_video(user_prompt):
    traj = get_trajectory_30pts(user_prompt)
    pipeline = load_model()
    
    fps = 3 
    num_frames = 30 
    frames = []
    
    print(f"🎬 开始生成视频...")
    generator = torch.Generator(device="cuda").manual_seed(42)

    for i in range(num_frames):
        print(f"渲染中: {i+1}/{num_frames}")
        map_img = draw_slow_map(traj, i, num_frames)
        
        # 每一帧的生成
        frame = pipeline(
            prompt=f"first person view driving video, {user_prompt}, realistic highway, 4k",
            negative_prompt="blurry, distorted, text",
            image=map_img,
            num_inference_steps=20,
            generator=generator
        ).images[0]
        
        frames.append(np.array(frame))

    video_path = "stable_driving_3fps.mp4"
    imageio.mimsave(video_path, frames, fps=fps)
    return draw_slow_map(traj, 0, num_frames), video_path

# ================= 🎨 UI =================
with gr.Blocks() as demo:
    gr.Markdown("# 🚗 UniMVM 自动驾驶视频生成 (修复版)")
    with gr.Row():
        txt_input = gr.Textbox(label="输入指令", value="向左平稳变道")
        btn_submit = gr.Button("🚀 渲染视频", variant="primary")
    with gr.Row():
        img_pre = gr.Image(label="初始轨迹预览")
        vid_out = gr.Video(label="生成结果")
    btn_submit.click(fn=generate_slow_video, inputs=txt_input, outputs=[img_pre, vid_out])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=6006)