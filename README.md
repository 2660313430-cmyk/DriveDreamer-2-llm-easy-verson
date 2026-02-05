# DriveDreamer-2-llm-easy-verson
这是一个端到端的自动驾驶生成系统。用户输入自然语言指令（如“向左变道”），系统利用 DeepSeek 大模型规划轨迹，并使用 ControlNet 生成逼真的第一人称驾驶视频。
既然你已经准备好了核心文件，一份优秀的 `README.md` 是你项目成功的关键。它不仅是“说明书”，更是你作为开发者专业度的体现。
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 项目简介 (Introduction)

本项目是论文 **[UniMVM: Unified Multi-View Map-Based Video Generation]** 的非官方复现与功能增强版本。

我们通过接入 **DeepSeek V3** 大语言模型，实现了从自然语言指令到自动驾驶视频生成的端到端交互。用户只需输入如“向左变道”等指令，系统即可自动规划轨迹、生成高精地图并渲染出逼真的驾驶视频。

### 🚀 核心改进与创新 (Key Innovations)
1. **LLM 决策大脑**：通过 `videocreate.py` 接入 DeepSeek API，实现复杂驾驶意图的逻辑解析。
2. **轻量化适配**：重构训练逻辑，完美支持 **nuScenes v1.0-mini** 数据集，大幅降低实验门槛。
3. **两阶段生成架构**：
   - **Stage 1 (Map Gen)**：通过轨迹图生成高精地图 (HD Map)。
   - **Stage 2 (Video Gen)**：通过地图渲染真实驾驶场景视频。

---

## 📂 项目结构 (Project Structure)

```text
UniMVM-AutoPilot/
├── videocreate.py        # 核心交互入口：LLM + 视频渲染 (3 FPS)
├── picturecreate.py      # 单帧图像生成交互界面
├── train_stage1_map.py   # (原 train_hdmap_gen.py) 地图生成模型训练
├── train_stage2_video.py # (原 runtest.py) 视频渲染模型训练
├── scripts/
│   ├── prepare_data.py   # 数据集清洗与准备
│   └── prepare_traj.py   # 轨迹数据预处理
├── requirements.txt      # 环境依赖清单
└── .gitignore            # 排除大文件上传

```

---

## 🛠️ 快速开始 (Getting Started)

### 1. 环境准备

建议在 Python 3.10 容器环境下运行：

```bash
pip install -r requirements.txt

```

### 2. 获取 API Key

本项目使用 DeepSeek 进行轨迹规划，请在 [DeepSeek Platform](https://platform.deepseek.com/) 申请 Key，并设置环境变量：

```bash
export LLM_API_KEY="你的_DeepSeek_API_Key"

```

### 3. 数据与模型

* **数据集**：请下载 [nuScenes v1.0-mini](https://www.nuscenes.org/download) 并解压至 `nuScenes/` 目录。
* **权重文件**：请下载预训练权重并放置于 `models/` 文件夹下。 [此处建议放你的网盘/HuggingFace链接]

### 4. 启动交互界面

```bash
python videocreate.py


```

---

## 🙏 致谢 (Acknowledgements)

本项目对以下论文及开源工作表示衷心的感谢与致敬：

1. **UniMVM**: 感谢原作者团队在多视角地图生成视频领域的开创性工作。
2. **Stable Diffusion**: 本项目使用 Stable Diffusion v1.5 作为底模。
3. **nuScenes**: 感谢 Motional 团队提供的开源自动驾驶数据集。

---

## ⚖️ 开源协议 (License)

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源协议。

```<img width="1233" height="595" alt="sample1" src="https://github.com/user-attachments/assets/8906fdd4-01b8-48e8-a926-79fbf563460d" />


---<img width="1165" height="565" alt="sample2" src="https://github.com/user-attachments/assets/0b4f9d78-558d-4b6f-b6d5-fe45546acd45" />


### 💡 如何把这个文件放到 GitHub？

1. **新建文件**：在你的本地 Windows 文件夹 `UniMVM-AutoPilot` 中，右键新建一个文本文档。
2. **命名**：重命名为 **`README.md`**（注意删掉后缀的 `.txt`）。
3. **粘贴**：将上面的内容完整复制进去。
4. **提交推送**：
    
   在文件夹内运行：
   ```bash
   git add README.md
   git commit -m "docs: add detailed README and project structure"
   git push origin main

```

### 🌟 额外小建议

* **加入 Demo 动图**：如果你能把你生成的 10 秒视频转成 GIF，放在 README 的开头，会让你的项目瞬间变得非常有吸引力。
* **分阶段展示**：可以在 README 里展示一下“轨迹图 -> 高精地图 -> 最终画面”的对比图。
