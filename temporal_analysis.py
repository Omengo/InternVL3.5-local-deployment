import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import warnings
import os
import numpy as np
from decord import VideoReader, cpu
import datetime

# 忽略警告
warnings.filterwarnings('ignore')

# =================配置区域=================

# 视频与模型路径 (请根据你的实际情况修改)
VIDEO_PATH = r"C:\Users\LENOVO\InternVL\examples\test_video3.mp4"
MODEL_PATH = r"C:\Users\LENOVO\InternVL\model\VL3.5-2b"

# 分析的时间窗口大小 (秒)。建议 5 秒，太短看不清动作，太长容易漏细节。
SEGMENT_DURATION = 5

# =================PPT 里的合法词汇表=================
# 强迫模型只在这些词里选，防止瞎编
VALID_ACTIONS = [
    "unscrewing-bolt", "disconnecting-connector", "removing-battery-cover",
    "removing-busbar-and-cable", "removing-battery-module", "removing-other-component"
]
VALID_COMPONENTS = [
    "battery_module-ioniq5", "cable", "top_cover", "bracket_plate",
    "connector", "screw", "bus_bar", "cable_tie", "terminal_cover"
]
VALID_TOOLS = ["screwdriver", "ratchet", "hand_drill", "pliers", "hands"]


# =================工具函数=================

def format_time(seconds):
    """把秒数转换成 00:00 格式"""
    return str(datetime.timedelta(seconds=int(seconds)))[2:]


def build_transform(input_size=448):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def get_video_info(video_path):
    """获取视频总时长和FPS"""
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()
    frame_count = len(vr)
    duration = frame_count / fps
    return duration, fps, vr


def get_segment_pixel_values(vr, start_sec, end_sec, fps, num_frames=6):
    """
    获取指定时间段内的帧
    start_sec: 开始秒数
    end_sec: 结束秒数
    num_frames: 这一段里抽几帧给模型看 (InternVL一般6-8帧效果好)
    """
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    total_frames = len(vr)

    # 边界保护
    if start_frame >= total_frames: return None
    if end_frame > total_frames: end_frame = total_frames

    # 在这一段里均匀抽帧
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames).astype(int)

    video_data = vr.get_batch(frame_indices).asnumpy()
    transform = build_transform()
    pixel_values = []
    for frame in video_data:
        img = T.ToPILImage()(frame)
        pixel_values.append(transform(img))

    # stack并转到GPU
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
    return pixel_values


# =================核心分析逻辑=================

def analyze_segment(model, tokenizer, pixel_values, start_str, end_str):
    """询问模型这一段发生了什么"""

    # 构建 Prompt：告诉模型这些是合法的词，让它描述画面
    prompt = f"""
    Describe the activity in this video segment.

    Identify ANY of the following that are clearly visible:
    - Actions: {", ".join(VALID_ACTIONS)}
    - Components: {", ".join(VALID_COMPONENTS)}
    - Tools: {", ".join(VALID_TOOLS)}

    If you see an action, describe it as: "Action: [action] using [tool] on [component]".
    If nothing significant is happening, say "No clear action".
    """

    generation_config = dict(max_new_tokens=256, do_sample=False)

    # 调用模型
    ret = model.chat(
        tokenizer,
        pixel_values=pixel_values,
        question=prompt,
        generation_config=generation_config
    )
    response = ret[0] if isinstance(ret, tuple) else ret

    return response


# =================主程序=================

def main():
    if not os.path.exists(VIDEO_PATH):
        print("Error: 视频文件不存在")
        return

    print("1. Loading Model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
    except Exception as e:
        print(f"Model Load Failed: {e}")
        return

    print("2. Reading Video Info...")
    duration, fps, vr = get_video_info(VIDEO_PATH)
    print(f"   Video Duration: {duration:.1f}s, FPS: {fps:.1f}")
    print(f"   Analysis Window: {SEGMENT_DURATION}s per step")
    print("\n" + "=" * 50)
    print("TIMELINE ANALYSIS REPORT (Exact Time Intervals)")
    print("=" * 50 + "\n")

    # === 时间切片循环 ===
    current_time = 0.0

    while current_time < duration:
        end_time = min(current_time + SEGMENT_DURATION, duration)

        # 格式化时间字符串 (例如 00:05 - 00:10)
        t_start_str = format_time(current_time)
        t_end_str = format_time(end_time)
        time_label = f"[{t_start_str} - {t_end_str}]"

        # 提取这一段的图像数据
        pixel_values = get_segment_pixel_values(vr, current_time, end_time, fps, num_frames=6)

        if pixel_values is not None:
            # 让模型分析这一段
            print(f"Processing segment {time_label}...", end="\r")
            description = analyze_segment(model, tokenizer, pixel_values, t_start_str, t_end_str)

            # 输出结果
            print(f"{time_label} Analysis:")
            print(f"{description.strip()}")
            print("-" * 30)

        # 移动到下一段
        current_time += SEGMENT_DURATION

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()