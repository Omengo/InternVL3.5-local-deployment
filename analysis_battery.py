import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import warnings
import os
import numpy as np
from decord import VideoReader, cpu
import datetime

warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
VIDEO_PATH = r"C:\Users\LENOVO\InternVL\examples\test_video3.mp4"
MODEL_PATH = r"C:\Users\LENOVO\InternVL\model\VL3.5-2b"

# 【关键设置】不惜时间，追求精度
CLIP_DURATION = 6.0  # 每次看 6 秒 (看清楚完整动作)
STRIDE = 2.0  # 每隔 2 秒就看一次 (高重叠，确保不漏)
NUM_FRAMES = 12  # 每个片段采样 12 帧 (高帧率，看清工具细节)

# ================= GROUND TRUTH (标准答案) =================
GROUND_TRUTH = [
    {"id": 1, "action": "removing-battery-cover", "component": "top_cover", "tool": "hands"},
    {"id": 2, "action": "unscrewing-bolt", "component": "bracket_plate", "tool": "ratchet"},
    {"id": 3, "action": "unscrewing-bolt", "component": "screw", "tool": "hands"},
    {"id": 4, "action": "removing-other-component", "component": "bracket_plate", "tool": "hands"},
    {"id": 5, "action": "removing-busbar-and-cable", "component": "bus_bar", "tool": "ratchet"},
    {"id": 6, "action": "unscrewing-bolt", "component": "screw", "tool": "ratchet"},
    {"id": 7, "action": "removing-busbar-and-cable", "component": "bus_bar", "tool": "hands"},
    {"id": 8, "action": "disconnecting-connector", "component": "connector", "tool": "screwdriver"}
]

# 核心词汇库 (用于模糊匹配)
KEYWORDS = {
    "action": ["unscrewing", "removing", "disconnecting", "loosening", "unbolting"],
    "component": ["cover", "plate", "screw", "bolt", "bus", "bar", "cable", "connector", "module"],
    "tool": ["ratchet", "screwdriver", "hand", "glove", "plier"]
}


# ================= UTILS =================

def format_time(seconds):
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


def get_clip_pixel_values(vr, start_sec, duration, num_frames=12):
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration) * fps)

    if start_frame >= total_frames: return None
    if end_frame > total_frames: end_frame = total_frames

    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames).astype(int)
    video_data = vr.get_batch(frame_indices).asnumpy()

    transform = build_transform()
    pixel_values = []
    for frame in video_data:
        img = T.ToPILImage()(frame)
        pixel_values.append(transform(img))

    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ================= INFERENCE (CoT) =================

def analyze_clip_cot(model, tokenizer, pixel_values):
    """
    使用思维链 (Chain of Thought) 强迫模型先观察细节，再下结论。
    """
    prompt = f"""
    Step-by-step Visual Analysis:
    1. Look at the worker's hands. Are they holding a tool?
       - If yes, is it a Ratchet (thick handle, socket) or a Screwdriver (thin handle) or Pliers?
       - If no, are they using just bare Hands/Gloves?
    2. Look at the object being touched. Is it a large Cover, a metal Plate, a thick Busbar, or a small Connector?
    3. Look at the motion. Is it Unscrewing (rotation), Removing (lifting), or Disconnecting?

    Based on your analysis above, output the final result in this strict format:
    FINAL: Action=[action], Component=[component], Tool=[tool]
    """

    generation_config = dict(max_new_tokens=256, do_sample=False)

    response = model.chat(
        tokenizer,
        pixel_values=pixel_values,
        question=prompt,
        generation_config=generation_config
    )

    if isinstance(response, tuple): response = response[0]
    return response


# ================= SCANNER & ALIGNMENT =================

def scan_video(model, tokenizer, vr, duration):
    timeline = []
    curr_time = 0.0

    print(f"\n>>> Starting Dense Scan (Total Duration: {duration:.1f}s)...")
    print(f">>> Scanning every {STRIDE}s with {CLIP_DURATION}s window.\n")

    step_count = 0
    while curr_time < duration:
        end_time = min(curr_time + CLIP_DURATION, duration)

        # 提取片段
        pixel_values = get_clip_pixel_values(vr, curr_time, CLIP_DURATION, num_frames=NUM_FRAMES)
        if pixel_values is None: break
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # 推理
        print(f"Scanning window [{format_time(curr_time)} - {format_time(end_time)}]...", end="\r")
        raw_output = analyze_clip_cot(model, tokenizer, pixel_values)

        # 提取 FINAL 行
        final_res = "Unknown"
        for line in raw_output.split('\n'):
            if "FINAL:" in line:
                final_res = line.replace("FINAL:", "").strip()
                break

        timeline.append({
            "start": curr_time,
            "end": end_time,
            "raw": final_res,
            "full_thought": raw_output  # 保存完整思考过程以便调试
        })

        curr_time += STRIDE
        step_count += 1

    print(f"\n\n>>> Scan Complete. Collected {len(timeline)} data points.")
    return timeline


def check_match(ground_truth_item, prediction_str):
    """
    模糊匹配评分系统
    """
    pred = prediction_str.lower()
    score = 0

    # 检查 Action
    gt_action_parts = ground_truth_item['action'].split('-')
    if any(part in pred for part in gt_action_parts if len(part) > 3):
        score += 1

    # 检查 Component
    gt_comp_parts = ground_truth_item['component'].split('_')
    if any(part in pred for part in gt_comp_parts if len(part) > 3):
        score += 1

    # 检查 Tool
    # 特殊处理: Ratchet 和 Screwdriver 很容易混，但必须分清
    if ground_truth_item['tool'] in pred:
        score += 1
    elif "hand" in ground_truth_item['tool'] and "glove" in pred:
        score += 1

    return score >= 2, score  # 只要对两个要素就算 Pass


def align_timeline_to_ground_truth(timeline):
    """
    全时段对齐算法：在时间轴中寻找最佳匹配
    """
    print("\n" + "=" * 20 + " ALIGNMENT REPORT " + "=" * 20)

    timeline_idx = 0
    total_score = 0

    for gt in GROUND_TRUTH:
        best_match = None
        best_score = 0
        found_at_time = ""

        # 从上一个找到的位置开始往后找 (保证时间顺序)
        # 搜索窗口：假设每个步骤最多持续 30秒，只往后搜 15 个时间片
        search_limit = min(timeline_idx + 20, len(timeline))

        for i in range(timeline_idx, search_limit):
            segment = timeline[i]
            is_match, score = check_match(gt, segment['raw'])

            if score > best_score:
                best_score = score
                best_match = segment
                match_idx = i

                # 如果是满分匹配 (3分)，直接锁定
                if score == 3:
                    break

        print(f"\nTarget Step {gt['id']}: {gt['action']} | {gt['component']} | {gt['tool']}")

        if best_match and best_score >= 1:  # 至少有一点沾边
            start_str = format_time(best_match['start'])
            end_str = format_time(best_match['end'])
            print(f"  --> FOUND at [{start_str}-{end_str}]")
            print(f"  --> Model Saw: {best_match['raw']}")

            if best_score == 3:
                print("  --> [Result]: ✅ PERFECT MATCH")
                total_score += 1
            elif best_score == 2:
                print("  --> [Result]: ⚠️ GOOD (Missed 1 item)")
                total_score += 0.8
            else:
                print("  --> [Result]: ⚠️ WEAK MATCH")
                total_score += 0.3

            # 更新时间指针，下一个步骤必须在这个之后发生
            # 但为了防止跳过太远，我们只推进到 match_idx
            timeline_idx = match_idx + 1
        else:
            print("  --> [Result]: ❌ NOT FOUND (Model completely missed this step)")

    print("-" * 50)
    print(f"Final High-Precision Score: {total_score:.1f} / {len(GROUND_TRUTH)}")


# ================= MAIN =================

def main():
    if not os.path.exists(VIDEO_PATH): return
    print(f"Loading InternVL (Precision Mode)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
        model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                          device_map="auto").eval()
    except Exception as e:
        print(f"Error: {e}")
        return

    # 1. 扫描视频
    vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
    duration = len(vr) / vr.get_avg_fps()
    timeline_data = scan_video(model, tokenizer, vr, duration)

    # 2. 智能对齐评分
    align_timeline_to_ground_truth(timeline_data)


if __name__ == "__main__":
    main()