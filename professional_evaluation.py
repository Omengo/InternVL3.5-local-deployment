# professional_evaluation_simple.py
import torch
import os
import json
from datetime import datetime
from decord import VideoReader, cpu
import numpy as np
from PIL import Image

# 专业评估问题集
EVALUATION_QUESTIONS = {
    # 维度1: 物体识别能力
    "object_recognition": [
        "视频中出现了哪些主要物体和设备？",
        "识别视频中的汽车电池型号和特征",
        "操作者使用了哪些工具？请列出工具名称",
        "电池的正负极是如何区分的？",
        "视频中有哪些安全防护设备？"
    ],

    # 维度2: 动作理解能力
    "action_understanding": [
        "描述拆卸汽车电池的主要步骤",
        "操作者的动作顺序是怎样的？",
        "哪些动作体现了专业操作规范？",
        "操作者在拆卸过程中遇到了什么困难？",
        "拆卸过程中的关键动作是什么？"
    ],

    # 维度3: 时序理解能力
    "temporal_understanding": [
        "按照时间顺序描述整个拆卸过程",
        "哪些步骤必须先于其他步骤执行？",
        "操作的时间分配有什么特点？",
        "视频中最重要的时间节点是什么？",
        "整个过程的耗时分布如何？"
    ],

    # 维度4: 细节观察能力
    "detail_observation": [
        "操作者是如何处理电池连接线的？",
        "描述电池固定装置的拆卸方式",
        "操作者有哪些安全操作细节？",
        "电池拆卸后的处理方式是什么？",
        "工作环境中有哪些需要注意的细节？"
    ],

    # 维度5: 推理分析能力
    "reasoning_analysis": [
        "拆卸汽车电池的主要风险有哪些？",
        "这个操作需要什么样的专业技能？",
        "为什么要按照这个顺序拆卸电池？",
        "操作中体现了哪些工程原理？",
        "如果操作不当可能造成什么后果？"
    ]
}


class InternVLEvaluator:
    def __init__(self, model_path="../model/VL3.5-2b"):
        self.model_path = model_path
        self.setup_model()
        self.results = []

    def setup_model(self):
        """初始化模型"""
        from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

        print("🔧 初始化评估模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        print("✅ 模型初始化完成")

    def extract_video_frames(self, video_path, num_segments=12):
        """提取视频关键帧"""
        print(f"🎥 提取视频帧: {video_path}")

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)

        # 均匀采样
        frame_indices = np.linspace(0, total_frames - 1, num_segments, dtype=int)

        frames = []
        for idx in frame_indices:
            frame = vr[idx].asnumpy()
            pil_image = Image.fromarray(frame)
            frames.append(pil_image)

        print(f"📊 提取了 {len(frames)} 个关键帧")
        return frames

    def process_frames(self, frames):
        """处理视频帧"""
        pixel_values_list = []
        for frame in frames:
            processed = self.image_processor(frame, return_tensors="pt").pixel_values
            pixel_values_list.append(processed)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        return pixel_values.to(self.model.device, dtype=torch.bfloat16)

    def ask_question(self, pixel_values, question, question_id, category):
        """向模型提问并记录结果"""
        generation_config = {
            "max_new_tokens": 500,
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.9
        }

        print(f"❓ [{category}] {question}")

        try:
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config
            )

            # 记录结果
            result = {
                "question_id": question_id,
                "category": category,
                "question": question,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "response_length": len(response)
            }

            self.results.append(result)
            print(f"💡 回答: {response}")
            print("-" * 80)

            return result

        except Exception as e:
            print(f"❌ 提问失败: {e}")
            return None

    def evaluate_video(self, video_path, questions_dict):
        """执行完整的视频评估"""
        print("🚀 开始专业视频分析评估")
        print("=" * 80)

        # 检查视频文件
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            print("请将你的汽车电池拆卸视频命名为 'car_battery_removal.mp4' 并放在 examples 文件夹中")
            return []

        # 处理视频
        frames = self.extract_video_frames(video_path)
        pixel_values = self.process_frames(frames)

        # 按类别进行评估
        total_questions = 0
        for category, questions in questions_dict.items():
            print(f"\n🔍 评估维度: {category.upper()}")
            print("=" * 50)

            for i, question in enumerate(questions, 1):
                question_id = f"{category}_{i}"
                self.ask_question(pixel_values, question, question_id, category)
                total_questions += 1

        print(f"\n🎉 评估完成! 共处理 {total_questions} 个问题")
        return self.results

    def save_results(self, output_file="evaluation_results.json"):
        """保存评估结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存到: {output_file}")

    def generate_report(self):
        """生成评估报告"""
        if not self.results:
            print("❌ 没有评估结果")
            return

        print("\n" + "=" * 80)
        print("📊 INTERNVL视频分析能力评估报告")
        print("=" * 80)

        # 按类别统计
        categories = {}
        for result in self.results:
            category = result['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        # 输出统计信息
        print(f"\n总问题数: {len(self.results)}")
        for category, results in categories.items():
            print(f"\n📈 {category.upper()} 维度:")
            print(f"   问题数量: {len(results)}")
            avg_length = sum(r['response_length'] for r in results) / len(results)
            print(f"   平均回答长度: {avg_length:.1f} 字符")


# 主程序
if __name__ == "__main__":
    # 初始化评估器
    evaluator = InternVLEvaluator()

    # 视频路径 - 修改为你的实际视频文件名
    video_path = "../examples/test_video.mp4"

    # 如果视频文件名不同，请在这里修改：
    # video_path = "../examples/你的视频文件名.mp4"

    # 执行评估
    results = evaluator.evaluate_video(video_path, EVALUATION_QUESTIONS)

    if results:
        # 保存结果
        evaluator.save_results()

        # 生成报告
        evaluator.generate_report()