# video_analysis_system.py
import torch
import os
import cv2
import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import random


class VideoAnalysisSystem:
    def __init__(self, model_path="../model/VL3.5-2b"):
        self.model_path = model_path
        self.setup_models()

        # 定义所有类别和对应的颜色
        self.categories = {
            # Action Classes - 红色系
            "unscrewing-bolt": (255, 0, 0),  # 红色
            "disconnecting-connector": (255, 100, 100),  # 浅红色
            "removing-battery-cover": (200, 0, 0),  # 暗红色
            "removing-busbar-and-cable": (255, 50, 50),  # 中红色
            "removing-battery-module": (255, 150, 150),  # 粉红色
            "removing-other-component": (255, 200, 200),  # 淡红色

            # Component Classes - 绿色系
            "battery_module": (0, 255, 0),  # 绿色
            "cable": (100, 255, 100),  # 浅绿色
            "top_cover": (0, 200, 0),  # 暗绿色
            "bracket_plate": (50, 255, 50),  # 中绿色
            "connector": (150, 255, 150),  # 淡绿色
            "screw": (0, 255, 100),  # 蓝绿色
            "bus_bar": (100, 255, 200),  # 青绿色
            "cable_tie": (50, 255, 150),  # 碧绿色
            "nut": (150, 255, 100),  # 黄绿色
            "rim_brace": (200, 255, 150),  # 浅黄绿
            "terminal_cover-in_series-hyundai": (0, 150, 0),  # 深绿
            "terminal_cover-bus_bar-hyundai": (0, 100, 0),  # 墨绿

            # Tool Classes - 蓝色系
            "screwdriver": (0, 0, 255),  # 蓝色
            "ratchet": (100, 100, 255),  # 浅蓝色
            "hand_drill": (0, 0, 200),  # 暗蓝色
            "pliers": (50, 50, 255),  # 中蓝色
            "drive_socket": (150, 150, 255)  # 淡蓝色
        }

        # 按类型分组
        self.action_classes = [
            "unscrewing-bolt", "disconnecting-connector", "removing-battery-cover",
            "removing-busbar-and-cable", "removing-battery-module", "removing-other-component"
        ]

        self.component_classes = [
            "battery_module", "cable", "top_cover", "bracket_plate", "connector",
            "screw", "bus_bar", "cable_tie", "nut", "rim_brace",
            "terminal_cover-in_series-hyundai", "terminal_cover-bus_bar-hyundai"
        ]

        self.tool_classes = [
            "screwdriver", "ratchet", "hand_drill", "pliers", "drive_socket"
        ]

    def setup_models(self):
        """初始化所有需要的模型"""
        from transformers import AutoTokenizer, AutoModel, AutoImageProcessor

        print("🔧 初始化视频分析系统...")

        # 加载InternVL模型
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

        print("✅ 系统初始化完成")

    def extract_video_info(self, video_path):
        """提取视频基本信息"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration": total_frames / fps
        }

    def detect_objects_in_frame(self, frame_pil):
        """在单帧中检测所有目标物体"""
        # 处理图像
        processed = self.image_processor(frame_pil, return_tensors="pt").pixel_values
        pixel_values = processed.to(self.model.device, dtype=torch.bfloat16)

        # 构建检测提示
        detection_prompt = f"""Please detect and locate all the following objects in this image:

        Actions: {', '.join(self.action_classes)}
        Components: {', '.join(self.component_classes)}
        Tools: {', '.join(self.tool_classes)}

        For each detected object, provide:
        1. Object name (must be one of the listed names)
        2. Bounding box coordinates [x1, y1, x2, y2] in format [left, top, right, bottom]
        3. Confidence level

        Return in JSON format with list of detections."""

        generation_config = {"max_new_tokens": 800, "do_sample": False}

        try:
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question=detection_prompt,
                generation_config=generation_config
            )

            # 解析响应
            detections = self.parse_detection_response(response, frame_pil.size)
            return detections

        except Exception as e:
            print(f"❌ 检测失败: {e}")
            return []

    def parse_detection_response(self, response, image_size):
        """解析模型检测响应"""
        detections = []

        try:
            # 尝试提取JSON格式的响应
            if "[" in response and "]" in response:
                # 提取JSON部分
                json_start = response.find("[")
                json_end = response.rfind("]") + 1
                json_str = response[json_start:json_end]

                import re
                # 使用正则表达式提取检测信息
                pattern = r'\{[^}]+\}'
                matches = re.findall(pattern, json_str)

                for match in matches:
                    try:
                        # 提取对象名称
                        name_match = re.search(r'"name":\s*"([^"]+)"', match)
                        bbox_match = re.search(r'"bbox":\s*\[([^\]]+)\]', match)
                        conf_match = re.search(r'"confidence":\s*([0-9.]+)', match)

                        if name_match and bbox_match:
                            name = name_match.group(1).strip()
                            # 检查是否是我们的目标类别
                            if name in self.categories:
                                bbox_str = bbox_match.group(1)
                                bbox_coords = [float(x.strip()) for x in bbox_str.split(",")]

                                if len(bbox_coords) == 4:
                                    # 转换为整数坐标
                                    x1, y1, x2, y2 = bbox_coords
                                    x1 = max(0, min(int(x1), image_size[0]))
                                    y1 = max(0, min(int(y1), image_size[1]))
                                    x2 = max(0, min(int(x2), image_size[0]))
                                    y2 = max(0, min(int(y2), image_size[1]))

                                    confidence = float(conf_match.group(1)) if conf_match else 0.5

                                    detections.append({
                                        "name": name,
                                        "bbox": [x1, y1, x2, y2],
                                        "confidence": confidence,
                                        "color": self.categories[name]
                                    })
                    except Exception as e:
                        continue

        except Exception as e:
            print(f"❌ 解析检测响应失败: {e}")

        return detections

    def create_annotated_video_with_detections(self, input_video, output_video, detection_interval=10):
        """创建带实时检测框选的视频"""
        print("🎯 开始实时检测和标注视频...")

        video_info = self.extract_video_info(input_video)
        cap = cv2.VideoCapture(input_video)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, video_info["fps"],
                              (video_info["width"], video_info["height"]))

        frame_count = 0
        total_detections = 0

        # 统计信息
        detection_stats = {category: 0 for category in self.categories.keys()}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为PIL图像进行检测
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            # 每隔一定帧数进行检测，或者每帧都检测
            detections = []
            if frame_count % detection_interval == 0:
                detections = self.detect_objects_in_frame(frame_pil)
                total_detections += len(detections)

                # 更新统计
                for detection in detections:
                    detection_stats[detection["name"]] += 1

            # 在帧上绘制检测框
            annotated_frame = self.draw_detections(frame.copy(), detections, frame_count)

            # 添加统计信息
            annotated_frame = self.draw_statistics(annotated_frame, detection_stats, frame_count, video_info)

            out.write(annotated_frame)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"📹 已处理 {frame_count} 帧，检测到 {total_detections} 个目标")

        cap.release()
        out.release()

        print(f"✅ 标注视频完成: {output_video}")
        print(f"📊 检测统计: 总帧数 {frame_count}, 总检测数 {total_detections}")

        return detection_stats

    def draw_detections(self, frame, detections, frame_count):
        """在帧上绘制检测框"""
        for detection in detections:
            name = detection["name"]
            bbox = detection["bbox"]
            color = detection["color"]
            confidence = detection["confidence"]

            x1, y1, x2, y2 = bbox

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # 绘制标签背景
            label = f"{name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)

            # 绘制标签文本
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 在框内显示置信度
            cv2.putText(frame, f"{confidence:.2f}", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def draw_statistics(self, frame, detection_stats, frame_count, video_info):
        """在帧上绘制统计信息"""
        height, width = frame.shape[:2]

        # 绘制背景
        stats_bg = np.zeros((200, 400, 3), dtype=np.uint8)
        stats_bg[:, :] = (0, 0, 0)

        # 添加标题
        cv2.putText(frame, "Detection Statistics", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 按类别类型显示统计
        y_offset = 50

        # 行为类别统计
        action_count = sum(detection_stats[action] for action in self.action_classes)
        cv2.putText(frame, f"Actions: {action_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        y_offset += 20

        # 组件类别统计
        component_count = sum(detection_stats[comp] for comp in self.component_classes)
        cv2.putText(frame, f"Components: {component_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        y_offset += 20

        # 工具类别统计
        tool_count = sum(detection_stats[tool] for tool in self.tool_classes)
        cv2.putText(frame, f"Tools: {tool_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        y_offset += 30

        # 显示前5个最常检测的类别
        sorted_stats = sorted(detection_stats.items(), key=lambda x: x[1], reverse=True)
        cv2.putText(frame, "Top Detections:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

        for i, (category, count) in enumerate(sorted_stats[:5]):
            if count > 0:
                color = self.categories[category]
                cv2.putText(frame, f"{category}: {count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 15

        # 添加帧信息
        current_time = frame_count / video_info["fps"]
        cv2.putText(frame, f"Time: {current_time:.1f}s | Frame: {frame_count}",
                    (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def process_video_with_real_time_detection(self, input_video, detection_interval=10):
        """处理视频并进行实时检测"""
        print("🚀 开始视频实时检测处理")
        print("=" * 60)

        if not os.path.exists(input_video):
            print(f"❌ 视频文件不存在: {input_video}")
            return None

        # 输出文件
        output_video = "output_real_time_detection.mp4"
        report_file = "detection_report.json"

        # 执行实时检测
        print("🎯 开始实时检测...")
        detection_stats = self.create_annotated_video_with_detections(
            input_video, output_video, detection_interval
        )

        # 保存检测报告
        report_data = {
            "video_info": self.extract_video_info(input_video),
            "detection_stats": detection_stats,
            "categories": self.categories,
            "timestamp": datetime.now().isoformat(),
            "total_detections": sum(detection_stats.values())
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print("\n📊 检测统计结果:")
        for category, count in sorted(detection_stats.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"  {category}: {count} 次")

        print(f"\n🎉 实时检测完成!")
        return {
            "annotated_video": output_video,
            "detection_report": report_file,
            "statistics": detection_stats
        }


# 主程序
if __name__ == "__main__":
    # 初始化系统
    system = VideoAnalysisSystem()

    # ==================== 配置区域 ====================
    # 设置你的视频文件名
    INPUT_VIDEO = "../examples/test_video3.mp4"

    # 检测间隔（每隔多少帧检测一次，1表示每帧都检测）
    DETECTION_INTERVAL = 5  # 可以调整这个值来平衡精度和速度
    # ==================== 配置结束 ====================

    # 执行实时检测
    results = system.process_video_with_real_time_detection(
        INPUT_VIDEO,
        detection_interval=DETECTION_INTERVAL
    )

    if results:
        print("\n📁 生成的输出文件:")
        for key, value in results.items():
            if key != "statistics" and value:
                print(f"  {key}: {value}")
    else:
        print("❌ 视频处理失败")