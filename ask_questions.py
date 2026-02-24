# my_scripts/ask_questions.py
import torch

print("❓ 自定义问题提问")

model_path = "../model/VL3.5-2b"

try:
    from transformers import AutoTokenizer, AutoModel

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    print("✅ 模型加载成功")

    # 生成配置
    generation_config = {
        "max_new_tokens": 300,  # 可以设置更长来获得详细回答
        "do_sample": True,
        "temperature": 0.7
    }

    # === 在这里输入你的问题 ===
    my_questions = [
        # 纯文本问题（不需要图片/视频）
        "请给我生成一段",

        # 你可以继续添加更多问题...

    ]

    print("开始提问...\n")

    for i, question in enumerate(my_questions, 1):
        print(f"🎯 问题{i}: {question}")
        print("-" * 50)

        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=None,  # 纯文本问题
            question=question,
            generation_config=generation_config
        )

        print(f"🤖 回答: {response}")
        print("=" * 80 + "\n")

except Exception as e:
    print(f"错误: {e}")