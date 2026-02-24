import torch
import sys

print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")

print("-" * 20)

# 1. 检查 CUDA (显卡驱动) 是否正常
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"✅ 发现 {device_count} 张显卡")
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   显卡 {i}: {gpu_name} | 总显存: {total_mem:.2f} GB")
else:
    print("❌ 未检测到 GPU！PyTorch 正在使用 CPU 运行。")
    print("   原因可能是：没装 CUDA 版 PyTorch，或者显卡驱动没装好。")
    sys.exit()

print("-" * 20)

# 2. 检查 bitsandbytes 是否能正常导入 (这是关键！)
try:
    import bitsandbytes as bnb
    print("✅ bitsandbytes 导入成功！版本:", bnb.__version__)
    print("   量化功能应该可用。")
except ImportError:
    print("❌ bitsandbytes 导入失败！")
    print("   -> 这就是原因！没有这个库，load_in_4bit=True 无法生效。")
except Exception as e:
    print(f"❌ bitsandbytes 导入出错: {e}")
    print("   -> Windows 上这个库经常报错，导致无法量化。")

print("-" * 20)
print("测试结束。")