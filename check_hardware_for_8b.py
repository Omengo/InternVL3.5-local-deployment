# hardware_check_8b.py
"""
Hardware compatibility checker for InternVL 8B model
Tests GPU memory, system RAM, disk space, and CUDA compatibility
"""

import torch
import sys
import os
import platform
import psutil
import json
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def check_gpu():
    """Check GPU specifications and compatibility"""
    print_header("GPU CHECK")

    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available")
        print("   - PyTorch cannot use GPU acceleration")
        print("   - 8B model will run very slowly on CPU only")
        return False, {}

    gpu_info = {}
    gpu_count = torch.cuda.device_count()
    print(f"✅ CUDA is available")
    print(f"📊 Number of GPUs: {gpu_count}")

    for i in range(gpu_count):
        print(f"\n  GPU {i}:")
        props = torch.cuda.get_device_properties(i)

        # GPU name
        gpu_name = props.name
        print(f"    Name: {gpu_name}")
        gpu_info[f"gpu_{i}_name"] = gpu_name

        # GPU memory in GB
        total_memory = props.total_memory / (1024 ** 3)
        gpu_info[f"gpu_{i}_memory_gb"] = total_memory
        print(f"    Total Memory: {total_memory:.2f} GB")

        # CUDA capability
        major = props.major
        minor = props.minor
        cuda_cap = f"{major}.{minor}"
        gpu_info[f"gpu_{i}_cuda_capability"] = cuda_cap
        print(f"    CUDA Capability: {cuda_cap}")

        # Multi-processor count
        print(f"    Multiprocessors: {props.multi_processor_count}")

        # Check minimum requirements
        if total_memory >= 16:
            print(f"    ✅ SUFFICIENT for FP16 precision (16GB+)")
        elif total_memory >= 8:
            print(f"    ⚠️  MINIMUM for 8-bit quantization (8GB+)")
        else:
            print(f"    ❌ INSUFFICIENT for 8B model (<8GB)")

    return True, gpu_info


def check_system_ram():
    """Check system RAM"""
    print_header("SYSTEM RAM CHECK")

    ram = psutil.virtual_memory()
    total_gb = ram.total / (1024 ** 3)
    available_gb = ram.available / (1024 ** 3)

    print(f"📊 Total RAM: {total_gb:.2f} GB")
    print(f"📊 Available RAM: {available_gb:.2f} GB")

    # Requirements for 8B model
    if total_gb >= 32:
        print("✅ SUFFICIENT RAM (32GB+)")
        ram_status = "sufficient"
    elif total_gb >= 16:
        print("⚠️  MINIMUM RAM (16GB+) - May work with CPU offloading")
        ram_status = "minimum"
    else:
        print("❌ INSUFFICIENT RAM (<16GB)")
        ram_status = "insufficient"

    return {
        "total_ram_gb": total_gb,
        "available_ram_gb": available_gb,
        "ram_status": ram_status
    }


def check_disk_space(model_path="../model/VL3.5-8b"):
    """Check available disk space"""
    print_header("DISK SPACE CHECK")

    # Get disk usage for the model path
    if os.path.exists(model_path):
        path = Path(model_path)
        disk_usage = psutil.disk_usage(path)
    else:
        # Use current directory
        disk_usage = psutil.disk_usage('.')

    total_gb = disk_usage.total / (1024 ** 3)
    free_gb = disk_usage.free / (1024 ** 3)

    print(f"📊 Total Disk Space: {total_gb:.2f} GB")
    print(f"📊 Free Disk Space: {free_gb:.2f} GB")

    # 8B model requires ~20GB
    if free_gb >= 30:
        print("✅ SUFFICIENT for 8B model (30GB+ recommended)")
        disk_status = "sufficient"
    elif free_gb >= 20:
        print("⚠️  MINIMUM for 8B model (20GB required)")
        disk_status = "minimum"
    else:
        print("❌ INSUFFICIENT disk space (<20GB)")
        disk_status = "insufficient"

    return {
        "total_disk_gb": total_gb,
        "free_disk_gb": free_gb,
        "disk_status": disk_status
    }


def check_python_environment():
    """Check Python and PyTorch versions"""
    print_header("PYTHON ENVIRONMENT")

    print(f"🐍 Python Version: {sys.version}")
    print(f"🔥 PyTorch Version: {torch.__version__}")
    print(f"⚡ CUDA Version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Not available'}")

    # Check for required packages
    required_packages = ['transformers', 'accelerate', 'bitsandbytes']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing_packages.append(package)

    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        "missing_packages": missing_packages
    }


def check_torch_cuda_compatibility():
    """Check if PyTorch CUDA version matches installed CUDA"""
    print_header("CUDA COMPATIBILITY")

    if not torch.cuda.is_available():
        print("CUDA not available - skipping compatibility check")
        return {"compatible": False, "message": "CUDA not available"}

    # Get PyTorch's CUDA version
    pytorch_cuda = torch.version.cuda
    print(f"PyTorch compiled with CUDA: {pytorch_cuda}")

    # Try to get system CUDA version from nvcc
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    system_cuda = line.split('release')[-1].strip().split(',')[0]
                    print(f"System CUDA version: {system_cuda}")

                    # Check compatibility
                    pytorch_major = pytorch_cuda.split('.')[0]
                    system_major = system_cuda.split('.')[0]

                    if pytorch_major == system_major:
                        print("✅ CUDA versions are compatible")
                        return {"compatible": True, "pytorch_cuda": pytorch_cuda, "system_cuda": system_cuda}
                    else:
                        print(f"⚠️  CUDA version mismatch - may cause issues")
                        print(f"   PyTorch expects CUDA {pytorch_major}.x, system has CUDA {system_major}.x")
                        return {"compatible": False, "pytorch_cuda": pytorch_cuda, "system_cuda": system_cuda}
    except:
        print("ℹ️  Could not detect system CUDA version (nvcc not found)")

    return {"compatible": True, "pytorch_cuda": pytorch_cuda, "system_cuda": "unknown"}


def estimate_8b_requirements():
    """Estimate 8B model requirements for current system"""
    print_header("8B MODEL REQUIREMENTS ESTIMATE")

    gpu_available, gpu_info = check_gpu()
    ram_info = check_system_ram()

    print("\n📋 8B MODEL LOADING OPTIONS:")
    print("-" * 40)

    if gpu_available:
        # Get first GPU memory
        gpu_memory = gpu_info.get('gpu_0_memory_gb', 0)

        if gpu_memory >= 16:
            print("✅ OPTION 1: FP16 Precision (Recommended)")
            print(f"   - GPU Memory: {gpu_memory:.1f}GB ≥ 16GB ✓")
            print("   - Speed: Fast")
            print("   - Quality: Best")
            print("   - Command: dtype=torch.float16, load_in_8bit=False")

        if gpu_memory >= 8:
            print("\n✅ OPTION 2: 8-bit Quantization")
            print(f"   - GPU Memory: {gpu_memory:.1f}GB ≥ 8GB ✓")
            print("   - Speed: Moderate")
            print("   - Quality: Good")
            print("   - Command: load_in_8bit=True")

        if gpu_memory >= 4:
            print("\n⚠️  OPTION 3: 4-bit Quantization (Experimental)")
            print(f"   - GPU Memory: {gpu_memory:.1f}GB ≥ 4GB ✓")
            print("   - Speed: Slow")
            print("   - Quality: Reduced")
            print("   - Requires: bitsandbytes with 4-bit support")

        if gpu_memory < 8:
            print("\n❌ OPTION 4: CPU Only (Not Recommended)")
            print(f"   - GPU Memory: {gpu_memory:.1f}GB < 8GB")
            print("   - Speed: Very Slow (5-10x slower)")
            print("   - RAM Required: ≥32GB")
            if ram_info['total_ram_gb'] >= 32:
                print(f"   - Your RAM: {ram_info['total_ram_gb']:.1f}GB ≥ 32GB ✓")
            else:
                print(f"   - Your RAM: {ram_info['total_ram_gb']:.1f}GB < 32GB ✗")

    else:
        print("❌ NO GPU AVAILABLE")
        print("8B model will be VERY slow on CPU")
        if ram_info['total_ram_gb'] >= 32:
            print(f"✅ Sufficient RAM: {ram_info['total_ram_gb']:.1f}GB")
        else:
            print(f"❌ Insufficient RAM: {ram_info['total_ram_gb']:.1f}GB < 32GB")


def run_comprehensive_check():
    """Run all hardware checks"""
    print_header("INTERNVL 8B HARDWARE COMPATIBILITY CHECK")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")

    all_results = {
        "system_info": {
            "os": f"{platform.system()} {platform.release()}",
            "processor": platform.processor(),
            "machine": platform.machine()
        }
    }

    # Run all checks
    gpu_available, gpu_info = check_gpu()
    all_results["gpu"] = gpu_info
    all_results["gpu_available"] = gpu_available

    all_results["ram"] = check_system_ram()
    all_results["disk"] = check_disk_space()
    all_results["python_env"] = check_python_environment()
    all_results["cuda_compat"] = check_torch_cuda_compatibility()

    # Summary
    print_header("SUMMARY & RECOMMENDATIONS")

    if gpu_available:
        gpu_memory = gpu_info.get('gpu_0_memory_gb', 0)
        if gpu_memory >= 8:
            print("🎉 YOUR SYSTEM CAN RUN 8B MODEL!")
            print(f"   - GPU Memory: {gpu_memory:.1f}GB (Minimum 8GB ✓)")

            if gpu_memory >= 16:
                print("   - Recommended: Use FP16 precision")
                print("   - Expected Performance: Good")
            else:
                print("   - Recommended: Use 8-bit quantization")
                print("   - Expected Performance: Moderate")
        else:
            print("⚠️  YOUR SYSTEM MAY STRUGGLE WITH 8B MODEL")
            print(f"   - GPU Memory: {gpu_memory:.1f}GB (Minimum 8GB ✗)")
            print("   - Consider: Using cloud services (Google Colab)")
            print("   - Alternative: Continue with 2B model")
    else:
        print("❌ YOUR SYSTEM IS NOT SUITABLE FOR 8B MODEL")
        print("   - No GPU available")
        print("   - CPU inference would be extremely slow")
        print("   - Strongly recommend: Use Google Colab free GPU")

    # Save results to file
    with open("hardware_check_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n📄 Detailed results saved to: hardware_check_results.json")

    # Show requirements estimate
    estimate_8b_requirements()

    return all_results


def quick_test_loading():
    """Quick test to see if we can load a small model"""
    print_header("QUICK LOAD TEST")

    try:
        # Try to load a tiny model first to test environment
        from transformers import AutoModel

        print("Testing model loading capability...")

        # This won't actually load 8B, just test the environment
        test_model = AutoModel.from_pretrained(
            "microsoft/DialoGPT-small",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )

        print(f"✅ Successfully loaded test model")
        print(f"   Device: {test_model.device}")
        print(f"   Dtype: {test_model.dtype}")

        del test_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True
    except Exception as e:
        print(f"❌ Failed to load test model: {e}")
        return False


if __name__ == "__main__":
    print("🔍 InternVL 8B Hardware Compatibility Checker")
    print("Version 1.0 - Comprehensive System Analysis")

    try:
        results = run_comprehensive_check()

        print_header("NEXT STEPS")

        if results.get("gpu_available", False):
            gpu_memory = results["gpu"].get("gpu_0_memory_gb", 0)

            if gpu_memory >= 8:
                print("1. Install required packages if missing:")
                print("   pip install transformers accelerate bitsandbytes")
                print("\n2. Try loading 8B model with this configuration:")

                if gpu_memory >= 16:
                    print("""
   model = AutoModel.from_pretrained(
       "OpenGVLab/InternVL3.5-8B",
       torch_dtype=torch.float16,
       device_map="auto",
       trust_remote_code=True
   )""")
                else:
                    print("""
   model = AutoModel.from_pretrained(
       "OpenGVLab/InternVL3.5-8B",
       load_in_8bit=True,
       device_map="auto",
       trust_remote_code=True
   )""")

                print("\n3. For video analysis, also install:")
                print("   pip install opencv-python decord pillow")
            else:
                print("1. Use Google Colab (free GPU with 16GB):")
                print("   https://colab.research.google.com/")
                print("\n2. Alternative: Continue using 2B model")
                print("   Your current setup is better suited for 2B")
        else:
            print("1. STRONGLY RECOMMEND using cloud GPU:")
            print("   - Google Colab (free)")
            print("   - AWS/GCP/Azure (paid)")
            print("\n2. CPU inference is NOT practical for 8B model")
            print("   Expected speed: 5-10 seconds per token")

        print("\n📋 Check the saved JSON file for detailed specifications")

    except Exception as e:
        print(f"\n❌ Error during hardware check: {e}")
        print("Please ensure you have required packages installed:")
        print("pip install torch psutil")