"""
Environment & CUDA Verification Script for Training Readiness.
Reports:
- Python Version
- PyTorch Version
- Torchvision Version
- CUDA Availability & Device Name
- cuDNN Version
- Output Mode: 'CUDA GPU MODE' or 'CPU MODE'
"""

import sys

def verify_environment():
    print("==================================================")
    print("      DEEP LEARNING ENVIRONMENT VERIFICATION      ")
    print("==================================================")
    print(f"Python Version:      {sys.version.split()[0]}")

    try:
        import torch
        import torchvision
        print(f"PyTorch Version:     {torch.__version__}")
        print(f"Torchvision Version: {torchvision.__version__}")

        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available:      {cuda_available}")

        if cuda_available:
            print(f"CUDA Version:        {torch.version.cuda}")
            print(f"cuDNN Version:       {torch.backends.cudnn.version()}")
            print(f"GPU Device Count:    {torch.cuda.device_count()}")
            print(f"GPU Device Name:     {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory:          {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            print("\nSTATUS: CUDA GPU MODE (Ready for Accelerated Training)")
        else:
            print("\nSTATUS: CPU MODE (Note: For fast GPU training on Google Colab, select 'Runtime -> Change runtime type -> T4 GPU')")

    except ImportError as e:
        print(f"Error importing PyTorch/Torchvision: {e}")
        print("\nSTATUS: MISSING DEPENDENCIES")

if __name__ == "__main__":
    verify_environment()
