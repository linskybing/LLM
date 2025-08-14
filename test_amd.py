import torch

def test_rocm_gpu():
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA (ROCm) device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = x @ y
        print("Matrix multiply result shape:", z.shape)
    else:
        print("CUDA (ROCm) GPU not available.")

if __name__ == "__main__":
    test_rocm_gpu()
