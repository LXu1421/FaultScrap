# test_gpu.py
import cupy as cp
import time
import numpy as np

def test_gpu_performance():
    """Test GPU performance vs CPU for a simple operation."""
    print("Testing GPU Performance...")
    print("=" * 50)
    
    # Create a large array
    size = 4096
    print(f"Array size: {size}x{size} ({size*size/1e6:.1f}M elements)")
    
    # CPU test
    print("\nCPU Test:")
    start = time.time()
    cpu_arr = np.random.randn(size, size)
    cpu_result = np.sin(cpu_arr) * np.cos(cpu_arr)
    cpu_time = time.time() - start
    print(f"  CPU time: {cpu_time:.3f}s")
    
    # GPU test
    print("\nGPU Test:")
    start = time.time()
    gpu_arr = cp.random.randn(size, size)
    gpu_result = cp.sin(gpu_arr) * cp.cos(gpu_arr)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
    gpu_time = time.time() - start
    print(cp.cuda.runtime.getDeviceCount())
    print(cp.cuda.runtime.getDeviceProperties(0)["name"])
    print(f"  GPU time: {gpu_time:.3f}s")
    
    print(f"\nSpeedup: {cpu_time/gpu_time:.1f}x")
    print("=" * 50)
    
    # Check GPU memory usage
    mem_info = cp.cuda.Device(0).mem_info
    print(f"\nGPU Memory Usage:")
    print(f"  Free: {mem_info[0]/1e9:.2f} GB")
    print(f"  Total: {mem_info[1]/1e9:.2f} GB")
    print(f"  Used: {(mem_info[1]-mem_info[0])/1e9:.2f} GB")

if __name__ == "__main__":
    test_gpu_performance()