# CUDA-kernels

一些用于学习和实验 CUDA 算子优化的实现集合，当前覆盖 GEMM、Softmax 和 RMSNorm 等常见 AI Infra / 深度学习基础算子。

这个仓库主要用于记录从朴素实现到优化实现的过程，包括线程块划分、共享内存、向量化读取、warp/block 规约以及基础性能测试。

## 目录结构

```text
.
├── GEMM/
│   ├── naive_sgemm.cu
│   ├── tiled_sgemm.cu
│   ├── register_blocking_sgemm.cu
│   ├── DoubleBuddering_sgemm.cu
│   └── Tensorcore_sgemm.cu
├── RMSnorm/
│   ├── basic_rmsnorm.cu
│   ├── basic_float4_rmsnorm.cu
│   ├── test_basic_float4_rmsnorm.cu
│   └── README.md
└── softmax/
    ├── safe_softmax.cu
    └── online_softmax.cu
```

## 当前内容

- GEMM
  - 朴素 SGEMM
  - tiled shared-memory SGEMM
  - register blocking / double buffering / Tensor Core 方向的实验实现
- Softmax
  - safe softmax
  - online softmax
  - warp-level 和 block-level reduction
- RMSNorm
  - FP32 RMSNorm forward
  - `float4` vectorized load
  - CPU reference 校验
  - cudaEvent 计时和带宽估算

## 环境要求

- NVIDIA GPU
- CUDA Toolkit
- 支持 CUDA 的 C++ 编译环境
- Windows 推荐使用 Visual Studio / x64 Native Tools Command Prompt
- Linux / WSL 推荐直接使用 `nvcc`

## 编译示例

可以按单文件方式编译对应 kernel。以 RMSNorm 为例：

```bash
nvcc -O3 RMSnorm/basic_float4_rmsnorm.cu -o rmsnorm
./rmsnorm
```

Windows PowerShell 示例：

```powershell
nvcc -O3 -Xcompiler "/utf-8" RMSnorm/basic_float4_rmsnorm.cu -o rmsnorm.exe
.\rmsnorm.exe
```

其他 `.cu` 文件可以用同样方式单独编译和实验：

```bash
nvcc -O3 softmax/safe_softmax.cu -o safe_softmax
nvcc -O3 GEMM/naive_sgemm.cu -o naive_sgemm
```

## 学习重点

- CUDA grid/block/thread 组织方式
- global memory 和 shared memory 访问模式
- `float4` 向量化读取
- warp-level / block-level reduction
- GEMM tiling、register blocking 和 double buffering 思路
- Softmax 数值稳定性和 online softmax
- RMSNorm 的正确性验证与性能测量

## 后续计划

- 补充更多 kernel 的 benchmark 和 correctness check
- 整理统一的 CMake 构建方式
- 增加 Nsight Compute 分析记录
- 补充不同 shape、block size、数据类型下的性能对比
- 完善 Tensor Core / FP16 / BF16 版本
