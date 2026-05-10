# CUDA RMSNorm 算子优化

本项目使用 CUDA 实现 RMSNorm（Root Mean Square Layer Normalization，均方根归一化）算子，并对其进行基础性能测试。

当前版本主要实现了一个基于 `float4` vectorized load（向量化读取）的 RMSNorm kernel（核函数），并使用 CPU reference（CPU 参考实现）验证正确性，使用 cudaEvent（CUDA 事件）测试 kernel 平均耗时。

---

## 1. 项目背景

RMSNorm 是大语言模型中常见的归一化操作，例如 LLaMA 类模型中就大量使用 RMSNorm。

相比 LayerNorm（层归一化），RMSNorm 不计算均值，只根据输入向量的平方均值进行归一化，因此计算过程更简单。

RMSNorm 的计算过程可以理解为：

- `mean_square = mean(x²)`
- `inv_rms = 1 / sqrt(mean_square + eps)`
- `y = x * inv_rms * weight`

其中：

- `x` 是输入向量
- `weight` 是可学习的缩放参数
- `eps` 是防止除零的小常数
- `y` 是输出向量

---

## 2. 当前实现

当前主要文件：

```text
RMSnorm/basic_float4_rmsnorm.cu
```

当前实现的 kernel（核函数）：

```text
rmsnorm_float4_kernel
```

当前实现方式：

1. 一个 CUDA block（线程块）负责处理输入矩阵中的一行。
2. 一个 row（行）对应一个 hidden vector（隐藏向量）。
3. 每个 thread（线程）处理该行中的一部分元素。
4. 使用 `float4` 一次读取 4 个连续的 `float`。
5. 先计算整行的平方和。
6. 使用 blockReduceSum（线程块规约）得到整行平方和。
7. 计算 `inv_rms`。
8. 再次读取输入 `x` 和 `weight`，写出最终结果 `y`。

---

## 3. 项目特性

- 实现 RMSNorm forward（前向计算）
- 使用 `float4` 进行 vectorized load（向量化读取）
- 使用 block-level reduction（线程块级规约）计算平方和
- 使用 CPU reference（CPU 参考实现）验证正确性
- 使用 cudaEvent（CUDA 事件）测试 kernel 平均耗时
- 计算 estimated bandwidth（估算内存带宽）
- 支持 FP32（单精度浮点数）输入

---

## 4. 编译与运行

### 4.1 Windows 编译方式

建议使用：

```text
x64 Native Tools Command Prompt for VS 2022
```

或者从该终端中打开 VS Code。

如果当前目录是仓库根目录：

```powershell
nvcc -O3 -Xcompiler "/utf-8" RMSnorm/basic_float4_rmsnorm.cu -o rmsnorm.exe
```

运行：

```powershell
.\rmsnorm.exe
```

如果当前目录已经在 `RMSnorm` 文件夹下：

```powershell
nvcc -O3 -Xcompiler "/utf-8" basic_float4_rmsnorm.cu -o rmsnorm.exe
```

运行：

```powershell
.\rmsnorm.exe
```

---

### 4.2 Linux / WSL 编译方式

如果当前目录是仓库根目录：

```bash
nvcc -O3 RMSnorm/basic_float4_rmsnorm.cu -o rmsnorm
./rmsnorm
```

如果当前目录已经在 `RMSnorm` 文件夹下：

```bash
nvcc -O3 basic_float4_rmsnorm.cu -o rmsnorm
./rmsnorm
```

---

## 5. 测试设置

当前 benchmark（性能测试）设置如下：

| 配置项 | 数值 |
|---|---:|
| Data type（数据类型） | FP32 |
| Input shape（输入形状） | 4096 x 4096 |
| Block size（线程块大小） | 256 threads |
| Warmup（预热次数） | 10 |
| Repeat（正式测试次数） | 100 |
| Timing method（计时方式） | cudaEvent |

---

## 6. 正确性验证

为了验证 CUDA kernel（CUDA 核函数）的计算结果是否正确，本项目实现了 CPU reference（CPU 参考实现）。

测试时会比较：

- GPU output（GPU 输出）
- CPU reference output（CPU 参考输出）

并计算 max error（最大误差）。

当前测试输出：

```text
kernel run successfully!
first output value: 0.180613
reference first value: 0.180613
max error: 3.09944e-06
Average RMSNorm time: 0.629821 ms
Estimated bandwidth: 426.209 GB/s
```

其中：

```text
max error = 3.09944e-06
```

对于 FP32（单精度浮点数）计算来说，这个误差是可以接受的。

由于 CPU 和 GPU 的 reduction（规约）顺序不同，浮点数加法顺序也可能不同，因此结果存在微小误差是正常现象。

---

## 7. 性能测试结果

| Kernel | 输入形状 | 数据类型 | 最大误差 | 平均耗时 | 估算带宽 |
|---|---:|---:|---:|---:|---:|
| RMSNorm float4 | 4096 x 4096 | FP32 | 3.10e-6 | 0.629821 ms | 426.209 GB/s |

---

## 8. 带宽估算方式

RMSNorm 大致包含以下内存访问：

1. 第一次读取 `x`：计算平方和
2. 第二次读取 `x`：计算输出
3. 读取 `weight`：乘缩放参数
4. 写入 `y`：保存输出

因此粗略估算内存访问量为：

```text
estimated_bytes = rows * cols * sizeof(float) * 4
```

估算带宽计算方式为：

```text
bandwidth = estimated_bytes / time
```

这里的 bandwidth（带宽）是 estimated bandwidth（估算带宽），不是 Nsight Compute（NVIDIA 性能分析器）统计出的精确 DRAM throughput（显存吞吐）。

---

## 9. Kernel 设计说明

### 9.1 一行一个 block

当前实现中，一个 CUDA block（线程块）负责处理输入矩阵中的一行：

```cpp
int row = blockIdx.x;
```

每个 thread（线程）处理该行中的一部分元素：

```cpp
for (int col4 = threadIdx.x; col4 < cols4; col4 += blockDim.x)
```

这样可以让一个 block 内的多个线程共同完成一行 RMSNorm。

---

### 9.2 float4 向量化读取

为了减少访存指令数量，当前实现使用 `float4` 读取连续的 4 个 `float`：

```cpp
const float4* row_x4 = reinterpret_cast<const float4*>(row_x);
```

相比逐个 `float` 读取，`float4` 可以让每个线程一次处理更多连续数据，提高访存效率。

---

### 9.3 平方和计算

每个线程先计算自己负责元素的平方和：

```cpp
local_sumsq += vx.x * vx.x;
local_sumsq += vx.y * vx.y;
local_sumsq += vx.z * vx.z;
local_sumsq += vx.w * vx.w;
```

然后通过 blockReduceSum（线程块规约）得到整行平方和：

```cpp
float row_sumsq = blockReduceSum(local_sumsq);
```

再计算均方值和反平方根：

```cpp
float mean_square = row_sumsq / cols;
float inv_rms = rsqrtf(mean_square + eps);
```

---

### 9.4 输出计算

RMSNorm 的输出公式为：

```text
y = x * inv_rms * weight
```

在 `float4` 路径中，每次处理 4 个元素：

```cpp
vy.x = vx.x * inv_rms * vw.x;
vy.y = vx.y * inv_rms * vw.y;
vy.z = vx.z * inv_rms * vw.z;
vy.w = vx.w * inv_rms * vw.w;
```

最后写回输出：

```cpp
row_y4[col4] = vy;
```

---

## 10. 当前结果总结

当前 `float4` 版本在 `4096 x 4096` 输入规模下的测试结果：

| 指标 | 结果 |
|---|---:|
| 平均耗时 | 0.629821 ms |
| 估算带宽 | 426.209 GB/s |
| 最大误差 | 3.09944e-06 |

这说明当前 kernel 已经能够正确完成 RMSNorm 计算，并完成了基本的 CUDA 性能测试流程。

---

## 11. 后续计划

后续可以继续完善以下内容：

- 增加 naive RMSNorm（朴素版本）作为 baseline（基准版本）
- 增加 scalar load（标量读取）版本，与 `float4` 版本对比
- 增加 warp-level reduction（线程束级规约）版本
- 支持 FP16（半精度浮点数）
- 测试更多 hidden size（隐藏维度）：
  - 1024
  - 2048
  - 4096
  - 8192
  - 11008
- 使用 Nsight Compute（NVIDIA 性能分析器）进一步分析性能瓶颈
- 增加 CMake 构建方式
- 整理更多 benchmark（性能测试）表格

---

## 12. 学习收获

通过这个项目，我主要练习了：

- CUDA kernel（CUDA 核函数）的编写
- CUDA thread/block（线程/线程块）组织方式
- global memory（全局内存）访问
- `float4` vectorized load（向量化读取）
- reduction（规约）优化
- cudaEvent（CUDA 事件）性能计时
- CPU reference（CPU 参考实现）正确性验证
- memory bandwidth（内存带宽）估算
- CUDA 程序的编译、运行和调试流程

---

## 13. 总结

本项目实现了一个基于 CUDA 的 RMSNorm 算子，并使用 `float4` 进行向量化读取优化。

目前已经完成：

1. CUDA RMSNorm kernel 实现
2. CPU reference 正确性验证
3. cudaEvent 性能测试
4. estimated bandwidth 估算

当前结果：

| 项目 | 数值 |
|---|---:|
| Shape | 4096 x 4096 |
| Data type | FP32 |
| Time | 0.629821 ms |
| Estimated bandwidth | 426.209 GB/s |
| Max error | 3.09944e-06 |

该项目是我学习 CUDA 算子优化和 AI Infra（人工智能基础设施）方向的一个起点，后续会继续加入更多 baseline（基准版本）和优化版本进行对比。