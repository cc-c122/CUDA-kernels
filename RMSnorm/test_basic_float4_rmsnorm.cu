#include<cuda_runtime.h>
#include<math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                           \
            exit(1);                                          \
        }                                                     \
    } while (0)

__device__ __forceinline__
float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__
float blockReduceSum(float val){
    static __shared__ float shared[32];
    
    val = warpReduceSum(val);

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if(lane == 0){
        shared[wid] = val;
    }

    __syncthreads();

    int Warpnums = (blockDim.x + 31) / 32;

    val = (threadIdx.x < Warpnums) ? shared[lane] : 0.0f;

    if(wid == 0){
        val = warpReduceSum(val);
    }

    if(threadIdx.x == 0){
        shared[0] = val;
    }

    __syncthreads();

    return shared[0];
}

__global__
void rmsnorm_float4_kernel(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ y,
int rows, int cols, float eps){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if(row >= rows) return;

    const float* row_x = x + row*cols;
    float* row_y = y + row*cols;

    int cols4 = cols / 4; //总行数变化
    int tail_start = cols4 * 4; //cols不足4时的尾巴
    const float4* row_x4 = reinterpret_cast<const float4*>(row_x); //把const float* 强转成 const float4*
    
    float local_sumsq = 0.0f;
    
    for(int col4 = tid; col4 < cols4; col4 += blockDim.x){
        float4 vx = row_x4[col4];

        local_sumsq += vx.x * vx.x;
        local_sumsq += vx.y * vx.y;
        local_sumsq += vx.z * vx.z;
        local_sumsq += vx.w * vx.w;
    }

    for(int col = tail_start + tid; col < cols; col += blockDim.x){
        float v = row_x[col];
        local_sumsq += v * v;
    }

    float row_sumsq = blockReduceSum(local_sumsq);

    float mean_square = row_sumsq / cols;
    float inv_rms = rsqrtf(mean_square + eps);

    const float4* weight4 = reinterpret_cast<const float4*>(weight);
    float4* row_y4 =reinterpret_cast<float4*>(row_y);

    for(int col4 = tid; col4 < cols4; col4 += blockDim.x){
        float4 vx = row_x4[col4];
        float4 vw = weight4[col4];

        float4 vy;
        vy.x = vx.x * inv_rms * vw.x;
        vy.y = vx.y * inv_rms * vw.y;
        vy.z = vx.z * inv_rms * vw.z;
        vy.w = vx.w * inv_rms * vw.w;

        row_y4[col4] = vy;
    }

    for(int col = tail_start + tid; col < cols; col += blockDim.x){
        float v = row_x[col];
        row_y[col] = v * inv_rms * weight[col];
    }
}

float max_error(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    float max_err = 0.0f;

    for (size_t i = 0; i < a.size(); i++) {
        float err = std::abs(a[i] - b[i]);
        if (err > max_err) {
            max_err = err;
        }
    }

    return max_err;
}

void rmsnorm_cpu(
    const std::vector<float>& x,
    const std::vector<float>& weight,
    std::vector<float>& y_ref,
    int rows,
    int cols,
    float eps
) {
    for (int r = 0; r < rows; r++) {
        float sumsq = 0.0f;

        for (int c = 0; c < cols; c++) {
            float v = x[r * cols + c];
            sumsq += v * v;
        }

        float mean_square = sumsq / cols;
        float inv_rms = 1.0f / std::sqrt(mean_square + eps);

        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            y_ref[idx] = x[idx] * inv_rms * weight[c];
        }
    }
}

int main() {
    // shape（输入形状）
    int rows = 4096;
    int cols = 4096;
    float eps = 1e-6f;

    // memory size（内存大小）
    size_t numel = static_cast<size_t>(rows) * cols;
    size_t x_bytes = numel * sizeof(float);
    size_t y_bytes = numel * sizeof(float);
    size_t weight_bytes = cols * sizeof(float);

    // host memory（CPU 端内存）
    std::vector<float> h_x(numel);
    std::vector<float> h_weight(cols);
    std::vector<float> h_y(numel);
    std::vector<float> h_ref(numel);

    // random init（随机初始化）
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < numel; i++) {
        h_x[i] = dist(gen);
    }

    for (int i = 0; i < cols; i++) {
        h_weight[i] = dist(gen);
    }

    // CPU reference（CPU 参考结果）
    rmsnorm_cpu(h_x, h_weight, h_ref, rows, cols, eps);

    // device memory（GPU 显存）
    float* d_x = nullptr;
    float* d_weight = nullptr;
    float* d_y = nullptr;

    CHECK_CUDA(cudaMalloc(&d_x, x_bytes));
    CHECK_CUDA(cudaMalloc(&d_weight, weight_bytes));
    CHECK_CUDA(cudaMalloc(&d_y, y_bytes));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), x_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));

    // launch config（启动配置）
    dim3 block(256);
    dim3 grid(rows);

    // warmup（预热）
    int warmup = 10;

    for (int i = 0; i < warmup; i++) {
        rmsnorm_float4_kernel<<<grid, block>>>(d_x, d_weight, d_y, rows, cols, eps);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // timing（计时）
    int repeat = 100;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < repeat; i++) {
        rmsnorm_float4_kernel<<<grid, block>>>(d_x, d_weight, d_y, rows, cols, eps);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaGetLastError());

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    float avg_ms = total_ms / repeat;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // copy result back（把结果拷回 CPU）
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, y_bytes, cudaMemcpyDeviceToHost));

    // correctness check（正确性检查）
    float max_err = max_error(h_y, h_ref);

    // bandwidth（估算内存带宽）
    // RMSNorm 大致访存：读 x 两次 + 读 weight 一次 + 写 y 一次
    double benchmark_bytes = static_cast<double>(numel) * sizeof(float) * 4;
    double bandwidth_gb_s = benchmark_bytes / (avg_ms / 1000.0) / 1e9;

    // output（输出结果）
    std::cout << "kernel run successfully!" << std::endl;
    std::cout << "first output value: " << h_y[0] << std::endl;
    std::cout << "reference first value: " << h_ref[0] << std::endl;
    std::cout << "max error: " << max_err << std::endl;
    std::cout << "Average RMSNorm time: " << avg_ms << " ms" << std::endl;
    std::cout << "Estimated bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;

    // free memory（释放显存）
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_weight));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}