#include<cuda_runtime.h>
#include<math.h>

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

    if(row > rows) return;

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