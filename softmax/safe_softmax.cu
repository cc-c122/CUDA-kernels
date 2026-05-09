#include<float.h>
#include<cuda_runtime.h>
#include<math.h>

__inline__ __device__
float warpReduceMax(float val){
    for(int offset = 16; offset > 0; offset >>= 1){
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__
float warpReduceSum(float val){
    for(int offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__inline__ __device__
float blockReduceMax(float val){
    static __shared__ float shared[32];
    
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax(val);

    if(lane == 0){
        shared[wid] = val;
    }

    __syncthreads();

    int numWarps = (blockDim.x + 31) / 32;
    
    val = (threadIdx.x < numWarps) ? shared[lane] : -FLT_MAX;

    if(wid == 0){
        val = warpReduceMax(val);
    }

    if(threadIdx.x == 0){
        shared[0] = val;
    }

    __syncthreads();
    return shared[0];
}

__inline__ __device__
float blockReduceSum(float val){
    static __shared__ float shared[32];

    val = warpReduceSum(val);

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x >> 5;

    if(lane == 0){
        shared[wid] = val;
    }

    int numWarps = (blockDim.x + 31) / 32;

    val = (threadIdx.x < numWarps) ? shared[lane] : -0.0f;

    if(wid == 0){
        val = warpReduceSum(val);
    }

    if(threadIdx.x == 0){
        shared[0] = val;
    }

    return shared[0];
}

__global__
void softmax_row_kernel(const float*__restrict__ input, float* __restrict__ output, int rows, int cols){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if(row > rows) return;

    const float* row_input = input + row*cols;
    float* row_output = output + row*cols;

    float localMax = -FLT_MAX;

    for(int col = tid; col < cols; col += blockDim.x){
        localMax = fmaxf(localMax, row_input[col]);
    }

    float rowMax = blockReduceMax(localMax);

    float localSum = 0.0f;

    for(int col = tid; col < cols; col += blockDim.x){
        float val = expf(row_input[col] - rowMax);
        row_output[col] = val;
        localSum += val;
    }

    float rowSum = blockReduceSum(localSum);

    for(int col = tid; col < cols; col += blockDim.x){
        row_output[col] = row_output[col] / rowSum;
    }
    
}