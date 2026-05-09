#include<cuda_runtime.h>
#include<float.h>
#include<math.h>

struct OnlineSoftmaxState{
    float m; //当前已看过元素里的最大值
    float l; //当前 sum(exp(x - m))
};

__device__ __forceinline__
OnlineSoftmaxState online_update(OnlineSoftmaxState state, float x){
    float old_m = state.m;
    float new_m = fmaxf(old_m, x);

    state.l = old_m*expf(old_m - new_m) + expf(x - new_m);
    state.m = new_m;
}

__device__ __forceinline__
OnlineSoftmaxState online_combine(OnlineSoftmaxState a, OnlineSoftmaxState b){
    OnlineSoftmaxState out;
    
    out.m = fmaxf(a.m, b.m);
    out.l = a.l*expf(a.m - out.m) + b.l*expf(b.m - out.m);

    return out;
}

__device__ __forceinline__
OnlineSoftmaxState warpReduceOnlineSoftmax(OnlineSoftmaxState state){
    for(int offset = 16; offset > 0; offset >>= 1){
        OnlineSoftmaxState other;

        other.m = __shfl_down_sync(0xffffffff, state.m, offset);
        other.l = __shfl_down_sync(0xffffffff, state.l, offset);

        state = online_combine(state, other);
    }

    return state;
}

__device__ __forceinline__
OnlineSoftmaxState blockReduceOnlineSoftmax(OnlineSoftmaxState state){
    static __shared__ float shared_m[32];
    static __shared__ float shared_l[32];

    state = warpReduceOnlineSoftmax(state);

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x >> 5; 

    if(lane == 0){
        shared_m[wid] = state.m;
        shared_l[wid] = state.l;
    }

    __syncthreads();

    int numWarps = (blockDim.x + 31) / 32;

    if(threadIdx.x < numWarps){
        state.m = shared_m[lane];
        state.l = shared_l[lane];
    }

    __syncthreads();

    if(wid == 0){
        state = warpReduceOnlineSoftmax(state);
    }
    
    if(threadIdx.x == 0){
        shared_l[0] = state.l;
        shared_m[0] = state.m;
    }

    __syncthreads();
    
    state.m = shared_m[0];
    state.l = shared_l[0];

    return state;
}

__global__
void online_softmax_row_kernel(const float*__restrict__ input, float*__restrict__ output, int rows, int cols){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if(row >= rows) return;

    const float* row_input = input + row*cols;
    float* row_output = output + row*cols;

    OnlineSoftmaxState state;
    state.m = -FLT_MAX;
    state.l = 0.0f;

    for(int col = tid; col < cols; col += blockDim.x){
        state = online_update(state, row_input[col]);
    }

    state = blockReduceOnlineSoftmax(state);

    float rowMax = state.m;
    float rowSum = state.l;

    for(int col = tid; col < cols; col += blockDim.x){
        row_output[col] = expf(row_input[col] - rowMax) / rowSum;
    }
}