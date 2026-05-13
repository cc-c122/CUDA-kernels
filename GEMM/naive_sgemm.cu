#include<cuda_runtime.h>

__global__
void naive_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int k){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float sum = 0.0f;

        for(int i = 0; i < k; ++i){
            sum += A[row*k + i]*B[i*N + col];
        }
        C[row*N + col] = sum;
    }
}