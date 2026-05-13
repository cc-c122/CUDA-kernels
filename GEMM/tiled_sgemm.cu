#include<cuda_runtime.h>
#define TILE_SIZE 16

__global__
void tiled_mamtul_kernel(const float* A, const float* B, float* C, int M, int N, int K){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    float sum = 0.0f;

    for(int title = 0; title < (K + TILE_SIZE - 1)/ TILE_SIZE; ++title){
        int A_col = title*TILE_SIZE + threadIdx.x; //假设blockDim.x == TILE_SIZE
        int B_row = title*TILE_SIZE + threadIdx.y;
        
        if(row < M && A_col < K){
            As[threadIdx.y][threadIdx.x] = A[row*K + A_col];
        }else{
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if(B_row < K && col < N){
            Bs[threadIdx.y][threadIdx.x] = B[B_row*N + col];
        }else{
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for(int i = 0; i < TILE_SIZE; ++i){
            sum += As[threadIdx.y][i]*Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < M && col < N){
        C[row * N + col] = sum;
    }
}