#include<cuda_runtime.h>

#define C_TILE 16 // C矩阵的title大小，也是A的行/B的列方向；决定block负责多大的输出区域
#define K_TILE 16 // K方向的title大小；决定每次往前推进多远
#define THREAD_TILE_Y 2
#define THREAD_TILE_X 2

#define BLOCK_Y (C_TILE / THREAD_TILE_Y)
#define BLOCK_X (C_TILE / THREAD_TILE_X)

__global__
void matmul_reg_block_2x2_kernel(const float* A, const float* B, float* C, int M, int N, int K){
    //发射的block为8x8
    __shared__ float As[C_TILE][K_TILE];
    __shared__ float Bs[K_TILE][C_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty*blockDim.x + threadIdx.x;

    int block_row = blockIdx.y * C_TILE; //当前block处理的title在c中的起始行
    int block_col = blockIdx.x * C_TILE; //当前block处理的title在c中的起始列
    //实际就是当前block负责的C tile，在全局矩阵C中的左上角位置


    //当前线程在 C title 内负责的 2 x 2 小块的位置
    int local_row0 = ty*THREAD_TILE_Y;
    int local_row1 = local_row0 + 1;

    int local_col0 = tx*THREAD_TILE_X; 
    int local_col1 = local_col0 + 1;
    
    //全局坐标
    int row0 = block_row + local_row0;
    int row1 = block_row + local_row1;

    int col0 = block_col + local_col0;
    int col1 = block_col + local_col1;

    //四个寄存器累加
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;

    int num_tiles = (K + K_TILE - 1) / K_TILE;

    for(int tile = 0; tile < num_tiles; ++tile){
        //一个 As 256个元素
        //总共只有64个线程，一个线程要加载4个A，4个B
        for(int load = 0; load < 4; load++){
            int idx = tid*4 + load;

            int a_local_row = idx / K_TILE;
            int a_local_col = idx % K_TILE;

            int a_global_row = block_row + a_global_row;
            int a_global_col = tile*K_TILE + a_global_col;

            if(a_global_row < M && a_global_col < K){
                As[a_local_row][a_local_col] = A[a_global_row * K + a_global_col];
            }else{
                As[a_global_row][a_global_col] = 0.0f;
            }

            int b_local_row = idx / C_TILE;
            int b_local_col = idx % C_TILE;

            int b_global_row = tile*K_TILE + b_local_row;
            int b_global_col = block_col + b_local_col;

            if(b_global_row < K && b_global_col < N){
                Bs[b_global_row][b_local_col] = B[b_global_row*N + b_global_col];
            }else{
                Bs[b_global_row][b_global_col] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for(int i = 0; i < K_TILE; ++i){
            float a0 = As[local_row0][i];
            float a1 = As[local_row1][i];

            float b0 = Bs[i][local_row0];
            float b1 = Bs[i][local_col1];

            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }
    }
    __syncthreads();

    if(row0 < M && col0 < N){
        C[row0*N + col0] = sum00;
    }
    if(row0 < M && col1 < N){
        C[row0*N + col1] = sum01;
    }
    if(row1 < M && col0 < N){
        C[row1 *N + col0] = sum10;
    }
    if(row1 < M && col1 < N){
        C[row1*N + col1] = sum11;
    }
}