#include<cuda_runtime.h>

#define BM 64
#define BN 64
#define BK 16

#define TM 4
#define TN 4

#define PAD_A 4

#define BLOCK_Y (BM / TM) // 16
#define BLOCK_X (BM / TN) // 16

__device__ __forceinline__
unsigned int smem_addr(const void* ptr){
    return static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__
void cp_async_16B(void* smem_ptr, const void* gmem_ptr){
    unsigned int smem = smem_addr(smem_ptr);

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem), "l"(gmem_ptr)
    );
}

__device__ __forceinline__
void cp_async_commit_group(){
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__
void cp_async_wait_all(){
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__global__ void matmul_cp_async_double_buffer_kernel(const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
){
    __shared__ float As[2][BM][BK + PAD_A];
    __shared__ float Bs[2][BK][BM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty*blockDim.x + tx;

    int block_row = blockIdx.y*BM;
    int block_col = blockIdx.x*BN;

    int local_row = ty*TM;
    int local_col = tx*TN;

    float acc[TM][TN];

    #pragma unroll
    for(int i = 0; i < TM; ++i){
        for(int j = 0; j < TN; ++j){
            acc[i][j] = 0.0f;
        }
    }

    int num_k_tiles = K / BK;

    auto load_tile_async = [&](int tile, int stage){
        int a_base = tid*4;

        int a_local_row = a_base / BK;
        int a_local_col = a_base % BK;
        
        int a_global_row = block_row + a_local_row;
        int a_global_col = tile*BK + a_local_col; //当前跳到第几个大块，再加局部的列

        const float* a_src = &A[a_global_row*K + a_global_col];
        float* a_dst = &As[stage][a_local_row][a_local_col];
        cp_async_16B(a_dst, a_src);

        int b_base = tid*4;

        int b_local_row = b_base / BM;
        int b_local_col = b_base % BM;

        int b_global_row = tile*BK + b_local_row;
        int b_global_col = block_col + b_local_col;

        const float* b_src = &B[b_global_row*N + b_global_col];
        float* b_dst = &Bs[stage][b_global_row][b_global_col];
        cp_async_16B(b_dst, b_src);
    };

    load_tile_async(0, 0);
    cp_async_commit_group();

    cp_async_wait_all();
    __syncthreads();

    for(int tile = 0; tile < num_k_tiles; ++tile){
        int compute_stage = tile & 1;
        int load_stage = compute_stage ^ 1;

        int next_tile = tile + 1;

        if(next_tile < num_k_tiles){
            load_tile_async(next_tile, load_stage);
            cp_async_commit_group();
        }

        #pragma unroll
        for(int k_inner = 0; k_inner < BK; k_inner++){
            float a_frag[TM];
            float b_frag[TN];

            #pragma unroll
            for(int i = 0; i < TM; ++i){
                a_frag[i] = As[compute_stage][local_row + i][k_inner];
            }

            #pragma unroll
            for(int j = 0; j < TN; ++j){
                b_frag[j] = Bs[compute_stage][k_inner][local_col + j];
            }

            #pragma unroll
            for(int i = 0; i < TM; ++i){
                #pragma unroll
                for(int j = 0; j < TN; ++j){
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        
        if(next_tile < num_k_tiles){
            cp_async_wait_all();
            __syncthreads();
        }
    }

    #pragma unroll
        for(int i = 0; i < TM; ++i){
            int row = block_row + local_row + i;
            int col = block_col + local_col;

            float4 out = make_float4{
                acc[i][0],
                acc[i][1],
                acc[i][2],
                acc[i][3]
            };

            *reinterpret_cast<float4>(&C[row*N + col]) = out;
}