#include<cuda_runtime.h>
#include<cuda_fp16.h>
#include<mma.h>

using namespace nvcuda;

#define BM  64
#define BN  64

#define BK  16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

#define WARPS_M (BM / WMMA_M)
#define WARPS_N (BN / WMMA_N)

#define WARPS_PER_BLOCK (WARPS_M * WARPS_N)
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE) //一个block处理c的一个64x64矩阵，我们规定WMMA使用16x16x16，c的64x64可以切成4x4个16x16，所以需要16个warp

#define CP_ASYNC_HALF_PER_COPY 8

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

__global__
void tensor_core_gemm_cp_async_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ c, int M, int N, int k){
    __shared__ half As[2][BM][BK];
    __shared__ half Bs[2][BK][BN];

    int tid = threadIdx.x;

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    int block_row = blockIdx.y*BM;
    int block_col = blockIdx.x*BN;

    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;
    
    int warp_row = warp_m*WMMA_M;
    int warp_col = warp_n*WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int num_k_tile = k / BK;

    auto load_tile_async = [&](int k_tile, int stage){
        int num_a_copies = (BM*BK) / CP_ASYNC_HALF_PER_COPY;

        for(int copy_idx = tid; copy_idx < num_a_copies; copy_idx += blockDim.x){
            
        }
    }
}
