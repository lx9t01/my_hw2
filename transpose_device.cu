#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j]; // this line is not optimal
        // because copy data from input to output is NOT aligned. 
        // the output data memeory address is not continuous, therefore within per warp,
        // it cannot be accessed with only one single cache line, therefore requires multiple cache
        // line access, and in serial times. So it's not coalesced access. 
    // when read from input data, it's (kind of) coalesced, but write to output is not. 
    // for EACH WARP there will need 32 cache line to write. (because the elements are not continuous). 
    // In total it will need (n ** 2) / 32 number of cache line.  
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // I have developed a version with non-coaleased memory access, but now it's totally coalesced access. 

    __shared__ float data[64][65]; // memory padding using one more row
    // because write to shmem does not transpose, read from shmem takes transpose
    // so one more row will shift the column-based read bank index by 1, which 
    // essentially eliminates bank conflict when read from shmem. 

    int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    for (; j < end_j; ++j) {
        data[j - 64 * blockIdx.y][threadIdx.x] = input[i + n * j];
    }
    __syncthreads();

    i = threadIdx.x + 64 * blockIdx.y;
    for (int m = 0; m < 4; ++m) {
        j = 4 * threadIdx.y + m + 64 * blockIdx.x;
        output[i + n * j] = data[threadIdx.x][4 * threadIdx.y + m];
    }
    // the reading/writing from shmem to gloabl mem is aligned and therefore it's coalesced access
    // I believe there will be no bank conflicts because of padding, 
    // and the cache line access is minimized due to warp-based aligned
    // memory access. 
    
}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {

    __shared__ float data[64][65]; 
    // for ILP, we can reduce the calculation by store the temporary variable.
    int bx = 64 * blockIdx.x;
    int ty = 4 * threadIdx.y;
    int by = 64 * blockIdx.y;

    // int i = threadIdx.x + bx;
    // int j = ty + by;
    int index = threadIdx.x + bx + n * (ty + by);
    // with unrolling loop and separate all the memory access with calculation (ILP)
    data[ty][threadIdx.x] = input[index];
    data[ty + 1][threadIdx.x] = input[index + n];
    data[ty + 2][threadIdx.x] = input[index + 2 * n];
    data[ty + 3][threadIdx.x] = input[index + 3 * n];

    __syncthreads();

    // also we get rid of a few initialization of i and j originally inside the loop;
    // i = threadIdx.x + by;
    // j = ty + bx;
    index = threadIdx.x + by + n * (ty + bx);
    output[index] = data[threadIdx.x][ty];
    output[index + n] = data[threadIdx.x][ty + 1];
    output[index + 2 * n] = data[threadIdx.x][ty + 2];
    output[index + 3 * n] = data[threadIdx.x][ty + 3];

}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
