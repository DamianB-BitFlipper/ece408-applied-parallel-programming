// MP 4 Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>
#include <cstdint>

#define BLOCK_SIZE 512 //@@ You can change this
#define WARP_SIZE 32
#define WINDOW_SIZE_PER_BLOCK 4

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

template <uint32_t blockSize>
__device__ void warpSum(volatile float* partialSum, uint32_t tid) {
    // Unroll the last warp of summing computation
    if (blockSize >= 64) {
        partialSum[tid] += partialSum[tid + 32];
    }
    if (blockSize >= 32) {
        partialSum[tid] += partialSum[tid + 16];
    }
    if (blockSize >= 16) {
        partialSum[tid] += partialSum[tid + 8];
    }
    if (blockSize >= 8) {
        partialSum[tid] += partialSum[tid + 4];
    }
    if (blockSize >= 4) {
        partialSum[tid] += partialSum[tid + 2];
    }
    if (blockSize >= 2) {
        partialSum[tid] += partialSum[tid + 1];
    }
    return;
}

template <uint32_t blockSize>
__global__ void sum(float* input, float* output, int len) {
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    __shared__ float partialSum[blockSize];

    uint32_t tid = threadIdx.x;
    uint32_t gridSize = blockSize * gridDim.x;
    uint32_t start = blockIdx.x * blockDim.x;

    // Each thread loads and adds its respective value from the grid over all grid strides
    partialSum[tid] = 0;
    for (uint32_t inputLoc = start + tid; inputLoc < len; inputLoc += gridSize) {
        partialSum[tid] += input[inputLoc];
    }

    // Wait for all threads to load their respective data
    __syncthreads();

    if (blockSize >= 1024) {
        if (tid < 512) {
            partialSum[tid] += partialSum[tid + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (tid < 256) {
            partialSum[tid] += partialSum[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            partialSum[tid] += partialSum[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            partialSum[tid] += partialSum[tid + 64];
        }
        __syncthreads();
    }

    // Perform the last bit of summation on the final warp with no need to
    // synchronize since it all executes in one warp anyway
    if (tid < WARP_SIZE) {
        warpSum<blockSize>(partialSum, tid);
    }

    // The final partial sum is located at `partialSum[0]`
    if (tid == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

int main(int argc, char** argv) {
    wbArg_t args;
    float* hostInput; // The input 1D list
    float hostOutput; // The output value
    float* deviceInput;
    float* deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float*)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (WINDOW_SIZE_PER_BLOCK * BLOCK_SIZE);
    if (numInputElements % (WINDOW_SIZE_PER_BLOCK * BLOCK_SIZE)) {
        numOutputElements++;
    }

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");

    //@@ Allocate GPU memory here
    cudaMalloc(&deviceInput, numInputElements * sizeof(float));
    cudaMalloc(&deviceOutput, numOutputElements * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");

    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(numOutputElements, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    wbTime_start(Compute, "Performing sum aggregation computation");

    //@@ Launch the GPU Kernel here
    sum<BLOCK_SIZE><<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);
    sum<BLOCK_SIZE><<<1, DimBlock>>>(deviceOutput, deviceOutput, numOutputElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(&hostOutput, deviceOutput, sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");


    wbTime_stop(Compute, "Performing sum aggregation computation");

    wbTime_start(GPU, "Freeing GPU Memory");

    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, &hostOutput, 1);

    free(hostInput);

    return 0;
}

