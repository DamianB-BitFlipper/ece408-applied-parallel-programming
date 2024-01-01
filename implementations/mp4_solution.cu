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

__global__ void sum(float* input, float* output, int len) {
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    __shared__ float partial_sum[2 * BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    uint32_t start = WINDOW_SIZE_PER_BLOCK * blockIdx.x * blockDim.x;

    // Each thread loads 4 values from global memory. It performs one sum outside
    // of the for-loop and stores the resulting 2 values in shared memory
    uint32_t input_loc0 = start + tid;
    bool load_loc0 = input_loc0 < len;
    uint32_t input_loc1 = input_loc0 + blockDim.x;
    bool load_loc1 = input_loc1 < len;
    uint32_t input_loc2 = input_loc1 + blockDim.x;
    bool load_loc2 = input_loc2 < len;
    uint32_t input_loc3 = input_loc2 + blockDim.x;
    bool load_loc3 = input_loc3 < len;

    // Some boundary checking
    if (load_loc0 && load_loc1) {
        partial_sum[2 * tid] = input[input_loc0] + input[input_loc1];
    } else if (load_loc0 && !load_loc1) {
        partial_sum[2 * tid] = input[input_loc0];
    } else {
        partial_sum[2 * tid] = 0;
    }

    // Some boundary checking
    if (load_loc2 && load_loc3) {
        partial_sum[2 * tid + 1] = input[input_loc2] + input[input_loc3];
    } else if (load_loc2 && !load_loc3) {
        partial_sum[2 * tid + 1] = input[input_loc2];
    } else {
        partial_sum[2 * tid + 1] = 0;
    }

    // Wait for all threads to load their respective data
    __syncthreads();

    for (int32_t stride = blockDim.x; stride > 0; stride >>= 1) {
        int32_t second_index = tid + stride;
        if (tid < stride) {
            partial_sum[tid] += partial_sum[second_index];
        }
        __syncthreads();
    }

    // The final partial sum is located at `partial_sum[0]`
    if (tid == 0) {
        output[blockIdx.x] = partial_sum[0];
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
    sum<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);
    sum<<<1, DimBlock>>>(deviceOutput, deviceOutput, numOutputElements);

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

