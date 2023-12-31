// MP 4 Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include <wb.h>
#include <cstdint>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void total(float* input, float* output, int len) {
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    __shared__ float partial_sum[2 * BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    uint32_t start = 2 * blockIdx.x * blockDim.x;

    // Each thread loads 2 values into shared memory `partial_sum`
    uint32_t input_loc = start + 2 * tid;

    // Some boundary checking
    if (input_loc < len) {
        partial_sum[2 * tid] = input[input_loc];
    } else {
        partial_sum[2 * tid] = 0;
    }

    // Some boundary checking
    if (input_loc + 1 < len) {
        partial_sum[2 * tid + 1] = input[input_loc + 1];
    } else {
        partial_sum[2 * tid + 1] = 0;
    }

    // Wait for all threads to load their respective data
    __syncthreads();

    for (int32_t stride = 1; stride <= blockDim.x; stride *= 2) {
        if (tid % stride == 0) {
            partial_sum[2 * tid] += partial_sum[2 * tid + stride];
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
    float* hostOutput; // The output list
    float* deviceInput;
    float* deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float*)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (2 * BLOCK_SIZE);
    if (numInputElements % (2 * BLOCK_SIZE)) {
        numOutputElements++;
    }
    hostOutput = (float*)malloc(numOutputElements * sizeof(float));

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
    total<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numInputElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    int ii;
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    wbTime_stop(Compute, "Performing sum aggregation computation");

    wbTime_start(GPU, "Freeing GPU Memory");

    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}

