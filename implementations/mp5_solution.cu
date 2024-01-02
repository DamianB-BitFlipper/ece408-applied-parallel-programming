// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include <wb.h>
#include <cstdint>
#include <iostream>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__device__ void scanAddUpSweep(
    volatile float* runningSum_sm,
    int32_t len,
    int32_t& pout,
    int32_t& pin
) {
    uint32_t tid{ threadIdx.x };

    for (int32_t stride{ 1 }; stride < len; stride *= 2) {
        int32_t threadMod{ stride * 2 };
        // If the current thread is on a strided position, add the stride amounts
        if ((tid + 1) % threadMod == 0) {
            runningSum_sm[pout + tid] =
                runningSum_sm[pin + tid - stride] + runningSum_sm[pin + tid];
        } else {
            // Simply copy the value over
            runningSum_sm[pout + tid] = runningSum_sm[pin + tid];
        }

        // Swap the `pout`  and `pin` locations for the next iteration
        pout = BLOCK_SIZE - pout;
        pin = BLOCK_SIZE - pin;

        __syncthreads();
    }
}

__device__ void scanAddDownSweep(
    volatile float* runningSum_sm,
    int32_t len,
    int32_t& pout,
    int32_t& pin
) {
    uint32_t tid{ threadIdx.x };

    for (int32_t stride{ len / 2 }, i{ 1 }; stride > 1; stride /= 2, i++) {
        int32_t lookBack{ stride / 2 };
        // If the current thread is on a strided position, add the stride amounts
        if (tid > lookBack && (tid + 1 - lookBack) % stride == 0) {
            runningSum_sm[pout + tid] =
                runningSum_sm[pin + tid] + runningSum_sm[pin + tid - lookBack];
        } else {
            // Simply copy the value over
            runningSum_sm[pout + tid] = runningSum_sm[pin + tid];
        }

        // Swap the `pout`  and `pin` locations for the next iteration
        pout = BLOCK_SIZE - pout;
        pin = BLOCK_SIZE - pin;

        __syncthreads();
    }
}

__global__ void scanAddExclusive(float* input, float* output, int32_t len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
    extern __shared__ float runningSum_sm[];

    uint32_t tid{ threadIdx.x };

    // Whether to read in, write out from the 0th or 1st `BLOCK_SIZE` of `runningSum_sm`
    int32_t pout{ BLOCK_SIZE }, pin{ 0 };

    // Load the values from global memory into shared memory
    runningSum_sm[pin + tid] = (tid < len) ? input[tid] : 0;
    runningSum_sm[pout + tid] = 0;
    __syncthreads();

    // Run the up-sweep routine
    scanAddUpSweep(runningSum_sm, len, pout, pin);

    // Run the down-sweep routine
    scanAddDownSweep(runningSum_sm, len, pout, pin);

    // Copy from the `runningSum_sm` to the `output`
    if (tid < len) {
        output[tid] = runningSum_sm[pin + tid];
    }

    return;
}

int main(int argc, char** argv) {
    wbArg_t args;
    float* hostInput; // The input 1D list
    float* hostOutput; // The output list
    float* deviceInput;
    float* deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float*)wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*)malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid{ 1, 1, 1 };
    dim3 DimBlock{ BLOCK_SIZE, 1, 1 };

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scanAddExclusive<<<DimGrid, DimBlock, 2 * BLOCK_SIZE * sizeof(float)>>>(deviceInput, deviceOutput, numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(
        cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

