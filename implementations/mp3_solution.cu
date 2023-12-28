#include    <wb.h>

#define wbCheck(stmt) do {                              \
        cudaError_t err = stmt;                         \
        if (err != cudaSuccess) {                       \
            wbLog(ERROR, "Failed to run stmt ", #stmt); \
            return -1;                                  \
        }                                               \
    } while(0)

const int BLOCK_WIDTH = 8;

// Compute C = A * B
__global__ void matrixMultiplyShared(float* A, float* B, float* C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
    __shared__ float subTileA[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float subTileB[BLOCK_WIDTH][BLOCK_WIDTH];

    // Compute the number of sub-tiles to cover the input matrices
    const int numSubTiles = ceil(numAColumns / (double)BLOCK_WIDTH);

    // Extract the `row` and `col` locations in the resulting `C` matrix
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Extract the respective row and column of the sub-tiles
    const int subTileARow = threadIdx.y;
    const int subTileACol = threadIdx.x;
    const int subTileBRow = threadIdx.y;
    const int subTileBCol = threadIdx.x;

    float dot_prod = 0;

    for (int subtileIdx = 0; subtileIdx < numSubTiles; subtileIdx++) {
        // Load the sub-tiles from the input matrices into shared memory
        subTileA[subTileARow][subTileACol] =
            A[row * numAColumns + subtileIdx * BLOCK_WIDTH + subTileACol];
        subTileB[subTileBRow][subTileBCol] =
            B[(subtileIdx * BLOCK_WIDTH + subTileBRow) * numBColumns + col];

        // Wait for all threads to finish loading the sub-tiles
        __syncthreads();

        // Compute the partial sum of the resulting `C` matrix
        for (int idx = 0; idx < BLOCK_WIDTH; idx++) {
            dot_prod += subTileA[subTileARow][idx] * subTileB[idx][subTileBCol];
        }
    }

    // Set the `dot_prod` to the `C` matrix
    C[row * numCColumns + col] = dot_prod;

    return;
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float* hostA; // The A matrix
    float* hostB; // The B matrix
    float* hostC; // The output C matrix
    float* deviceA;
    float* deviceB;
    float* deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;

    //@@ Allocate the hostC matrix
    hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");

    //@@ Allocate GPU memory here
    cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float));
    cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float));
    cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");

    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(
        ceil(numCColumns / (double)BLOCK_WIDTH),
        ceil(numCRows / (double)BLOCK_WIDTH),
        1);
    dim3 DimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    
    wbTime_start(Compute, "Performing CUDA computation");

    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<DimGrid, DimBlock>>>(
        deviceA, deviceB, deviceC,
        numARows, numAColumns,
        numBRows, numBColumns,
        numCRows, numCColumns);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");

    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

