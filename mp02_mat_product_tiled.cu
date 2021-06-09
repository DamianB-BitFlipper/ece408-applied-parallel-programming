#include<stdio.h>
#include<time.h>
#include<stdlib.h>
#include<time.h>

#define TILE_WIDTH 8
__global__ tiled_mat_product(float* M, float* N, float* P, int Width){
    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float PValue = 0;

    for(int m = 0; m < Width / TILE_WIDTH; m++){
        subTileM[ty][tx] = M[Row * Width + m * TILE_WIDTH + tx];
        subTileN[ty][tx] = M[(m * TILE_WIDTH + ty) * Width + ty + Col];
        __syncthreads();
        for(int k = 0; k < TILE_WIDTH; k++){
            PValue += subTileM[ty][k] * subTileN[k][tx];
        }
        __syncthreads();
    }
    P[Row * Width + Col] = PValue;
}


int main(){
    srand(time(NULL));
    int Width = 256 * TILE_WIDTH;
    dim3 block_dim(TILE_WIDTH, 1, 1)
    dim3 grid_dim(Width / TILE_WIDTH, 1, 1)
    // allocate host memory
    float* M_host = (float*) malloc(Width * Width * sizeof(float));
    float* N_host = (float*) malloc(Width * Width * sizeof(float));
    float* P_host = (float*) malloc(Width * Width * sizeof(float));

    // init array
    for(int index = 0; index < Width * Width; index++){
        M_host[index] = rand() % 17;
        N_host[index] = rand() % 17;
    }
    
    // device memory pointer
    float* M_device;
    float* N_device;
    float* P_device;
    // device memory
    cudaMalloc((void**)&M_device, Width * Width * sizeof(float));
    cudaMalloc((void**)&N_device, Width * Width * sizeof(float));
    cudaMalloc((void**)&P_device, Width * Width * sizeof(float));

    // copy data from host to device

    cudaMemcpy(M_device, M_host, Width * Width * sizeof(float));
    cudaMemcpy(N_device, N_host, Width * Width * sizeof(float));
    // init value
    tiled_mat_product<<<grid_dim, block_dim>>>(M_device, N_device, P_device, Width);
    
    // copy result from device to host

    cudaMemcpy(P_host, P_device, Width * Width * sizeof(float));


}