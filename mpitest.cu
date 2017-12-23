/***
 * Author:Yun-Chen Lo
 * File:mpi peer2peertest.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <cuda.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include "mpi.h"

__global__ void assign(int *dev, int cpu_thread_id){
	printf("my rank%d\n", cpu_thread_id);
	if(cpu_thread_id==0){
		dev[0] = 7;
		dev[1] = 6;
		dev[2] = 5;
		dev[3] = 0;
		dev[4] = -7;
		dev[5] = 0;
		for(int i = 0; i < 6; i++)
			printf("%d ", dev[i]);
		printf("\n");
	}
	else{
		dev[0] = 0;
		dev[1] = 0;
		dev[2] = 0;
		dev[3] = 3;
		dev[4] = 2;
		dev[5] = 1;
	}
}
__global__ void show(int *dev){
	for(int i = 0; i < 6; i++)
		printf("%d ", dev+i);
	printf("\n");
}

int main(int argc, char** argv) {
	int nprocs;
    int myrank;
	/* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int *devMap0;			//device memory
    int *devMap1;			//device memory
    cudaSetDevice(myrank);
    cudaDeviceEnablePeerAccess (1- myrank,0);
    if(myrank == 0){
    	cudaMalloc(&devMap0, 6 * sizeof(int));
    	assign<<<1, dim3(1, 1)>>>(devMap0, myrank);
    }else{
		cudaMalloc(&devMap1, 6 * sizeof(int));
		assign<<<1, dim3(1, 1)>>>(devMap1, myrank);
    }
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));
    printf("ptr = %d %d\n", devMap0, devMap1);
    //cudaMemcpy(&Hostmap[save_border], &mymem[save_border], pass_size*sizeof(int), cudaMemcpyDeviceToHost);

}