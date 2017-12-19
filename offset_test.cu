#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include <omp.h>

__global__ void assign(int *dev, int cpu_thread_id){
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

int main(int argc, char** argv){
	/* Assign GPU to process */
	int num_cpus, num_gpus, cpu_thread_id, gpuid;
	cudaGetDeviceCount(&num_gpus);
	num_cpus = omp_get_max_threads();
	printf("there is %d cpus and %d gpus\n", num_cpus, num_gpus);
	int *dev1, *dev2;
	cudaMalloc(&dev1, 6 * sizeof(int));
	assign<<<1, dim3(1, 1)>>>(dev1, 0);
	int *Host;
	Host = (int *)malloc(6* sizeof(int));
	Host[5] = 379;
	for(int i=0; i<6; i++){
		printf("%d ", Host[i]);
	}
	printf("\n");
	cudaMemcpy(&Host[0], &dev1[3], 3*sizeof(int), cudaMemcpyDeviceToHost);
	for(int i=0; i<6; i++){
		printf("%d ", Host[i]);
	}
	printf("\n");
/*
#pragma omp parallel shared(num_gpus, dev1, dev2) private(cpu_thread_id, gpuid)
{
	#pragma omp critical
	{
		cpu_thread_id = omp_get_thread_num();
		cudaSetDevice(cpu_thread_id);
		cudaGetDevice(&gpuid);
		printf("cpu:%d <-> gpu:%d\n", cpu_thread_id, gpuid);
	}
	cudaGetDevice(&gpuid);
	if(gpuid == 0)
		
	else
		cudaMalloc(&dev2, 6 * sizeof(int));
	cudaDeviceEnablePeerAccess (1- gpuid,0);
	if(gpuid == 0)
		assign<<<1, dim3(1, 1)>>>(dev1, cpu_thread_id);
	else
		assign<<<1, 1>>>(dev2, cpu_thread_id);
	//cudaMemcpy(&dev1[3], dev2, 3*sizeof(int), cudaMemcpyDeviceToDevice);
	//if(gpuid==0)
	//	show<<<1, 1>>>(dev1);
}*/
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));
}