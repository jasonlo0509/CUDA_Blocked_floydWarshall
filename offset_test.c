#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include <omp.h>
int main(int argc, char** argv){
	/* Assign GPU to process */
	int num_cpus, num_gpus, cpu_thread_id, gpuid;
	cudaGetDeviceCount(&num_gpus);
	num_cpus = omp_get_max_threads();
	printf("there is %d cpus and %d gpus\n", num_cpus, num_gpus);
}