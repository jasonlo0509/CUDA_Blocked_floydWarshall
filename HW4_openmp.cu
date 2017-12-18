/***
 * Author:Yun-Chen Lo
 * File:HW4_openmp.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>
#include <omp.h>

#define INF 1000000000

int *Hostmap; 	// host memory for input adjacent file
int N; 			// number of vertex


int ceil(int a, int b){
	return (a + b -1)/b;
}

int readInput(const char* infile, int B){
	FILE * pFile;
	int in;
	int i, j, width;
	pFile = fopen ( infile , "r" );
	int m;
	fscanf(pFile, "%d %d", &N, &m);
	if(ceil(N, B) == N/B || ceil(N, B) == 1){
		width = N;
		cudaMallocHost(&Hostmap, (width*width)*sizeof(int));
	}
	else{
		width = ceil(N, B) * B;
		cudaMallocHost(&Hostmap, (width*width)*sizeof(int));
	}
	printf("width = %d\n ", width);
	for(i=0; i<width; i++){
  		for(j=0; j<width; j++){
  			if(i!=j)
  				Hostmap[width*i + j] = INF;
  			else if(i == j && i < N)
  				Hostmap[width*i + j] = 0;
  			else
  				Hostmap[width*i + j] = INF;
  		}
  	}
  	while (--m >= 0)
    {  
		fscanf(pFile, "%d %d %d", &i, &j, &in);
		Hostmap[width*i + j] = in;
    }
    return width;
}


__global__ void floyd_phaseI(int k, int *devMap, int B, int d_N){
	extern __shared__ int shared_mem[];
	int i = threadIdx.y;
	int j = threadIdx.x;
	int base = k * B;
	int d_i = base + i;
	int d_j = base + j;
	int g_mem_index = d_i * d_N + d_j;
	shared_mem[i*B + j] = devMap[g_mem_index];
	__syncthreads();
	
	for(int l = 0; l < B; l++){
		if (shared_mem[i*B + l] + shared_mem[l*B + j] < shared_mem[i*B + j]){
			shared_mem[i*B + j] = shared_mem[i*B + l] + shared_mem[l*B + j];
		}
		__syncthreads();
	}
	devMap[g_mem_index] = shared_mem[i*B + j];
}

__global__ void floyd_phaseII(int k, int *devMap, int B, int d_N){
	extern __shared__ int S[];
	if(blockIdx.x != k){
		int *shared_mem = &S[0];
		int *shared_buffer = &S[B*B];
		int i = threadIdx.y;
		int j = threadIdx.x;
		int base = k*B;
		int d_i, d_j;
		int g_mem_index = (i+base) * d_N + (j+base);
		if(blockIdx.y == 0){ 	// row
			d_i = base + threadIdx.y;
			d_j = blockDim.x * blockIdx.x + threadIdx.x;
			shared_mem[j*B + i] = devMap[g_mem_index];
			g_mem_index = d_i * d_N + d_j;
			shared_buffer[i*B + j] = devMap[g_mem_index];
		}
		else { 					// col
			d_i = blockDim.x * blockIdx.x + threadIdx.y;
			d_j = base + threadIdx.x;
			shared_mem[i*B + j] = devMap[g_mem_index];
			g_mem_index = d_i * d_N + d_j;
			shared_buffer[j*B + i] = devMap[g_mem_index];
		}
		__syncthreads();

		if(blockIdx.y == 0){
			for(int l = 0; l < B; l++){
				if(shared_mem[l*B + i] + shared_buffer[l*B + j] < shared_buffer[i*B +j]){
					shared_buffer[i*B + j] = shared_mem[l*B +i] + shared_buffer[l*B +j];
				}
				__syncthreads();
			}
			devMap[g_mem_index] = shared_buffer[i*B +j];
		}
		else{
			for(int l = 0; l < B; l++){
				if(shared_buffer[l*B +i] + shared_mem[l*B +j] < shared_buffer[j*B +i]){
					shared_buffer[j*B +i] = shared_buffer[l*B +i] + shared_mem[l*B +j];
				}
				__syncthreads();
			}
			devMap[g_mem_index] = shared_buffer[j*B +i];
		}
	}
}

__global__ void floyd_phaseIII(int k, int *devMap, int B, int d_N, int offset, int bound){
	extern __shared__ int S[];
	
	if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y==0){
		printf("Iteration[%d]\n", k);
		for(int t_i = 0; t_i < d_N; t_i++){
			for(int t_j = 0; t_j < d_N; t_j++){
				printf("%d ", devMap[t_i*d_N + t_j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	
	if(blockIdx.x!= k && blockIdx.y!= k && blockIdx.y >= offset && blockIdx.y < bound){
		int *d_c = &S[0];
		int *d_r = &S[B*B];
		int base = k*B;
		int d_i = blockDim.y * blockIdx.y + threadIdx.y;
		int d_j = blockDim.x * blockIdx.x + threadIdx.x;
		int i = threadIdx.y;
		int j = threadIdx.x;
		int col_base = (base + i) * d_N + d_j;
		int row_base = d_i * d_N + (base + j);
		base = d_i * d_N + d_j;
		d_r[i*B + j] = devMap[col_base];
		d_c[i*B + j] = devMap[row_base];
		int oldD = devMap[base];

		__syncthreads();

		int newD;
		for (int t = 0; t < B; t++) {
			newD = d_c[i*B + t] + d_r[t*B + j];
			if (newD < oldD)
				oldD = newD;
			__syncthreads();
		}
		devMap[base] = oldD;
	}
}

void Block_floydWarshall(int* Hostmap, int** ptr_arr, int B, int width, int cpu_thread_id){
	int k;
	int round = ceil(N, B);
	int BLKSIZE;
	int offset, bound, pos, size;
	int *mymem, *othermem;
	mymem = ptr_arr[cpu_thread_id];
	othermem = ptr_arr[1-cpu_thread_id];

	if(round == 1){
    	BLKSIZE = N;
    }
    else{
    	BLKSIZE = B;
    }
    dim3 blockSize1(BLKSIZE, BLKSIZE);

    dim3 gridSize2(round, 2);
    dim3 blockSize2(BLKSIZE, BLKSIZE);

	dim3 gridSize3(round, round);
	dim3 blockSize3(BLKSIZE, BLKSIZE);

    //printf("[%d]BLKSIZE = %d\n", cpu_thread_id, BLKSIZE);
    //printf("round = %d\n", round);
    
    offset = (cpu_thread_id != 0)? ceil(round, 2) : 0;
    bound = (cpu_thread_id != 0)? round : ceil(round, 2);
    size = B * width * sizeof(int);

    //printf("offset:%d, bound:%d, size:%d\n", offset, bound, size);
    
    for(k = 0; k<round; k++){
    	#pragma omp barrier
    	floyd_phaseI<<<1, blockSize1, B*B*sizeof(int)>>>(k, mymem, BLKSIZE, width);
    	floyd_phaseII<<<gridSize2, blockSize2, 2*B*B*sizeof(int)>>>(k, mymem, BLKSIZE, width);
    	floyd_phaseIII<<<gridSize3, blockSize3, 2*B*B*sizeof(int)>>>(k, mymem, BLKSIZE, width, offset, bound);
    	pos = (k+1) * B * width;
    	if(k < round - 1){
    		if((k+1) >= offset && (k+1) < bound){
    			printf("cpu(%d) pos = %d\n",cpu_thread_id, pos);
    			cudaMemcpy(&othermem[pos], &mymem[pos], size, cudaMemcpyDeviceToDevice);
    		}
    	}
    	else{
    		//printf("o*B*w %d, size %d\n",offset* B * width, (bound-offset)*B*width);
    		cudaMemcpy(&Hostmap[offset* B * width], &mymem[offset * B * width], (bound-offset)*B*width*sizeof(int), cudaMemcpyDeviceToHost);
    	}
    }
    
}

void saveSolution(int* FinalMap, const char* outfile){
	FILE *out;
	out=fopen(outfile, "wb");
	fwrite(FinalMap,sizeof(int),N*N,out);
    fclose(out);
}

int main(int argc, char** argv) {
	const char* infile = argv[1];
	const char* outfile = argv[2];
	int B = atoi(argv[3]);			//block size
	int width;						//width of the appended array
	int *devMap;					//device memory
	int* ptr_arr[2];				//ptr_arr that saves two device memory ptr 
	width = readInput(infile, B);
	
	/* Assign GPU to process */
	int num_cpus, num_gpus, cpu_thread_id, gpuid;
	cudaGetDeviceCount(&num_gpus);
	num_cpus = omp_get_max_threads();
	printf("there is %d cpus and %d gpus\n", num_cpus, num_gpus);

#pragma omp parallel shared(num_gpus) private(cpu_thread_id, gpuid)
{
	#pragma omp critical
	{
		cpu_thread_id = omp_get_thread_num();
		cudaSetDevice(cpu_thread_id);
		cudaGetDevice(&gpuid);
		printf("cpu:%d <-> gpu:%d\n", cpu_thread_id, gpuid);
	}
	cudaMalloc(&devMap, width * width * sizeof(int));
	cudaGetDevice(&gpuid);
	printf("gpu(%d) d_ptr = %d\n", gpuid, devMap);
	ptr_arr[cpu_thread_id] = devMap;
}

#pragma omp parallel shared(Hostmap, ptr_arr) private(cpu_thread_id, gpuid)
{
	cpu_thread_id = omp_get_thread_num();
	cudaMemcpy(ptr_arr[cpu_thread_id], Hostmap, sizeof(int) * width * width, cudaMemcpyHostToDevice);
	Block_floydWarshall(Hostmap, ptr_arr, B, width, cpu_thread_id);
	cudaMemcpy(Hostmap, ptr_arr[cpu_thread_id], sizeof(int) * width * width, cudaMemcpyDeviceToHost);
	cudaFree(ptr_arr[cpu_thread_id]);
}

	/* Final Saving of matrix */
	int *FinalMap;
	cudaMallocHost(&FinalMap, (N*N)*sizeof(int));
	for(int i = 0; i < width; i++){
		for(int j = 0; j < width; j++){
			if(i < N && j < N)
				FinalMap[N * i + j] = Hostmap[width*i + j];
		}
	}
	saveSolution(FinalMap, outfile);
	cudaFreeHost(Hostmap);
	cudaFreeHost(FinalMap);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		printf("Error: %s\n", cudaGetErrorString(err));
	
	return 0;
}