/***
 * Author:Yun-Chen Lo
 * File:HW4_cuda.cu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <assert.h>


#define INF 1000000000


int *Hostmap; 	// host memory for input adjacent file
int *devMap;	// device memory
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
	#pragma unroll 16
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
			#pragma unroll 16
			for(int l = 0; l < B; l++){
				if(shared_mem[l*B + i] + shared_buffer[l*B + j] < shared_buffer[i*B +j]){
					shared_buffer[i*B + j] = shared_mem[l*B +i] + shared_buffer[l*B +j];
				}
				__syncthreads();
			}
			devMap[g_mem_index] = shared_buffer[i*B +j];
		}
		else{
			#pragma unroll 16
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

__global__ void floyd_phaseIII(int k, int *devMap, int B, int d_N){
	extern __shared__ int S[];
	if(blockIdx.x!= k && blockIdx.y!= k){
		//__shared__ int d_c[16][16], d_r[16][16];
		int *d_c = &S[0];
		int *d_r = &S[B*B];
		int base = k * B;
		int d_i = blockDim.y * blockIdx.y + threadIdx.y;
		int d_j = blockDim.x * blockIdx.x + threadIdx.x;
		int i = threadIdx.y;
		int j = threadIdx.x;
		int col_base = (base + i) * d_N + d_j;
		int row_base = d_i * d_N + (base + j);
		base = d_i * d_N + d_j;
		d_r[i*B + j] = devMap[col_base];
		d_c[j*B + i] = devMap[row_base];
		int oldD = devMap[base];
		__syncthreads();

		int newD;
		#pragma unroll 16
		for (int t = 0; t < B; t++) {
			newD = d_c[t*B + i] + d_r[t*B + j];
			if (newD < oldD)
				oldD = newD;
			__syncthreads();
		}
		devMap[base] = oldD;
	}
}

void Block_floydWarshall(int* devMap, int B, int width){
	int k;
	int round = ceil(N, B);
	int BLKSIZE;
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

    printf("BLKSIZE = %d",BLKSIZE);
    printf("round = %d\n", round);
    for(k = 0; k<round; k++){
    	floyd_phaseI<<<1, blockSize1, B*B*sizeof(int)>>>(k, devMap, BLKSIZE, width);
    	floyd_phaseII<<<gridSize2, blockSize2, 2*B*B*sizeof(int)>>>(k, devMap, BLKSIZE, width);
    	floyd_phaseIII<<<gridSize3, blockSize3, 2*B*B*sizeof(int)>>>(k, devMap, BLKSIZE, width);
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
	int B = atoi(argv[3]);	//block size
	int width;
	width = readInput(infile, B);

	cudaMalloc(&devMap, width * width * sizeof(int));
	cudaMemcpy(devMap, Hostmap, sizeof(int) * width * width, cudaMemcpyHostToDevice);

	Block_floydWarshall(devMap, B, width);
	
	cudaMemcpy(Hostmap, devMap, sizeof(int) * width * width, cudaMemcpyDeviceToHost);
	cudaFree(devMap);

	printf("Hostmap\n");
	printf("%d %d\n", Hostmap[1], Hostmap[width*width-1]);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		    printf("Error: %s\n", cudaGetErrorString(err));
	int *FinalMap = (int*)malloc((N*N)*sizeof(int));
	for(int i = 0; i < width; i++){
		for(int j = 0; j < width; j++){
			if(i < N && j < N)
				FinalMap[N * i + j] = Hostmap[width*i + j];
		}
	}
	saveSolution(FinalMap, outfile);
	return 0;
}