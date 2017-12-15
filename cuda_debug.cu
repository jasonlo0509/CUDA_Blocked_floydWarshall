/***
 * Author:Yun-Chen Lo
 * File:cuda_debug.cu
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
	int in, counter=0;
	int i, j, width;
	pFile = fopen ( infile , "r" );
	fscanf (pFile, "%d", &in);
	N = in;
	if(ceil(N, B) == N/B ){
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
  	while (!feof (pFile))
    {  
		fscanf (pFile, "%d", &in); 
		counter ++;
		if(counter > 1){
			if((counter-2) % 3 == 0){
				i=in;
			}
			else if ((counter-2) % 3 == 1 ){
				j=in;
			}
			else if((counter-2) % 3 == 2){
				Hostmap[width*i + j] = in;
			}
      	}
    }
    return width;
}


__global__ void floyd_phaseI(int k, int *devMap, int B, int d_N){
	__shared__ int shared_mem[32][32];
	int i = threadIdx.y;
	int j = threadIdx.x;
	int d_i = k * B + i;
	int d_j = k * B + j;
	int g_mem_index = d_i * d_N + d_j;
	shared_mem[i][j] = devMap[g_mem_index];
	__syncthreads();

	for(int l = 0; l < B; l++){
		if (shared_mem[i][l] + shared_mem[l][j] < shared_mem[i][j]){
			shared_mem[i][j] = shared_mem[i][l] + shared_mem[l][j];
		}
		__syncthreads();
	}
	devMap[g_mem_index] = shared_mem[i][j];
}

__global__ void floyd_phaseII(int k, int *devMap, int B, int d_N){
	if(blockIdx.x != k){
		__shared__ int shared_mem[32][32], shared_buffer[32][32];
		int i = threadIdx.y;
		int j = threadIdx.x;
		int d_i, d_j;
		if(blockIdx.y == 0){ 	// row
			d_i = k * B + threadIdx.y;
			d_j = blockDim.x * blockIdx.x + threadIdx.x;//problem
		}
		else { 					// col
			d_i = blockDim.x * blockIdx.x + threadIdx.y;
			d_j = k * B + threadIdx.x;
		}
		
		int g_mem_index = (i+B*k) * d_N + (j+B*k);
		shared_mem[i][j] = devMap[g_mem_index];
		g_mem_index = d_i * d_N + d_j;
		shared_buffer[i][j] = devMap[g_mem_index];
		__syncthreads();

		if(blockIdx.y == 0){
			for(int l = 0; l < B; l++){
				if(shared_mem[i][l] + shared_buffer[l][j] < shared_buffer[i][j]){
					shared_buffer[i][j] = shared_mem[i][l] + shared_buffer[l][j];
				}
				__syncthreads();
			}
		}
		else{
			for(int l = 0; l < B; l++){
				if(shared_buffer[i][l] + shared_mem[l][j] < shared_buffer[i][j]){
					shared_buffer[i][j] = shared_buffer[i][l] + shared_mem[l][j];
				}
				__syncthreads();
			}
		}
		devMap[g_mem_index] = shared_buffer[i][j];

	}
}

__global__ void floyd_phaseIII(int k, int *devMap, int B, int d_N){
	if(blockIdx.x!= k && blockIdx.y!= k){
		__shared__ int d_c[32][32], d_r[32][32];
		int base = k * B;
		int d_i = blockDim.y * blockIdx.y + threadIdx.y;
		int d_j = blockDim.x * blockIdx.x + threadIdx.x;
		int i = threadIdx.y;
		int j = threadIdx.x;
		int col_base = (base + i) * d_N + d_j;
		int row_base = d_i * d_N + (base + j);
		base = d_i * d_N + d_j;
		d_r[i][j] = devMap[col_base];
		d_c[i][j] = devMap[row_base];
		int oldD = devMap[base];
		__syncthreads();

		int newD;
		for (int t = 0; t < B; t++) {
			newD = d_c[i][t] + d_r[t][j];
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
    
    //int d_N = N;

    printf("BLKSIZE = %d",BLKSIZE);
    printf("round = %d\n", round);
    for(k = 0; k<round; k++){
    	floyd_phaseI<<<1, blockSize1>>>(k, devMap, BLKSIZE, width);
    	floyd_phaseII<<<gridSize2, blockSize2>>>(k, devMap, BLKSIZE, width);
    	floyd_phaseIII<<<gridSize3, blockSize3>>>(k, devMap, BLKSIZE, width);
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
	for(int i = 0; i < width*width; i++)
		printf("%d ", Hostmap[i]);
	printf("\n");
	
	printf("Hostmap\n");
	printf("%d %d\n", Hostmap[1], Hostmap[width*width-1]);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		    printf("Error: %s\n", cudaGetErrorString(err));
	int *FinalMap;
	cudaMallocHost(&FinalMap, (N*N)*sizeof(int));
	for(int i = 0; i < width; i++){
		for(int j = 0; j < width; j++){
			if(i < N && j < N)
				FinalMap[N * i + j] = Hostmap[width*i + j];
		}
	}
	saveSolution(FinalMap, outfile);
	return 0;
}