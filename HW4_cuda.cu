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

void readInput(const char* infile){
	FILE * pFile;
	int in, counter=0;
	int i, j;
	pFile = fopen ( infile , "r" );
	fscanf (pFile, "%d", &in);
	N = in;
	cudaMallocHost(&Hostmap, (N*N)*sizeof(int));
	for(i=0; i<N; i++){
  		for(j=0; j<N; j++){
  			if(i!=j)
  				Hostmap[N*i + j] = INF;
  			else
  				Hostmap[N*i + j] = 0;
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
				Hostmap[N*i + j] = in;
			}
      	}
    }
}

int ceil(int a, int b){
	return (a + b -1)/b;
}

__global__ void floyd_phaseI(int k, int *devMap, int B, int d_N){
	__shared__ int shared_mem[32][32];
	int i = k * B + threadIdx.y;
	int j = k * B + threadIdx.x;
	if(i < d_N && j < d_N){
		int g_mem_index = i * d_N + j;
		shared_mem[i][j] = devMap[g_mem_index];
		__syncthreads();

		for(int l = 0; l < B; l++){
			if (shared_mem[i][l] + shared_mem[l][j] < shared_mem[i][j]){
				shared_mem[i][j] = shared_mem[i][l] + shared_mem[l][j];
			}
		}
		devMap[g_mem_index] = shared_mem[i][j];
	}
}
// add memory boundary
__global__ void floyd_phaseII(int k, int *devMap, int B, int d_N){
	if(blockIdx.x != k){
		__shared__ int shared_mem[32][32], shared_buffer[32][32];
		int i = k * B + threadIdx.y;
		int j = k * B + threadIdx.x;
		int g_mem_index = i * d_N + j;
		shared_mem[i][j] = devMap[g_mem_index];

		if(blockIdx.y == 0){ 	// row
			i = k * B + threadIdx.y;
			j = B * blockIdx.x + threadIdx.x;
		}
		else { 					// col
			i = B * blockIdx.x + threadIdx.y;
			j = k * B + threadIdx.x;
		}
		g_mem_index = i * d_N + j;
		shared_buffer[i][j] = devMap[g_mem_index];
		__syncthreads();

		if(blockIdx.y == 0){
			for(int l = 0; l < B; l++){
				if(shared_mem[i][l] + shared_buffer[l][j] < shared_buffer[i][j]){
					shared_buffer[i][j] = shared_mem[i][l] + shared_buffer[l][j];
				}
			}
		}
		else{
			for(int l = 0; l < B; l++){
				if(shared_buffer[i][l] + shared_mem[l][j] < shared_buffer[i][j]){
					shared_buffer[i][j] = shared_mem[i][l] + shared_buffer[l][j];
				}
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
		int row_base = d_i * d_N + base + j;
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
		}
		devMap[base] = oldD;
	}
}

void Block_floydWarshall(int* devMap, int B){
	int k;
	int round = ceil(N, B);
	int BLKSZ;
	if(round == 1){
    	BLKSZ = N;
    }
    else{
    	BLKSZ = round;
    }
    dim3 blockSize1(BLKSZ, BLKSZ);

    dim3 gridSize2(N / BLKSZ, 2);
    dim3 blockSize2(BLKSZ, BLKSZ);

	dim3 gridSize3(N / BLKSZ, N / BLKSZ);
	dim3 blockSize3(BLKSZ, BLKSZ);
    
    int d_N = N;

    printf("N/BLKSZ = %d",N/BLKSZ);
    printf("round = %d\n", round);
    for(k = 0; k<round; k++){
    	floyd_phaseI<<<1, blockSize1>>>(k, devMap, BLKSZ, d_N);
    	//floyd_phaseII<<<gridSize2, blockSize2>>>(k, devMap, BLKSZ, d_N);
    	//floyd_phaseIII<<<gridSize3, blockSize3>>>(k, devMap, BLKSZ, d_N);
    }
}

void saveSolution(const char* outfile){
	FILE *out;
	out=fopen(outfile, "wb");
	fwrite(Hostmap,sizeof(int),N*N,out);
    fclose(out);
}

int main(int argc, char** argv) {
	const char* infile = argv[1];
	const char* outfile = argv[2];
	int B = atoi(argv[3]);	//block size
	readInput(infile);
	cudaMalloc(&devMap, N * N * sizeof(int));
	cudaMemcpy(devMap, Hostmap, sizeof(int) * N * N, cudaMemcpyHostToDevice);
	Block_floydWarshall(devMap, B);
	cudaMemcpy(Hostmap, devMap, sizeof(int) * N * N, cudaMemcpyDeviceToHost);
	printf("%d %d %d \n", Hostmap[0], Hostmap[1], Hostmap[2]);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
		    printf("Error: %s\n", cudaGetErrorString(err));
	saveSolution(outfile);
	return 0;
}