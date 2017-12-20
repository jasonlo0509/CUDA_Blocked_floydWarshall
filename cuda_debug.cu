/***
 * Author:Yun-Chen Lo
 * File:cuda_debug.cu
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

#define INF 1000000000


int *Hostmap; 	// host memory for input adjacent file
int *devMap;	// device memory
int N; 			// number of vertex

clock_t begin_total, end_total, begin_I, end_I, begin_II, end_II, begin_III, end_III;
clock_t total_time, I_time=0, II_time=0, III_time=0;


int ceil(int a, int b){
	return (a + b -1)/b;
}

int readInput(const char* infile, int B){
	FILE * pFile;
	int in=0;
	int i, j, width;
	pFile = fopen ( infile , "r" );
	int fd = open(infile, O_RDONLY, (mode_t)0600);
	struct stat fileInfo = {0};
	if (fstat(fd, &fileInfo) == -1)
    {
        perror("Error getting the file size");
        exit(EXIT_FAILURE);
    }
	printf("File size is %ji\n", (intmax_t)fileInfo.st_size);
	char *map = (char *)mmap(0, fileInfo.st_size, PROT_READ, MAP_SHARED, fd, 0);
	int m = 10, start_i;
	N = 5;
	printf("map[0]:%d, map[1]:%d\n", map[0], map[1]);
	for (i = 0; i < fileInfo.st_size; i++){
        if(map[i] == ' ' || map[i] == '\n'){
            if(map[i] == '\n'){
                start_i = i;
                break;
            }
            else{
                N = in;
            }
            in = 0;
        }
        else{
            in = (int)map[i]-(int)'0' + 10 * in;
        }
    }
    printf("N:%d start_i:%d\n", N, start_i);
	/*fscanf(pFile, "%d %d", &N, &m);*/
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
  	int h_i=0, h_j=0;
  	in = 0;
  	for (i = start_i; i < fileInfo.st_size; i++){
        if(map[i] == ' ' || map[i] == '\n'){
            j++;
            if(map[i] == '\n'){
                j = 0;
                Hostmap[h_i*width + h_j] = in;
            }
            else{
                h_i = (j == 1)?in:h_i;
                h_j = (j == 2)?in:h_j;
            }
            in = 0;
        }
        else{
            in = (int)map[i]-(int)'0' + 10 * in;
        }
    }
	/*for(i =0; i< width; i++){
        for(j = 0; j<width; j++){
            printf("%d ", Hostmap[i*width + j]);
        }
        printf("\n");
    }*/
    
    return width;
}


__global__ void floyd_phaseI(int k, int *devMap, int B, int d_N){
	__shared__ int shared_mem[16][16];
	int i = threadIdx.y;
	int j = threadIdx.x;
	int base = k * B;
	int d_i = base + i;
	int d_j = base + j;
	int g_mem_index = d_i * d_N + d_j;
	shared_mem[i][j] = devMap[g_mem_index];
	__syncthreads();
	#pragma unroll 16
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
		__shared__ int shared_mem[16][16], shared_buffer[16][16];
		int i = threadIdx.y;
		int j = threadIdx.x;
		int base = k*B;
		int d_i, d_j;
		int g_mem_index = (i+base) * d_N + (j+base);
		if(blockIdx.y == 0){ 	// row
			d_i = base + threadIdx.y;
			d_j = blockDim.x * blockIdx.x + threadIdx.x;
			shared_mem[j][i] = devMap[g_mem_index];
			g_mem_index = d_i * d_N + d_j;
			shared_buffer[i][j] = devMap[g_mem_index];
		}
		else { 					// col
			d_i = blockDim.x * blockIdx.x + threadIdx.y;
			d_j = base + threadIdx.x;
			shared_mem[i][j] = devMap[g_mem_index];
			g_mem_index = d_i * d_N + d_j;
			shared_buffer[j][i] = devMap[g_mem_index];
		}
		__syncthreads();

		if(blockIdx.y == 0){
			#pragma unroll 16
			for(int l = 0; l < B; l++){
				if(shared_mem[l][i] + shared_buffer[l][j] < shared_buffer[i][j]){
					shared_buffer[i][j] = shared_mem[l][i] + shared_buffer[l][j];
				}
				__syncthreads();
			}
			devMap[g_mem_index] = shared_buffer[i][j];
		}
		else{
			#pragma unroll 16
			for(int l = 0; l < B; l++){
				if(shared_buffer[l][i] + shared_mem[l][j] < shared_buffer[j][i]){
					shared_buffer[j][i] = shared_buffer[l][i] + shared_mem[l][j];
				}
				__syncthreads();
			}
			devMap[g_mem_index] = shared_buffer[j][i];
		}
	}
}

__global__ void floyd_phaseIII(int k, int *devMap, int B, int d_N){
	if(blockIdx.x!= k && blockIdx.y!= k){
		__shared__ int d_c[16][16], d_r[16][16];
		int base = k * B;
		int d_i = blockDim.y * blockIdx.y + threadIdx.y;
		int d_j = blockDim.x * blockIdx.x + threadIdx.x;
		int i = threadIdx.y;
		int j = threadIdx.x;
		int col_base = (base + i) * d_N + d_j;
		int row_base = d_i * d_N + (base + j);
		base = d_i * d_N + d_j;
		d_r[i][j] = devMap[col_base];
		d_c[j][i] = devMap[row_base];
		int oldD = devMap[base];
		__syncthreads();

		int newD;
		#pragma unroll 16
		for (int t = 0; t < B; t++) {
			newD = d_c[t][i] + d_r[t][j];
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
    	begin_I = clock();
    	floyd_phaseI<<<1, blockSize1>>>(k, devMap, BLKSIZE, width);
    	end_I = clock();
    	I_time += (end_I - begin_I);
    	begin_II = clock();
    	floyd_phaseII<<<gridSize2, blockSize2>>>(k, devMap, BLKSIZE, width);
    	end_II = clock();
    	II_time += (end_II - begin_II);
    	begin_III = clock();
    	floyd_phaseIII<<<gridSize3, blockSize3>>>(k, devMap, BLKSIZE, width);
    	end_III = clock();
    	III_time += (end_III - begin_III);
    }
}

void saveSolution(int* FinalMap, const char* outfile){
	FILE *out;
	out=fopen(outfile, "wb");
	fwrite(FinalMap,sizeof(int),N*N,out);
    fclose(out);
}

int main(int argc, char** argv) {
//time
	begin_total = clock();
	
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
	/*for(int i = 0; i < width*width; i++)
		printf("%d ", Hostmap[i]);
	printf("\n");
	*/
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
//time
	end_total = clock();
	total_time = end_total - begin_total;
	printf("total time = %lf\n", (double)(total_time) / CLOCKS_PER_SEC);
	printf("phase I time = %lf\n", (double)(I_time) / CLOCKS_PER_SEC);
	printf("phase II time = %lf\n", (double)(II_time) / CLOCKS_PER_SEC);
	printf("phase III time = %lf\n", (double)(III_time) / CLOCKS_PER_SEC);

	return 0;
}