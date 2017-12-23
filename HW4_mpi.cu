/***
 * Author:Yun-Chen Lo
 * File:HW4_mpi.cu
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

#define INF 1000000000

int *Hostmap; 	// host memory for input adjacent file
int N; 			// number of vertex

int ceil(int a, int b){
	return (a + b -1)/b;
}

int readInput(const char* infile, int B){
	int in=0;
	int i, j, width;
	int fd = open(infile, O_RDONLY, (mode_t)0600);
	struct stat fileInfo = {0};
	if (fstat(fd, &fileInfo) == -1)
    {
        perror("Error getting the file size");
        exit(EXIT_FAILURE);
    }
	printf("File size is %ji\n", (intmax_t)fileInfo.st_size);
	char *map = (char *)mmap(0, fileInfo.st_size, PROT_READ, MAP_SHARED, fd, 0);
	int start_i;
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
	if(ceil(N, B) == N/B || ceil(N, B) == 1){
		width = N;
		cudaMallocHost(&Hostmap, (width*width)*sizeof(int));
	}
	else{
		width = ceil(N, B) * B;
		cudaMallocHost(&Hostmap, (width*width)*sizeof(int));
	}
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

__global__ void floyd_phaseIII(int k, int *devMap, int B, int d_N, int offset, int bound, int round, int myrank){
	extern __shared__ int S[];
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

void ErrorMessage(int error, int rank, char* string)
{
    fprintf(stderr, "Process %d: Error %d in %s\n", rank, error, string);
    MPI_Finalize();
    exit(-1);
}

void MPI_SAVE(int* FinalMap, const char* outfile, int myrank, int write_offset, int save_size){
	printf("MPI_COMM_WORLD:%d|myrank:%d write_offset:%d save_size:%d\n", MPI_COMM_WORLD, myrank, write_offset, save_size);
	MPI_File out;
	MPI_Status status;
	MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &out);
	MPI_File_write_at(out, write_offset*sizeof(int), &FinalMap[write_offset], save_size, MPI_INT, &status);
	MPI_File_close(&out);
}

void Block_floydWarshall(int* Hostmap, int* devMap, int B, int width, int myrank, const char* outfile){
	int k;
	int round = ceil(N, B);
	int BLKSIZE;
	int offset, bound, size, save_border, pass_size, pos;
	int write_offset, save_size;
	MPI_Status status;

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

    printf("[%d]BLKSIZE = %d\n", myrank, BLKSIZE);
    printf("round = %d, N = %d\n", round, N);
    
    offset = (myrank!= 0)? ceil(round, 2) : 0;
    bound = (myrank != 0)? round : ceil(round, 2);
    size = BLKSIZE * width;
    save_border = offset * BLKSIZE * width;
    pass_size = (bound-offset)*BLKSIZE*width;

    printf("offset:%d, bound:%d, size:%d, ***: %d\n", offset, bound, size, (bound-offset)*B*width);
    
    for(k = 0; k<round; k++){
    	MPI_Barrier(MPI_COMM_WORLD);
    	floyd_phaseI<<<1, blockSize1, B*B*sizeof(int)>>>(k, devMap, BLKSIZE, width);
    	floyd_phaseII<<<gridSize2, blockSize2, 2*B*B*sizeof(int)>>>(k, devMap, BLKSIZE, width);
    	floyd_phaseIII<<<gridSize3, blockSize3, 2*B*B*sizeof(int)>>>(k, devMap, BLKSIZE, width, offset, bound, round, myrank);
    	pos = (k+1) * B * width;
    	if(k < round - 1){
    		if((k+1) >= offset && (k+1) < bound){
    			cudaMemcpy(&Hostmap[pos], &devMap[pos], size*sizeof(int), cudaMemcpyDeviceToHost);
    			MPI_Send(&Hostmap[pos], size, MPI_INT, 1-myrank, 0, MPI_COMM_WORLD);
    		}
    		else{
    			MPI_Recv(&Hostmap[pos], size, MPI_INT, 1-myrank, 0, MPI_COMM_WORLD, &status);
    			cudaMemcpy(&devMap[pos], &Hostmap[pos], size* sizeof(int), cudaMemcpyHostToDevice);
    		}
    	}
    	else{
    		cudaMemcpy(&Hostmap[save_border], &devMap[save_border], pass_size*sizeof(int), cudaMemcpyDeviceToHost);
    	}
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
	write_offset = offset * BLKSIZE * N;
	save_size = (bound-offset) * BLKSIZE * N;
	if(write_offset + save_size > N*N-1){
		save_size = N*N - write_offset;
		printf("kkkkkkkkkk\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
    MPI_SAVE(FinalMap, outfile, myrank, write_offset, save_size);
}


int main(int argc, char** argv) {
	const char* infile = argv[1];
	const char* outfile = argv[2];
	int B = atoi(argv[3]);	//block size
	int nprocs;
    int myrank;
	/* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int *devMap;			//device memory
	int width;				//width of the appended array
	int gpuid;
	width = readInput(infile, B);
    
    cudaSetDevice(myrank);
    cudaGetDevice(&gpuid);
    cudaDeviceEnablePeerAccess (1- gpuid,0);
    printf("cpu:%d <-> gpu:%d\n", myrank, gpuid);
    cudaMalloc(&devMap, width * width * sizeof(int));
    cudaMemcpy(devMap, Hostmap, sizeof(int) * width * width, cudaMemcpyHostToDevice);
    
    Block_floydWarshall(Hostmap, devMap, B, width, myrank, outfile);
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) 
	printf("Error: %s\n", cudaGetErrorString(err));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize(); 
    printf("Number of tasks= %d My rank= %d devptr = %d\n", nprocs, myrank, devMap);
    return 0;
}