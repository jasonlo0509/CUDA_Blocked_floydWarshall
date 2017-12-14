#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <cuda_runtime.h>
#include <assert.h>

#define INF 1000000000

int *dist;
int *map;
int vertex;

void readInput(const char* infile){
	FILE * pFile;
	int in, counter=0;
	int i, j;
	pFile = fopen ( infile , "r" );
	fscanf (pFile, "%d", &in);
	vertex = in;
	map = (int *)malloc((vertex*vertex)*sizeof(int));
	dist = (int *)malloc((vertex*vertex)*sizeof(int));
	for(i=0; i<vertex; i++){
  		for(j=0; j<vertex; j++){
  			if(i!=j)
  				map[vertex*i + j] = INF;
  			else
  				map[vertex*i + j] = 0;
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
				map[vertex*i + j] = in;
			}
      	}
    }
}

void floydWarshall()
{
	int i, j, k;
	for (i = 0; i < vertex; i++)
        for (j = 0; j < vertex; j++)
            dist[i*vertex + j] = map[i*vertex + j];

    for (k = 0; k < vertex; k++){
        for (i = 0; i < vertex; i++){
            for (j = 0; j < vertex; j++){
                if (dist[i*vertex + k] + dist[k*vertex + j] < dist[i*vertex + j]){
                    dist[i*vertex + j] = dist[i*vertex + k] + dist[k*vertex + j];
                }
            }
        }
    }
}

void saveSolution(const char* outfile){
	int i, j;
	FILE *out;
	out=fopen(outfile, "wb");
	fwrite(dist,sizeof(int),vertex*vertex,out);
    fclose(out);
}

int main(int argc, char** argv) {
	const char* infile = argv[1];
	const char* outfile = argv[2];
	int blk = strtol(argv[3], 0, 10);
	readInput(infile);
	floydWarshall();

	saveSolution(outfile);
}