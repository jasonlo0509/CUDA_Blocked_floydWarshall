#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

void ErrorMessage(int error, char* string)
{
    fprintf(stderr, "Error %d in %s\n", error, string);
    exit(-1);
}

int main(int argc, const char *argv[])
{    
    const char *filepath = "/home/pp17/p103061108/hw4/1.in";

    int fd = open(filepath, O_RDONLY, (mode_t)0600);
    
    if (fd == -1)
    {
        perror("Error opening file for writing");
        exit(EXIT_FAILURE);
    }        
    
    struct stat fileInfo = {0};
    
    if (fstat(fd, &fileInfo) == -1)
    {
        perror("Error getting the file size");
        exit(EXIT_FAILURE);
    }
    
    if (fileInfo.st_size == 0)
    {
        fprintf(stderr, "Error: File is empty, nothing to do\n");
        exit(EXIT_FAILURE);
    }
    
    printf("File size is %ji\n", (intmax_t)fileInfo.st_size);
    
    char *map = (char *)mmap(0, fileInfo.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED)
    {
        close(fd);
        perror("Error mmapping the file");
        exit(EXIT_FAILURE);
    }
    int in = 0, N, m;
    int start_i;
    int i, j;
    for (off_t i = 0; i < fileInfo.st_size; i++){
        if(map[i] == ' ' || map[i] == '\n'){
            printf("num = %d ", in);
            if(map[i] == '\n'){
                m = in;
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
    
    int* Hostmap;
    int j_cnt=0;
    int h_i, h_j;
    printf("N= %d\n", N);
    Hostmap = (int *)malloc(N*N*sizeof(int));
    if (Hostmap == NULL) ErrorMessage(-1, "malloc");
    
    for(i =0; i< N; i++){
        for(j = 0; j<N; j++){
            printf("%d ", Hostmap[i*N + j]);
        }
        printf("\n");
    }

    for(i = 0; i< N; i++){
        for(j = 0; j<N; j++){
            Hostmap[N*i + j] = 100000;
        }
    }

    for (off_t i = start_i; i < fileInfo.st_size; i++){
        if(map[i] == ' ' || map[i] == '\n'){
            printf("num = %d ", in);
            j++;
            if(map[i] == '\n'){
                j = 0;
                Hostmap[h_i*N+h_j] = in;
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
    printf("\n");
    for(int i =0; i< N; i++){
        for(int j = 0; j<N; j++){
            printf("%d ", Hostmap[i*N + j]);
        }
        printf("\n");
    }
    /*
    int in = 0;
    for (off_t i = 0; i < fileInfo.st_size; i++)
    {
        //printf("Found character %c at %ji\n", (int)map[i], (intmax_t)i);
        if(map[i] == ' ' || map[i] == '\n'){
            printf("num = %d ", in);
            in = 0;
        }
        else{
            in = (int)map[i]-(int)'0' + 10 * in;
        }
    }*/

    // Don't forget to free the mmapped memory
    if (munmap(map, fileInfo.st_size) == -1)
    {
        close(fd);
        perror("Error un-mmapping the file");
        exit(EXIT_FAILURE);
    }

    // Un-mmaping doesn't close the file, so we still need to do that.
    close(fd);
    
    return 0;
}