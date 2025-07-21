#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void vectorAdd(float* A, float* B, float* C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}

int main(){
    float* A_h,*B_h,*C_h,*C_gold; // _h is for host
    float* A_d,*B_d,*C_d; // _d represents device
	int n = 10000;
	int size = n * sizeof(float);
    // Allocate memory for HOST array
    if ((A_h = (float*)malloc(size)) == NULL || (B_h = (float*)malloc(size)) == NULL || (C_h = (float*)malloc(size)) == NULL || (C_gold = (float*)malloc(size)) == NULL){
        printf("Error in malloc\n");
        return -1;
    }
    srand(123);
    for (int i = 0; i < n; i++){
        // Generate number between 1 to 100 (2 decimal places)
        A_h[i] = (rand() % 9901 + 100) / 100.0f;
        B_h[i] = (rand() % 9901 + 100) / 100.0f;
        C_gold[i] = A_h[i] + B_h[i];
    }
    if (cudaMalloc((void**)&A_d,size) != cudaSuccess || cudaMalloc((void**)&B_d,size) != cudaSuccess || cudaMalloc((void**)&C_d,size) != cudaSuccess){
        printf("Error in cudaMalloc\n");
        return -1;
    }
    cudaMemcpy(A_d,A_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B_h,size,cudaMemcpyHostToDevice);

    // cuda kernel launch
    vectorAdd<<<ceil(n/256.0),256>>>(A_d,B_d,C_d,n);
    cudaMemcpy(C_h,C_d,size,cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++){
        if (C_h[i] != C_gold[i]){
            printf("Wrong answer C[%d]\n",i);
            cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
            free(A_h); free(B_h); free(C_h); free(C_gold);  
            return -1;
        }
    }

    printf("kernel success!!\n");
    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    free(A_h); free(B_h); free(C_h); free(C_gold);  

	return 0;
}
