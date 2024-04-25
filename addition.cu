#include <stdio.h>

#define N 1000000 // Size of vectors

__global__ void vectorAdd(int *a, int *b, int *c) {
    // Calculate thread index within the grid
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Perform vector addition only if within vector size limit
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

int main() {
    int *a, *b, *c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    // Allocate memory on host for input and output vectors
    a = (int*)malloc(N * sizeof(int));
    b = (int*)malloc(N * sizeof(int));
    c = (int*)malloc(N * sizeof(int));

    // Allocate memory on the device for input and output vectors
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    // Initialize input vectors on the host
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Copy input vectors from host to device
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    int blockSize = 256;
    int gridSize = (int)ceil((float)N / blockSize);

    // Launch kernel to perform vector addition on the GPU
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c);

    // Copy result vector from device to host
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Display first few elements of the result vector
    printf("Vector Addition Results (First 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free allocated memory on host and device
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

/*
Assignment No: 4 
==================================================================== 
Lightbox
Decompose using 4 blocks of 2 threads each
blockDim.x = 1, threadId.x = 0,1 within each block, blockId.x = 0, 1, 2, 3
Title:  Write a CUDA Program for  
1. Addition of two large vectors 
2. Matrix Multiplication using CUDA C. 
=====================================================================Objective: 
To learn about CUDA 
To understand the concept of vector addition and matrix multiplication using CUDA 
===================================================================== 
Theory: 
. What is CUDA? 
 NVIDIA"s CUDA is a general purpose parallel computing platform and programming model that accelerates deep learning and other computeintensive apps by taking advantage of the parallel processing power of GPUs. 
 CUDA enables developers to speed up computeintensive applications by harnessing the power of GPUs for the parallelizable part of the computation. 
 In 2003, a team of researchers led by Ian Buck unveiled Brook, the first widely adopted programming model to extend C with dataparallel constructs. Buck later joined NVIDIA and led the launch of CUDA in 2006, the first commercial solution for general purpose computing on GPUs. 
 Within the supported CUDA compiler, any piece of code can be run on GPU by using the __global__ keyword. Programmers must change their malloc/new and free/delete calls as per the CUDA guidelines so that appropriate space could be allocated on GPU. Once the computation on GPU is finished, the results are synchronized and ported to the CPU. 
 Architecture of CUDA 
. GPUs run one kernel (a group of tasks) at a time. 
. Each kernel consists of blocks, which are independent groups of ALUs. 
. Each block contains threads, which are levels of computation. 
. The threads in each block typically work together to calculate a value. 
. Threads in the same block can share memory. 
. In CUDA, sending information from the CPU to the GPU is often the most typical part of the computation. 
. For each thread, local memory is the fastest, followed by shared memory, global, static, and texture memory the slowest. 
 For compilation in CUDA using nvcc. The CUDA compiler is called nvcc. This will exist on your machine if you have installed the CUDA development toolkit. Program name must be save with “.cu” extension. 
$ nvcc filename.cu  
$ ./filename 
. What is vector? 
 A vector is a quantity which has both magnitude and direction. A vector quantity, unlike scalar, has a direction component along with the magnitude which helps to determine the position of one point relative to the other. 
 the length of the segment of the directed line is called the magnitude of a vector and the angle at which the vector is inclined shows the direction of the vector. 
. How to do vector addition? 
 CUDA convention has been to depict treads as scribbly lines with arrow heads, as shown below. In this case, the 4 blocks of threads that form the 1dimensional grid become analogous to the processing units that we would assign to the portions of the array that would be run in parallel. In contrast to OpenMP and MPI, however, the individual threads within the blocks would each work on one data element of the array. 
 In CUDA programs, we always determine which thread is running and use that to determine what portion of data to work on. The mapping of work is up to the programmer. 
 to variables supplied by the CUDA library that help us define just what thread is executing. The following diagram shows how three of the available variables, corresponding to the grid, blocks, and threads within the blocks would be assigned for our above example decomposition: 
 The three variables that we can access in CUDA code for this example shown above are: 
1. threadIdx.x  represents a thread"s index along the x dimension within the block. 
2. blockIdx.x  represents a thread"s block"s index along the x dimension within the grid. 
3. blockDim.x  represents the number of threads per block in the x direction. 
 In our simple example, there are a total of eight threads executing, each one numbered from 0 through 7. Each thread executing code on the GPU can determine which thread it is by using the above variables like this: 
int tid = blockDim.x  blockIdx.x + threadIdx.x; 
 In CUDA programming, the primary motherboard with the CPU is referred to as the host and the GPU coprocessor is usually called the device. The GPU device has separate memory and different circuitry for executing instructions. Code to be executed on the GPU must be compiled for its instrution set. 
 The overall structure of a CUDA program that uses the GPU for computation is as follows: 
1. Define the the code that will run on the device in a separate function, called the kernel function. 
2. In the main program running on the host"s CPU: 
a) allocate memory on the host for the data arrays. 
b) initialze the data arrays in the host"s memory. 
c) allocate separate memory on the GPU device for the data arrays. 
d) copy data arrays from the host memory to the GPU device memory. 
3. On the GPU device, execute the kernel function that computes new data values given the original arrays. Specify how many blocks and threads per block to use for this computation. 
4. After the kernel function completes, copy the computed values from the GPU device memory back to the host"s memory. 
. How to do matrix multiplication using CUDA? 
 2D matrices can be stored in the computer memory using two layouts  rowmajor and columnmajor. 
1. Column Major 
2. Row Major 
In rowmajor layout, element(x,y) can be addressed as: xwidth + y. In the above example, the width of the matrix is 4. For example, element (1,1) will be found at position  14 + 1 = 5 in the 1D array. 
 Mapping each data element to a thread. The following mapping scheme is used to map data to thread. This gives each thread its unique identity. 
row=blockIdx.xblockDim.x+threadIdx.x; 
col=blockIdx.yblockDim.y+threadIdx.y; 
 a grid is madeup of blocks, and that the blocks are made up of threads. All threads in the same block have the same block index. 
 Matrix multiplication between a (IxJ) matrix d_M and (JxK) matrix d_N produces a matrix d_P with dimensions (IxK). The formula used to calculate elements of d_P is – 
d_Px,y = .. d_Mx,,kd_Nk,y, for k=0,1,2,....width 
 A d_P element calculated by a thread is in „blockIdx.yblockDim.y+threadIdx.y" row and „blockIdx.xblockDim.x+threadIdx.x" column. Here is the actual kernel that implements the above logic. 
__global__ void simpleMatMulKernell(float d_M, float d_N, float d_P, int width) 
{ 
} 
 This helps to calculate row and col to address what element of d_P will be calculated by this thread. 
int row = blockIdx.ywidth+threadIdx.y; 
int col = blockIdx.xwidth+threadIdx.x; 
 Matrix Multiplication: 
if(row<width && col <width) { 
 float product_val = 0 
for(int k=0;k<width;k++) { 
 product_val += d_M[rowwidth+k]d_N[kwidth+col]; 
 } 
d_p[rowwidth+col] = product_val; 
} 
===================================================================== 
Conclusion: 
Thus we have studied vector addition and matrix multiplication using CUDA in C.  
===================================================================== 


    Methodology and Detailed Explanation:

    - This CUDA program performs vector addition of two arrays (vectors) on the GPU.

    - The `vectorAdd` kernel function is executed on the GPU. Each thread calculates the sum of corresponding elements from arrays `a` and `b`
      and stores the result in array `c`.

    - Memory management involves allocating memory for vectors (`a`, `b`, `c`) on both host (CPU) and device (GPU) using `malloc` and `cudaMalloc`.

    - Vector data is initialized on the host, and then copied from host to device using `cudaMemcpy`.

    - The CUDA kernel is launched with a grid of `gridSize` blocks, each containing `blockSize` threads. The total number of threads launched
      is `gridSize * blockSize`.

    - The kernel computes the result of vector addition in parallel on the GPU.

    - After kernel execution, the result vector (`c`) is copied back from the device to the host using `cudaMemcpy`.

    - Finally, the first few elements of the result vector (`c`) are printed to verify the correctness of vector addition.

    Key Points:
    - CUDA kernel functions are defined with `__global__` keyword and are executed on the GPU.
    - Kernel launches are configured with grid and block dimensions to parallelize the computation.
    - Memory is managed explicitly between host (CPU) and device (GPU) using `cudaMalloc` and `cudaMemcpy`.
    - The vector addition operation (`c[tid] = a[tid] + b[tid];`) is performed in parallel by multiple threads on the GPU.
    - After kernel execution, the result is copied back from the GPU to the host for further processing or verification.

*/
