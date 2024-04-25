#include<iostream>
#include<stdlib.h>
#include<omp.h>
using namespace std;

// Function prototypes
void mergesort(int a[], int i, int j);
void merge(int a[], int i1, int j1, int i2, int j2);

// Recursive function to perform merge sort
void mergesort(int a[], int i, int j)
{
    int mid;
    
    // Base case: If the array has more than one element
    if(i < j)
    {
        // Calculate the middle index
        mid = (i + j) / 2;
        
        // Parallelize the divide step using OpenMP sections
        #pragma omp parallel sections 
        {
            // Sort the left half recursively in parallel
            #pragma omp section
            {
                mergesort(a, i, mid);        
            }

            // Sort the right half recursively in parallel
            #pragma omp section
            {
                mergesort(a, mid + 1, j);    
            }
        }

        // Merge the sorted halves
        merge(a, i, mid, mid + 1, j);    
    }
}

// Function to merge two sorted subarrays
void merge(int a[], int i1, int j1, int i2, int j2)
{
    int temp[1000];    
    int i, j, k;
    i = i1;    
    j = i2;    
    k = 0;
    
    // Merge two subarrays into a temporary array 'temp'
    while(i <= j1 && j <= j2)    
    {
        if(a[i] < a[j])
        {
            temp[k++] = a[i++];
        }
        else
        {
            temp[k++] = a[j++];
        }    
    }
    
    // Copy remaining elements from the first subarray
    while(i <= j1)    
    {
        temp[k++] = a[i++];
    }
        
    // Copy remaining elements from the second subarray
    while(j <= j2)    
    {
        temp[k++] = a[j++];
    }
        
    // Copy the merged elements back to the original array
    for(i = i1, j = 0; i <= j2; i++, j++)
    {
        a[i] = temp[j];
    }    
}

// Main function
int main()
{
    int *a, n, i;
    double start_time, end_time, seq_time, par_time;
    
    // Prompt user to enter the number of elements
    cout << "\nEnter total number of elements: ";
    cin >> n;
    
    // Dynamically allocate memory for the array
    a = new int[n];

    // Prompt user to enter the elements of the array
    cout << "\nEnter elements: ";
    for(i = 0; i < n; i++)
    {
        cin >> a[i];
    }
    
    // Sequential merge sort
    start_time = omp_get_wtime();
    mergesort(a, 0, n - 1);
    end_time = omp_get_wtime();
    seq_time = end_time - start_time;
    cout << "\nSequential Time: " << seq_time << " seconds" << endl;
    
    // Parallel merge sort using OpenMP
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            mergesort(a, 0, n - 1);
        }
    }
    end_time = omp_get_wtime();
    par_time = end_time - start_time;
    cout << "\nParallel Time: " << par_time << " seconds" << endl;
    
    // Display the sorted array
    cout << "\nSorted array: ";
    for(i = 0; i < n; i++)
    {
        cout << a[i] << " ";
    }

    // Free the dynamically allocated memory
    delete[] a;
    
    return 0;
}

/*
Execution: 
 g++ -o Merge -fopenmp -pthread Merge.cpp

 ./Merge

Enter total number of elements: 8
Enter the elements:
10
5
7
3
1
8
2
6

Sequential Time: 0.000192 seconds
Parallel Time: 0.000367 seconds

Sorted Array:
1
2
3
5
6
7
8
10


Assignment No: 2 

==================================================================== 
Title:  Write a program to implement Parallel Bubble Sort and Merge sort using OpenMP. Use existing algorithms and measure the performance of sequential and parallel algorithms. 
=====================================================================Objective: 
To learn about OpenMP. 
To understand the concept of parallelism or parallel algorithms. 
===================================================================== 
Theory: 
. What is parallel algorithm? 
 An algorithm is a sequence of steps that take inputs from the user and after some computation, produces an output. A parallel algorithm is an algorithm that can execute several instructions simultaneously on different processing devices and then combine all the individual outputs to produce the final result. 
 In parallel algorithm, The problem is divided into subproblems and are executed in parallel to get individual outputs. Later on, these individual outputs are combined together to get the final desired output. 
 While designing a parallel algorithm, proper CPU utilization should be considered to get an efficient algorithm. 

. What is OpenMP? 
 OpenMP (Open MultiProcessing) is an application programming interface (API) that supports multiplatform sharedmemory multiprocessing programming in C, C++, and Fortran, on many platforms, instructionset architectures and operating systems, including Solaris, AIX, FreeBSD, HPUX, Linux, macOS, and Windows. It consists of a set of compiler directives, library routines, and environment variables that influence runtime behavior. 
 An application built with the hybrid model of parallel programming can run on a computer cluster using both OpenMP and Message Passing Interface (MPI), such that OpenMP is used for parallelism within a (multicore) node while MPI is used for parallelism between nodes. There have also been efforts to run OpenMP on software distributed shared memory systems, to translate OpenMP into MPI and to extend OpenMP for nonshared memory systems. 
 OpenMP is an implementation of multithreading, a method of parallelizing whereby a primary thread (a series of instructions executed consecutively) forks a specified number of subthreads and the system divides a task among them. The threads then run concurrently, with the runtime environment allocating threads to different processors. 

 The section of code that is meant to run in parallel is marked accordingly, with a compiler directive that will cause the threads to form before the section is executed.[3] Each thread has an ID attached to it which can be obtained using a function (called omp_get_thread_num()). The thread ID is an integer, and the primary thread has an ID of 0. After the execution of the parallelized code, the threads join back into the primary thread, which continues onward to the end of the program. 
 By default, each thread executes the parallelized section of code independently. Worksharing constructs can be used to divide a task among the threads so that each thread executes its allocated part of the code. Both task parallelism and data parallelism can be achieved using OpenMP 
 The runtime environment allocates threads to processors depending on usage, machine load and other factors. The runtime environment can assign the number of threads based on environment variables, or the code can do so using functions.  
 The OpenMP functions are included in a header file labelled omp.h in C/C++. 
 The core elements of OpenMP are the constructs for thread creation, workload distribution (work sharing), dataenvironment management, thread synchronization, userlevel runtime routines and environment variables. 
 In C/C++, OpenMP uses #pragmas for thread creation. 

 For compilation in GCC using fopenmp: 
$ gcc fopenmp hello.c o hello ldl 

. Merge Sort: 
 Parallel merge sort is a parallelized version of the merge sort algorithm that takes advantage of multiple processors or cores to improve its performance.  
 In parallel merge sort, the input array is divided into smaller subarrays, which are sorted in parallel using multiple processors or cores. The sorted subarrays are then merged together in parallel to produce the final sorted output. 
 The parallel merge sort algorithm can be broken down into the following steps: 
1. Divide the input array into smaller subarrays. 
2. Assign each subarray to a separate processor or core for sorting. 
3. Sort each subarray in parallel using the merge sort algorithm. 
4. Merge the sorted subarrays together in parallel to produce the final sorted output. 
5. The merging step in parallel merge sort is performed in a similar way to the merging step in the sequential merge sort algorithm. However, because the subarrays are sorted in parallel, the merging step can also be performed in parallel using multiple 
processors or cores. This can significantly reduce the time required to merge the sorted subarrays and produce the final output. 
6. Parallel merge sort can provide significant performance benefits for large input arrays with many elements, especially when running on hardware with multiple processors or cores. However, it also requires additional overhead to manage the parallelization, and may not always provide performance improvements for smaller input sizes or when run on hardware with limited parallel processing capabilities. 

 There are several metrics that can be used to measure the performance of sequential and parallel merge sort algorithms: 
1. Execution time: Execution time is the amount of time it takes for the algorithm to complete its sorting operation. This metric can be used to compare the speed of sequential and parallel merge sort algorithms. 
2. Speedup: Speedup is the ratio of the execution time of the sequential merge sort algorithm to the execution time of the parallel merge sort algorithm. A speedup of greater than 1 indicates that the parallel algorithm is faster than the sequential algorithm. 
3. Efficiency: Efficiency is the ratio of the speedup to the number of processors or cores used in the parallel algorithm. This metric can be used to determine how well the parallel algorithm is utilizing the available resources. 
4. Scalability: Scalability is the ability of the algorithm to maintain its performance as the input size and number of processors or cores increase. A scalable algorithm will maintain a consistent speedup and efficiency as more resources are added. 

. Examples of Merge Sort: 
 To understand the working of the merge sort algorithm, let's take an unsorted array. It will be easier to understand the merge sort via an example. Let the elements of array are:  
 According to the merge sort, first divide the given array into two equal halves. Merge sort keeps dividing the list into equal parts until it cannot be further divided. As there are eight elements in the given array, so it is divided into two arrays of size 4. 
 Now, again divide these two arrays into halves. As they are of size 4, divide them into new arrays of size 2. 
 Now, again divide these arrays to get the atomic value that cannot be further divided. 
 Now, combine them in the same manner they were broken. In combining, first compare the element of each array and then combine them into another array in sorted order. 
 So, first compare 12 and 31, both are in sorted positions. Then compare 25 and 8, and in the list of two values, put 8 first followed by 25. Then compare 32 and 17, sort them and put 17 first followed by 32. After that, compare 40 and 42, and place them sequentially. 
 In the next iteration of combining, now compare the arrays with two data values and merge them into an array of found values in sorted order.  
 Now, there is a final merging of the arrays. After the final merging of above arrays, the array will look like – 
===================================================================== 
Conclusion: 
Thus we have studied parallelism and OpenMP used in programming. Also, How Bubble sort and Merge sort algorithm works in parallel programming. 
===================================================================== 

    Methodology and Detailed Explanation:
    
    - This program implements parallel merge sort using OpenMP to sort an array of integers.
    
    - The `mergesort` function recursively divides the array into two halves until single elements are obtained, and then merges them back in sorted order.
        - OpenMP is used to parallelize the divide step (`#pragma omp parallel sections`), where each section sorts a different half of the array.
    
    - The `merge` function combines two sorted subarrays into a single sorted array.
        - It uses an auxiliary array `temp` to store the merged result temporarily.
        - Elements from the two subarrays are compared and merged into `temp` in sorted order.
        - The sorted elements from `temp` are copied back to the original array.
    
    - In the `main` function:
        - It prompts the user to enter the number of elements (`n`) in the array and the elements themselves.
        - It performs merge sort sequentially and measures the execution time (`seq_time`).
        - It then performs merge sort in parallel using OpenMP and measures the parallel execution time (`par_time`).
        - Finally, it displays the sorted array and the execution times for sequential and parallel merge sort.
    
    Key Points:
    - OpenMP directives are used to parallelize the recursive divide step (`#pragma omp parallel sections`).
    - The `single` directive ensures that only one thread initiates the parallel divide step.
    - The time taken for sequential and parallel merge sort is measured using `omp_get_wtime()`.
    - Dynamic memory allocation (`new` and `delete[]`) is used for the array to handle variable-sized input.
    
    Note:
    - The implementation assumes that the array size (`n`) is within the limits specified by the program.
    - Parallel merge sort can offer performance benefits for large arrays by leveraging multiple threads to speed up sorting.

    This C++ program implements a parallel version of the merge sort algorithm using OpenMP for parallelism. Let's break down each part of the code:

1. **Header and Library Inclusion**
    ```cpp
    #include<iostream>
    #include<stdlib.h>
    #include<omp.h>
    using namespace std;
    ```
    - Includes necessary header files for input/output (`iostream`), memory allocation (`stdlib.h`), and OpenMP (`omp.h`).

2. **Function Prototypes**
    ```cpp
    void mergesort(int a[], int i, int j);
    void merge(int a[], int i1, int j1, int i2, int j2);
    ```
    - Declares function prototypes for `mergesort` and `merge` functions used in merge sort algorithm.

3. **Recursive `mergesort` Function**
    ```cpp
    void mergesort(int a[], int i, int j)
    {
        int mid;
        
        // Base case: If the array has more than one element
        if(i < j)
        {
            mid = (i + j) / 2;
            
            // Parallelize the divide step using OpenMP sections
            #pragma omp parallel sections 
            {
                #pragma omp section
                {
                    mergesort(a, i, mid);        
                }

                #pragma omp section
                {
                    mergesort(a, mid + 1, j);    
                }
            }

            // Merge the sorted halves
            merge(a, i, mid, mid + 1, j);    
        }
    }
    ```
    - Recursive function to perform merge sort on array `a[]` from index `i` to `j`.
    - The array is divided into two halves (`i` to `mid` and `mid + 1` to `j`) using OpenMP `sections` to parallelize the divide step.

4. **`merge` Function**
    ```cpp
    void merge(int a[], int i1, int j1, int i2, int j2)
    {
        int temp[1000];    
        int i, j, k;
        i = i1;    
        j = i2;    
        k = 0;
        
        // Merge two subarrays into a temporary array 'temp'
        while(i <= j1 && j <= j2)    
        {
            if(a[i] < a[j])
            {
                temp[k++] = a[i++];
            }
            else
            {
                temp[k++] = a[j++];
            }    
        }
        
        // Copy remaining elements from the first subarray
        while(i <= j1)    
        {
            temp[k++] = a[i++];
        }
            
        // Copy remaining elements from the second subarray
        while(j <= j2)    
        {
            temp[k++] = a[j++];
        }
            
        // Copy the merged elements back to the original array
        for(i = i1, j = 0; i <= j2; i++, j++)
        {
            a[i] = temp[j];
        }    
    }
    ```
    - Function to merge two sorted subarrays (`a[i1]` to `a[j1]` and `a[i2]` to `a[j2]`) into a temporary array `temp[]`.
    - Merged elements are copied back to the original array `a[]`.

5. **`main` Function**
    ```cpp
    int main()
    {
        int *a, n, i;
        double start_time, end_time, seq_time, par_time;
        
        // Prompt user to enter the number of elements
        cout << "\nEnter total number of elements: ";
        cin >> n;
        
        // Dynamically allocate memory for the array
        a = new int[n];

        // Prompt user to enter the elements of the array
        cout << "\nEnter elements: ";
        for(i = 0; i < n; i++)
        {
            cin >> a[i];
        }
        
        // Sequential merge sort
        start_time = omp_get_wtime();
        mergesort(a, 0, n - 1);
        end_time = omp_get_wtime();
        seq_time = end_time - start_time;
        cout << "\nSequential Time: " << seq_time << " seconds" << endl;
        
        // Parallel merge sort using OpenMP
        start_time = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            {
                mergesort(a, 0, n - 1);
            }
        }
        end_time = omp_get_wtime();
        par_time = end_time - start_time;
        cout << "\nParallel Time: " << par_time << " seconds" << endl;
        
        // Display the sorted array
        cout << "\nSorted array: ";
        for(i = 0; i < n; i++)
        {
            cout << a[i] << " ";
        }

        // Free the dynamically allocated memory
        delete[] a;
        
        return 0;
    }
    ```
    - The `main` function:
      - Prompts the user to enter the number of elements (`n`) and the elements of the array `a[]`.
      - Performs sequential merge sort and measures the time taken (`seq_time`).
      - Performs parallel merge sort using OpenMP and measures the time taken (`par_time`).
      - Displays the sorted array and the time taken for both sequential and parallel merge sort.

6. **Time Measurement**
    - Time is measured using `omp_get_wtime()` before and after sorting to calculate the execution time (`seq_time` and `par_time`).

7. **Output**
    - The program outputs the sorted array and the time taken for both sequential and parallel merge sort.

8. **Memory Management**
    - Memory for the array `a[]` is dynamically allocated using `new` and deallocated using `delete[]` to avoid memory leaks.

Overall, this program demonstrates the use of OpenMP to parallelize the divide step of merge sort (`mergesort` function) by splitting the array into two halves and recursively sorting them in parallel sections. The performance improvement achieved by parallelizing merge sort is measured and compared with sequential merge sort.
*/
