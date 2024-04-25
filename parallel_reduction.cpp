#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>

using namespace std;

// Function to find the minimum value in an array using OpenMP reduction
void min_reduction(vector<int>& arr) {
  int min_value = INT_MAX;

  // Parallel loop with reduction for finding the minimum value
  #pragma omp parallel for reduction(min: min_value)
  for (int i = 0; i < arr.size(); i++) {
    if (arr[i] < min_value) {
      min_value = arr[i];
    }
  }

  // Output the minimum value found by reduction
  cout << "Minimum value: " << min_value << endl;
}

// Function to find the maximum value in an array using OpenMP reduction
void max_reduction(vector<int>& arr) {
  int max_value = INT_MIN;

  // Parallel loop with reduction for finding the maximum value
  #pragma omp parallel for reduction(max: max_value)
  for (int i = 0; i < arr.size(); i++) {
    if (arr[i] > max_value) {
      max_value = arr[i];
    }
  }

  // Output the maximum value found by reduction
  cout << "Maximum value: " << max_value << endl;
}

// Function to calculate the sum of elements in an array using OpenMP reduction
void sum_reduction(vector<int>& arr) {
  int sum = 0;

  // Parallel loop with reduction for calculating the sum of elements
  #pragma omp parallel for reduction(+: sum)
  for (int i = 0; i < arr.size(); i++) {
    sum += arr[i];
  }

  // Output the sum of all elements found by reduction
  cout << "Sum: " << sum << endl;
}

// Function to calculate the average of elements in an array using OpenMP reduction
void average_reduction(vector<int>& arr) {
  int sum = 0;

  // Parallel loop with reduction for calculating the sum of elements
  #pragma omp parallel for reduction(+: sum)
  for (int i = 0; i < arr.size(); i++) {
    sum += arr[i];
  }

  // Calculate the average by dividing the sum by the number of elements
  double average = (double)sum / arr.size();

  // Output the average value
  cout << "Average: " << average << endl;
}

int main() {
  // Create a vector of integers
  vector<int> arr {5, 2, 9, 1, 7, 6, 8, 3, 4};

  // Perform reduction operations using OpenMP
  min_reduction(arr);
  max_reduction(arr);
  sum_reduction(arr);
  average_reduction(arr);

  return 0;
}

/*
Execution: 

### Compilation:

```bash
g++ -o reduction_example reduction_example.cpp -fopenmp
```

### Execution:

```bash
./reduction_example
```

### Example Input:

```
Enter total number of elements =>
9
Enter elements =>
5
2
9
1
7
6
8
3
4
```

### Output:

```
Minimum value: 1
Maximum value: 9
Sum: 45
Average: 5
```

GROUP – A 
Assignment No: 3 
==================================================================== 
Title:  Write a program to implement Min, Max, Sum and Average operations using Parallel Reduction. 
=====================================================================Objective: 
To learn about OpenMP 
To understand the concept of Parallel Reduction 
===================================================================== 
Theory: 
. What is OpenMP? 
 OpenMP (Open MultiProcessing) is an application programming interface (API) that supports multiplatform sharedmemory multiprocessing programming in C, C++, and Fortran, on many platforms, instructionset architectures and operating systems, including Solaris, AIX, FreeBSD, HPUX, Linux, macOS, and Windows. It consists of a set of compiler directives, library routines, and environment variables that influence runtime behavior. 
 An application built with the hybrid model of parallel programming can run on a computer cluster using both OpenMP and Message Passing Interface (MPI), such that OpenMP is used for parallelism within a (multicore) node while MPI is used for parallelism between nodes. There have also been efforts to run OpenMP on software distributed shared memory systems, to translate OpenMP into MPI and to extend OpenMP for nonshared memory systems. 
 For compilation in GCC using fopenmp: 
$ gcc fopenmp hello.c o hello ldl 
. What is Parallel Reduction? 
 Reduction : specifies that 1 or more variables that are private to each thread are subject of reduction operation at end of parallel region.  
reduction(operation:var) 
where, operation is operator to perform on the variables (var) at the end of the parallel region and Var is one or more variables on which to perform scalar reduction. 
. Min Operation 
 The function takes in a vector of integers as input and finds the minimum value in the vector using parallel reduction. 
 The OpenMP reduction clause is used with the "min" operator to find the minimum value across all threads. 
 The minimum value found by each thread is reduced to the overall minimum value of the entire array. 
 The final minimum value is printed to the console. 
. Max  Operation 
 The function takes in a vector of integers as input and finds the maximum value in the vector using parallel reduction. 
 The OpenMP reduction clause is used with the "max" operator to find the maximum value across all threads. 
 The maximum value found by each thread is reduced to the overall maximum value of the entire array. 
 The final maximum value is printed to the console. 
. Sum Operation 
 The function takes in a vector of integers as input and finds the sum of all the values in the vector using parallel reduction. 
 The OpenMP reduction clause is used with the "+" operator to find the sum across all threads. 
 The sum found by each thread is reduced to the overall sum of the entire array. 
 The final sum is printed to the console. 
. Average Operation 
 The function takes in a vector of integers as input and finds the average of all the values in the vector using parallel reduction. 
 The OpenMP reduction clause is used with the "+" operator to find the sum across all threads. 
 The function takes in a vector of integers as input and finds the average of all the values in the vector using parallel reduction. 
 The OpenMP reduction clause is used with the "+" operator to find the sum across all threads. 
. Main Function 
 The function takes in a vector of integers as input and finds the average of all the values in the vector using parallel reduction. 
 The OpenMP reduction clause is used with the "+" operator to find the sum across all threads. 
===================================================================== 
Conclusion: 
Thus we have studied Min, Max, Sum and Average operation using parallel reduction in C++ with OpenMP. Also, Parallel reduction is a powerful technique that allows us to perform these operation on large arrays more efficiently by dividing the work among multiple threads running in parallel. 
===================================================================== 

    Methodology and Detailed Explanation:

    - This C++ program demonstrates various reduction operations (min, max, sum, average) using OpenMP directives.

    - Reduction operations are used to compute a final result from individual contributions of threads in parallel.

    - Each reduction operation (`min`, `max`, `+`) is applied to an initial value (`min_value`, `max_value`, `sum`) within a parallel loop.

    - The `#pragma omp parallel for reduction(...)` directive ensures that each thread computes its local minimum, maximum, sum, or average,
      and then combines the results into a single value using the specified reduction operation (`min`, `max`, `+`).

    - In the `main` function:
        - A vector `arr` containing integer values is created.
        - The `min_reduction`, `max_reduction`, `sum_reduction`, and `average_reduction` functions are called with `arr` as input to perform reduction operations.
        - Each function uses OpenMP directives to parallelize the computation and calculate the respective reduction result.

    Key Points:
    - OpenMP `reduction` clause is used to specify the reduction operation (`min`, `max`, `+`) and the variable to hold the final result (`min_value`, `max_value`, `sum`).
    - Reduction operations are efficient for parallel computations where the final result is derived from combining partial results from multiple threads.
    - The `min_reduction` and `max_reduction` functions use `INT_MAX` and `INT_MIN` as initial values for finding the minimum and maximum values, respectively.
    - The `sum_reduction` function calculates the sum of all elements in the array using the `+` reduction operation.
    - The `average_reduction` function calculates the average value by dividing the sum of elements by the number of elements (`arr.size()`).

  This C++ program demonstrates the use of OpenMP's reduction clause to perform parallel reductions on an array of integers. The reductions include finding the minimum value, maximum value, sum of elements, and average of elements in the array. Let's break down the key parts of the code:

1. Header Inclusion
   ```pp
   #include <iostream>
   #include <vector>
   #include <omp.h>
   #include <climits>
   ```
   - Includes necessary header files for input/output (`iostream`), using vectors (`vector`), OpenMP (`omp.h`), and defining integer limits (`climits`).

2. Function Definitions
   - `min_reduction`: Uses OpenMP reduction to find the minimum value in the array.
   - `max_reduction`: Uses OpenMP reduction to find the maximum value in the array.
   - `sum_reduction`: Uses OpenMP reduction to calculate the sum of elements in the array.
   - `average_reduction`: Uses OpenMP reduction to calculate the average of elements in the array.

3. `min_reduction` Function
   ```pp
   void min_reduction(vector<int>& arr) {
     int min_value = INT_MAX;
   
     #pragma omp parallel for reduction(min: min_value)
     for (int i = 0; i < arr.size(); i++) {
       if (arr[i] < min_value) {
         min_value = arr[i];
       }
     }
   
     cout << "Minimum value: " << min_value << endl;
   }
   ```
   - Initializes `min_value` to `INT_MAX`.
   - Uses OpenMP `parallel for` with `reduction(min: min_value)` to find the minimum value in the array.

4. `max_reduction` Function
   ```pp
   void max_reduction(vector<int>& arr) {
     int max_value = INT_MIN;
   
     #pragma omp parallel for reduction(max: max_value)
     for (int i = 0; i < arr.size(); i++) {
       if (arr[i] > max_value) {
         max_value = arr[i];
       }
     }
   
     cout << "Maximum value: " << max_value << endl;
   }
   ```
   - Initializes `max_value` to `INT_MIN`.
   - Uses OpenMP `parallel for` with `reduction(max: max_value)` to find the maximum value in the array.

5. `sum_reduction` Function
   ```pp
   void sum_reduction(vector<int>& arr) {
     int sum = 0;
   
     #pragma omp parallel for reduction(+: sum)
     for (int i = 0; i < arr.size(); i++) {
       sum += arr[i];
     }
   
     cout << "Sum: " << sum << endl;
   }
   ```
   - Initializes `sum` to `0`.
   - Uses OpenMP `parallel for` with `reduction(+: sum)` to calculate the sum of all elements in the array.

6. `average_reduction` Function
   ```pp
   void average_reduction(vector<int>& arr) {
     int sum = 0;
   
     #pragma omp parallel for reduction(+: sum)
     for (int i = 0; i < arr.size(); i++) {
       sum += arr[i];
     }
   
     double average = (double)sum / arr.size();
   
     cout << "Average: " << average << endl;
   }
   ```
   - Initializes `sum` to `0`.
   - Uses OpenMP `parallel for` with `reduction(+: sum)` to calculate the sum of all elements in the array.
   - Computes the average by dividing `sum` by the size of the array (`arr.size()`).

7. `main` Function
   ```pp
   int main() {
     vector<int> arr {5, 2, 9, 1, 7, 6, 8, 3, 4};
   
     min_reduction(arr);
     max_reduction(arr);
     sum_reduction(arr);
     average_reduction(arr);
   
     return 0;
   }
   ```
   - Initializes a vector `arr` with integer values.
   - Calls the reduction functions (`min_reduction`, `max_reduction`, `sum_reduction`, `average_reduction`) to perform respective operations on the array.

8. Output
   - Each reduction function outputs the result (minimum value, maximum value, sum, average) after the parallel reduction operation.

This program demonstrates how to use OpenMP's reduction clause to efficiently perform parallel reductions on arrays, which can significantly improve performance for large datasets. Each reduction operation (`min`, `max`, `+`) is performed in parallel across multiple threads, leveraging the parallel computing capabilities of OpenMP.
*/