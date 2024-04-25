#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_NODES 100

int graph[MAX_NODES][MAX_NODES]; // Adjacency matrix to represent the graph
int visited[MAX_NODES]; // Array to track visited nodes during DFS

// Function for parallel Depth-First Search (DFS) traversal
void parallel_dfs(int node, int num_nodes) {
    visited[node] = 1; // Mark the current node as visited
    printf("Node visited: %d\n", node); // Print the visited node

    // Explore neighbors of the current node in parallel using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) {
        // Check if there is an edge from the current node to node i and if i has not been visited
        if (graph[node][i] && !visited[i]) {
            // Recursively visit node i
            parallel_dfs(i, num_nodes);
        }
    }
}

// Main function
int main() {
    int num_nodes, source;

    // Prompt the user to enter the number of nodes in the graph
    printf("Enter the number of nodes: ");
    scanf("%d", &num_nodes);

    // Read the adjacency matrix from user input to represent the graph
    printf("Enter the adjacency matrix:\n");
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            scanf("%d", &graph[i][j]);
        }
    }

    // Initialize the visited array in parallel to mark all nodes as not visited
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) {
        visited[i] = 0;
    }

    // Prompt the user to enter the source node for DFS traversal
    printf("Enter the source node: ");
    scanf("%d", &source);

    // Perform parallel DFS traversal starting from the specified source node
    parallel_dfs(source, num_nodes);

    return 0;
}

/*
Execution: 
To execute the Depth First Search (DFS) program implemented in C++ using OpenMP, follow these instructions:

1. **Compile the Program:**
   Use a C++ compiler with OpenMP support to compile the source file (`dfs_parallel.cpp`). For example:
   ```bash
   g++ -o dfs_program dfs_parallel.cpp -fopenmp
   ```

2. **Run the Executable:**
   After successful compilation, execute the compiled program:
   ```bash
   ./dfs_program
   ```

Example Input:
For example, consider testing with a simple graph:
- Number of nodes (num_nodes): 5
- Adjacency matrix:
```
0 1 1 0 0
1 0 1 1 0
1 1 0 0 1
0 1 0 0 1
0 0 1 1 0


GROUP – A 
Assignment No: 1 
==================================================================== 
Title: Design and implement Parallel Breadth First Search and Depth First Search based on existing algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS 
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

 In above sample code, The pragma omp parallel is used to fork additional threads to carry out the work enclosed in the construct in parallel. The original thread will be denoted as master thread with thread ID 0. 

 For compilation in GCC using fopenmp: 
$ gcc fopenmp hello.c o hello ldl
 
. Depth First Search (DFS): 
 Depth first Search or Depth first traversal is a recursive algorithm for searching all the vertices of a graph or tree data structure. Traversal means visiting all the nodes of a graph. 
 A standard DFS implementation puts each vertex of the graph into one of two categories: 
1. Visited 
2. Not Visited 
 The purpose of the algorithm is to mark each vertex as visited while avoiding cycles. 

. Algorithm for DFS: 
Step 1: Start by putting any one of the graph's vertices on top of a stack. 
Step 2: Take the top item of the stack and add it to the visited list. 
Step 3: Create a list of that vertex's adjacent nodes. Add the ones which aren't in the visited list to the top of the stack. 
Step 4: Keep repeating steps 2 and 3 until the stack is empty. 

. Examples of DFS: 
o An undirected graph with 5 vertices. 
o Start from vertex 0, the DFS algorithm starts by putting it in the Visited list and putting all its adjacent vertices in the stack. 
o Visit the element at the top of stack i.e. 1 and go to its adjacent nodes. Since 0 has already been visited, we visit 2 instead. 
o Vertex 2 has an unvisited adjacent vertex in 4, so we add that to the top of the stack and visit it. 
o After visit the last element 3, it doesn't have any unvisited adjacent nodes, so completed the Depth First Traversal of the graph. 
===================================================================== 
Conclusion: 
Thus we have studied parallelism and OpenMP used in programming. Also, How DFS and BFS algorithm works for traversal of graph. 


 Methodology and Detailed Explanation:
    
    - This program implements parallel Depth-First Search (DFS) to traverse a graph represented by an adjacency matrix.
    
    - The `parallel_dfs` function performs DFS starting from a given node `node` in parallel.
        - It first marks the current node `node` as visited by setting `visited[node] = 1`.
        - It then prints a message to indicate that the node has been visited (`printf("Node visited: %d\n", node)`).
        
        - The function explores all neighbors of the current node `node` using OpenMP parallelization.
            - For each neighbor `i` of `node` (where `graph[node][i] == 1` indicates an edge exists between `node` and `i`):
                - If `i` has not been visited (`!visited[i]`), it recursively calls `parallel_dfs(i, num_nodes)` to visit `i`.
    
    - In the `main` function:
        - It prompts the user to enter the number of nodes (`num_nodes`) in the graph.
        - It then reads the adjacency matrix representing the graph from user input.
        - Next, it initializes the `visited` array in parallel to mark all nodes as not visited (`visited[i] = 0`).
        - It prompts the user to enter the source node (`source`) from which the DFS traversal will start.
        - Finally, it calls `parallel_dfs(source, num_nodes)` to initiate the DFS traversal in parallel from the specified `source` node.
    
    Key Points:
    - The adjacency matrix (`graph`) represents the connectivity between nodes in the graph.
    - OpenMP directives (`#pragma omp parallel for`) are used to parallelize the loop that explores node neighbors during DFS.
    - The `visited` array ensures that each node is visited only once during the DFS traversal to avoid infinite loops.
    - The DFS traversal can visit nodes in any order due to parallel execution, but it guarantees that all reachable nodes are visited.
    
    Note:
    - This implementation assumes an undirected graph, and the adjacency matrix should reflect the connectivity of nodes accurately.
    - Parallel DFS traversal may exhibit different node visitation orders across runs due to the nature of parallel execution.

Let's go through each line of the provided C code and explain its purpose:

1. ```
   #include <stdio.h>
   #include <stdlib.h>
   #include <omp.h>
   ```
   These lines include necessary header files. `<stdio.h>` for standard input/output operations, `<stdlib.h>` for memory allocation functions like `malloc` and `free`, and `<omp.h>` for OpenMP directives used for parallel programming.

2. ```
   #define MAX_NODES 100
   ```
   This line defines a constant `MAX_NODES` with a value of 100, indicating the maximum number of nodes that the graph can have.

3. ```
   int graph[MAX_NODES][MAX_NODES];
   ```
   This declares a 2D array `graph` to represent the adjacency matrix of the graph. It can store up to `MAX_NODES x MAX_NODES` entries, where `graph[i][j]` represents an edge from node `i` to node `j`.

4. ```
   int visited[MAX_NODES];
   ```
   This declares an array `visited` of size `MAX_NODES` to keep track of visited nodes during the DFS traversal.

5. ```
   void parallel_dfs(int node, int num_nodes) {
   ```
   This line defines a function `parallel_dfs` that performs a parallel Depth-First Search (DFS) traversal starting from a given `node` across `num_nodes` in the graph.

6. ```
   visited[node] = 1;
   printf("Node visited: %d\n", node);
   ```
   Marks the current `node` as visited (`visited[node] = 1`) and prints a message indicating the node has been visited.

7. ```
   #pragma omp parallel for
   for (int i = 0; i < num_nodes; i++) {
       if (graph[node][i] && !visited[i]) {
           parallel_dfs(i, num_nodes);
       }
   }
   ```
   This is the heart of the DFS algorithm. The `#pragma omp parallel for` directive parallelizes the loop, allowing multiple threads to explore different nodes concurrently. For each node `i` (neighbor of `node`), it checks if there's an edge (`graph[node][i]`) from `node` to `i` and if `i` has not been visited (`!visited[i]`). If both conditions are true, it recursively calls `parallel_dfs` to visit node `i`.

8. ```
   int main() {
   ```
   Defines the `main()` function, which is the entry point of the program.

9. ```
   int num_nodes, source;
   ```
   Declares variables `num_nodes` (to store the number of nodes in the graph) and `source` (to store the starting node for DFS traversal).

10. ```
    printf("Enter the number of nodes: ");
    scanf("%d", &num_nodes);
   ```
   Prompts the user to enter the number of nodes in the graph and reads the input into `num_nodes`.

11. ```
    printf("Enter the adjacency matrix:\n");
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            scanf("%d", &graph[i][j]);
        }
    }
   ```
   Prompts the user to enter the adjacency matrix of the graph. The nested loops read and populate the `graph` array based on user input, where `graph[i][j]` represents the presence (`1`) or absence (`0`) of an edge between node `i` and node `j`.

12. ```
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) {
        visited[i] = 0;
    }
   ```
   Initializes the `visited` array in parallel, marking all nodes as not visited (`0`) before starting the DFS traversal.

13. ```
    printf("Enter the source node: ");
    scanf("%d", &source);
   ```
   Prompts the user to enter the source node from where the DFS traversal should begin and reads the input into `source`.

14. ```
    parallel_dfs(source, num_nodes);
   ```
   Calls the `parallel_dfs` function with `source` node and `num_nodes` to start the DFS traversal from the specified source node.

15. ```
    return 0;
   ```
   Indicates successful termination of the program (`main()` function).

This program utilizes OpenMP to parallelize the DFS traversal, which can significantly improve performance for large graphs with many nodes. The DFS traversal starts from a specified source node and explores all reachable nodes in parallel, marking them as visited and printing the traversal path.
*/
