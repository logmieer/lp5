#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_NODES 100
#define INF 999999

int graph[MAX_NODES][MAX_NODES]; // Adjacency matrix
int visited[MAX_NODES];
int distance[MAX_NODES];

// Parallel Breadth-First Search (BFS) function to calculate shortest distances
void parallel_bfs(int source, int num_nodes) {
    // Initialize visited and distance arrays in parallel
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) {
        visited[i] = 0; // Mark all nodes as not visited
        distance[i] = INF; // Set initial distances to infinity
    }

    visited[source] = 1; // Mark the source node as visited
    distance[source] = 0; // Set distance to source node as 0

    int queue[MAX_NODES]; // Queue for BFS traversal
    int front = 0, rear = 0; // Front and rear pointers for queue
    queue[rear++] = source; // Enqueue the source node

    // Perform BFS traversal
    while (front < rear) {
        int u = queue[front++]; // Dequeue a node from the queue

        // Explore neighbors of node u in parallel
        #pragma omp parallel for
        for (int v = 0; v < num_nodes; v++) {
            if (graph[u][v] && !visited[v]) {
                // If there's an edge from u to v and v is not visited
                visited[v] = 1; // Mark v as visited
                distance[v] = distance[u] + 1; // Update distance to v
                queue[rear++] = v; // Enqueue v for further exploration
            }
        }
    }
}

// Main function
int main() {
    int num_nodes, source;

    // Prompt user to enter the number of nodes
    printf("Enter the number of nodes: ");
    scanf("%d", &num_nodes);

    // Read the adjacency matrix from user input
    printf("Enter the adjacency matrix:\n");
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            scanf("%d", &graph[i][j]);
        }
    }

    // Prompt user to enter the source node for BFS
    printf("Enter the source node: ");
    scanf("%d", &source);

    // Call parallel BFS function to calculate shortest distances
    parallel_bfs(source, num_nodes);

    // Output shortest distances from the source node
    printf("Shortest distances from node %d:\n", source);
    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d: Distance %d\n", i, distance[i]);
    }

    return 0;
}

/*
Assignment No: 1 
==================================================================== 
Title: Design and implement Parallel Breadth First Search and Depth First Search based on existing algorithms using OpenMP. Use a Tree or an undirected graph for BFS and DFS 
=====================================================================
Objective: 
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

.  Breadth First Search (BFS): 
 Traversal means visiting all the nodes of a graph. Breadth First Traversal or Breadth First Search is a recursive algorithm for searching all the vertices of a graph or tree data structure. 
 A standard BFS implementation puts each vertex of the graph into one of two categories: 
1. Visited 
2. Not Visited 

 The purpose of the algorithm is to mark each vertex as visited while avoiding cycles. 

. Algorithm for BFS: 
Step 1: Start by putting any one of the graph's vertices at the back of a queue. 
Step 2: Take the front item of the queue and add it to the visited list. 
Step 3: Create a list of that vertex's adjacent nodes. Add the ones which aren't in the visited list to the back of the queue. 
Step 4: Keep repeating steps 2 and 3 until the queue is empty. 

. Example of BFS: 
o An undirected graph with 5 vertices. 
o Start from vertex 0, the BFS algorithm starts by putting it in the Visited list and putting all its adjacent vertices in the stack. 
o Visit the element at the front of queue i.e. 1 and go to its adjacent nodes. Since 0 has already been visited, we visit 2 instead. 
o Vertex 2 has an unvisited adjacent vertex in 4, add that to the back of the queue and visit 3, which is at the front of the queue. 
o Only 4 remains in the queue since the only adjacent node of 3 i.e. 0 is already visited. 
o Visit last remaining item in the queue to check if it has unvisited neighbors. Since the queue is empty, we have completed the Breadth First Traversal of the graph. 

Algorithm for BFS:
1. Enqueue the root node.
2. Dequeue a node and visit its neighbors.
3. Enqueue unvisited neighbors and mark them as visited.
4. Repeat until all nodes are visited.

==========================================================================================================================================

Execution:

Compilation:
First, compile the C program using a C compiler like gcc if you're on a Unix-based system:

g++ -o bfs_program bfs_parallel.cpp -fopenmp

Execute the compiled program:

./bfs_program


Input:
The program will prompt you to enter the number of nodes (num_nodes), the adjacency matrix representing the graph, and the source node (source) from which you want to calculate the shortest distances.

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
```
- Source node (source): 0 (assuming 0 is the index of the source node)

Output:
After providing the input, the program will calculate the shortest distances from the specified source node using parallel BFS. It will then display the shortest distance from the source node to each other node in the graph.

Example Output:
If you use the example input above (assuming source node 0), the output might look like:
```
Shortest distances from node 0:
Node 0: Distance 0
Node 1: Distance 1
Node 2: Distance 1
Node 3: Distance 2
Node 4: Distance 2
```
This output indicates the shortest distances from node 0 to all other nodes in the graph based on the BFS traversal.

---


    Methodology:
    
    - The program uses OpenMP directives for parallelization to speed up the BFS traversal process.
    
    - The `parallel_bfs` function initializes two arrays (`visited` and `distance`) in parallel using OpenMP, marking all nodes as unvisited and setting initial distances to infinity.
    
    - It then sets the source node (`source`) as visited (`visited[source] = 1`) and initializes its distance as `0` (`distance[source] = 0`), as it is the starting point of BFS.
    
    - A queue (`queue`) is used to manage the nodes to be processed during BFS traversal. The `front` and `rear` variables are used to track the front and rear of the queue, respectively.
    
    - The BFS traversal continues as long as there are nodes in the queue (`while (front < rear)`). In each iteration, a node (`u`) is dequeued from the queue.
    
    - The function then explores all neighbors (`v`) of the dequeued node (`u`) in parallel. If there exists an edge from `u` to `v` (`graph[u][v] == 1`) and `v` has not been visited (`!visited[v]`), then:
        - `v` is marked as visited (`visited[v] = 1`).
        - The distance to `v` from the source node (`distance[v]`) is updated to be `distance[u] + 1`, since BFS explores nodes level by level.
        - Node `v` is enqueued (`queue[rear++] = v`) to be processed in subsequent iterations.
    
    - The `main` function reads input values (`num_nodes`, adjacency matrix, and `source` node) from the user.
    
    - It then calls the `parallel_bfs` function with the specified `source` node and number of nodes to compute the shortest distances from the source node to all other nodes.
    
    - Finally, it outputs the computed shortest distances for each node from the source node.
    
    Note:
    - This implementation assumes an unweighted graph (each edge has a weight of `1`), suitable for BFS traversal.
    - For weighted graphs, modifications to the distance update logic would be necessary to account for edge weights appropriately.

    Explaination:
    Let's break down each line of the provided C code:

1. ```
   #include <stdio.h>
   ```
   This line includes the standard input/output library, which provides functions like `printf` and `scanf` for input/output operations.

2. ```
   #include <stdlib.h>
   ```
   This line includes the standard library, which provides functions like memory allocation (`malloc`, `free`) and other general utilities.

3. ```
   #include <omp.h>
   ```
   This line includes OpenMP, a library used for parallel programming in C/C++. It enables developers to write multi-threaded programs by providing constructs for parallelism.

4. ```
   #define MAX_NODES 100
   #define INF 999999
   ```
   These lines define constants `MAX_NODES` and `INF`. `MAX_NODES` sets the maximum number of nodes in the graph (100 in this case), and `INF` represents an infinite distance value.

5. ```
   int graph[MAX_NODES][MAX_NODES]; // Adjacency matrix
   int visited[MAX_NODES];
   int distance[MAX_NODES];
   ```
   These lines declare three arrays:
   - `graph`: An adjacency matrix representing a graph where `graph[i][j]` indicates if there's an edge between nodes `i` and `j`.
   - `visited`: An array to keep track of visited nodes during BFS traversal.
   - `distance`: An array to store the shortest distance from the source node to each node in the graph.

6. ```
   void parallel_bfs(int source, int num_nodes) {
   ```
   This line declares a function named `parallel_bfs` which will perform a parallel Breadth-First Search (BFS) to calculate shortest distances from a given `source` node across `num_nodes`.

7. ```
   #pragma omp parallel for
   for (int i = 0; i < num_nodes; i++) {
       visited[i] = 0; // Mark all nodes as not visited
       distance[i] = INF; // Set initial distances to infinity
   }
   ```
   This `#pragma` directive initiates a parallel loop where each iteration (`i`) is executed in parallel by multiple threads. It initializes the `visited` array to mark all nodes as unvisited (`0`) and sets all distances to `INF`.

8. ```
   visited[source] = 1; // Mark the source node as visited
   distance[source] = 0; // Set distance to source node as 0
   ```
   These lines mark the `source` node as visited (`1`) and set its distance from itself to `0`.

9. ```
   int queue[MAX_NODES]; // Queue for BFS traversal
   int front = 0, rear = 0; // Front and rear pointers for queue
   queue[rear++] = source; // Enqueue the source node
   ```
   This sets up a queue (`queue`) for BFS traversal. The `front` and `rear` pointers track the beginning and end of the queue. Initially, the `source` node is enqueued (`rear++`).

10. ```
    while (front < rear) {
        int u = queue[front++]; // Dequeue a node from the queue
   ```
   This initiates a while loop to process nodes in the BFS traversal. It dequeues (`front++`) a node `u` from the `queue`.

11. ```
    #pragma omp parallel for
    for (int v = 0; v < num_nodes; v++) {
        if (graph[u][v] && !visited[v]) {
            visited[v] = 1; // Mark v as visited
            distance[v] = distance[u] + 1; // Update distance to v
            queue[rear++] = v; // Enqueue v for further exploration
        }
    }
   ```
   Inside the loop, a parallel loop (`#pragma omp parallel for`) explores neighbors (`v`) of node `u`. If there's an edge (`graph[u][v]`) from `u` to `v` and `v` is not visited (`!visited[v]`), `v` is marked as visited, its distance from the `source` node is updated (`distance[v] = distance[u] + 1`), and `v` is enqueued for further exploration.

12. ```
    int main() {
   ```
   This line defines the `main()` function, which is the entry point of the program.

13. ```
    int num_nodes, source;
   ```
   Declares variables `num_nodes` and `source` to store the number of nodes and the source node for BFS.

14. ```
    printf("Enter the number of nodes: ");
    scanf("%d", &num_nodes);
   ```
   Prompts the user to enter the number of nodes in the graph.

15. ```
    printf("Enter the adjacency matrix:\n");
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            scanf("%d", &graph[i][j]);
        }
    }
   ```
   Prompts the user to enter the adjacency matrix representing the graph, where each `graph[i][j]` value indicates the presence of an edge between nodes `i` and `j`.

16. ```
    printf("Enter the source node: ");
    scanf("%d", &source);
   ```
   Prompts the user to enter the source node for the BFS traversal.

17. ```
    parallel_bfs(source, num_nodes);
   ```
   Calls the `parallel_bfs` function with the specified `source` node and `num_nodes` to calculate shortest distances.

18. ```
    printf("Shortest distances from node %d:\n", source);
    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d: Distance %d\n", i, distance[i]);
    }
   ```
   Outputs the shortest distances from the `source` node to all other nodes in the graph, as calculated by the `parallel_bfs` function.

19. ```
    return 0;
   ```
   Indicates successful termination of the program (`main()` function).

This program efficiently uses OpenMP for parallelization to speed up the BFS traversal process, especially useful for large graphs with many nodes.
*/
