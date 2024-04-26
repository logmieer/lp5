#include <iostream>
#include <omp.h>
using namespace std;

void bubbleSort(int arr[], int n);

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

int main() {
    int *arr, n;
    double start_time, end_time, seq_time, par_time;
    
    cout << "\nEnter total number of elements: ";
    cin >> n;
    arr = new int[n];
    
    cout << "\nEnter elements: ";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }
    
    // Sequential algorithm
    start_time = omp_get_wtime();
    bubbleSort(arr, n);
    end_time = omp_get_wtime();
    seq_time = end_time - start_time;
    cout << "\nSequential Time: " << seq_time << endl;
    
    // Parallel algorithm
    start_time = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            bubbleSort(arr, n);
        }
    }
    end_time = omp_get_wtime();
    par_time = end_time - start_time;
    cout << "\nParallel Time: " << par_time << endl;
    
    cout << "\nSorted array is: ";
    for (int i = 0; i < n; i++) {
        cout << "\n" << arr[i];
    }

    delete[] arr;
    return 0;
}

