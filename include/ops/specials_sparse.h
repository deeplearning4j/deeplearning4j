//
// This file contains various helper methods/classes suited for sparse support
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SPECIALS_SPARSE_H
#define LIBND4J_SPECIALS_SPARSE_H

/**
 * Just simple helper for debugging :)
 *
 * @param indices
 * @param rank
 * @param x
 */
void printIndex(int *indices, int rank, int x) {
    printf(" [");
    for (int e = 0; e < rank; e++) {
        if (e > 0)
            printf(", ");

        printf("%i", indices[x * rank + e]);
    }
    printf("] ");
}

bool ltIndices(int *indices, int rank, Nd4jIndex x, Nd4jIndex y) {
    for (int e = 0; e < rank; e++) {
        int idxX = indices[x * rank + e];
        int idxY = indices[y * rank + e];
        // we're comparing indices one by one, starting from outer dimension
        if (idxX < idxY) {
            return true;
        } else if (idxX == idxY) {
            // do nothing, continue to next dimension
        } else
            return false;
    }

    return false;
}

/**
 * Returns true, if x > y, false otherwise
 * @param indices
 * @param rank
 * @param x
 * @param y
 * @return
 */

bool gtIndices(int *indices, int rank, Nd4jIndex x, Nd4jIndex y) {
    for (int e = 0; e < rank; e++) {
        // we're comparing indices one by one, starting from outer dimension
        int idxX = indices[x * rank + e];
        int idxY = indices[y * rank + e];
        if ( idxX > idxY) {
            return true;
        } else if (idxX == idxY) {
            // do nothing, continue to next dimension
        } else
            return false;
    }
    return false;
}

template <typename T>
void swapEverything(int *indices, T *array, int rank, Nd4jIndex x, Nd4jIndex y) {
    // swap indices
    for (int e = 0; e < rank; e++) {
        int tmp = indices[x * rank + e];
        indices[x * rank + e] = indices[y * rank + e];
        indices[y * rank + e] = tmp;
    }

    // swap values
    T tmp = array[x];
    array[x] = array[y];
    array[y] = tmp;
}

template<typename T>
void coo_quickSort_parallel_internal(int *indices, T* array, Nd4jIndex left, Nd4jIndex right, int cutoff, int rank)
{

    Nd4jIndex i = left, j = right;
    Nd4jIndex pvt = (left + right) / 2;


    {
        /* PARTITION PART */
        while (i <= j) {
            while (ltIndices(indices, rank, i, pvt))
                i++;

            while (gtIndices(indices, rank, j, pvt))
                j--;


            if (i <= j) {
                swapEverything(indices, array, rank, i, j);
                i++;
                j--;
            }
        }

    }


    if ( ((right-left)<cutoff) ){
        if (left < j){ coo_quickSort_parallel_internal(indices, array, left, j, cutoff, rank); }
        if (i < right){ coo_quickSort_parallel_internal(indices, array, i, right, cutoff, rank); }

    }else{
#pragma omp task
        { coo_quickSort_parallel_internal(indices, array, left, j, cutoff, rank); }
#pragma omp task
        { coo_quickSort_parallel_internal(indices, array, i, right, cutoff, rank); }
    }

}

template<typename T>
void coo_quickSort_parallel(int *indices, T* array, Nd4jIndex lenArray, int numThreads, int rank){

    int cutoff = 1000;

#pragma omp parallel num_threads(numThreads) if (numThreads>1)
    {
#pragma omp single nowait
        {
            coo_quickSort_parallel_internal(indices, array, 0, lenArray-1, cutoff, rank);
        }
    }

}


template <typename T>
void sortCooIndicesGeneric(int *indices, T *values, Nd4jIndex length, int rank) {
    coo_quickSort_parallel<T>(indices, values, length, omp_get_max_threads(), rank);
    //coo_quickSort_parallel<T>(indices, values, length, 1, rank);
}

#endif //LIBND4J_SPECIALS_SPARSE_H
