//
// @author raver119@gmail.com
//

#include <ops/specials_sparse.h>
#include <dll.h>
#include <pointercast.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <types/float16.h>

namespace nd4j {
    namespace sparse {

        template <typename T>
        void SparseUtils<T>::printIndex(int *indices, int rank, int x) {
            printf(" [");
            for (int e = 0; e < rank; e++) {
                if (e > 0)
                    printf(", ");

                printf("%i", indices[x * rank + e]);
            }
            printf("] ");
        }

        template <typename T>
        bool SparseUtils<T>::ltIndices(int *indices, int rank, Nd4jIndex x, Nd4jIndex y) {
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

        template <typename T>
        bool SparseUtils<T>::gtIndices(int *indices, int rank, Nd4jIndex x, Nd4jIndex y) {
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
        void SparseUtils<T>::swapEverything(int *indices, T *array, int rank, Nd4jIndex x, Nd4jIndex y) {
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
        void SparseUtils<T>::coo_quickSort_parallel_internal(int *indices, T* array, Nd4jIndex left, Nd4jIndex right, int cutoff, int rank) {

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

        template <typename T>
        void SparseUtils<T>::coo_quickSort_parallel(int *indices, T* array, Nd4jIndex lenArray, int numThreads, int rank){

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
        void SparseUtils<T>::sortCooIndicesGeneric(int *indices, T *values, Nd4jIndex length, int rank) {
            coo_quickSort_parallel(indices, values, length, omp_get_max_threads(), rank);
            //coo_quickSort_parallel<T>(indices, values, length, 1, rank);
        }


        template class ND4J_EXPORT SparseUtils<float>;
        template class ND4J_EXPORT SparseUtils<float16>;
        template class ND4J_EXPORT SparseUtils<double>;
    }
}
