//
// @author raver119@gmail.com
//

#include <ops/specials_sparse.h>
#include <dll.h>
#include <pointercast.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <types/float16.h>

namespace nd4j {
    namespace sparse {

        template <typename T>
        void SparseUtils<T>::printIndex(Nd4jLong *indices, int rank, int x) {
            printf(" [");
            for (int e = 0; e < rank; e++) {
                if (e > 0)
                    printf(", ");

                printf("%lld", (long long) indices[x * rank + e]);
            }
            printf("] ");
        }

        template <typename T>
        bool SparseUtils<T>::ltIndices(Nd4jLong *indices, int rank, Nd4jLong x, Nd4jLong y) {
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
        bool SparseUtils<T>::gtIndices(Nd4jLong *indices, int rank, Nd4jLong x, Nd4jLong y) {
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
        void SparseUtils<T>::swapEverything(Nd4jLong *indices, T *array, int rank, Nd4jLong x, Nd4jLong y) {
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
        void SparseUtils<T>::coo_quickSort_parallel_internal(Nd4jLong *indices, T* array, Nd4jLong left, Nd4jLong right, int cutoff, int rank) {

            Nd4jLong i = left, j = right;
            Nd4jLong pvt = (left + right) / 2;


            {
                // flag that indicates that pivot index lies between i and j and *could* be swapped.
                bool checkPivot = true;
                /* PARTITION PART */
                while (i <= j) {
                    while (ltIndices(indices, rank, i, pvt))
                        i++;

                    while (gtIndices(indices, rank, j, pvt))
                        j--;


                    if (i <= j) {
                        if(i != j) { // swap can be fairly expensive. don't swap i -> i
                            swapEverything(indices, array, rank, i, j);
                        }

                        // only check pivot if it hasn't already been swapped.
                        if (checkPivot) {
                            // check if we moved the pivot, if so, change pivot index accordingly
                            if (pvt == j) {
                                pvt = i;
                                checkPivot = false;
                            } else if (pvt == i) {
                                pvt = j;
                                checkPivot = false;
                            }
                        }

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
        void SparseUtils<T>::coo_quickSort_parallel(Nd4jLong *indices, T* array, Nd4jLong lenArray, int numThreads, int rank){

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
        void SparseUtils<T>::sortCooIndicesGeneric(Nd4jLong *indices, T *values, Nd4jLong length, int rank) {
#ifdef _OPENMP
            coo_quickSort_parallel(indices, values, length, omp_get_max_threads(), rank);
#else
            coo_quickSort_parallel(indices, values, length, 1, rank);
#endif
        }


        template class ND4J_EXPORT SparseUtils<float>;
        template class ND4J_EXPORT SparseUtils<float16>;
        template class ND4J_EXPORT SparseUtils<double>;
    }
}
