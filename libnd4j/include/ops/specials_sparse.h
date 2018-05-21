//
// This file contains various helper methods/classes suited for sparse support
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SPECIALS_SPARSE_H
#define LIBND4J_SPECIALS_SPARSE_H

#include <pointercast.h>

namespace nd4j {
    namespace sparse {

        template <typename T>
        class SparseUtils {
        public:
            /**
        * Just simple helper for debugging :)
        *
        * @param indices
        * @param rank
        * @param x
        */
            static void printIndex(Nd4jLong *indices, int rank, int x);
            static bool ltIndices(Nd4jLong *indices, int rank, Nd4jLong x, Nd4jLong y);

            /**
            * Returns true, if x > y, false otherwise
            * @param indices
            * @param rank
            * @param x
            * @param y
            * @return
            */
            static bool gtIndices(Nd4jLong *indices, int rank, Nd4jLong x, Nd4jLong y);

            static void swapEverything(Nd4jLong *indices, T *array, int rank, Nd4jLong x, Nd4jLong y);

            static void coo_quickSort_parallel_internal(Nd4jLong *indices, T* array, Nd4jLong left, Nd4jLong right, int cutoff, int rank);

            static void coo_quickSort_parallel(Nd4jLong *indices, T* array, Nd4jLong lenArray, int numThreads, int rank);

            static void sortCooIndicesGeneric(Nd4jLong *indices, T *values, Nd4jLong length, int rank);
        };
    }
}


#endif //LIBND4J_SPECIALS_SPARSE_H
