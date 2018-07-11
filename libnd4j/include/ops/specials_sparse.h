//
// This file contains various helper methods/classes suited for sparse support
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_SPECIALS_SPARSE_H
#define LIBND4J_SPECIALS_SPARSE_H

#define ND4J_CLIPMODE_THROW 0
#define ND4J_CLIPMODE_WRAP 1
#define ND4J_CLIPMODE_CLIP 2

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

            static Nd4jLong coo_quickSort_findPivot(Nd4jLong *indices, T *array, Nd4jLong left, Nd4jLong right,
                                                    int rank);

            static void sortCooIndicesGeneric(Nd4jLong *indices, T *values, Nd4jLong length, int rank);


        };

        class IndexUtils {
            public:
            /**
             * Converts indices in COO format into an array of flat indices
             * 
             * based on numpy.ravel_multi_index
             */
            static void ravelMultiIndex(Nd4jLong *indices, Nd4jLong *flatIndices, Nd4jLong length,  Nd4jLong *fullShapeBuffer, int mode);

            /**
             * Converts flat indices to index matrix in COO format
             * 
             * based on numpy.unravel_index
             */
            static void unravelIndex(Nd4jLong *indices, Nd4jLong *flatIndices, Nd4jLong length,  Nd4jLong *fullShapeBuffer);
        };
    }
}


#endif //LIBND4J_SPECIALS_SPARSE_H
