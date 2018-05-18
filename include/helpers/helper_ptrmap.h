//
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_PTRMAP_H
#define LIBND4J_HELPER_PTRMAP_H

#ifdef __CUDACC__
#define ptr_def __host__ __device__ inline
#else
#define ptr_def inline
#endif

namespace nd4j {

    /**
     * This class is a simple wrapper to represent batch arguments as single surface of parameters.
     * So we pass batch parameters as single surface, and then we use this helper to extract arguments for each aggregates.
     *
     * Surface map format is simple:
     * [0] we put numbers for num*Arguments
     * [1] then we put indexing arguments, since their size is constant
     * [2] here we put block of JVM IntArrays by value, batchLimit * maxIntArrays * maxArraySize;
     * [3] then we put real arguments
     * [4] then we put arguments pointers
     * [5] then we put shape pointers
     *
     */
    template <typename T>
    class PointersHelper {
    private:
        int aggregates;
        void *ptrGeneral;

        // we enforce maximal batch size limit, to simplify
#ifdef __CUDACC__
        const int batchLimit = 8192;
#else
        const int batchLimit = 512;
#endif

        // we have 5 diff kinds of arguments: arguments, shapeArguments, intArrayArguments, indexArguments, realArguments
        const int argTypes = 5;

        int maxIntArrays;
        int maxArraySize;

        // right now we hardcode maximas, but we'll probably change that later
        int maxIndexArguments;
        int maxRealArguments;

        // since that's pointers (which is 64-bit on 64bit systems), we limit number of maximum arguments to 1/2 of maxIndex arguments
        int maxArguments;
        int maxShapeArguments;

        int sizeT;
        int sizePtr;
    public:

        /**
         * We accept single memory chunk and number of jobs stored.
         *
         * @param ptrToParams pointer to "surface"
         * @param numAggregates actual number of aggregates being passed in
         * @return
         */
#ifdef __CUDACC__
        __host__ __device__
#endif
        PointersHelper(void *ptrToParams, int numAggregates, int maxArgs, int maxShapes, int maxIntArrays, int maxIntArraySize, int maxIdx, int maxReals) {
            aggregates = numAggregates;
            ptrGeneral = ptrToParams;

            // ptrSize for hypothetical 32-bit compatibility
            sizePtr = sizeof(ptrToParams);

            // unfortunately we have to know sizeOf(T)
            sizeT = sizeof(T);

            this->maxIntArrays = maxIntArrays;
            this->maxArraySize = maxIntArraySize;
            this->maxIndexArguments = maxIdx;
            this->maxArguments = maxArgs;
            this->maxShapeArguments = maxShapes;
            this->maxRealArguments = maxReals;
        }

        /**
         * This method returns point
         *
         * @param aggregateIdx
         * @return
         */

        ptr_def T **getArguments(int aggregateIdx) {
            T **aPtr = (T **) getRealArguments(batchLimit);

            return aPtr + (aggregateIdx * maxArguments);
        }

        /**
         * This method returns number of array arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        ptr_def int getNumArguments(int aggregateIdx) {
            int *tPtr = (int *) ptrGeneral;
            return tPtr[aggregateIdx * argTypes];
        }

        /**
         * This method returns set of pointers to shape aruments for specified aggregates
         *
         * @param aggregateIdx
         * @return
         */
        ptr_def Nd4jLong **getShapeArguments(int aggregateIdx) {
            Nd4jLong **sPtr = (Nd4jLong **)getArguments(batchLimit);

            return sPtr + (aggregateIdx * maxShapeArguments);
        }

        /**
         * This methor returns number of shape arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        ptr_def int getNumShapeArguments(int aggregateIdx) {
            int *tPtr = (int *) ptrGeneral;
            return tPtr[aggregateIdx * argTypes + 1];
        }

        /**
         * This method returns pointer to array of int/index arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        ptr_def int *getIndexArguments(int aggregateIdx) {
            // we skip first numeric num*arguments
            int *ptr = ((int *) ptrGeneral) + (batchLimit * argTypes);

            // and return index for requested aggregate
            return ptr + (aggregateIdx * maxIndexArguments) ;
        }

        /**
         * This method returns number of int/index arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        ptr_def int getNumIndexArguments(int aggregateIdx) {
            int *tPtr = (int *) ptrGeneral;
            return tPtr[aggregateIdx * argTypes + 2];
        }

        /**
         * This method returns pointer to array of jvm IntArray arguments
         */
        ptr_def int *getIntArrayArguments(int aggregateIdx, int argumentIdx) {
            int *ptr = (int * )getIndexArguments(batchLimit);

            return ptr + (aggregateIdx * maxIntArrays * maxArraySize) + (argumentIdx * maxArraySize);
        }

        /**
         * This method returns number of jvm IntArray arguments
         */
        ptr_def int getNumIntArrayArguments(int aggregateIdx) {
            int *tPtr = (int *) ptrGeneral;
            return tPtr[aggregateIdx * argTypes + 4];
        }

        /**
         * This method returns real arguments for specific aggregate
         *
         * @param aggregateIdx
         * @return
         */
        ptr_def T *getRealArguments(int aggregateIdx) {
            // we get pointer for last batchElement + 1, so that'll be pointer for 0 realArgument
            T *ptr = (T * ) getIntArrayArguments(batchLimit, 0);

            return ptr + (aggregateIdx * maxRealArguments);
        }

        /**
         * This methor returns number of real arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        ptr_def int getNumRealArguments(int aggregateIdx) {
            int *tPtr = (int *) ptrGeneral;
            return tPtr[aggregateIdx * argTypes + 3];
        }
    };
}

#endif //LIBND4J_HELPER_PTRMAP_H
