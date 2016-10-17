//
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_PTRMAP_H
#define LIBND4J_HELPER_PTRMAP_H



namespace nd4j {

    /**
     * This class is a simple wrapper to represent batch arguments as single surface of parameters.
     * So we pass batch parameters as single surface, and then we use this helper to extract arguments for each aggregates.
     *
     * Surface map format is simple:
     * [0] we put numbers for num*Arguments
     * [1] then we put indexing arguments, since their size is constant
     * [2] then we put real arguments
     * [3] then we put shape pointers
     * [4] then we put arguments pointers
     *
     */
    template <typename T>
    class PointersHelper {
    private:
        int aggregates;
        void *ptrGeneral;

        // we enforce maximal batch size limit, to simplify
        const int batchLimit = 512;

        // we have 4 diff kinds of arguments: arguments, shapeArguments, indexArguments, realArguments
        const int argTypes = 4;

        // right now we hardcode maximas, but we'll probably change that later
        const int maxIndexArguments = 32;
        const int maxRealArguments = 32;

        // since that's pointers (which is 64-bit on 64bit systems), we limit number of maximum arguments to 1/2 of maxIndex arguments
        const int maxArguments = 16;
        const int maxShapeArguments = 16;

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
        PointersHelper(void *ptrToParams, int numAggregates) {
            aggregates = numAggregates;
            ptrGeneral = ptrToParams;

            // ptrSize for hypothetical 32-bit compatibility
            sizePtr = sizeof(ptrToParams);

            // unfortunately we have to know sizeOf(T)
            sizeT = sizeof(T);
        }

        /**
         * This method returns point
         *
         * @param aggregateIdx
         * @return
         */
        T **getArguments(int aggregateIdx) {
            return NULL;
        }

        /**
         * This method returns number of array arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        int getNumArguments(int aggregateIdx) {
            int *tPtr = (int *) ptrGeneral;
            return tPtr[aggregateIdx * argTypes];
        }

        /**
         * This method returns set of pointers to shape aruments for specified aggregates
         *
         * @param aggregateIdx
         * @return
         */
        int **getShapeArguments(int aggregateIdx) {
            return NULL;
        }

        /**
         * This methor returns number of shape arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        int getNumShapeArguments(int aggregateIdx) {
            int *tPtr = (int *) ptrGeneral;
            return tPtr[aggregateIdx * argTypes + 1];
        }

        /**
         * This method returns pointer to array of int/index arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        int *getIndexArguments(int aggregateIdx) {
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
        int getNumIndexArguments(int aggregateIdx) {
            int *tPtr = (int *) ptrGeneral;
            return tPtr[aggregateIdx * argTypes + 2];
        }

        /**
         * This method returns real arguments for specific aggregate
         *
         * @param aggregateIdx
         * @return
         */
        T *getRealArguments(int aggregateIdx) {
            // we get pointer for last batchElement + 1, so that'll be pointer for 0 realArgument
            T *ptr = (T * )getIndexArguments(batchLimit);

            return ptr + (aggregateIdx * maxRealArguments);
        }

        /**
         * This methor returns number of real arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        int getNumRealArguments(int aggregateIdx) {
            int *tPtr = (int *) ptrGeneral;
            return tPtr[aggregateIdx * argTypes + 3];
        }
    };
}

#endif //LIBND4J_HELPER_PTRMAP_H
