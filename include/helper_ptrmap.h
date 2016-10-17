//
// @author raver119@gmail.com
//

#ifndef LIBND4J_HELPER_PTRMAP_H
#define LIBND4J_HELPER_PTRMAP_H



namespace nd4j {

    /**
     * This class is a simple wrapper to represent batch arguments as single surface of parameters.
     * So we pass batch parameters as single surface, and then we use this helper to extract arguments for each aggregates.
     */
    template <typename T>
    class PointersHelper {
    private:
        int aggregates;
        void *ptrGeneral;
        int ptrSize;
        int aggregateWidth;
    public:

        /**
         * We accept single memory chunk and number of jobs stored.
         *
         * @param ptrToParams pointer to "surface"
         * @param numAggregates number of aggregates being passed in
         * @param width
         * @return
         */
        PointersHelper(void *ptrToParams, int numAggregates, int width) {
            aggregates = numAggregates;
            ptrGeneral = ptrToParams;
            aggregateWidth = width;

            ptrSize = sizeof(ptrToParams);
        }

        /**
         * This method returns point
         *
         * @param jobId
         * @return
         */
        T **getArguments(int aggregateIdx) {
            return NULL;
        }

        /**
         * This method returns number of array arguments for specified aggregate
         *
         * @param jobId
         * @return
         */
        int getNumArguments(int aggregateIdx) {
            return 0;
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
         * @param jobId
         * @return
         */
        int getNumShapeArguments(int aggregateIdx) {
            return 0;
        }

        /**
         * This method returns pointer to array of int/index arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        int *getIndexArguments(int aggregateIdx) {
            return NULL;
        }

        /**
         * This method returns number of int/index arguments for specified aggregate
         *
         * @param aggregateIdx
         * @return
         */
        int getNumIndexArguments(int aggregateIdx) {
            return 0;
        }

        /**
         * This method returns real arguments for specific aggregate
         *
         * @param aggregateIdx
         * @return
         */
        T *getRealArguments(int aggregateIdx) {
            return NULL;
        }

        /**
         * This methor returns number of real arguments for specified aggregate
         *
         * @param jobId
         * @return
         */
        int getNumRealArguments(int aggregateIdx) {
            return 0;
        }
    };
}

#endif //LIBND4J_HELPER_PTRMAP_H
