//
// Created by raver119 on 24.04.17.
//

#ifndef LIBND4J_CONCAT_H
#define LIBND4J_CONCAT_H


#ifdef __CUDACC__
#define ELEMENT_THRESHOLD 8192
#define TAD_THRESHOLD 2
#endif


namespace nd4j {
    //FIXME: get rid of this redefinition
    typedef union
    {
        float f_;
        int   i_;
    } FloatBits2;



    template <typename T>
    class SpecialMethods {
    public:
        static void concatCpuGeneric(int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfo, T *result, int *resultShapeInfo);
        static void accumulateGeneric(T **x, T *z, int n, const Nd4jIndex length);
        static void averageGeneric(T **x, T *z, int n, const Nd4jIndex length, bool propagate);

        static int getPosition(int *xShapeInfo, int index);
        static void quickSort_parallel_internal(T* array, int *xShapeInfo, int left, int right, int cutoff, bool descending);
        static void quickSort_parallel(T* array, int *xShapeInfo, Nd4jIndex lenArray, int numThreads, bool descending);

        static int nextPowerOf2(int number);
        static int lastPowerOf2(int number);

        static void sortGeneric(T *x, int *xShapeInfo, bool descending);
        static void sortTadGeneric(T *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, bool descending);

        static void decodeBitmapGeneric(void *dx, Nd4jIndex N, T *dz);
        static Nd4jIndex encodeBitmapGeneric(T *dx, Nd4jIndex N, int *dz, float threshold);
    };
}


#endif //LIBND4J_CONCAT_H
