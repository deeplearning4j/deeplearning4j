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
        static void concatCpuGeneric(int dimension, int numArrays, Nd4jPointer *data, Nd4jPointer *inputShapeInfo, T *result, Nd4jLong *resultShapeInfo);
        static void accumulateGeneric(T **x, T *z, int n, const Nd4jLong length);
        static void averageGeneric(T **x, T *z, int n, const Nd4jLong length, bool propagate);

        static Nd4jLong getPosition(Nd4jLong *xShapeInfo, Nd4jLong index);
        static void quickSort_parallel_internal(T* array, Nd4jLong *xShapeInfo, int left, int right, int cutoff, bool descending);
        static void quickSort_parallel(T* array, Nd4jLong *xShapeInfo, Nd4jLong lenArray, int numThreads, bool descending);

        static int nextPowerOf2(int number);
        static int lastPowerOf2(int number);

        static void sortGeneric(T *x, Nd4jLong *xShapeInfo, bool descending);
        static void sortTadGeneric(T *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending);

        static void decodeBitmapGeneric(void *dx, Nd4jLong N, T *dz);
        static Nd4jLong encodeBitmapGeneric(T *dx, Nd4jLong N, int *dz, float threshold);
    };
}


#endif //LIBND4J_CONCAT_H
