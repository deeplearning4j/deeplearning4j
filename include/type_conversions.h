//
// @author raver119@gmail.com
//

#ifndef LIBND4J_TYPE_CONVERSIONS_H
#define LIBND4J_TYPE_CONVERSIONS_H

#define ND4J_FLOAT8 0
#define ND4J_INT8 1
#define ND4J_UINT8 2
#define ND4J_FLOAT16 3
#define ND4J_INT16 4
#define ND4J_FLOAT24 5
#define ND4J_FLOAT32 6
#define ND4J_DOUBLE 7


template<typename S, typename T>
void convertGeneric(void *dx, long N, void *dz) {
    S *x = reinterpret_cast<S *> (dx);
    T *z = reinterpret_cast<T *> (dz);

    if (N < 8000) {

#pragma omp simd
        for (int i = 0; i < N; i++) {
            z[i] = (T) x[i];
        }
    } else {

#pragma omp parallel for simd schedule(guided)
        for (int i = 0; i < N; i++) {
            z[i] = (T) x[i];
        }
    }
};

/*
 * TypeDef:
 *     void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, long N, int dstType, Nd4jPointer z);
 */
void NativeOps::convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, long N, int dstType, Nd4jPointer z) {
    void *dx = reinterpret_cast<void *> (x);
    void *dz = reinterpret_cast<void *> (z);

    if (srcType == ND4J_FLOAT8) {

    } else if (srcType == ND4J_INT8) {

    } else if (srcType == ND4J_UINT8) {

    } else if (srcType == ND4J_FLOAT16) {

    } else if (srcType == ND4J_INT16) {

    } else if (srcType == ND4J_FLOAT24) {

    } else if (srcType == ND4J_FLOAT32) {
        if (dstType == ND4J_FLOAT8) {

        } else if (dstType == ND4J_INT8) {

        } else if (dstType == ND4J_UINT8) {

        } else if (dstType == ND4J_FLOAT16) {
            //convertGeneric<float, nd4j::float16>(dx, N, dz);
        } else if (dstType == ND4J_INT16) {

        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_DOUBLE) {
            convertGeneric<float, double>(dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_DOUBLE) {

    } else {
        printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
}


#endif //LIBND4J_TYPE_CONVERSIONS_H
