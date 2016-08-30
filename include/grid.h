//
// @author raver119@gmail.com
//
//
#ifndef LIBND4J_GRID_H
#define LIBND4J_GRID_H

// number of pointers within single grid row
#define GRID_WIDTH 19


template <typename T>
__device__ inline static void metaPredicateReduceGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
                                                         T *dx, int *xShapeInfo, T *dy, int *yShapeInfo, T *dz, int *zShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets, T *reductionBuffer, T *extraA, T *extraB, T scalarA, T scalarB, bool scalarReturned) {
    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        if (opTypeA == 0) params[0] = (Nd4jPointer *) &scalarA;
        else params[0] = (Nd4jPointer *) extraA;

        if (opTypeB == 0) params[1] = (Nd4jPointer *) &scalarB;
        else params[1] = (Nd4jPointer *) extraB;

        paramsPtr = (T *) params;

        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce::ReduceFunction<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));
    }
    __syncthreads();

    // this op can be used only for reduce calls now

    if (opTypeA == 0) { // scalar

     //   DISPATCH_METAOP(functions::reduce::ReduceFunction<T>::template transformCuda6D, PARAMS(dx, xShapeInfo, paramsPtr, dz, zShapeInfo,  dimension, dimensionLength, reductionBuffer, manager, tadShapeInfo, tadOffsets), ReduceMetaOp, OPS_A(SCALAR_OPS), OPS_B(REDUCE_OPS));
    } else if (opTypeA == 1) { // transform

    } else {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            printf("Unknown opTypeA: [%i]\n", opTypeA);
    }
}

template <typename T>
__device__ inline static void metaPredicateShapeGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N,
                                                          T *dx, int *xShapeInfo, T *dy, int *yShapeInfo, T *dz, int *zShapeInfo, T *extraA, T *extraB, T scalarA, T scalarB) {
    __shared__
    Nd4jPointer params[2];
    __shared__
    T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) params[0] = (Nd4jPointer * ) & scalarA;
        else params[0] = (Nd4jPointer *) extraA;

        if (opTypeB == 0) params[1] = (Nd4jPointer * ) & scalarB;
        else params[1] = (Nd4jPointer *) extraB;

        paramsPtr = (T *) params;
    }
    __syncthreads();

    if (opTypeA == 2) {
        if (opTypeB == 0) {
      //      DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, paramsPtr, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    }
}

template <typename T>
__device__ inline static void metaPredicateStridedGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N,
                                                       T *dx, int xStride, T *dy, int yStride, T *dz, int zStride, T *extraA, T *extraB, T scalarA, T scalarB
    ) {

    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) params[0] = (Nd4jPointer *) &scalarA;
        else params[0] = (Nd4jPointer *) extraA;

        if (opTypeB == 0) params[1] = (Nd4jPointer *) &scalarB;
        else params[1] = (Nd4jPointer *) extraB;

        paramsPtr = (T *) params;
    }
    __syncthreads();
#ifdef __EXPERIMENTAL__
    if (opTypeB == 0) { // SCALAR
        if (opTypeA == 0) {
            // double scalar
            DISPATCH_METAOP(functions::transform::Transform<T>::template transformCuda, PARAMS(N, dx, xStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(SCALAR_OPS), OPS_B(SCALAR_OPS));
        } else if (opTypeA == 1) {
            // transform
            DISPATCH_METAOP(functions::transform::Transform<T>::template transformCuda, PARAMS(N, dx, xStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        } else if (opTypeA == 2) {
            // pwt
            // this is the most important thing here: its Dup() + Scalar
            DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    } else if (opTypeB == 1) { // TRANSFORM
        if (opTypeA == 0) {
            DISPATCH_METAOP(functions::transform::Transform<T>::template transformCuda, PARAMS(N, dx, xStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(SCALAR_OPS), OPS_B(TRANSFORM_OPS));
        }
    } else if (opTypeB == 2) { // PWT
        if (opTypeA == 0) { // SCALAR

            DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(SCALAR_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS));
        } else if (opTypeA == 1) { // TRANSFORM

            DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(TRANSFORM_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS));
        } else if (opTypeA == 2) {

        }
    } else {
        if (threadIdx.x == 0 && blockIdx.x)
            printf("Unknown opTypeB: [%i]\n", opTypeB);
    }
#else
    if (opTypeA == 2) {
        if (opTypeB == 0) {
     //       DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    }
#endif
}

__global__ void metaPredicateStridedFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int xStride, float *dy, int yStride, float *dz, int zStride, float *extraA, float *extraB, float scalarA, float scalarB) {

    metaPredicateStridedGeneric<float>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);
}

__global__ void metaPredicateReduceFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
                                                float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo,  int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets, float *reductionBuffer, float *extraA, float *extraB, float scalarA, float scalarB, bool scalarReturned) {

    metaPredicateReduceGeneric<float>(opTypeA, opNumA, opTypeB, opNumB, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, reductionBuffer, extraA, extraB, scalarA, scalarB, scalarReturned);
}

__global__ void metaPredicateShapeFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {

    metaPredicateShapeGeneric<float>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);
}

#endif //LIBND4J_GRID_H
