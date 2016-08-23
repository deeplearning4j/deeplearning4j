//
// @author raver119@gmail.com
//
//
#ifndef LIBND4J_GRID_H
#define LIBND4J_GRID_H

// number of pointers within single grid row
#define GRID_WIDTH 19


//#include <scalar.h>
//#include <transform.h>
//#include <pairwise_transform.h>



namespace functions {
    namespace meta {

        template<typename T>
        class MetaTransform {
        public:
            template<typename OpTypeA, typename OpTypeB>
            static inline __device__ void transformCuda(
                    Nd4jIndex n,
                    T *dy,
                    int incy,
                    T *paramsA,
                    T *paramsB,
                    T *result,
                    int resultStride) {

                int totalThreads = gridDim.x * blockDim.x;
                int tid = blockIdx.x * blockDim.x + threadIdx.x;

                Nd4jIndex i = tid;
                if (incy == 1 && resultStride == 1) {

                    for (; i < n; i += totalThreads) {
                        result[i] = OpTypeB::op(OpTypeA::op(dy[i], paramsA), paramsB);
                    }
                } else {

                    for (; i < n; i += totalThreads) {
                        result[i * resultStride] = OpTypeB::op(OpTypeA::op(dy[i * incy], paramsA), paramsB);
                    }
                }
            }

            static inline __device__ void processMetaLinear(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
                                                            Nd4jIndex n,
                                                            T *dy,
                                                            int incy,
                                                            T *paramsA,
                                                            T *paramsB,
                                                            T *result,
                                                            int resultStride) {
                if (opTypeA == 0) {
                    transformCuda<simdOps::Add <T>, simdOps::Abs <T>> (n, dy, incy, paramsA, paramsB, result, resultStride);
                } else if (opTypeA == 1) {
                    transformCuda<simdOps::Abs <T>, simdOps::Add <T>> (n, dy, incy, paramsB, paramsA, result, resultStride);
                }
            }
        };
    }
}

template <typename T>
__device__ inline void metaStridedGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
                                          Nd4jIndex n,
                                          T dx,
                                          T *dy,
                                          int incy,
                                          T *params,
                                          T *result,
                                          int resultStride) {

    /*
    __shared__ T paramsA[1];
    if (threadIdx.x == 0)
        paramsA[0] = dx;
    __syncthreads();
*/
    functions::meta::MetaTransform<T>::processMetaLinear(opTypeA, opNumA, opTypeB, opNumB, n, dy, incy, &dx, params, result, resultStride);
}


__global__ void metaStridedFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
                                 Nd4jIndex n,
                                 float dx,
                                 float *dy,
                                 int incy,
                                 float *paramsB,
                                 float *result,
                                 int resultStride) {


    metaStridedGeneric<float>(opTypeA, opNumA, opTypeB, opNumB, n, dx, dy, incy, paramsB, result, resultStride);
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

    if (opTypeB == 0) { // SCALAR

    } else if (opTypeB == 1) { // TRANSFORM
        if (opTypeA == 0) {
            DISPATCH_METAOP(functions::transform::Transform<T>::template transformCuda, PARAMS(N, dx, xStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), OPS_A(SCALAR_OPS), OPS_B(TRANSFORM_OPS));
        }
    } else if (opTypeB == 2) { // PWT
        if (opTypeA == 0) { // SCALAR

            DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), OPS_A(SCALAR_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS));
        } else if (opTypeA == 1) { // TRANSFORM

            DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), OPS_A(TRANSFORM_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS));
        } else if (opTypeA == 2) {

        }
    } else {
        if (threadIdx.x == 0)
            printf("Unknown opTypeB: [%i]\n", opTypeB);
    }
}

__global__ void metaPredicateStridedFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int xStride, float *dy, int yStride, float *dz, int zStride, float *extraA, float *extraB, float scalarA, float scalarB) {

    metaPredicateStridedGeneric<float>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);
}

#endif //LIBND4J_GRID_H
