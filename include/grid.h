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
__device__ inline static void metaPredicateElementwiseGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N,
                                                       T *dx, int xStride, T *dy, int yStride, T *dz, int zStride, T *extraA, T *extraB, T scalarA, T scalarB
    ) {
    /*
Nd4jIndex n,
			T *dx,
			T *dy,
			int incx,
			int incy,
			T *params,
			T *result,
			int incz,int *allocationPointer,
			UnifiedSharedMemory *manager,
			int *tadOnlyShapeInfo
     */

    if (opTypeB == 0) { // SCALAR

    } else if (opTypeB == 1) { // TRANSFORM

    } else if (opTypeB == 2) { // PWT
        if (opTypeA == 0) { // SCALAR
            __shared__ Nd4jPointer params[2];
            __shared__ T *paramsPtr;
            if (threadIdx.x == 0) {
                params[0] = (Nd4jPointer *) &scalarA;
                params[1] = (Nd4jPointer *) extraB;
                paramsPtr = (T *) params;
            }
            __syncthreads();

            // functions::pairwise_transforms::PairWiseTransform<T>::template
            // PAIRWISE_TRANSFORM_OPS
            DISPATCH_METAOP(transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), OPS_A(SCALAR_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS));


/*
            if (opNumA == 0 && opNumB ==0) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::Add<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==1) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::Copy<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==2) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::Divide<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==3) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::EqualTo<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==4) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::GreaterThan<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==5) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::LessThan<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==6) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::Multiply<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==7) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::ReverseDivide<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==8) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::ReverseSubtract<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==9) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::Subtract<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==10) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::Epsilon<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==11) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::GreaterThanOrEqual<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==12) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::LessThanOrEqual<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==13) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::Max<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==14) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::Min<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==15) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::NotEqualTo<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==16) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::Copy<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==17) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::Axpy<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==45) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::CompareAndSet<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0 && opNumB ==46) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Add<T>, simdOps::CompareAndReplace<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 0) printf("Unknown MetaOp opB id: [%i]\n", opNumB);
            else if (opNumA == 1 && opNumB ==0) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::Add<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==1) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::Copy<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==2) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::Divide<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==3) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::EqualTo<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==4) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::GreaterThan<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==5) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::LessThan<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==6) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::Multiply<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==7) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::ReverseDivide<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==8) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::ReverseSubtract<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==9) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::Subtract<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==10) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::Epsilon<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==11) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::GreaterThanOrEqual<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==12) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::LessThanOrEqual<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==13) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::Max<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==14) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::Min<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==15) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::NotEqualTo<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==16) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::Copy<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==17) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::Axpy<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==45) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::CompareAndSet<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1 && opNumB ==46) functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::MetaOp<T, simdOps::Subtract<T>, simdOps::CompareAndReplace<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr );
            else if (opNumA == 1) printf("Unknown MetaOp opB id: [%i]\n", opNumB);
*/

        } else if (opTypeA == 1) { // TRANSFORM

        }
    } else {
        if (threadIdx.x == 0)
            printf("Unknown opTypeB: [%i]\n", opTypeB);
    }
}

__global__ void metaPredicateElementwiseFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, long N, float *dx, int xStride, float *dy, int yStride, float *dz, int zStride, float *extraA, float *extraB, float scalarA, float scalarB) {

    metaPredicateElementwiseGeneric<float>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);
}

#endif //LIBND4J_GRID_H
