


#include <op_boilerplate.h>
#include <pointercast.h>
#include <helpers/TAD.h>
#include <loops/grid_strided.h>
#include <types/float16.h>


#define GRID_WIDTH 19 // number of pointers within single grid row

#include <ops/ops.h>
#include <ops/meta_ops.h>
#include <loops/legacy_ops.h>

template <typename T>
__device__ inline static void metaPredicateStridedGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
                                                          Nd4jLong N, T *dx, Nd4jLong xStride, T *dy, Nd4jLong yStride, T *dz, Nd4jLong zStride, T *extraA, T *extraB, T scalarA, T scalarB
) {
    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) {
            params[0] = (Nd4jPointer *) &scalarA;
        }
        else params[0] = (Nd4jPointer *) extraA;

        if (opTypeB == 0) {
            params[1] = (Nd4jPointer *) &scalarB;
        }
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
            DISPATCH_METAOP(functions::grid::GRID<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
        }
    } else if (opTypeB == 1) { // TRANSFORM
        if (opTypeA == 0) {
            DISPATCH_METAOP(functions::transform::Transform<T>::template transformCuda, PARAMS(N, dx, xStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(SCALAR_OPS), OPS_B(TRANSFORM_OPS));
        }
    } else if (opTypeB == 2) { // PWT
        if (opTypeA == 0) { // SCALAR

            DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(SCALAR_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS));
        } else if (opTypeA == 1) { // TRANSFORM

            DISPATCH_METAOP(functions::grid::GRID<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), MetaOp, OPS_A(TRANSFORM_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS));
        } else if (opTypeA == 2) {

        }
    } else {
        if (threadIdx.x == 0 && blockIdx.x)
            printf("Unknown opTypeB: [%i]\n", opTypeB);
    }
#else
    if (opTypeA == 2) {
        if (opTypeB == 0) {
            //      DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
            //      functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::InvertedMetaOp<T, simdOps::Copy<T>, simdOps::Multiply<T>>>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr);
        }
    }
#endif
}

template<typename T, typename OpClass>
__device__ static inline void invertedMetaPairwiseStridedGeneric(const int opTypeA, const int opTypeB, Nd4jLong N, T *dx, Nd4jLong xStride, T *dy, Nd4jLong yStride, T *dz, Nd4jLong zStride, T *extraA, T *extraB, T scalarA, T scalarB) {
    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) {
            params[0] = (Nd4jPointer *) &scalarA;
        }
        else params[0] = (Nd4jPointer *) extraA;

        if (opTypeB == 0) {
            params[1] = (Nd4jPointer *) &scalarB;
        }
        else params[1] = (Nd4jPointer *) extraB;

        paramsPtr = (T *) params;
    }
    __syncthreads();

    functions::grid::GRIDStrided<T>::template transformCuda<OpClass>(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr);
};


template<typename T>
__device__ static inline void invertedMetaPairwiseStridedNumericGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, T *dx, Nd4jLong xStride, T *dy, Nd4jLong yStride, T *dz, Nd4jLong zStride, T *extraA, T *extraB, T scalarA, T scalarB) {
    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) {
            params[0] = (Nd4jPointer *) &scalarA;
        }
        else params[0] = (Nd4jPointer *) extraA;

        if (opTypeB == 0) {
            params[1] = (Nd4jPointer *) &scalarB;
        }
        else params[1] = (Nd4jPointer *) extraB;

        paramsPtr = (T *) params;
    }
    __syncthreads();

    functions::grid::GRIDStrided<T>::transformCuda(opTypeA, opNumA, opTypeB, opNumB, N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr);
};

extern "C" __global__ void invertedMetaPairwiseStridedNumericFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float *dx, Nd4jLong xStride, float *dy, Nd4jLong yStride, float *dz, Nd4jLong zStride, float *extraA, float *extraB, float scalarA, float scalarB) {
    invertedMetaPairwiseStridedNumericGeneric<float>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);
}

extern "C" __global__ void invertedMetaPairwiseStridedNumericDouble(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, double *dx, Nd4jLong xStride, double *dy, Nd4jLong yStride, double *dz, Nd4jLong zStride, double *extraA, double *extraB, double scalarA, double scalarB) {
    invertedMetaPairwiseStridedNumericGeneric<double>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);
}

extern "C" __global__ void invertedMetaPairwiseStridedNumericHalf(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float16 *dx, Nd4jLong xStride, float16 *dy, Nd4jLong yStride, float16 *dz, Nd4jLong zStride, float16 *extraA, float16 *extraB, float16 scalarA, float16 scalarB) {
    invertedMetaPairwiseStridedNumericGeneric<float16>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);
}

#ifndef __CLION_IDE__
// kernels set for pairwise + scalar based on stride                                                                                         const int opTypeA, const int opTypeB, Nd4jLong N, T *dx, int xStride, T *dy, int yStride, T *dz, int zStride, T *extraA, T *extraB, T scalarA, T scalarB
//DISPATCH_KERNEL_META(invertedMetaPairwiseStrided_Pairwise_Scalar_, invertedMetaPairwiseStridedGeneric, float, metaOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, Nd4jLong N, float *dx, int xStride, float *dy, int yStride, float *dz, int zStride, float *extraA, float *extraB, float scalarA, float scalarB), PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
//DISPATCH_KERNEL_META(invertedMetaPairwiseStrided_Pairwise_Scalar_, invertedMetaPairwiseStridedGeneric, double, metaOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, Nd4jLong N, double *dx, int xStride, double *dy, int yStride, double *dz, int zStride, double *extraA, double *extraB, double scalarA, double scalarB), PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
//DISPATCH_KERNEL_META(invertedMetaPairwiseStrided_Pairwise_Scalar_, invertedMetaPairwiseStridedGeneric, float16, metaOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, Nd4jLong N, float16 *dx, int xStride, float16 *dy, int yStride, float16 *dz, int zStride, float16 *extraA, float16 *extraB, float16 scalarA, float16 scalarB), PARAMS(opTypeA, opTypeB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
#endif


namespace functions {
    namespace grid {

        /**
         * This method is able to execute various ops that takes 2 operands (x, y) + extras
         * @tparam T
         */
        template <typename T>
        __device__ T _execute_2OEF(const int opType, const int opNum, T x, T y, T *extras) {
            T z;
            switch(opType) {
                case 2: {
                    EXECUTE_NOE((x, y, extras), OPS_A(PAIRWISE_TRANSFORM_OPS));
                };
                break;
                default: {
                    PRINT_FIRST("Unknown opType provided: [%i]\n", opType);
                }
                break;
            }
            return z;
        }


        /**
        * This method is able to execute various ops that takes 1 operand (x) + extras
        * @tparam T
        */
        template <typename T>
        __device__ T _execute_1OEF(const int opType, const int opNum, T x, T *extras) {
            T z;
            switch(opType) {
                case 0: {
                    EXECUTE_NOE((x, extras), OPS_A(SCALAR_OPS));
                }
                break;
                default: {
                    PRINT_FIRST("Unknown opType provided: [%i]\n", opType);
                }
                break;
            }

            return z;
        }


        template <typename T>
        __device__ T _invertedOpExecutorB(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, T x, T y, T *extras) {
            // this code is basically InvertedMetaOp, reorganized to suit per-type execution

            Nd4jPointer *wrap = reinterpret_cast<Nd4jPointer *> (extras);
            T *paramsA = reinterpret_cast<T *> (wrap[0]);
            T *paramsB = reinterpret_cast<T *> (wrap[1]);
            T intermediate;

            // Executing first op, opA
            intermediate = _execute_2OEF<T>(opTypeA, opNumA, x, y, paramsA);

            // Executing second op, opB
            intermediate = _execute_1OEF<T>(opTypeB, opNumB, intermediate, paramsB);

            // just returning result now
            return intermediate;
        }

        template<typename T>
        template<typename OpType>
        __device__ void GRIDStrided<T>::transformCuda(Nd4jLong n, T *dx, T *dy, Nd4jLong incx, Nd4jLong incy, T *params, T *result, Nd4jLong incz,int *allocationPointer, UnifiedSharedMemory *manager,Nd4jLong *tadOnlyShapeInfo) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (incx == incy && incy == incz && incx == 1) {
                for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
                    result[i] = OpType::op(dx[i], dy[i], params);
                }
            } else {
                for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
                    result[i * incz] = OpType::op(dx[i * incx], dy[i * incy], params);
                }
            }
        }


        template<typename T>
        __device__ void GRIDStrided<T>::transformCuda(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong n, T *dx, T *dy, Nd4jLong incx, Nd4jLong incy, T *params, T *result, Nd4jLong incz,int *allocationPointer, UnifiedSharedMemory *manager,Nd4jLong *tadOnlyShapeInfo) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            if (incx == incy && incy == incz && incx == 1) {
                for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
                    result[i] = _invertedOpExecutorB(opTypeA, opNumA, opTypeB, opNumB, dx[i], dy[i], params);
                }
            } else {
                for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
                    result[i * incz] = _invertedOpExecutorB(opTypeA, opNumA, opTypeB, opNumB, dx[i * incx], dy[i * incy], params);
                }
            }
        }


        template <>
        void GRIDStrided<float>::execMetaPredicateStrided(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float *dx, Nd4jLong xStride, float *dy, Nd4jLong yStride, float *dz, Nd4jLong zStride, float *extraA, float *extraB, float scalarA, float scalarB) {
            invertedMetaPairwiseStridedNumericFloat<<<128, 1024, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);
        }

        template <>
        void GRIDStrided<float16>::execMetaPredicateStrided(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float16 *dx, Nd4jLong xStride, float16 *dy, Nd4jLong yStride, float16 *dz, Nd4jLong zStride, float16 *extraA, float16 *extraB, float16 scalarA, float16 scalarB) {
            invertedMetaPairwiseStridedNumericHalf<<<128, 1024, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);
        }

        template <>
        void GRIDStrided<double>::execMetaPredicateStrided(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, double *dx, Nd4jLong xStride, double *dy, Nd4jLong yStride, double *dz, Nd4jLong zStride, double *extraA, double *extraB, double scalarA, double scalarB) {
            invertedMetaPairwiseStridedNumericDouble<<<128, 1024, 1024, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xStride, dy, yStride, dz, zStride, extraA, extraB, scalarA, scalarB);
        }

        //template class GRID<float>;
        //template class GRID<float16>;
        //template class GRID<double>;
    }
}