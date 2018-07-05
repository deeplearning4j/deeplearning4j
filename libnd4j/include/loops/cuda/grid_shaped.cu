


#include <op_boilerplate.h>
#include <pointercast.h>
#include <helpers/TAD.h>
#include <types/float16.h>
#include <loops/grid_shaped.h>
#include <helpers/DebugHelper.h>


#include <ops/meta_ops.h>
#include <loops/legacy_ops.h>


#define GRID_WIDTH 19 // number of pointers within single grid row

template <typename T>
__device__ inline static void metaPredicateShapeGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB,
                                                        Nd4jLong N, T *dx, Nd4jLong *xShapeInfo, T *dy, Nd4jLong *yShapeInfo, T *dz, Nd4jLong *zShapeInfo, T *extraA, T *extraB, T scalarA, T scalarB) {
    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) {
            params[0] = reinterpret_cast<Nd4jPointer *>(&scalarA);
        }
        else params[0] = reinterpret_cast<Nd4jPointer *>(extraA);

        if (opTypeB == 0) {
            params[1] = reinterpret_cast<Nd4jPointer *>(&scalarB);
        }
        else params[1] = reinterpret_cast<Nd4jPointer *>(extraB);

        paramsPtr = reinterpret_cast<T *>(params);
    }
    __syncthreads();

    if (opTypeA == 2) {
        if (opTypeB == 0) {
            //    DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, paramsPtr, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));
            //  functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda<simdOps::InvertedMetaOp<T, simdOps::Copy<T>, simdOps::Multiply<T>>>(dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, paramsPtr, nullptr, nullptr, nullptr);
        }
    }
}

template<typename T, typename OpClass>
__device__ static inline void invertedMetaPairwiseShapedGeneric(const int opTypeA, const int opTypeB, Nd4jLong N, T *dx, Nd4jLong *xShapeInfo, T *dy, Nd4jLong *yShapeInfo, T *dz, Nd4jLong *zShapeInfo, T *extraA, T *extraB, T scalarA, T scalarB) {
    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) {
            params[0] = reinterpret_cast<Nd4jPointer *>(&scalarA);
        }
        else params[0] = reinterpret_cast<Nd4jPointer *>(extraA);

        if (opTypeB == 0) {
            params[1] = reinterpret_cast<Nd4jPointer *>(&scalarB);
        }
        else params[1] = reinterpret_cast<Nd4jPointer *>(extraB);

        paramsPtr = reinterpret_cast<T *>(params);
    }
    __syncthreads();

    functions::grid::GRIDShaped<T>::template transformCuda<OpClass>(dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, paramsPtr, nullptr, nullptr, nullptr);
};

template<typename T, typename OpClass>
__device__ static inline void invertedMetaPairwiseShapedGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, T *dx, Nd4jLong *xShapeInfo, T *dy, Nd4jLong *yShapeInfo, T *dz, Nd4jLong *zShapeInfo, T *extraA, T *extraB, T scalarA, T scalarB) {
    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) {
            params[0] = reinterpret_cast<Nd4jPointer *>(&scalarA);
        }
        else params[0] = reinterpret_cast<Nd4jPointer *>(extraA);

        if (opTypeB == 0) {
            params[1] = reinterpret_cast<Nd4jPointer *>(&scalarB);
        }
        else params[1] = reinterpret_cast<Nd4jPointer *>(extraB);

        paramsPtr = reinterpret_cast<T *>(params);
    }
    __syncthreads();

    functions::grid::GRIDShaped<T>::template transformCuda<OpClass>(opTypeA, opNumA, opTypeB, opNumB, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, paramsPtr, nullptr, nullptr, nullptr);
};

template<typename T>
__device__ static inline void invertedMetaPairwiseShapedNumericGeneric(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, T *dx, Nd4jLong *xShapeInfo, T *dy, Nd4jLong *yShapeInfo, T *dz, Nd4jLong *zShapeInfo, T *extraA, T *extraB, T scalarA, T scalarB) {
    __shared__ Nd4jPointer params[2];
    __shared__ T *paramsPtr;
    if (threadIdx.x == 0) {
        if (opTypeA == 0) {
            params[0] = reinterpret_cast<Nd4jPointer *>(&scalarA);
        }
        else params[0] = reinterpret_cast<Nd4jPointer *>(extraA);

        if (opTypeB == 0) {
            params[1] = reinterpret_cast<Nd4jPointer *>(&scalarB);
        }
        else params[1] = reinterpret_cast<Nd4jPointer *>(extraB);

        paramsPtr = reinterpret_cast<T *>(params);
    }

    __syncthreads();

    functions::grid::GRIDShaped<T>::transformCuda(opTypeA, opNumA, opTypeB, opNumB, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, paramsPtr, nullptr, nullptr, nullptr);
};


extern "C" __global__ void invertedMetaPairwiseShapedNumericFloat(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float *dx, Nd4jLong *xShapeInfo, float *dy, Nd4jLong *yShapeInfo, float *dz, Nd4jLong *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
    invertedMetaPairwiseShapedNumericGeneric<float>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);
}

extern "C" __global__ void invertedMetaPairwiseShapedNumericDouble(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, double *dx, Nd4jLong *xShapeInfo, double *dy, Nd4jLong *yShapeInfo, double *dz, Nd4jLong *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB) {
    invertedMetaPairwiseShapedNumericGeneric<double>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);
}

extern "C" __global__ void invertedMetaPairwiseShapedNumericHalf(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float16 *dx, Nd4jLong *xShapeInfo, float16 *dy, Nd4jLong *yShapeInfo, float16 *dz, Nd4jLong *zShapeInfo, float16 *extraA, float16 *extraB, float16 scalarA, float16 scalarB) {
    invertedMetaPairwiseShapedNumericGeneric<float16>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);
}



#ifndef __CLION_IDE__
// kernels set for pairwise + scalar based on shape
//DISPATCH_KERNEL_META(invertedMetaPairwiseShaped_Pairwise_Scalar_, invertedMetaPairwiseShapedGeneric, float, metaOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, Nd4jLong N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB), PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
//DISPATCH_KERNEL_META(invertedMetaPairwiseShaped_Pairwise_Scalar_, invertedMetaPairwiseShapedGeneric, double, metaOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, Nd4jLong N, double *dx, int *xShapeInfo, double *dy, int *yShapeInfo, double *dz, int *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB), PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
//DISPATCH_KERNEL_META(invertedMetaPairwiseShaped_Pairwise_Scalar_, invertedMetaPairwiseShapedGeneric, float16, metaOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, Nd4jLong N, float16 *dx, int *xShapeInfo, float16 *dy, int *yShapeInfo, float16 *dz, int *zShapeInfo, float16 *extraA, float16 *extraB, float16 scalarA, float16 scalarB), PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))
#endif

namespace functions {
    namespace grid {
        template <typename T>
        __device__ __noinline__ T invertedOpExecutorA(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, T x, T y, T *extras);

        template <typename T>
        __device__  __noinline__ T execute_2OE(const int opType, const int opNum, T x, T y, T *extras);

        template <typename T>
        __device__ __noinline__ T execute_1OE(const int opType, const int opNum, T x, T *extras);


        __device__ __noinline__ void _ind2subC(int rank, Nd4jLong *shape, Nd4jLong idx, Nd4jLong length, Nd4jLong *coords) {
            shape::ind2subC(rank, shape, idx, length, coords);
        }

        __device__ __noinline__ void _ind2subC(int rank, Nd4jLong *shape, Nd4jLong idx, Nd4jLong *coords) {
            shape::ind2subC(rank, shape, idx, coords);
        }

        __device__ __noinline__ Nd4jLong _getOffset(Nd4jLong offset,  Nd4jLong *shape, Nd4jLong *stride, Nd4jLong *coords, int rank) {
            return shape::getOffset(offset, shape, stride, coords, rank);
        }

        __device__ __noinline__ Nd4jLong* _shapeOf(Nd4jLong *shape) {
            return shape::shapeOf(shape);
        }

        __device__ __noinline__ Nd4jLong* _stride(Nd4jLong *shape) {
            return shape::stride(shape);
        }

        __device__ __noinline__ int _rank(Nd4jLong* shape) {
            return shape::rank(shape);
        }

        /**
         * This method is able to execute various ops that takes 2 operands (x, y) + extras
         * @tparam T
         */
        template <typename T>
        __device__  __noinline__ T execute_2OE(const int opType, const int opNum, T x, T y, T *extras) {
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
        __device__ __noinline__ T execute_1OE(const int opType, const int opNum, T x, T *extras) {
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
        __device__ __noinline__ T invertedOpExecutorA(const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, T x, T y, T *extras) {
            // this code is basically InvertedMetaOp, reorganized to suit per-type execution

            auto wrap = reinterpret_cast<Nd4jPointer *> (extras);
            auto paramsA = reinterpret_cast<T *> (wrap[0]);
            auto paramsB = reinterpret_cast<T *> (wrap[1]);
            T intermediate;

            // Executing first op, opA
            intermediate = functions::grid::execute_2OE<T>(opTypeA, opNumA, x, y, paramsA);

            // Executing second op, opB
            T intermediate2 = functions::grid::execute_1OE<T>(opTypeB, opNumB, intermediate, paramsB);

            //printf("X: [%f]; Y: [%f]; I0: [%f]; Z: [%f];\n", (float) x, (float) y, (float) intermediate, (float) intermediate2);

            // just returning result now
            return intermediate2;
        }

        template<typename T>
        __device__ void GRIDShaped<T>::transformCuda(int opTypeA, int opNumA, int opTypeB, int opNumB,  T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            __shared__ int xRank;
            __shared__ int yRank;
            __shared__ int resultRank;
            __shared__ Nd4jLong n;

            __shared__ Nd4jLong *xShape;
            __shared__ Nd4jLong *yShape;
            __shared__ Nd4jLong *zShape;

            __shared__ Nd4jLong *xStride;
            __shared__ Nd4jLong *yStride;
            __shared__ Nd4jLong *zStride;

            if (threadIdx.x == 0) {
                xRank = _rank(xShapeBuffer);
                yRank = _rank(yShapeBuffer);
                resultRank = _rank(resultShapeBuffer);
                n = shape::length(xShapeBuffer);

                xShape = _shapeOf(xShapeBuffer);
                yShape = _shapeOf(yShapeBuffer);

                if (dx != result) {
                    zShape = _shapeOf(resultShapeBuffer);
                    zStride = _stride(resultShapeBuffer);
                }

                xStride = _stride(xShapeBuffer);
                yStride = _stride(yShapeBuffer);
            }
            __syncthreads();

            if (dx == result) {
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];

                for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
                    _ind2subC(xRank, xShape, i, n, xCoord);
                    _ind2subC(yRank, yShape, i, n, yCoord);

                    auto xOffset = _getOffset(0, xShape, xStride, xCoord, xRank);
                    auto yOffset = _getOffset(0, yShape, yStride, yCoord, yRank);
                    result[xOffset] = functions::grid::invertedOpExecutorA<T>(opTypeA, opNumA, opTypeB, opNumB, dx[xOffset], y[yOffset], extraParams); //OpType::op(dx[xOffset], y[yOffset], extraParams);
                }

            } else {
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];
                Nd4jLong resultCoord[MAX_RANK];

                for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
                    _ind2subC(xRank, xShape, i, n, xCoord);
                    _ind2subC(yRank, yShape, i, n, yCoord);
                    _ind2subC(resultRank, zShape, i, n, resultCoord);

                    auto xOffset = _getOffset(0, xShape, xStride, xCoord, xRank);
                    auto yOffset = _getOffset(0, yShape, yStride, yCoord, yRank);
                    auto resultOffset = _getOffset(0, zShape, zStride, resultCoord, resultRank);
                    result[resultOffset] = functions::grid::invertedOpExecutorA<T>(opTypeA, opNumA, opTypeB, opNumB, dx[xOffset], y[yOffset], extraParams); //OpType::op(dx[xOffset], y[yOffset], extraParams);
                }
            }
        }

        template<typename T>
        template<typename OpType>
        __device__ void GRIDShaped<T>::transformCuda(T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo) {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            __shared__ int xRank;
            __shared__ int yRank;
            __shared__ int resultRank;
            __shared__ Nd4jLong n;

            __shared__ Nd4jLong *xShape;
            __shared__ Nd4jLong *yShape;
            __shared__ Nd4jLong *zShape;

            __shared__ Nd4jLong *xStride;
            __shared__ Nd4jLong *yStride;
            __shared__ Nd4jLong *zStride;

            if (threadIdx.x == 0) {
                xRank = _rank(xShapeBuffer);
                yRank = _rank(yShapeBuffer);
                resultRank = _rank(resultShapeBuffer);
                n = shape::length(xShapeBuffer);

                xShape = _shapeOf(xShapeBuffer);
                yShape = _shapeOf(yShapeBuffer);

                if (dx != result) {
                    zShape = _shapeOf(resultShapeBuffer);
                    zStride = _stride(resultShapeBuffer);
                }

                xStride = _stride(xShapeBuffer);
                yStride = _stride(yShapeBuffer);
            }
            __syncthreads();

            if (dx == result) {
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];

                for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
                    _ind2subC(xRank, xShape, i, n, xCoord);
                    _ind2subC(yRank, yShape, i, n, yCoord);

                    auto xOffset = _getOffset(0, xShape, xStride, xCoord, xRank);
                    auto yOffset = _getOffset(0, yShape, yStride, yCoord, yRank);
                    result[xOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);
                }
            } else {
                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];
                Nd4jLong resultCoord[MAX_RANK];

                for (Nd4jLong i = tid; i < n; i += gridDim.x * blockDim.x) {
                    _ind2subC(xRank, xShape, i, n, xCoord);
                    _ind2subC(yRank, yShape, i, n, yCoord);
                    _ind2subC(resultRank, zShape, i, n, resultCoord);

                    auto xOffset = _getOffset(0, xShape, xStride, xCoord, xRank);
                    auto yOffset = _getOffset(0, yShape, yStride, yCoord, yRank);
                    auto resultOffset = _getOffset(0, zShape, zStride, resultCoord, resultRank);
                    result[resultOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);
                }
            }
        }


        template <>
        void GRIDShaped<float>::execMetaPredicateShaped(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float *dx, Nd4jLong *xShapeInfo, float *dy, Nd4jLong *yShapeInfo, float *dz, Nd4jLong *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
            invertedMetaPairwiseShapedNumericFloat<<<128, 1024, 2048, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);

            DEBUG_KERNEL(stream, opNumA);
        }

        template <>
        void GRIDShaped<float16>::execMetaPredicateShaped(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, float16 *dx, Nd4jLong *xShapeInfo, float16 *dy, Nd4jLong *yShapeInfo, float16 *dz, Nd4jLong *zShapeInfo, float16 *extraA, float16 *extraB, float16 scalarA, float16 scalarB) {
            invertedMetaPairwiseShapedNumericHalf<<<128, 1024, 2048, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);

            DEBUG_KERNEL(stream, opNumB);
        }

        template <>
        void GRIDShaped<double>::execMetaPredicateShaped(cudaStream_t * stream, Nd4jPointer *extras, const int opTypeA, const int opNumA, const int opTypeB, const int opNumB, Nd4jLong N, double *dx, Nd4jLong *xShapeInfo, double *dy, Nd4jLong *yShapeInfo, double *dz, Nd4jLong *zShapeInfo, double *extraA, double *extraB, double scalarA, double scalarB) {
            invertedMetaPairwiseShapedNumericDouble<<<128, 1024, 2048, *stream>>>(opTypeA, opNumA, opTypeB, opNumB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);

            DEBUG_KERNEL(stream, opNumA);
        }
    }
}