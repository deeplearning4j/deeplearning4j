//
// Created by raver119 on 08.10.2017.
//

#include "../scalar.h"
#include <op_boilerplate.h>

#include "../legacy_ops.h"

namespace functions {
    namespace scalar {


        template<typename T>
        template<typename OpType>
        void ScalarTransform<T>::transform(T *x, int *xShapeInfo, T *extraParams, T *z, int *zShapeInfo, T *scalars, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *tadShapeInfoZ, Nd4jIndex *tadOffsetsZ) {

            if (tadShapeInfoZ == nullptr) {
                tadShapeInfoZ = tadShapeInfo;
                tadOffsetsZ = tadOffsets;
            }

            // tad preparation
            int tadEWS = shape::elementWiseStride(tadShapeInfo);
            int zEWS = shape::elementWiseStride(tadShapeInfo);
            //int tadRank = shape::rank(tadShapeInfo);
            int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
            int numTads =shape::length(xShapeInfo) / tadLength;

            int tadsPerThread = numTads / TAD_THRESHOLD;
            int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
            num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

            // main loop, rolling along tads
#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
            for (int r = 0; r < numTads; r++) {

                Nd4jIndex offset = tadOffsets[r];
                Nd4jIndex offsetZ = tadOffsetsZ[r];
                T scalar = scalars[r];

                if (tadEWS >= 1 && zEWS >= 1) {
                    T *oZ = z + offsetZ;
                    T *oX = x + offset;

                    if (tadEWS == 1 && zEWS == 1) {

#pragma omp simd
                        for (int f = 0; f < tadLength; f++) {
                            oZ[f] = OpType::op(oX[f], scalar, extraParams);
                        }
                    } else {

// TODO: nested loop should be used here probably, instead of simd
#pragma omp simd
                        for (int f = 0; f < tadLength; f++) {
                            oZ[f * zEWS] = OpType::op(oX[f * tadEWS], scalar, extraParams);
                        }
                    }

                } else {
                    // ind2sub loop
                    printf("Super-bad loop visited. Shouldn't ever happen\n");
                }
            }
        }

        template<typename T>
        void ScalarTransform<T>::transform(int opNum,
                              T *x,
                              int *xShapeInfo,
                              T *extraParams,
                              T *z,
                              int *zShapeInfo,
                              T *scalars,
                              int *dimension,
                              int dimensionLength,
                              int *tadShapeInfo,
                              Nd4jIndex *tadOffsets,
                              int *tadShapeInfoZ,
                              Nd4jIndex *tadOffsetsZ) {
            DISPATCH_BY_OPNUM(transform, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), SCALAR_OPS);
        }

        template<typename T>
        void ScalarTransform<T>::transform(const int opNum,
                              T *x,
                              int *xShapeInfo,
                              T *result,
                              int *resultShapeInfo,
                              T scalar,
                              T *extraParams,
                              int *indexes,
                              int *resultIndexes) {
            DISPATCH_BY_OPNUM(transform, PARAMS(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes), SCALAR_OPS);
        }


        template<typename T>
        void ScalarTransform<T>::transform(const int opNum, T *x, int xStride, T *result, int resultStride,
                              T scalar, T *extraParams, const Nd4jIndex n) {
            DISPATCH_BY_OPNUM(transform, PARAMS(x, xStride, result, resultStride, scalar, extraParams, n), SCALAR_OPS);
        }

        template<typename T>
        void ScalarTransform<T>::transform(const int opNum,
                              T *x,
                              int *xShapeInfo,
                              T *result,
                              int *resultShapeInfo,
                              T scalar, T *extraParams) {
            DISPATCH_BY_OPNUM(transform, PARAMS(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams), SCALAR_OPS);
        }

        template<typename T>
        template<typename OpType>
        void ScalarTransform<T>::transform(T *x,
                              int *xShapeInfo,
                              T *result,
                              int *resultShapeInfo,
                              T scalar,
                              T *extraParams,
                              int *indexes,
                              int *resultIndexes) {
            const Nd4jIndex n = shape::length(xShapeInfo);
#pragma omp parallel for simd schedule(guided) if (n > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
            for (Nd4jIndex i = 0; i < n; i++) {
                result[resultIndexes[i]] = OpType::op(x[indexes[i]], scalar,extraParams);
            }
        }

        template<typename T>
        template<typename OpType>
        void ScalarTransform<T>::transform(T *x,
                               int *xShapeInfo,
                               T *result,
                               int *resultShapeInfo,
                               T scalar, T *extraParams) {
            char xOrdering = shape::order(xShapeInfo);
            char resultOrdering = shape::order(resultShapeInfo);
            int xElementWiseStride = shape::elementWiseStride(xShapeInfo);

            nd4j_logger("Launching scalar: xOrder: %i; zOrder: %i; xEWS: %i\n", xOrdering, resultOrdering, xElementWiseStride);

            int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
            if(xOrdering != resultOrdering || xElementWiseStride < 1 || resultElementWiseStride < 0) {
                int shapeIter[MAX_RANK];
                int coord[MAX_RANK];
                int dim;
                int xStridesIter[MAX_RANK];
                int resultStridesIter[MAX_RANK];
                int *xShape = shape::shapeOf(xShapeInfo);
                int *xStride = shape::stride(xShapeInfo);
                int *resultStride = shape::stride(resultShapeInfo);
                int rank = shape::rank(xShapeInfo);
                if(PrepareTwoRawArrayIter<T>(rank,
                                             xShape,
                                             x,
                                             xStride,
                                             result,
                                             resultStride,
                                             &rank,
                                             shapeIter,
                                             &x,
                                             xStridesIter,
                                             &result,
                                             resultStridesIter) >= 0) {
                    ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                            /* Process the innermost dimension */
                            T *xIter = x;
                            T *resultIter = result;
                            resultIter[0] = OpType::op(xIter[0],scalar,extraParams);
                        } ND4J_RAW_ITER_TWO_NEXT(dim,
                                                 rank,
                                                 coord,
                                                 shapeIter,
                                                 x,
                                                 xStridesIter,
                                                 result,
                                                 resultStridesIter);
                }
                else {
                    printf("Unable to prepare array\n");
                }

            }
            else {
                const Nd4jIndex n = shape::length(xShapeInfo);

                if(xElementWiseStride >= 1 && resultElementWiseStride >= 1) {
                    transform<OpType>(x,xElementWiseStride,result,resultElementWiseStride,scalar,extraParams,n);
                }
                else {
                    int *xShape = shape::shapeOf(xShapeInfo);
                    int *resultShape = shape::shapeOf(resultShapeInfo);

                    int *xStride = shape::stride(xShapeInfo);
                    int *resultStride = shape::stride(resultShapeInfo);
                    int xRank = shape::rank(xShapeInfo);
                    int resultRank = shape::rank(resultShapeInfo);

                    int xOffset = shape::offset(xShapeInfo);
                    int resultOffset = shape::offset(resultShapeInfo);

#pragma omp parallel for simd schedule(guided) if (n > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
                    for (Nd4jIndex i = 0; i < n; i++) {
                        int *xIdx = shape::ind2sub(xRank, xShape, i);
                        int *resultIdx = shape::ind2sub(resultRank, resultShape, i);
                        Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xIdx, xRank);
                        Nd4jIndex resultOffset2 = shape::getOffset(resultOffset, resultShape, resultStride, resultIdx,
                                                                   resultRank);

                        result[resultOffset2] = OpType::op(x[xOffset2], scalar, extraParams);

                        delete[] xIdx;
                        delete[] resultIdx;

                    }
                }

                }
            }


            template<typename T>
            template<typename OpType>
            void ScalarTransform<T>::transform(T *x, int xStride, T *result, int resultStride, T scalar, T *extraParams, const Nd4jIndex n) {

                Nd4jIndex elementsPerThread = n / ELEMENT_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                Nd4jIndex span = (n / num_threads) + 8;

                if (xStride == 1 && resultStride == 1) {

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        Nd4jIndex tid = omp_get_thread_num();
                        Nd4jIndex start = span * tid;
                        Nd4jIndex end = span * (tid + 1);
                        if (end > n) end = n;
#pragma omp simd
                        for (Nd4jIndex i = start; i < end; i++) {
                            result[i] = OpType::op(x[i], scalar, extraParams);
                        }
                    }
                }

                else {
#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        Nd4jIndex tid = omp_get_thread_num();
                        Nd4jIndex start = span * tid;
                        Nd4jIndex end = span * (tid + 1);
                        if (end > n) end = n;
#pragma omp simd
                        for (Nd4jIndex i = start; i < end; i++) {
                            result[i * resultStride] = OpType::op(x[i * xStride], scalar, extraParams);
                        }
                    }
                }
            }


        template class ND4J_EXPORT ScalarTransform<float>;
        template class ND4J_EXPORT ScalarTransform<float16>;
        template class ND4J_EXPORT ScalarTransform<double>;
    }
}