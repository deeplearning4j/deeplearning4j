//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/reduce.h>
#include <loops/legacy_ops.h>

namespace functions {
    namespace reduce {

        template <typename T>
        template <typename OpType>
            T _CUDA_H ReduceFunction<T>::execScalar(T *x, Nd4jLong *xShapeInfo, T *extraParams) {
                const Nd4jLong length = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                if (xElementWiseStride >= 1) {
                    return execScalar<OpType>(x, xElementWiseStride, length, extraParams);
                }
                else {
                    Nd4jLong shapeIter[MAX_RANK];
                    Nd4jLong coord[MAX_RANK];
                    int dim;
                    Nd4jLong xStridesIter[MAX_RANK];

                    auto xShape = shape::shapeOf(xShapeInfo);
                    auto xStride = shape::stride(xShapeInfo);
                    T start = OpType::startingValue(x);
                    int rank = shape::rank(xShapeInfo);

                    if (PrepareOneRawArrayIter<T>(rank,
                                                  xShape,
                                                  x,
                                                  xStride,
                                                  &rank,
                                                  shapeIter,
                                                  &x,
                                                  xStridesIter) >= 0) {

                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                /* Process the innermost dimension */
                                const T *xIter = x;
                                start = OpType::update(start, OpType::op(xIter[0], extraParams), extraParams);
                            }
                        ND4J_RAW_ITER_ONE_NEXT(dim,
                                               rank,
                                               coord,
                                               shapeIter,
                                               x,
                                               xStridesIter);
                        start = OpType::postProcess(start, shape::length(xShapeInfo), extraParams);
                    }
                    else {
                        printf("Unable to prepare array\n");
                    }

                    return start;
                }
            }

        template <typename T>
        T ReduceFunction<T>::execScalar(const int opNum, T *x, Nd4jLong *xShapeInfo, T *extraParams) {
                RETURNING_DISPATCH_BY_OPNUM(execScalar, PARAMS(x, xShapeInfo, extraParams), REDUCE_OPS);
        }

        template <typename T>
        void ReduceFunction<T>::exec(const int opNum,
                             T *x,
                             Nd4jLong *xShapeInfo,
                             T *extraParams,
                             T *result,
                             Nd4jLong *resultShapeInfoBuffer,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset) {
                DISPATCH_BY_OPNUM(exec, PARAMS(x,
                                               xShapeInfo,
                                               extraParams,
                                               result,
                                               resultShapeInfoBuffer,
                                               dimension,
                                               dimensionLength,
                                               tadShapeInfo,
                                               tadOffset),
                                  REDUCE_OPS);
        }

        template <typename T>
        template <typename OpType>
        void _CUDA_H ReduceFunction<T>::exec(T *x,
                             Nd4jLong *xShapeInfo,
                             T *extraParams,
                             T *result,
                             Nd4jLong *resultShapeInfoBuffer,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset) {

                auto resultLength = shape::length(resultShapeInfoBuffer);

                //pre squeezed: this is for keeping the pointer to the original
                //shape information for tad offset
                //the squeezed information doesn't render the right strides for
                //tad offset
                // || tad.wholeThing
                if (resultLength == 1 || dimension == nullptr || dimensionLength == shape::rank(xShapeInfo)) {
                    result[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
                    return;
                }

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecial(x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength, tadShapeInfo, tadOffset);
                    return;
                }

                auto tadOnlyShapeInfo = tadShapeInfo;
                auto tadOffsets = tadOffset;
                shape::TAD *tad = nullptr;

                if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
                    tad = new shape::TAD(xShapeInfo, dimension, dimensionLength);
                    tad->createTadOnlyShapeInfo();
                    tad->createOffsets();

                    if (tad->dimensionLength < 1) {
                        delete tad;
                        return;
                    }

                    tadOnlyShapeInfo = tad->tadOnlyShapeInfo;
                    tadOffsets = tad->tadOffsets;
                }


                const auto tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                auto numTads = shape::length(xShapeInfo) / tadLength;
                auto tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);

                int tadsPerThread = resultLength / TAD_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                if (tadEWS > 0 && (numTads == 1 || shape::isVector(tadOnlyShapeInfo) || shape::isScalar(tadOnlyShapeInfo))) {

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
                    for (int i = 0; i < resultLength; i++) {
                        T *iter = x + tadOffsets[i];
                        T start = OpType::startingValue(iter);
                        if (tadEWS == 1) {

// FIXME: proper reduction should be used here
                            for (int j = 0; j < tadLength; j++) {
                                start = OpType::update(start, OpType::op(iter[j], extraParams), extraParams);

                            }
                        }
                        else {
// FIXME: proper reduction to be used here
                            for (int j = 0; j < tadLength; j++) {
                                start = OpType::update(start, OpType::op(iter[j * tadEWS], extraParams), extraParams);
                            }
                        }
                        result[i] = OpType::postProcess(start, tadLength, extraParams);
                    }
                }
                else {
                    auto tadShape = shape::shapeOf(tadOnlyShapeInfo);
                    auto tadStride = shape::stride(tadOnlyShapeInfo);
                    int tadRank = shape::rank(tadOnlyShapeInfo);

#pragma omp  parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
                    for (int i = 0; i < resultLength; i++) {
                        auto offset = tadOffsets[i];
                        Nd4jLong xCoord[MAX_RANK];

                        T start = OpType::startingValue(x + offset);

                        for (int j = 0; j < tadLength; j++) {
                            shape::ind2subC(tadRank, tadShape, j, tadLength, xCoord);
                            auto xOffset = shape::getOffset(offset, tadShape, tadStride, xCoord, tadRank);

                            start = OpType::update(start, OpType::op(x[xOffset], extraParams), extraParams);
                        }

                        result[i] = OpType::postProcess(start, tadLength, extraParams);;
                    }
                }

                if (tad != nullptr)
                    delete tad;
            }


        template <typename T>
        template<typename OpType>
        void _CUDA_H ReduceFunction<T>::exec(T *x,
                             Nd4jLong *xShapeInfo,
                             T *extraParams,
                             T *result,
                             Nd4jLong *resultShapeInfo) {
                return execScalar<OpType>(x, xShapeInfo, extraParams);
        }

        template <typename T>
        template <typename OpType>
        T _CUDA_H ReduceFunction<T>::execScalar(const T *x, Nd4jLong xElementWiseStride, Nd4jLong length, T *extraParams) {
                T startingVal = OpType::startingValue(x);
                if (xElementWiseStride == 1) {
                    if (length < ELEMENT_THRESHOLD) {
                        T local = OpType::startingValue(x);

// FIXME: proper reduction to be used here
                        for (Nd4jLong i = 0; i < length; i++) {
                            T curr = OpType::op(x[i], extraParams);
                            local = OpType::update(local, curr, extraParams);

                        }
                        local = OpType::postProcess(local, length, extraParams);

                        return local;
                    }

                    else {
                        T finalVal = startingVal;
                        BlockInformation info(length, ELEMENT_THRESHOLD);
                        T *blocks = new T[info.threads];

#pragma omp parallel num_threads(info.threads) if (info.threads > 1) proc_bind(AFFINITY) default(shared)
                        {
                            T local = OpType::startingValue(x);
                            for (int i = omp_get_thread_num(); i < info.chunks; i += info.threads) {
                                Nd4jLong newOffset = (i * info.items);
                                const T *chunk = x + newOffset;
                                Nd4jLong itemsToLoop = info.items;
                                if (i * info.items >= length) {
                                    break;
                                }

                                //handle modulo case
                                if (newOffset + info.items >= length) {
                                    itemsToLoop = length - newOffset;
                                }

// FIXME: proper reduction should be used here
                                for (Nd4jLong j = 0; j < itemsToLoop && i * info.items + j < length; j++) {
                                    T curr = OpType::op(chunk[j], extraParams);
                                    local = OpType::update(local, curr, extraParams);
                                }

                            }

                            blocks[omp_get_thread_num()] = local;
                        }

// FIXME: proper reduction should be used here
                        for (int i = 0; i < info.threads; i++) {
                            finalVal = OpType::update(finalVal, blocks[i], extraParams);
                        }


                        finalVal = OpType::postProcess(finalVal, length, extraParams);
                        delete[] blocks;
                        return finalVal;

                    }

                }

                else {
                    if (length < ELEMENT_THRESHOLD) {
                        T local = OpType::startingValue(x);

// FIXME: proper reduction should be used here
                        for (Nd4jLong i = 0; i < length; i++) {
                            T curr = OpType::op(x[i * xElementWiseStride], extraParams);
                            local = OpType::update(local, curr, extraParams);

                        }

                        local = OpType::postProcess(local, length, extraParams);

                        return local;
                    }

                    T finalVal = startingVal;
                    BlockInformation info(length, ELEMENT_THRESHOLD);
                    T *blocks = new T[info.threads];


#pragma omp parallel num_threads(info.threads) if (info.threads > 1) proc_bind(AFFINITY) default(shared)
                    {
                        T local = OpType::startingValue(x);
                        for (int i = omp_get_thread_num(); i < info.chunks; i += info.threads) {
                            Nd4jLong newOffset = (i * info.items) * xElementWiseStride;
                            const T *chunk = x + newOffset;
                            Nd4jLong itemsToLoop = info.items;
                            if (i * info.items >= length)
                                break;

// FIXME: proper reduction should be used here
                            for (Nd4jLong j = 0; j < itemsToLoop && i * info.items + j < length; j++) {
                                T curr = OpType::op(chunk[j * xElementWiseStride], extraParams);
                                local = OpType::update(local, curr, extraParams);
                            }
                        }

                        blocks[omp_get_thread_num()] = local;
                    }

// FIXME: proper reduction should be used here
                    for (int i = 0; i < info.threads; i++) {
                        finalVal = OpType::update(finalVal, blocks[i], extraParams);
                    }

                    finalVal = OpType::postProcess(finalVal, length, extraParams);
                    delete[] blocks;
                    return finalVal;

                }

            }


        template class ND4J_EXPORT ReduceFunction<float>;
        template class ND4J_EXPORT ReduceFunction<float16>;
        template class ND4J_EXPORT ReduceFunction<double>;

        //template void ReduceFunction<float16>::exec<simdOps::LogSumExp<float16>>(float16*, int*, float16*, float16*, int*, int*, int, int*, Nd4jLong*);
        //template void ReduceFunction<float>::exec<simdOps::LogSumExp<float>>(float*, int*, float*, float*, int*, int*, int, int*, Nd4jLong*);
        //template void ReduceFunction<double>::exec<simdOps::LogSumExp<double>>(double*, int*, double*, double*, int*, int*, int, int*, Nd4jLong*);

        BUILD_CALL_1(template void ReduceFunction<float>::exec, float, (float*, Nd4jLong*, float*, float*, Nd4jLong*, int*, int, Nd4jLong*, Nd4jLong*), REDUCE_OPS)
        BUILD_CALL_1(template void ReduceFunction<float16>::exec, float16, (float16*, Nd4jLong*, float16*, float16*, Nd4jLong*, int*, int, Nd4jLong*, Nd4jLong*), REDUCE_OPS)
        BUILD_CALL_1(template void ReduceFunction<double>::exec, double, (double*, Nd4jLong*, double*, double*, Nd4jLong*, int*, int, Nd4jLong*, Nd4jLong*), REDUCE_OPS)

        BUILD_CALL_1(template float ReduceFunction<float>::execScalar, float, (float *x, Nd4jLong *, float*), REDUCE_OPS)
        BUILD_CALL_1(template float16 ReduceFunction<float16>::execScalar, float16, (float16 *x, Nd4jLong *, float16*), REDUCE_OPS)
        BUILD_CALL_1(template double ReduceFunction<double>::execScalar, double, (double *x, Nd4jLong *, double*), REDUCE_OPS)
    }
}