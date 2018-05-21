//
//  @author  raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/transform.h>
#include <loops/legacy_ops.h>

namespace functions {
    namespace transform {

        template <typename T>
        void Transform<T>::exec(int opNum, T *dx, Nd4jLong xStride, T *result, Nd4jLong resultStride, T *extraParams, const Nd4jLong n) {
            DISPATCH_BY_OPNUM(exec, PARAMS(dx, xStride, result, resultStride, extraParams, n), TRANSFORM_OPS);
		}

        template <typename T>
        void Transform<T>::exec(
				int opNum,
				T *dx,
				Nd4jLong *xShapeInfo,
				T *result,
				Nd4jLong *resultShapeInfo,
				T *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {
                    DISPATCH_BY_OPNUM(exec, PARAMS(dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), TRANSFORM_OPS);
		}

        template <typename T>
        template<typename OpType>
		void _CUDA_H Transform<T>::exec(
                    T *dx,
                    Nd4jLong *xShapeInfo,
                    T *result,
                    Nd4jLong *resultShapeInfo,
                    T *extraParams, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

                if(OpType::requiresSpecial) {
                    OpType::execSpecial(dx,xShapeInfo,result,resultShapeInfo,extraParams, tadShapeInfo, tadOffsets);
                    return;
                }

                auto n = shape::length(xShapeInfo);
                auto xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                auto resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);

                if(xElementWiseStride >= 1 && resultElementWiseStride >= 1 && shape::order(xShapeInfo) == shape::order(resultShapeInfo)) {
                    exec<OpType>(dx,xElementWiseStride,result,resultElementWiseStride,extraParams,n);
                }
                else {
                    Nd4jLong shapeIter[MAX_RANK];
                    Nd4jLong coord[MAX_RANK];
                    int dim;
                    Nd4jLong xStridesIter[MAX_RANK];
                    Nd4jLong resultStridesIter[MAX_RANK];
                    auto xShape = shape::shapeOf(xShapeInfo);
                    auto xStride = shape::stride(xShapeInfo);
                    auto resultStride = shape::stride(resultShapeInfo);
                    int rank = shape::rank(xShapeInfo);
                    if(PrepareTwoRawArrayIter<T>(rank,
                                                 xShape,
                                                 dx,
                                                 xStride,
                                                 result,
                                                 resultStride,
                                                 &rank,
                                                 shapeIter,
                                                 &dx,
                                                 xStridesIter,
                                                 &result,
                                                 resultStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter);
                        {
                            // Process the innermost dimension
                            T *xIter = dx;
                            T *resultIter = result;
                            resultIter[0] = OpType::op(xIter[0], extraParams);
                        }
                        ND4J_RAW_ITER_TWO_NEXT(dim,
                                               rank,
                                               coord,
                                               shapeIter,
                                               dx,
                                               xStridesIter,
                                               result,
                                               resultStridesIter);

                }
            }
        }

        template <typename T>
        template <typename OpType>
		void _CUDA_H Transform<T>::exec(T *dx,
                             Nd4jLong xStride,
                             T *result,
                             Nd4jLong resultStride,
                             T *extraParams,
                             const Nd4jLong n) {

                int elementsPerThread = n / ELEMENT_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                int span = (n / num_threads) + 8;

                if (xStride == 1 && resultStride == 1) {

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        int tid = omp_get_thread_num();
                        int start = span * tid;
                        int end = span * (tid + 1);
                        if (end > n) end = n;

#pragma omp simd
                        for (Nd4jLong i = start; i < end; i++) {
                            result[i] = OpType::op(dx[i], extraParams);
                        }
                    }
                } else {

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        int tid = omp_get_thread_num();
                        int start = span * tid;
                        int end = span * (tid + 1);
                        if (end > n) end = n;

#pragma omp simd
                        for (Nd4jLong i = start; i < end; i++) {
                            result[i*resultStride] = OpType::op(dx[i * xStride], extraParams);
                    }
                }
            }
        }

        template class ND4J_EXPORT Transform<float>;
        template class ND4J_EXPORT Transform<float16>;
        template class ND4J_EXPORT Transform<double>;


        BUILD_CALL_1(template void Transform<float>::exec, float, (float*, Nd4jLong*, float*, Nd4jLong*, float*, Nd4jLong*, Nd4jLong*), TRANSFORM_OPS)
        BUILD_CALL_1(template void Transform<float16>::exec, float16, (float16*, Nd4jLong*, float16*, Nd4jLong*, float16*, Nd4jLong*, Nd4jLong*), TRANSFORM_OPS)
        BUILD_CALL_1(template void Transform<double>::exec, double, (double*, Nd4jLong*, double*, Nd4jLong*, double*, Nd4jLong*, Nd4jLong*), TRANSFORM_OPS)

        BUILD_CALL_1(template void Transform<float>::exec, float, (float*, Nd4jLong, float*, Nd4jLong, float*, const Nd4jLong), TRANSFORM_OPS)
        BUILD_CALL_1(template void Transform<float16>::exec, float16, (float16*, Nd4jLong, float16*, Nd4jLong, float16*, const Nd4jLong), TRANSFORM_OPS)
        BUILD_CALL_1(template void Transform<double>::exec, double, (double*, Nd4jLong, double*, Nd4jLong, double*, const Nd4jLong), TRANSFORM_OPS)
    }
}