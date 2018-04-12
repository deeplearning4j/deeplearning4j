//
//  @author  raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/transform.h>
#include <loops/legacy_ops.h>

namespace functions {
    namespace transform {

        template <typename T>
        void Transform<T>::exec(int opNum, T *dx, int xStride, T *result, int resultStride, T *extraParams, const int n) {
            DISPATCH_BY_OPNUM(exec, PARAMS(dx, xStride, result, resultStride, extraParams, n), TRANSFORM_OPS);
		}

        template <typename T>
        void Transform<T>::exec(
				int opNum,
				T *dx,
				int *xShapeInfo,
				T *result,
				int *resultShapeInfo,
				T *extraParams, int *tadShapeInfo, Nd4jIndex *tadOffsets) {
                    DISPATCH_BY_OPNUM(exec, PARAMS(dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), TRANSFORM_OPS);
		}

        template <typename T>
        template<typename OpType>
		void _CUDA_H Transform<T>::exec(
                    T *dx,
                    int *xShapeInfo,
                    T *result,
                    int *resultShapeInfo,
                    T *extraParams, int *tadShapeInfo, Nd4jIndex *tadOffsets) {

                if(OpType::requiresSpecial) {
                    OpType::execSpecial(dx,xShapeInfo,result,resultShapeInfo,extraParams, tadShapeInfo, tadOffsets);
                    return;
                }

                int n = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);

                if(xElementWiseStride >= 1 && resultElementWiseStride >= 1 && shape::order(xShapeInfo) == shape::order(resultShapeInfo)) {
                    exec<OpType>(dx,xElementWiseStride,result,resultElementWiseStride,extraParams,n);
                }
                else {
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
                             int xStride,
                             T *result,
                             int resultStride,
                             T *extraParams,
                             const int n) {

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
                        for (Nd4jIndex i = start; i < end; i++) {
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
                        for (Nd4jIndex i = start; i < end; i++) {
                            result[i*resultStride] = OpType::op(dx[i * xStride], extraParams);
                    }
                }
            }
        }

        template class ND4J_EXPORT Transform<float>;
        template class ND4J_EXPORT Transform<float16>;
        template class ND4J_EXPORT Transform<double>;


        BUILD_CALL_1(template void Transform<float>::exec, float, (float*, int*, float*, int*, float*, int*, Nd4jIndex*), TRANSFORM_OPS)
        BUILD_CALL_1(template void Transform<float16>::exec, float16, (float16*, int*, float16*, int*, float16*, int*, Nd4jIndex*), TRANSFORM_OPS)
        BUILD_CALL_1(template void Transform<double>::exec, double, (double*, int*, double*, int*, double*, int*, Nd4jIndex*), TRANSFORM_OPS)

        BUILD_CALL_1(template void Transform<float>::exec, float, (float*, int, float*, int, float*, const int), TRANSFORM_OPS)
        BUILD_CALL_1(template void Transform<float16>::exec, float16, (float16*, int, float16*, int, float16*, const int), TRANSFORM_OPS)
        BUILD_CALL_1(template void Transform<double>::exec, double, (double*, int, double*, int, double*, const int), TRANSFORM_OPS)
    }
}