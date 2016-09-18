//
// Created by raver119 on 21.08.16.
//

#ifndef LIBND4J_PLAY_H
#define LIBND4J_PLAY_H

#include <op_boilerplate.h>

#define SCALAR_OPS \
        (10, SCALAR::Add),\
        (11, SCALAR::Subtract),\
        (12, SCALAR::Multiply)


#define PAIRWISE_TRANSFORM_OPS \
        (20, PWT::Add),\
        (21, PWT::Subtract),\
        (22, PWT::Multiply), \
        (23, PWT::Random)


/*
#define SCALAR_OPS \
        (0, simdOps::Add),\
        (1, simdOps::Subtract),\
        (2, simdOps::Multiply),\
        (3, simdOps::Divide),\
        (4, simdOps::ReverseDivide),\
        (5, simdOps::ReverseSubtract),\
        (6, simdOps::Max),\
        (7, simdOps::LessThan),\
        (8, simdOps::GreaterThan),\
        (9, simdOps::EqualTo),\
        (10,simdOps::LessThanOrEqual),\
        (11,simdOps::NotEqualTo),\
        (12,simdOps::Min),\
        (13,simdOps::Copy),\
        (14,simdOps::Mod),\
        (15,simdOps::ReverseMod),\
        (16,simdOps::GreaterThanOrEqual)

#define PAIRWISE_TRANSFORM_OPS \
        (0, simdOps::Add),\
        (1, simdOps::Copy),\
        (2, simdOps::Divide),\
        (3, simdOps::EqualTo),\
        (4, simdOps::GreaterThan),\
        (5, simdOps::LessThan),\
        (6, simdOps::Multiply),\
        (7, simdOps::ReverseDivide),\
        (8, simdOps::ReverseSubtract),\
        (9, simdOps::Subtract),\
        (10,simdOps::Epsilon),\
        (11,simdOps::GreaterThanOrEqual),\
        (12,simdOps::LessThanOrEqual),\
        (13,simdOps::Max),\
        (14,simdOps::Min),\
        (15,simdOps::NotEqualTo),\
        (16,simdOps::Copy),\
        (17,simdOps::Axpy),\
        (45,simdOps::CompareAndSet),\
        (46,simdOps::CompareAndReplace)
*/

//DISPATCH_SIMPLE(scalarAlongDimension_, float, PARAMS(x, xShapeInfo, extraParamx, z, zShapeInfo, scalars, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))

//_EXEC_KERNEL_F(scalarAlongDimension_, scalarAlongDimensionGeneric, float, (float inputA, float inputB), (paramA, paramB), (10, SCALAR::Add), (11, SCALAR::Subtract), (12, SCALAR::Multiply))

//DISPATCH_KERNEL_SIMPLE(scalarAlongDimension_, scalarAlongDimensionGeneric, float, INPUT(float inputA, float inputB), PARAMS(paramA, paramB), OPS_A(SCALAR_OPS))

// original version
//    DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(SCALAR_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS))

/*
DISPATCH_METAOP(invertedMetaPairwiseShaped_Pairwise_Scalar, PARAMS(opTypeA, opTypeB, N, x, xShape, y, yShape, z, zShape, extrasA, extrasB, scalarA, scalarB), float, OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS));*/

//DISPATCH_KERNEL_META(invertedMetaPairwiseShaped_Pairwise_Scalar_, invertedMetaPairwiseShapedGeneric, float, simdOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB), PARAMS(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))

//_EXPAND_KERNEL_CALL(invertedMetaPairwiseShaped_Pairwise_Scalar_, invertedMetaPairwiseShapedGeneric, float, simdOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB), PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), 66, simdOps::SomeOpA, 99, simdOps::SomeOpB)

/*
 extern "C" __global__ void invertedMetaOpKernel_Pairwise_Scalar_16_1_float(const int opTypeA, const int opTypeB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
    invertedMetaPairwiseShapedGeneric<float, simdOps::InvertedMetaOp<float, simdOps::Copy<float>, simdOps::Multiply<float>>>(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);
 }
 */
#endif //LIBND4J_PLAY_H
