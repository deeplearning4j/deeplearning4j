//
// Created by raver119 on 21.08.16.
//

#ifndef LIBND4J_PLAY_H
#define LIBND4J_PLAY_H

#include <op_boilerplate.h>


#define LAUNCH(A, B, C, D) <<<A, B, C, D>>>


#define PAIRWISE_TRANSFORM_OPS \
        (0, PWT::Set),\
        (1, PWT::Copy),\
        (2, PWT::Divide)


#define SCALAR_OPS \
        (0, SCALAR::Add),\
        (1, SCALAR::Subtract),\
        (2, SCALAR::Multiply),\
        (3, SCALAR::Divide),\
        (4, SCALAR::ReverseDivide)

// original version
//    DISPATCH_METAOP(functions::pairwise_transforms::PairWiseTransform<T>::template transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), InvertedMetaOp, OPS_A(SCALAR_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS))


// host-based selector
//DISPATCH_METAOP(invertedMetaOpKernel_Pairwise_Scalar, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), Float, OPS_A(SCALAR_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS));


DISPATCH_KERNEL_META(invertedMetaPairwiseShaped_Pairwise_Scalar_, invertedMetaPairwiseShapedGeneric, float, simdOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB), PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr),  OPS_A(PAIRWISE_TRANSFORM_OPS), OPS_B(SCALAR_OPS))

//_EXPAND_KERNEL_CALL(invertedMetaPairwiseShaped_Pairwise_Scalar_, invertedMetaPairwiseShapedGeneric, float, simdOps::InvertedMetaOp, INPUT(const int opTypeA, const int opTypeB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB), PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), 66, simdOps::SomeOpA, 99, simdOps::SomeOpB)

/*
 extern "C" __global__ void invertedMetaOpKernel_Pairwise_Scalar_16_1_float(const int opTypeA, const int opTypeB, long N, float *dx, int *xShapeInfo, float *dy, int *yShapeInfo, float *dz, int *zShapeInfo, float *extraA, float *extraB, float scalarA, float scalarB) {
    invertedMetaPairwiseShapedGeneric<float, simdOps::InvertedMetaOp<float, simdOps::Copy<float>, simdOps::Multiply<float>>>(opTypeA, opTypeB, N, dx, xShapeInfo, dy, yShapeInfo, dz, zShapeInfo, extraA, extraB, scalarA, scalarB);
 }
 */
#endif //LIBND4J_PLAY_H
