//
// Created by raver119 on 21.08.16.
//

#ifndef LIBND4J_PLAY_H
#define LIBND4J_PLAY_H

//#include <op_boilerplate.h>

#define ACTIVATIONS \
        (0, nd4j::activations::Identity) ,\
        (1, nd4j::activations::ReLU)

#define NATIVE_LAYERS \
        (0, nd4j::layers::DenseLayer)
//        (1, nd4j::layers::ConvolutionLayer) ,\
//        (2, nd4j::layers::Pooling2DLayer) ,\
//        (3, nd4j::layers::LSTMLayer)

#define DATA_TYPES \
        (0, float) ,\
        (1, double) ,\
        (2, float16)


#define DECLARE_OP(NAME, NIN, NOUT)   template <typename T> \
                                      class NAME: public nd4j::ops::DeclarableOp<T> { \
                                      public:\
                                      NAME() : nd4j::ops::DeclarableOp<T>(-1, 1, #NAME) { } \
                                      protected:


//DECLARE_OP("Concat", -1, 1)


//BUILD_LAYERS_FACTORY(float, OPS_A(NATIVE_LAYERS), OPS_B(ACTIVATIONS))


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
