/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver119 on 21.08.16.
//

#ifndef LIBND4J_PLAY_H
#define LIBND4J_PLAY_H

#include <enum_boilerplate.h>

#define DATA_TYPES \
        (DATA_FLOAT, float) ,\
        (DATA_DOUBLE, double) ,\
        (DATA_HALF, float16)

BUILD_ENUMERATION(DATA_TYPES)

//BUILD_SINGLE_SELECTOR(xType, functions::IndexReduce, ::op(a, b, c, d, e), DATA_TYPES)
//BUILD_DOUBLE_SELECTOR(xType, yType, functions::IndexReduce, ::op(a, b, c, d, e), DATA_TYPES, DATA_TYPES)

//BUILD_SINGLE_TEMPLATE(template class Alpha, (signature), DATA_TYPES);

//BUILD_DOUBLE_TEMPLATE(template class Alpha, (signature) , DATA_TYPES, DATA_TYPES);

/*
#define SCALAR_OPS \
        (0, simdOps::Identity) ,\
        (1, simdOps::ReLU)
*/
/*
#define NATIVE_LAYERS \
        (0, nd4j::layers::DenseLayer)
//        (1, nd4j::layers::ConvolutionLayer) ,\
//        (2, nd4j::layers::Pooling2DLayer) ,\
//        (3, nd4j::layers::LSTMLayer)


*/
/*
#define PAIRWISE_TRANSFORM_OPS \
        (0, simdOps::Add),\
        (1, simdOps::Copy),\
        (2, simdOps::Divide),\
        (3, simdOps::EqualTo),\
        (4, simdOps::GreaterThan),\
        (5, simdOps::LessThan),\
        (6, simdOps::Multiply),\
        (7, simdOps::Pow),\
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
        (18,simdOps::ReverseDivide),\
        (45,simdOps::CompareAndSet),\
        (46,simdOps::CompareAndReplace),\
        (56,simdOps::And),\
        (57,simdOps::Or),\
        (58,simdOps::Xor),\
        (59,simdOps::Remainder),\
        (60,simdOps::FMod),\
        (69,simdOps::Atan2)

EXECUTE_NOE((x, y, extras), OPS_A(PAIRWISE_TRANSFORM_OPS))
*/


//EXECUTE_NOE((x, extras), OPS_A(SCALAR_OPS))

//BUILD_CALL_1(template void nd4j::NDArray<float16>::applyTransform, float16, (NDArray<float16>* a, float16* b), TRANSFORM_OPS)

//BUILD_CALL_1(template void nd4j::NDArray<float16>::applyPairwiseTransform, float16, (NDArray<float16>* other, float16* extraParams), PAIRWISE_TRANSFORM_OPS)
//BUILD_TRACKER(TRANSFORM, ACTIVATIONS)

//BUILD_CALL_1(template void nd4j::NDArray<float16>::applyScalar, float16, (float16 scalar, NDArray<float16>* target, float16 *extraParams) , ACTIVATIONS);

/*
#define DECLARE_OP(NAME, NIN, NOUT)   DECLARE_OP_UNIQ(__COUNTER__, NAME, NIN, NOUT)
#define DECLARE_OP_UNIQ(CTR, NAME, NIN, NOUT)   template <typename T> \
                                                class NAME: public nd4j::ops::DeclarableOp<T> { \
                                                public:\
                                                NAME() : nd4j::ops::DeclarableOp<T>(NIN, NOUT, #NAME) { } \
                                                protected: \
                                                    Nd4jStatus validateAndExecute(Block<T>& block); \
                                                };\
                                                template <typename T> \
                                                Nd4jStatus nd4j::ops::NAME<T>::validateAndExecute(Block<T>& block)
*/
//#define END_OP(NAME) }; static nd4j::ops::__registrator<NAME<float>> register_op##Name;

//#DECLARE_OP(Concat, -1, 1)

//END_OP(Concat)


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
