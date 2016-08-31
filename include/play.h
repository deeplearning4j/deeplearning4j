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



#endif //LIBND4J_PLAY_H
