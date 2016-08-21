//
// Created by raver119 on 21.08.16.
//

#ifndef LIBND4J_PLAY_H
#define LIBND4J_PLAY_H

#include <op_boilerplate.h>

#define PAIRWISE_TRANSFORM_OPS \
        (0, PWT::Add),\
        (1, PWT::Copy),\
        (2, PWT::Divide),\
        (3, PWT::EqualTo),\
        (4, PWT::GreaterThan), \
        (17,PWT::RandomOp), \
        (18,PWT::FunnyOp), \
        (19,PWT::AstralOp)


#define SCALAR_OPS \
        (12,SCALAR::Min), \
        (13,SCALAR::Copy),\
        (14,SCALAR::Mod),\
        (15,SCALAR::ReverseMod),\
        (16,SCALAR::GreaterThanOrEqual)


//        _EXPAND_META_CALL(transformCuda, PARAMS(N, dx, dy, xStride), 10, 15, simdOps::Divide, simdOps::FunnyStuff)

//        FOR_EACH(WHAT, function, (10,20,30), (SCALAR_OPS))
        //FOR_EACH_META(call, LOL, SCALAR_OPS, PAIRWISE_TRANSFORM_OPS)

    DISPATCH_METAOP(transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), OPS_A(SCALAR_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS))

#endif //LIBND4J_PLAY_H
