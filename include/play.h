//
// Created by raver119 on 21.08.16.
//

#ifndef LIBND4J_PLAY_H
#define LIBND4J_PLAY_H

#include <op_boilerplate.h>

#define PAIRWISE_TRANSFORM_OPS \
        (0, PWT::Add),\
        (1, PWT::Copy), \
        (2, PWT::Divide)


#define SCALAR_OPS \
        (12,SCALAR::Min), \
        (13,SCALAR::Copy),\
        (14,SCALAR::Mod)


    DISPATCH_METAOP(transformCuda, PARAMS(N, dx, dy, xStride, yStride, paramsPtr, dz, zStride, nullptr, nullptr, nullptr), OPS_A(SCALAR_OPS), OPS_B(PAIRWISE_TRANSFORM_OPS))



#endif //LIBND4J_PLAY_H
