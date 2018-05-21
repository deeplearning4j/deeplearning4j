//
// Created by raver on 4/5/2018.
//

#ifndef LIBND4J_CUDALAUNCHHELPER_H
#define LIBND4J_CUDALAUNCHHELPER_H


#include <pointercast.h>
#include <dll.h>
#include <op_boilerplate.h>
#include <types/triple.h>

namespace nd4j {
    class CudaLaunchHelper {
    public:
        static Triple getFlatLaunchParams(Nd4jLong length, int SM, int CORES, int SHARED_MEMORY);

    };
}


#endif //LIBND4J_CUDALAUNCHHELPER_H
