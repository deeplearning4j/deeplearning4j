//
// Created by raver on 6/30/2018.
//

#ifndef LIBND4J_OMPLAUNCHHELPER_H
#define LIBND4J_OMPLAUNCHHELPER_H

#include <pointercast.h>
#include <op_boilerplate.h>

namespace nd4j {
    class OmpLaunchHelper {
    public:
        static Nd4jLong betterSpan(Nd4jLong N);
        static Nd4jLong betterSpan(Nd4jLong N, Nd4jLong numThreads);
        
        static int betterThreads(Nd4jLong N);
        static int betterThreads(Nd4jLong N, int maxThreads);
    };
}


#endif //LIBND4J_OMPLAUNCHHELPER_H
