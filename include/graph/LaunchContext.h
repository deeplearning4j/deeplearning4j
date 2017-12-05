//
// Created by raver119 on 30.11.17.
//

#ifndef LIBND4J_CUDACONTEXT_H
#define LIBND4J_CUDACONTEXT_H

namespace nd4j {
    namespace graph {
        class LaunchContext {
        public:
            LaunchContext();
            ~LaunchContext() = default;
        };
    }
}


#endif //LIBND4J_CUDACONTEXT_H
