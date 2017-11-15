//
// This class provides ops execution for CUDA
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_CUDA_LAUNCHER_H
#define LIBND4J_CUDA_LAUNCHER_H

#include "ops/declarable/DeclarableOp.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <pointercast.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        class CudaLauncherOp : public DeclarableOp {
        public:
            CudaLauncherOp(int nIn, int nOut, const char *name, bool inplaceable, int numTargs, int numIargs) : nd4j::ops::DeclarableOp<T>(nIn, nOut, name, inplaceable, numTargs, numIargs) { };
        protected:
            Nd4jStatus validateAndExecute(Context<T>& block);
        };

        template <typename T>
        Nd4jStatus nd4j::ops::NAME<T>::validateAndExecute(Context<T>& block) {
            cudaStream_t* stream = block->_stream;

            // step 1: validate whatever can be validated

            // step 2: push pointers to gpu

            // step 3: execute kernel

            // step 4: synchronize to ensure we can continue

            cudaError_t res = cudaStreamSynchronize(*stream);

            if (res != 0)
                return ND4J_STATUS_KERNEL_FAILURE;
        }
    }
}

#endif //LIBND4J_LAUNCHER_H
