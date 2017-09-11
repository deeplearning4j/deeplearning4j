//
// @author raver119@gmail.com
//

#ifndef LIBND4J_CUDA_PARITY_OPS_H_H
#define LIBND4J_CUDA_PARITY_OPS_H_H

#include <op_boilerplate.h>
#include <memory>
#include <shape.h>
#include <ops/ops.h>
#include <loops/random.h>
#include <NDArray.h>
#include <ops/declarable/declarable_ops.h>
#include <NDArrayFactory.h>

namespace nd4j {
    namespace ops {

        // test replacement op, of second type.
        // op should be executed in device context
        DECLARE_DEVICE_OP(testop2i2o, 2, 2, true, 0, 0) {
            nd4j_printf("GPU op used!","");

            return ND4J_STATUS_OK;
        }

    }
}

#endif //LIBND4J_CUDA_PARITY_OPS_H_H
