//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_stop_gradient)

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(stop_gradient, 1, 1, true) {
            NDArray<T>* x = INPUT_VARIABLE(0);
            NDArray<T>* out = OUTPUT_VARIABLE(0);
            // just for lulz
            x->template applyTransform<simdOps::Identity<T>>(out, nullptr);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(StopGradient, stop_gradient);
    }
}

#endif