//
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_clipbyvalue)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(clipbyvalue, 1, 1, true, 2, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* output = this->getZ(block);

            input->template applyTransform<simdOps::ClipByValue<T>>(output, block.getTArguments()->data());

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(ClipByValue, clipbyvalue);
    }
}

#endif