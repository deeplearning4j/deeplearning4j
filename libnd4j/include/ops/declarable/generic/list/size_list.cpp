//
// Created by raver119 on 06.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_size_list)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(size_list, 1, 1, 0, 0) {
            auto list = INPUT_LIST(0);

            auto result = NDArrayFactory<T>::scalar((T) list->height());

            OVERWRITE_RESULT(result);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TensorArraySizeV3, size_list);
        DECLARE_SYN(tensorarraysizev3, size_list);
    }
}

#endif