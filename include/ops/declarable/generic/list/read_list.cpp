//
// Created by raver119 on 06.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(read_list, 1, 1, 0, 1) {
            auto list = INPUT_LIST(0);

            REQUIRE_TRUE(list->height() > 0, 0, "Number of elements in list should be positive prior to Read call");

            auto index = INT_ARG(0);

            REQUIRE_TRUE(list->isWritten(index), 0, "Requested index [%i] wasn't written yet", index);

            auto result = list->read(index);

            OVERWRITE_RESULT(result);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TensorArrayReadV3, read_list);
        DECLARE_SYN(tensorarrayreadv3, read_list);
    }
}
