//
// Created by raver119 on 06.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_read_list)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(read_list, 1, 1, 0, 0) {
            auto list = INPUT_LIST(0);
            NDArray<T> * result = nullptr;

            REQUIRE_TRUE(list->height() > 0, 0, "ReadList: number of elements in list should be positive prior to Read call");

            if (block.getIArguments()->size() > 0) {
                auto index = INT_ARG(0);

                REQUIRE_TRUE(list->isWritten(index), 0, "ReadList: requested index [%i] wasn't written yet", index);

                result = list->read(index);
            } else if (block.width() > 0) {
                auto vec = INPUT_VARIABLE(1);

                REQUIRE_TRUE(vec->isScalar(), 0, "ReadList: index operand should be a scalar");
                
                auto index = (int) vec->getScalar(0);

                REQUIRE_TRUE(list->isWritten(index), 0, "ReadList: requested index [%i] wasn't written yet", index);

                result = list->read(index);
            } else {
                REQUIRE_TRUE(false, 0, "ReadList: index value should be set either via IntArgs or via second operand");
            }

            OVERWRITE_RESULT(result);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TensorArrayReadV3, read_list);
        DECLARE_SYN(tensorarrayreadv3, read_list);
    }
}

#endif