//
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_write_list)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(write_list, 2, 1, 0, -2) {
            auto list = INPUT_LIST(0);

            // nd4j mode
            if (block.getIArguments()->size() == 1) {
                auto input = INPUT_VARIABLE(1);
                auto idx = INT_ARG(0);

                Nd4jStatus result = list->write(idx, input->dup());

                auto res = NDArray<T>::scalar(list->counter());
                OVERWRITE_RESULT(res);

                return result;
            } else if (block.width() >= 3) {
                auto input = INPUT_VARIABLE(block.width() - 2);
                auto idx = INPUT_VARIABLE(block.width() - 1);

                REQUIRE_TRUE(idx->isScalar(), 0, "Index should be Scalar");

                Nd4jStatus result = list->write(idx->getScalar(0), input->dup());

                auto res = NDArray<T>::scalar(list->counter());
                OVERWRITE_RESULT(res);

                return result;
            } else return ND4J_STATUS_BAD_INPUT;
        }
        DECLARE_SYN(TensorArrayWriteV3, write_list);
        DECLARE_SYN(tensorarraywritev3, write_list);
    }
}

#endif