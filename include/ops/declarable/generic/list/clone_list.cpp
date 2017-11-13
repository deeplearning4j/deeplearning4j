//
// @author raver119@gmail.com
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(clone_list, 1, 1, 0, 0) {
            auto list = INPUT_LIST(0);

            auto newList = list->clone();

            block.getVariableSpace()->trackList(newList);

            OVERWRITE_RESULT(newList);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TensorArrayIdentityV3, clone_list);
        DECLARE_SYN(tensorarrayidentityv3, clone_list);
    }
}