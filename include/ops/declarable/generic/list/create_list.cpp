//
// Created by raver119 on 06.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(create_list, 1, 2, 0, -2) {
            int height = 0;
            bool expandable = false;
            if (block.getIArguments()->size() == 1) {
                height = INT_ARG(0);
                expandable = (bool) INT_ARG(1);
            } else if (block.getIArguments()->size() == 2) {
                height = INT_ARG(0);
            } else {
                height = 0;
                expandable = true;
            }

            auto list = new NDArrayList<T>(height, expandable);

            // we recieve input array for graph integrity purposes only
            auto input = INPUT_VARIABLE(0);

            block.getVariableSpace()->trackList(list);

            OVERWRITE_RESULT(list);

            if (!block.getVariableSpace()->hasVariable(block.nodeId(), 1)) {
                block.getVariableSpace()->putVariable(block.nodeId(), 1, new Variable<T>(nullptr, nullptr, block.nodeId(), 1));
            }

            auto var = block.getVariableSpace()->getVariable(block.getNodeId(), 1);
            auto scalar = NDArrayFactory<T>::scalar(list->counter());
            var->setNDArray(scalar);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TensorArrayV3, create_list);
        DECLARE_SYN(tensorarrayv3, create_list);
        DECLARE_SYN(TensorArrayCreateV3, create_list);
        DECLARE_SYN(tensorarraycreatev3, create_list);
    }
}
