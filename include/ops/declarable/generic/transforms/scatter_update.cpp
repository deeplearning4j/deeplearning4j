//
// Created by raver119 on 24.11.17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_scatter_update)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
    namespace ops {
        /**
         * scatter update operation
         *
         * IArgs map:
         * IArgs[0] - update operation: 0 - add; 1 - sub; 2 - mul; 3 - div; 4 - rsub; 5 - rdiv; 6 - assign
         * IArgs[1] - number of dimensions
         * IArgs[...] - dimensions
         * IArgs[...] - number of indices
         * IArgs[...] - indices
         *
         * @tparam T
         */
        CONFIGURABLE_OP_IMPL(scatter_update, 2, 1, true, 0, -1) {
            
            NDArray<T> *operand = INPUT_VARIABLE(0);
            NDArray<T> *updates = INPUT_VARIABLE(1);
            
            helpers::scatterUpdate(*operand, *updates, block.getIArguments());
            
            return Status::OK();
        }
        DECLARE_SYN(scatterupdate, scatter_update);
    }
}

#endif