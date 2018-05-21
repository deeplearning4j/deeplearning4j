//
// Created by raver119 on 24.11.17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_scatter_div)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/ScatterHelper.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(scatter_div, 3, 1, true) {
            auto input = INPUT_VARIABLE(0);
            auto indices = INPUT_VARIABLE(1);
            auto updates = INPUT_VARIABLE(2);

            auto output = OUTPUT_VARIABLE(0);

            if (!block.isInplace())
                output->assign(input);

            ScatterHelper<T>::template scatter_apply<simdOps::Divide<T>>(output, indices, updates);        

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(ScatterDiv, scatter_div);
    }
}

#endif