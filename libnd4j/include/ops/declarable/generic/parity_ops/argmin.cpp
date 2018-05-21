//
// Created by raver119 on 01.11.2017.
// Modified by GS <sgazeos@gmail.com> 4/5/2018.

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_argmin)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>

namespace nd4j {
    namespace ops {
        REDUCTION_OP_IMPL(argmin, 1, 1, false, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            auto axis = *block.getIArguments();

            // axis might be dynamic (i.e. tf mode)
            if (block.width() > 1 && axis.size() == 0) {
                auto vector = INPUT_VARIABLE(1);
                axis.resize(vector->lengthOf());
                helpers::adjustAxis(input, vector, axis);

                auto shape = ShapeUtils<T>::evalReduceShapeInfo(input->ordering(), axis, *input, false);
                auto output = new NDArray<T>(shape, false, block.getWorkspace());

                input->template applyIndexReduce<simdOps::IndexMin<T>>(output, axis);

                OVERWRITE_RESULT(output);
                RELEASE(shape, input->getWorkspace());
            } else {
                auto output = OUTPUT_VARIABLE(0);

                input->template applyIndexReduce<simdOps::IndexMin<T>>(output, axis);
                STORE_RESULT(output);
            }

            return ND4J_STATUS_OK;
        }
    }
}

#endif