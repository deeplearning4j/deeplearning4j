//
// Created by raver119 on 01.11.2017.
// Modified by GS <sgazeos@gmail.com> 4/5/2018

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_argmax)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>
namespace nd4j {
    namespace ops {
        REDUCTION_OP_IMPL(argmax, 1, 1, false, 0, -2) {
            NDArray<T>* input = INPUT_VARIABLE(0);

            auto axis = *block.getIArguments();

            // axis might be dynamic (i.e. tf mode)
            if (block.width() > 1 && axis.size() == 0) {
                NDArray<T>* axisVector = INPUT_VARIABLE(1);
                axis.resize(axisVector->lengthOf());
                helpers::adjustAxis(input, axisVector, axis);

                auto shape = ShapeUtils<T>::evalReduceShapeInfo(input->ordering(), axis, *input, false, true);
                auto output = new NDArray<T>(shape, false, block.getWorkspace());
                
                input->template applyIndexReduce<simdOps::IndexMax<T>>(output, axis);

                OVERWRITE_RESULT(output);
                RELEASE(shape, input->getWorkspace());
            } else {
                auto output = OUTPUT_VARIABLE(0);

                input->template applyIndexReduce<simdOps::IndexMax<T>>(output, axis);
                STORE_RESULT(output);
            }

            return ND4J_STATUS_OK;
        }
    }
}

#endif