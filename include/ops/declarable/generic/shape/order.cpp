//
// Created by raver119 on 12.02.18.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_order)

#include <ops/declarable/headers/shape.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(order, 1, 1, false, 0, 1) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            output->assign(input);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(order) {
            auto input = inputShape->at(0);
            Nd4jLong *newShape;

            auto isFOrder = INT_ARG(0) == 1;

            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(input), Nd4jLong
            );

            if (isFOrder)
                shape::shapeBufferFortran(shape::rank(input), shape::shapeOf(input), newShape);
            else
                shape::shapeBuffer(shape::rank(input), shape::shapeOf(input), newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif