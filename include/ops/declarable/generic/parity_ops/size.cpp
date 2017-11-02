//
// Created by raver119 on 01.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(size, 1, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(output->isScalar(), 0, "Size output should be scalar");

            output->putScalar(0, (T) input->lengthOf());

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(size) {
            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);

            newShape[0] = 2;
            newShape[1] = 1;
            newShape[2] = 1;
            newShape[3] = 1;
            newShape[4] = 1;
            newShape[5] = 0;
            newShape[6] = 1;
            newShape[7] = 99;

            return new ShapeList(newShape);
        }
    }
}