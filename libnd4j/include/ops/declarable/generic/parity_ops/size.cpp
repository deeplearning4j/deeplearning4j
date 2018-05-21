//
// Created by raver119 on 01.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_size)

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
            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(0), Nd4jLong);

            shape::shapeScalar(newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif