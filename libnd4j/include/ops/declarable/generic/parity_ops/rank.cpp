//
// Created by raver119 on 01.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_rank)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(rank, 1, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(output->isScalar(), 0, "Rank output should be scalar");

            output->putScalar(0, (T) input->rankOf());

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(rank) {
            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(0), Nd4jLong);

            shape::shapeScalar(newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif