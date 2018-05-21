//
// Created by raver119 on 24.02.18.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_test_scalar)

#include <ops/declarable/headers/tests.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(test_scalar, 1, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            T val = input->getIndexedScalar(0) + 2.0;
            output->putIndexedScalar(0, val);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(test_scalar) {
            Nd4jLong *newShape;
            ALLOCATE(newShape, block.workspace(), shape::shapeInfoLength(2), Nd4jLong);

            newShape[0] = 2;
            newShape[1] = 1;
            newShape[2] = 1;
            newShape[3] = 1;
            newShape[4] = 1;
            newShape[5] = 0;
            newShape[6] = 1;
            newShape[7] = 99;


            return SHAPELIST(newShape);
        }
    }
}

#endif