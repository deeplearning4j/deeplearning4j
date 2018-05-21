//
// Created by GS <sgazeos@gmail.com> 31.01.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_l2_loss)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(l2_loss, 1, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(output->isScalar(), 0, "Rank output should be scalar");
            int numZeros = 0;

            T sum = input->template reduceNumber<simdOps::SquaredNorm<T>>();
            sum /= 2;
            (*output)(0) = sum;

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(l2_loss) {
            Nd4jLong *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(0), Nd4jLong);

            shape::shapeScalar(newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif