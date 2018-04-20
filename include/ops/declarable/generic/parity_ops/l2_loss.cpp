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
//            for (int e = 0; e < input->lengthOf(); e++)
//                if ((*input)(e) == T(0))
//                    numZeros++;
            T sum = input->template reduceNumber<simdOps::SquaredNorm<T>>();//((T)(0) + numZeros) / input->lengthOf();
            sum /= 2;
            output->putScalar(0, sum);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(l2_loss) {
            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(0), int);

            shape::shapeScalar(newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif