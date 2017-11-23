//
// Created by raver119 on 23.11.17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(squaredsubtract, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = OUTPUT_VARIABLE(0);

            if (!x->isScalar() && !y->isScalar() && x->lengthOf() == y->lengthOf()) {
                REQUIRE_OK(this->validateInputLengthMatch(block));
                x->template applyPairwiseTransform<simdOps::SquaredSubtract<T>>(y, z, nullptr);

            } else if (!x->isScalar() && y->isScalar()) {
                x->template applyScalar<simdOps::SquaredSubtract<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::SquaredSubtract<T>>(*x, z);
            }
            else if (x->isScalar() && y->isScalar()) { // x->isScalar() && y->isScalar()
                z->putScalar(0, nd4j::math::nd4j_pow(x->getScalar(0) - y->getScalar(0), (T) 2));
            } else {
                auto tZ = x->template applyTrueBroadcast<simdOps::SquaredSubtract<T>>(y);
                OVERWRITE_RESULT(tZ);
            }

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(squareddifference, squaredsubtract);
    }
}