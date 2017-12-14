//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(softmax, 1, 1, true) {
            // YaY
            auto input = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            FIXME: 
            input->template applyTransform<simdOps::SoftMax<T>>(z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        OP_IMPL(softmax_bp, 2, 1, true) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* epsInput = INPUT_VARIABLE(1);

            NDArray<T>* z = OUTPUT_VARIABLE(0);
            /*
                INDArray out = Nd4j.getExecutioner().execAndReturn(new SoftMax(in));

                INDArray x = out.mul(epsilon).sum(1);
                INDArray dLdz = out.mul(epsilon.subColumnVector(x));
            */

            auto tmp_ = new NDArray<T>(input);
            input->template applyTransform<simdOps::SoftMax<T>>(z, nullptr);
            z->template applyPairwiseTransform<simdOps::Multiply<T>>(epsInput, tmp_, nullptr);

            auto sum = tmp_->template reduceAlongDimension<simdOps::Sum<T>>({1});

            tmp_->assign(epsInput);
            tmp_->template applyBroadcast<simdOps::Subtract<T>>({0}, sum);

            z->template applyPairwiseTransform<simdOps::Multiply<T>>(tmp_, z, nullptr);

            STORE_RESULT(*z);

            delete sum;
            delete tmp_;

            return ND4J_STATUS_OK;
        }
    }
}