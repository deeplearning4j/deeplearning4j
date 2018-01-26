//
// @author raver119@gmail.com
//
#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This is Sigmoid activation function implementation
         * Math is: 1 / 1 + exp(-x)
         */
        DECLARE_CONFIGURABLE_OP(sigmoid, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(sigmoid_bp, 2, 1, true, 0, 0);

        /**
         * This is Softsign activation function implementation
         * Math is: x / 1 + abs(x)
         */
        DECLARE_CONFIGURABLE_OP(softsign, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(softsign_bp, 2, 1, true, 0, 0);

        /**
         * This is Tanh activation function implementation
         */
        DECLARE_CONFIGURABLE_OP(tanh, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(tanh_bp, 2, 1, true, 0, 0);

        /**
         * This is Softplus activation function implementation
         * Math is: log(1 + exp(x))
         */
        DECLARE_CONFIGURABLE_OP(softplus, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(softplus_bp, 2, 1, true, 0, 0);

        /**
         * This is RELU activation function implementation
         */
        DECLARE_CONFIGURABLE_OP(relu, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(relu_bp, 2, 1, true, 0, 0);

        /**
         * This is SELU activation function implementation
         */
        DECLARE_CONFIGURABLE_OP(selu, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(selu_bp, 2, 1, true, 0, 0);

        /**
         * This is Leaky RELU activation function.
         * Math is: x < 0 ?  alpha * x : x;
         */
        DECLARE_CONFIGURABLE_OP(lrelu, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(lrelu_bp, 2, 1, true, 0, 0);

        /**
         * This op is ELU activation function.
         * Math is: x >= 0 ? x : exp(x) - 1;
         */
        DECLARE_CONFIGURABLE_OP(elu, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(elu_bp, 2, 1, true, 0, 0);

        /**
         * This is Cube activation function.
         * Math is: x^3
         */
        DECLARE_CONFIGURABLE_OP(cube, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(cube_bp, 2, 1, true, 0, 0);

        /**
         * This is RectifiedTanh activation function.
         * Math is: max(0, tanh(x))
         */
        DECLARE_CONFIGURABLE_OP(rectifiedtanh, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(rectifiedtanh_bp, 2, 1, true, 0, 0);

        /**
         * This is RationalTanh activation function.
         */
        DECLARE_CONFIGURABLE_OP(rationaltanh, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(rationaltanh_bp, 2, 1, true, 0, 0);

        /**
         * This is HardTanh activation function.
         * Math is: x < -1.0 ? -1.0 : x > 1.0 ? 1.0 : x;
         */
        DECLARE_CONFIGURABLE_OP(hardtanh, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(hardtanh_bp, 2, 1, true, 0, 0);

        /**
         * This is HardSigmoid activation function.
         * Math is: min(1, max(0, 0.2 * x + 0.5))
         */
        DECLARE_CONFIGURABLE_OP(hardsigmoid, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(hardsigmoid_bp, 2, 1, true, 0, 0);

        /**
         * This is Indentity operation. It passes signal umodified in both directions.
         */
        DECLARE_OP(identity, 1, 1, true);
        DECLARE_OP(identity_bp, 2, 1, true);

        /**
         * This is Concatenated RELU implementation.
         * What happens inside: RELU(Concat((x, -x, {-1})))
         * 
         * PLEASE NOTE: Concatenation will double amount of features available in input
         */
        DECLARE_CUSTOM_OP(crelu, 1, 1, false, 0, 0);        
        DECLARE_CUSTOM_OP(crelu_bp, 2, 1, false, 0, 0);
        
        /**
         * x^2 + y operation
         */
        DECLARE_OP(polisq, 2, 1, true);
    }
}