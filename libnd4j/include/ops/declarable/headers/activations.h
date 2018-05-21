//
// @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_ACTIVATIONS_H
#define LIBND4J_HEADERS_ACTIVATIONS_H


#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This is Sigmoid activation function implementation
         * Math is: 1 / 1 + exp(-x)
         */
        #if NOT_EXCLUDED(OP_sigmoid)
        DECLARE_CONFIGURABLE_OP(sigmoid, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(sigmoid_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is Softsign activation function implementation
         * Math is: x / 1 + abs(x)
         */
        #if NOT_EXCLUDED(OP_softsign)
        DECLARE_CONFIGURABLE_OP(softsign, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(softsign_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is Tanh activation function implementation
         */
        #if NOT_EXCLUDED(OP_tanh)
        DECLARE_CONFIGURABLE_OP(tanh, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(tanh_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is Softplus activation function implementation
         * Math is: log(1 + exp(x))
         */
        #if NOT_EXCLUDED(OP_softplus)
        DECLARE_CONFIGURABLE_OP(softplus, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(softplus_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is RELU activation function implementation
         */
        #if NOT_EXCLUDED(OP_relu)
        DECLARE_CONFIGURABLE_OP(relu, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(relu_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is SELU activation function implementation
         */
        #if NOT_EXCLUDED(OP_selu)
        DECLARE_CONFIGURABLE_OP(selu, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(selu_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is Leaky RELU activation function.
         * Math is: x < 0 ?  alpha * x : x;
         */
        #if NOT_EXCLUDED(OP_lrelu)
        DECLARE_CONFIGURABLE_OP(lrelu, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(lrelu_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This op is ELU activation function.
         * Math is: x >= 0 ? x : exp(x) - 1;
         */
        #if NOT_EXCLUDED(OP_elu)
        DECLARE_CONFIGURABLE_OP(elu, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(elu_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is Cube activation function.
         * Math is: x^3
         */
        #if NOT_EXCLUDED(OP_cube)
        DECLARE_CONFIGURABLE_OP(cube, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(cube_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is RectifiedTanh activation function.
         * Math is: max(0, tanh(x))
         */
        #if NOT_EXCLUDED(OP_rectifiedtanh)
        DECLARE_CONFIGURABLE_OP(rectifiedtanh, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(rectifiedtanh_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is RationalTanh activation function.
         */
        #if NOT_EXCLUDED(OP_rationaltanh)
        DECLARE_CONFIGURABLE_OP(rationaltanh, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(rationaltanh_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is HardTanh activation function.
         * Math is: x < -1.0 ? -1.0 : x > 1.0 ? 1.0 : x;
         */
        #if NOT_EXCLUDED(OP_hardtanh)
        DECLARE_CONFIGURABLE_OP(hardtanh, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(hardtanh_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is HardSigmoid activation function.
         * Math is: min(1, max(0, 0.2 * x + 0.5))
         */
        #if NOT_EXCLUDED(OP_hardsigmoid)
        DECLARE_CONFIGURABLE_OP(hardsigmoid, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(hardsigmoid_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * This is Indentity operation. It passes signal umodified in both directions.
         */
        #if NOT_EXCLUDED(OP_identity)
        DECLARE_OP(identity, 1, 1, true);
        DECLARE_OP(identity_bp, 2, 1, true);
        #endif

        /**
         * This is Concatenated RELU implementation.
         * What happens inside: RELU(Concat((x, -x, {-1})))
         * 
         * PLEASE NOTE: Concatenation will double amount of features available in input
         */
        #if NOT_EXCLUDED(OP_crelu)
        DECLARE_CUSTOM_OP(crelu, 1, 1, false, 0, 0);
        DECLARE_CUSTOM_OP(crelu_bp, 2, 1, false, 0, 0);
        #endif

        /**
         * This is RELU6 activation function implementation
         */
        #if NOT_EXCLUDED(OP_relu6)
        DECLARE_CONFIGURABLE_OP(relu6, 1, 1, true, 1, 0);
        DECLARE_CONFIGURABLE_OP(relu6_bp, 2, 1, true, 0, 0);
        #endif

        /**
         * relu_layer = relu(x*w + b)
         */
        DECLARE_CUSTOM_OP(relu_layer, 3, 1, false, 0, 0);
    }
}

#endif