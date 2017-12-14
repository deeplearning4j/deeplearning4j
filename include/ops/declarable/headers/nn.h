//
//  @author raver119@gmail.com
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        /**
         * This is generic SoftMax implementation
         * Expected arguments:
         * 0: 2D array
         */
        DECLARE_OP(softmax, 1, 1, true);
        DECLARE_OP(softmax_bp, 2, 1, true);

        /**
         * Local response normalization implementation.
         * Reference: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
         * Expected arguments:
         * input: 4D array
         * 
         * T args:
         * 0: alpha
         * 1: beta
         * 2: bias
         * 3: depth
         */
        DECLARE_CUSTOM_OP(lrn, 1, 3, true, 4, 0);

        /**
        * Batch normalization implementation. 
        * Reference: https://arxiv.org/abs/1502.03167v3
        * 
        * Expected arguments:
        * input: input array (any number of dimensions)
        * mean:
        * variance:
        * gamma:
        * beta:
        * 
        * Int args:
        * 0: apply scale
        * 1: apply offset
        * 
        * 
        * T args:
        * 0: epsilon
        */
        DECLARE_CUSTOM_OP(batchnorm, 5, 1, false, 1, 2);

        /**
         * This operation updates parameters with provided gradients, wrt learning rate
         * Expected arguments:
         * x: parameters, any shape
         * y: gradients. same shape as x
         * lr: optional, learning rate
         * 
         * T args:
         * 0: optional, learning rate
         */
        DECLARE_CONFIGURABLE_OP(apply_sgd, 2, 1, true, -2, 0);   
    }
}