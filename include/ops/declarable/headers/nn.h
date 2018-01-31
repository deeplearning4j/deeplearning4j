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

        /**
         * This operation performs batch normalization of layer, it is based on following article http://arxiv.org/abs/1502.03167.
         * Expected arguments:
         * x: input 4D array of shape [bS,iH,iW,iD] (data format = NHWC) or [bS,iD,iH,iW] (data format = NCHW), where
         *    bS - batch size 
         *    iH - input height    
         *    iW - input width 
         *    iD - input depth (or number of channels)
         * scale:  1D input array of scale factors, shape [iD]
         * offset: 1D input array of offsets (shifts), shape [iD]
         * mean: 1D input array of population mean used for inference, shape [iD], this array is required only if isTraining = false
         * variance: 1D input array of population mean used for inference, shape [iD], this array is required only if isTraining = false         
         * 
         * T input arguments:
         * 0: epsilon, it is optional argument, default value is 0.001, this is small number to be added to the variance of x
         * 
         * integer input arguments:
         * 0: dataFormat, may have two values: zero -> NHWC, unity -> NCHW
         * 1: isTraining, may have two values: zero -> inference, unity -> training
         */
        DECLARE_CUSTOM_OP(fused_batch_norm, 3, 1, false, 0, 2);
    }
}