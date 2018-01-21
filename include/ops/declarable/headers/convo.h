//
//
//

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {

        /**
         * 1D temporal convolution implementation
         * Expected input: 
         * x: 3D array
         * weight: 3D Array
         * bias: optional vector
         * 
         * Int args:
         * 0: kernel
         * 1: stride
         * 2: padding
         */
        DECLARE_CUSTOM_OP(conv1d, 2, 1, false, 0, 3);
        DECLARE_CUSTOM_OP(conv1d_bp, 3, 2, false, 0, 3);

        /**
         * 2D convolution implementation
         * Expected input: 
         * x: 4D array
         * weight: 4D Array
         * bias: optional vector, length of outputChannels
         * 
         * IntArgs:
         * 0: kernel height
         * 1: kernel width
         * 2: stride height
         * 3: stride width
         * 4: padding height
         * 5: padding width
         * 6: dilation height
         * 7: dilation width
         * 8: same mode: 0 false, 1 true
         */
        DECLARE_CUSTOM_OP(conv2d, 2, 1, false, 0, 3);
        DECLARE_CUSTOM_OP(conv2d_bp, 3, 2, false, 0, 9);

        /**
         * Depthwise convolution2d op:
         * Expected inputs:
         * x: 4D array, NCHW format
         * weightsDepth: 4D array,
         * weightsPointwise: optional, 4D array
         * bias: optional, vector
         */
        DECLARE_CUSTOM_OP(sconv2d, 2, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(sconv2d_bp, 4, 2, false, 0, 9);

        /**
         * 2D deconvolution implementation
         * 
         * IntArgs:
         * 0: kernel height
         * 1: kernel width
         * 2: stride height
         * 3: stride width
         * 4: padding height
         * 5: padding width
         * 6: dilation height
         * 7: dilation width
         * 8: same mode: 0 false, 1 true
         */
        DECLARE_CUSTOM_OP(deconv2d, 2, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(deconv2d_bp, 4, 2, false, 0, 9);

        /**
         * This op implements max pooling for convolution networks.
         * Expected Input: 4D array, NCHW format.
         *
         * IntArgs:
         * 0: kernel height
         * 1: kernel width
         * 2: stride height
         * 3: stride width
         * 4: padding height
         * 5: padding width
         * 6: dilation height
         * 7: dilation width
         * 8: same mode: 0 false, 1 true
         */
        DECLARE_CUSTOM_OP(maxpool2d, 1, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(maxpool2d_bp, 2, 1, false, 0, 9);

        /**
         * This op implements average pooling for convolution networks.
         * Expected Input: 4D array, NCHW format.
         *
         * IntArgs:
         * 0: kernel height
         * 1: kernel width
         * 2: stride height
         * 3: stride width
         * 4: padding height
         * 5: padding width
         * 6: dilation height
         * 7: dilation width
         * 8: same mode: 0 false, 1 true
         */
        DECLARE_CUSTOM_OP(avgpool2d, 1, 1, false, 0, 9);
        DECLARE_CUSTOM_OP(avgpool2d_bp, 2, 1, false, 0, 9);

        /**
         * This op implements pnorm pooling for convolution networks.
         * Expected Input: 4D array, NCHW format.
         *
         * IntArgs:
         * 0: kernel height
         * 1: kernel width
         * 2: stride height
         * 3: stride width
         * 4: padding height
         * 5: padding width
         * 6: dilation height
         * 7: dilation width
         * 8: same mode: 0 false, 1 true
         * 9: p for p-norm
         */
        DECLARE_CUSTOM_OP(pnormpool2d, 1, 1, false, 0, 10);
        DECLARE_CUSTOM_OP(pnormpool2d_bp, 2, 1, false, 1, 10);


        DECLARE_CUSTOM_OP(maxpool3d, 1, 2, true, 0, 13); 
        DECLARE_CUSTOM_OP(maxpool3d_bp, 3, 1, true, 0, 13);
        
        
        DECLARE_CUSTOM_OP(avgpool3d, 1, 1, true, 0, 11);
        DECLARE_CUSTOM_OP(avgpool3d_bp, 2, 1, true, 0, 11);
        
        
        DECLARE_CUSTOM_OP(fullconv3d, 5, 1, false, 0, 13);
        DECLARE_CUSTOM_OP(fullconv3d_bp, 5, 1, false, 0, 13);
        DECLARE_CUSTOM_OP(fullconv3d_grad, 4, 2, false, 1, 13);
        
        /**
         *  Universal pooling op, combines max/avg/pnorm pooling.
         *  Shouldn't be used directly, consider using corresponding operations instead.
         */
        DECLARE_CUSTOM_OP(pooling2d, 1, 1, false, 0, 11);
        
        /**
         * This op implements im2col algorithm, widely used in convolution neural networks
         * Input: 4D input expected
         * 
         * Int args:
         * 0: kernel height
         * 1: kernel width
         * 2: stride height
         * 3: stride width
         * 4: padding height
         * 5: padding width
         * 6: dilation height
         * 7: dilation width
         * 8: isSameMode
         */
        DECLARE_CUSTOM_OP(im2col, 1, 1, false, 0, 9);

        /**
         * This op implements col2im algorithm, widely used in convolution neural networks
         * Input: 6D input expected (like output of im2col op)
         * 
         * Int args:
         * 0: stride height
         * 1: stride width
         * 2: padding height
         * 3: padding width
         * 4: image height
         * 5: image width
         * 6: dilation height
         * 7: dilation width
         */
        DECLARE_CUSTOM_OP(col2im, 1, 1, false, 0, 9);
        
        /**
         * Upsampling implementation, based on pytorch
         *
         * IArgs map:
         * IArgs[0] - scale factor
         */
        DECLARE_CUSTOM_OP(upsampling2d, 1, 1, false, 0, 1);
        DECLARE_CUSTOM_OP(upsampling2d_bp, 2, 1, false, 0, 1);

        /**
         * 3D convolution implementation
         * 
         * IntArgs:
         * 0: dilation T
         * 1: dilation W
         * 2: dilation H
         * 4: padding T
         * 5: padding W
         * 6: padding H
         */
        DECLARE_CUSTOM_OP(conv3d, 2, 1, false, 0, 7); 
        DECLARE_CONFIGURABLE_OP(conv3d_bp, 3, 1, false, 0, 7); // TODO: to be implemented        

        /**
         * This op produces binary matrix wrt to target dimension.
         * Maximum value within each TAD is replaced with 1, other values are set to true.
         * 
         * Int args:
         * 0: axis
         */
        DECLARE_CONFIGURABLE_OP(ismax, 1, 1, true, 0, -1);

        /**
         * Dilation2D op
         * 
         * Int args:
         * 0: isSameMode
         */
        DECLARE_CUSTOM_OP(dilation2d, 2, 1, false, 0, 1);
    }
}
