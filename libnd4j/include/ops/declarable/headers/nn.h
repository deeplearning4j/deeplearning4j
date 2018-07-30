/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_NN_H
#define LIBND4J_HEADERS_NN_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {

        #if NOT_EXCLUDED(OP_softmax)
        DECLARE_CONFIGURABLE_OP(softmax, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(softmax_bp, 2, 1, true, 0, 0);
        #endif

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
        #if NOT_EXCLUDED(OP_lrn_old)
        DECLARE_CUSTOM_OP(lrn_old, 1, 3, true, 4, 0);
        #endif

        /**
         * Local response normalization implementation as TF.
         * input: 4D array
         * 
         * T args:
         *
         * 0: bias
         * 1: alpha
         * 2: beta
         *
         * Int arg: depth - optional local radius
         * 
         * output - 4D array 
         */
        #if NOT_EXCLUDED(OP_lrn)
        DECLARE_CONFIGURABLE_OP(lrn, 1, 1, true, 3, 0);
        #endif

        /**
         * Local response normalization - backprop variant.
         * input: 
         *  0 - 4D array of data
         *  1 - epsilon - 4D array of approximation
         * 
         * T args:
         *
         * 0: bias
         * 1: alpha
         * 2: beta
         *
         * Int arg: depth - optional local radius
         *
         * output - next approximation as 4D array
         */
        #if NOT_EXCLUDED(OP_lrn)
        DECLARE_CONFIGURABLE_OP(lrn_bp, 2, 1, true, 3, 0);
        #endif

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
        #if NOT_EXCLUDED(OP_batchnorm)
        DECLARE_CUSTOM_OP(batchnorm, 5, 1, false, 1, 2);
        #endif

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
        #if NOT_EXCLUDED(OP_apply_sgd)
        DECLARE_CONFIGURABLE_OP(apply_sgd, 2, 1, true, -2, 0);   
        #endif

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
        #if NOT_EXCLUDED(OP_fused_batch_norm)
        DECLARE_CUSTOM_OP(fused_batch_norm, 3, 1, false, 0, 2);
        #endif

        #if NOT_EXCLUDED(OP_log_softmax)
        DECLARE_CONFIGURABLE_OP(log_softmax, 1, 1, true, 0, 0);
        DECLARE_CONFIGURABLE_OP(log_softmax_bp, 2, 1, true, 0, 0);
        #endif

    }
}

#endif