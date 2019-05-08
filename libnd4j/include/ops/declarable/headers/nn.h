/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
        DECLARE_CUSTOM_OP(batchnorm, 3, 1, false, 1, 2);
        #endif
        #if NOT_EXCLUDED(OP_batchnorm)
        DECLARE_CUSTOM_OP(batchnorm_new, 3, 1, false, 1, 2);
        #endif

        /**
        * back prop in batch normalization
        * 
        * Expected arguments:
        * input: input array (any number of dimensions)
        * mean:
        * variance:
        * gamma: optional
        * beta: optional
        * dLdOut: next epsilon
        * 
        * Int args:
        * 0: apply scale
        * 1: apply offset 
        * 
        * T args:
        * 0: epsilon
        *
        * output arrays:
        * dL/dInput
        * dL/dMean
        * dL/dVariance
        * dL/dGamma
        * dL/dBeta
        */
        #if NOT_EXCLUDED(OP_batchnorm)
        DECLARE_CUSTOM_OP(batchnorm_bp, 4, 3, false, 1, 2);
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


        /**
         * relu_layer = relu(x*w + b)
         */
        DECLARE_CUSTOM_OP(relu_layer, 3, 1, false, 0, 0);

        /**
         * applies layer normalization to input
         * y = g * standardize(x) + b
         *
         * see nd4j::ops::standardize
         *
         */
        #if NOT_EXCLUDED(OP_layer_norm)
                DECLARE_CONFIGURABLE_OP(layer_norm, 3, 1, true, 0, -2);
                DECLARE_CUSTOM_OP(layer_norm_bp, 4, 1, false, 0, -2);
        #endif

        /**
         * This operation performs dot product attention on the given timeseries input with the given queries
         * out = sum(similarity(k_i, q) * v_i)
         *
         * similarity(k, q) = softmax(k * q) where x * q is the dot product of x and q
         *
         * Optionally with normalization step:
         * similarity(k, q) = softmax(k * q / sqrt(size(q))
         *
         * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, p. 4, eq. 1)
         *
         * Note: This supports multiple queries at once, if only one query is available the queries vector still has to
         * be 3D but can have queryCount = 1
         *
         * Note: keys and values usually is the same array. If you want to use it as the same array, simply pass it for
         * both.
         *
         * Expected arguments:
         * q: input 3D array "queries" of shape [batchSize, featureKeys, queryCount] or 4D array of shape [batchSize, numHeads, featureKeys, queryCount]
         * k: input 3D array "keys" of shape [batchSize, featureKeys, timesteps] or 4D array of shape [batchSize, numHeads, featureKeys, timesteps]
         * v: input 3D array "values" of shape [batchSize, featureValues, timesteps] or 4D array of shape [batchSize, numHeads, featureValues, timesteps]
         * mask: OPTIONAL; array that defines which values should be skipped of shape [batchSize, timesteps]
         *
         * integer input arguments:
         * 0: normalization, may have two values: zero -> do not apply normalization, one -> apply normalization
         * 1: withWeights, may have two values: zero -> do not return weights, one -> return weights
         *
         * Output Arrays:
         * 0: Attention result arrays of shape [batchSize, featureValues, queryCount] or [batchSize, numHeads, featureValues, queryCount]
         * 1: OPTIONAL; Attention weights of shape [batchSize, timesteps, queryCount] or [batchSize, numHeads, timesteps, queryCount]
         */
        #if NOT_EXCLUDED(OP_dot_product_attention)
                DECLARE_CUSTOM_OP(dot_product_attention, 3, -1, false, 0, 2);
                DECLARE_CUSTOM_OP(dot_product_attention_bp, 4, 3, false, 0, 1);
        #endif


        /**
         * This performs multi-headed dot product attention on the given timeseries input
         * out = concat(head_1, head_2, ..., head_n) * Wo
         * head_i = dot_product_attention(Wq_i*q, Wk_i*k, Wv_i*v)
         *
         * Optionally with normalization when calculating the attention for each head.
         *
         * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, pp. 4,5, "3.2.2 Multi-Head Attention")
         *
         * This makes use of dot_product_attention OP support for rank 4 inputs.
         *
         * Expected arguments:
         * q: input 3D array "queries" of shape [batchSize, featureKeys, queryCount]
         * k: input 3D array "keys" of shape [batchSize, featureKeys, timesteps]
         * v: input 3D array "values" of shape [batchSize, featureValues, timesteps]
         * Wq: input query projection weights of shape [numHeads, projectedKeys, featureKeys]
         * Wk: input key projection weights of shape [numHeads, projectedKeys, featureKeys]
         * Wv: input value projection weights of shape [numHeads, projectedValues, featureValues]
         * Wo: output projection weights of shape [numHeads * projectedValues, outSize]
         * mask: OPTIONAL; array that defines which values should be skipped of shape [batchSize, timesteps]
         *
         * integer input arguments:
         * 0: normalization, may have two values: zero -> do not apply normalization, one -> apply normalization
         * 1: withWeights, may have two values: zero -> do not return weights, one -> return weights
         *
         * Output Arrays:
         * 0: Attention result arrays of shape [batchSize, outSize, queryCount]
         * 1: OPTIONAL; Attention weights of shape [batchSize, numHeads, timesteps, queryCount]
         */
        #if NOT_EXCLUDED(OP_multi_head_dot_product_attention)
                DECLARE_CUSTOM_OP(multi_head_dot_product_attention, 7, -1, false, 0, 2);
                DECLARE_CUSTOM_OP(multi_head_dot_product_attention_bp, 8, 7, false, 0, 1);
        #endif
    }
}

#endif