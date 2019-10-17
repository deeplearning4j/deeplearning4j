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
// @author saudet
//

#ifndef DEV_TESTS_MKLDNNUTILS_H
#define DEV_TESTS_MKLDNNUTILS_H

#include <NativeOps.h>
#include <NDArray.h>
#include <mkldnn.hpp>
#include <MKLDNNStream.h>
#include <graph/Context.h>
#include <ops/declarable/PlatformHelper.h>
#include <platform_boilerplate.h>


namespace nd4j{
    namespace ops {
        namespace platforms {
            /**
             * Here we actually declare our platform helpers
             */
            DECLARE_PLATFORM(conv2d);

            DECLARE_PLATFORM(conv2d_bp);

            DECLARE_PLATFORM(avgpool2d);

            DECLARE_PLATFORM(avgpool2d_bp);

            DECLARE_PLATFORM(maxpool2d);

            DECLARE_PLATFORM(maxpool2d_bp);

            DECLARE_PLATFORM(conv3dnew);

            DECLARE_PLATFORM(conv3dnew_bp);

            DECLARE_PLATFORM(maxpool3dnew);

            DECLARE_PLATFORM(maxpool3dnew_bp);

            DECLARE_PLATFORM(avgpool3dnew);

            DECLARE_PLATFORM(avgpool3dnew_bp);

            DECLARE_PLATFORM(lrn);

            DECLARE_PLATFORM(batchnorm_new);

            DECLARE_PLATFORM(lstmLayer);
        }
    }

    namespace mkldnnUtils {

        /**
         * Utility methods for MKLDNN
         */
        void getMKLDNNMemoryDescConv2d(
                int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, bool isNCHW,
                int bS, int iC, int iH, int iW, int oC, int oH, int oW, const NDArray* src, const NDArray* diff_src,
                const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
                mkldnn::memory::desc* conv_src_md, mkldnn::memory::desc* conv_diff_src_md, mkldnn::memory::desc* conv_weights_md,
                mkldnn::memory::desc* conv_diff_weights_md, mkldnn::memory::desc* conv_bias_md, mkldnn::memory::desc* conv_dst_md,
                mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_weights_md,
                mkldnn::memory::desc* user_diff_weights_md, mkldnn::memory::desc* user_bias_md, mkldnn::memory::desc* user_dst_md,
                mkldnn::memory::dims& conv_strides, mkldnn::memory::dims& conv_padding, mkldnn::memory::dims& conv_padding_r);

        void getMKLDNNMemoryDescConv3d(
                int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, bool isSameMode, bool isNCDHW,
                int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW, const NDArray* src, const NDArray* diff_src,
                const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
                mkldnn::memory::desc* conv_src_md, mkldnn::memory::desc* conv_diff_src_md, mkldnn::memory::desc* conv_weights_md,
                mkldnn::memory::desc* conv_diff_weights_md, mkldnn::memory::desc* conv_bias_md, mkldnn::memory::desc* conv_dst_md,
                mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_weights_md,
                mkldnn::memory::desc* user_diff_weights_md, mkldnn::memory::desc* user_bias_md, mkldnn::memory::desc* user_dst_md,
                mkldnn::memory::dims& conv_strides, mkldnn::memory::dims& conv_padding, mkldnn::memory::dims& conv_padding_r);

        void getMKLDNNMemoryDescPool2d(
                int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int poolingMode, int extraParam0, bool isNCHW,
                int bS, int iC, int iH, int iW, int oC, int oH, int oW,
                const NDArray* src, const NDArray* diff_src, const NDArray* dst, mkldnn::algorithm& algorithm,
                mkldnn::memory::desc* pool_src_md, mkldnn::memory::desc* pool_diff_src_md, mkldnn::memory::desc* pool_dst_md,
                mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_dst_md,
                mkldnn::memory::dims& pool_strides, mkldnn::memory::dims& pool_kernel, mkldnn::memory::dims& pool_padding, mkldnn::memory::dims& pool_padding_r);

        void getMKLDNNMemoryDescPool3d(
                int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, int poolingMode, int extraParam0, bool isNCDHW,
                int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW,
                const NDArray* src, const NDArray* diff_src, const NDArray* dst, mkldnn::algorithm& algorithm,
                mkldnn::memory::desc* pool_src_md, mkldnn::memory::desc* pool_diff_src_md, mkldnn::memory::desc* pool_dst_md,
                mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_dst_md,
                mkldnn::memory::dims& pool_strides, mkldnn::memory::dims& pool_kernel, mkldnn::memory::dims& pool_padding, mkldnn::memory::dims& pool_padding_r);

        void getMKLDNNMemoryDescBatchNorm(const NDArray* src, const NDArray* diff_src, const NDArray* dst,
                                          mkldnn::memory::desc* batchnorm_src_md, mkldnn::memory::desc* batchnorm_diff_src_md, mkldnn::memory::desc* batchnorm_dst_md,
                                          mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_dst_md, int axis);

        void getMKLDNNMemoryDescLrn(const NDArray* src, const NDArray* diff_src, const NDArray* dst,
                                    mkldnn::memory::desc* lrn_src_md, mkldnn::memory::desc* lrn_diff_src_md, mkldnn::memory::desc* lrn_dst_md,
                                    mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_dst_md, int axis);

        mkldnn::engine& getEngine(void *ptr);
    }
}



#endif //DEV_TESTS_MKLDNNUTILS_H
