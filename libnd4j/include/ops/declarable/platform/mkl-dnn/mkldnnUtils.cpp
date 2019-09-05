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


#include "mkldnnUtils.h"

using namespace mkldnn;

namespace nd4j {
    namespace mkldnnUtils {
        void getMKLDNNMemoryDescPool2d(
                int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int poolingMode, int extraParam0, bool isNCHW,
                int bS, int iC, int iH, int iW, int oC, int oH, int oW,
                const NDArray* src, const NDArray* diff_src, const NDArray* dst, mkldnn::algorithm& algorithm,
                mkldnn::memory::desc* pool_src_md, mkldnn::memory::desc* pool_diff_src_md, mkldnn::memory::desc* pool_dst_md,
                mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_dst_md,
                mkldnn::memory::dims& pool_strides, mkldnn::memory::dims& pool_kernel, mkldnn::memory::dims& pool_padding, mkldnn::memory::dims& pool_padding_r) {
            mkldnn::memory::dims pool_src_tz = { bS, iC, iH, iW };
            mkldnn::memory::dims pool_dst_tz = { bS, oC, oH, oW };

            pool_strides = { sH, sW };
            pool_kernel = { kH, kW };
            pool_padding = { pH, pW };
            pool_padding_r = { (oH - 1) * sH - iH + kH - pH,
                               (oW - 1) * sW - iW + kW - pW };

            algorithm = poolingMode == 0 ? pooling_max
                                         : extraParam0 == 0 ? pooling_avg_exclude_padding
                                                            : pooling_avg_include_padding;
            auto type = mkldnn::memory::data_type::f32;
            auto format = isNCHW ? mkldnn::memory::format::nchw : mkldnn::memory::format::nhwc;
            auto supposed_to_be_any_format = mkldnn::memory::format::nChw8c; // doesn't work with "any"

            if (src != nullptr && src->getBuffer() != nullptr && pool_src_md != nullptr) {
                *pool_src_md = mkldnn::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
                *user_src_md = mkldnn::memory::desc({ pool_src_tz }, type, format);
                user_src_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[isNCHW ? 0 : 0];
                user_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[isNCHW ? 1 : 3];
                user_src_md->data.layout_desc.blocking.strides[0][2] = src->stridesOf()[isNCHW ? 2 : 1];
                user_src_md->data.layout_desc.blocking.strides[0][3] = src->stridesOf()[isNCHW ? 3 : 2];
            }

            if (diff_src != nullptr && diff_src->getBuffer() != nullptr && pool_diff_src_md != nullptr) {
                *pool_diff_src_md = mkldnn::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
                *user_diff_src_md = mkldnn::memory::desc({ pool_src_tz }, type, format);
                user_diff_src_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[isNCHW ? 0 : 0];
                user_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[isNCHW ? 1 : 3];
                user_diff_src_md->data.layout_desc.blocking.strides[0][2] = diff_src->stridesOf()[isNCHW ? 2 : 1];
                user_diff_src_md->data.layout_desc.blocking.strides[0][3] = diff_src->stridesOf()[isNCHW ? 3 : 2];
            }

            if (dst != nullptr && dst->getBuffer() != nullptr && pool_dst_md != nullptr) {
                *pool_dst_md = mkldnn::memory::desc({ pool_dst_tz }, type, supposed_to_be_any_format);
                *user_dst_md = mkldnn::memory::desc({ pool_dst_tz }, type, format);
                user_dst_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_dst_md->data.layout_desc.blocking.strides[0][0] = dst->stridesOf()[isNCHW ? 0 : 0];
                user_dst_md->data.layout_desc.blocking.strides[0][1] = dst->stridesOf()[isNCHW ? 1 : 3];
                user_dst_md->data.layout_desc.blocking.strides[0][2] = dst->stridesOf()[isNCHW ? 2 : 1];
                user_dst_md->data.layout_desc.blocking.strides[0][3] = dst->stridesOf()[isNCHW ? 3 : 2];
            }
        };


        void getMKLDNNMemoryDescPool3d(
                int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, int poolingMode, int extraParam0, bool isNCDHW,
                int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW,
                const NDArray* src, const NDArray* diff_src, const NDArray* dst, mkldnn::algorithm& algorithm,
                mkldnn::memory::desc* pool_src_md, mkldnn::memory::desc* pool_diff_src_md, mkldnn::memory::desc* pool_dst_md,
                mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_dst_md,
                mkldnn::memory::dims& pool_strides, mkldnn::memory::dims& pool_kernel, mkldnn::memory::dims& pool_padding, mkldnn::memory::dims& pool_padding_r) {
            mkldnn::memory::dims pool_src_tz = { bS, iC, iD, iH, iW };
            mkldnn::memory::dims pool_dst_tz = { bS, oC, oD, oH, oW };

            pool_strides = { sD, sH, sW };
            pool_kernel = { kD, kH, kW };
            pool_padding = { pD, pH, pW };
            pool_padding_r = { (oD - 1) * sD - iD + kD - pD,
                               (oH - 1) * sH - iH + kH - pH,
                               (oW - 1) * sW - iW + kW - pW };

            algorithm = poolingMode == 0 ? pooling_max
                                         : extraParam0 == 0 ? pooling_avg_exclude_padding
                                                            : pooling_avg_include_padding;
            auto type = mkldnn::memory::data_type::f32;
            auto format = isNCDHW ? mkldnn::memory::format::ncdhw : mkldnn::memory::format::ndhwc;
            auto supposed_to_be_any_format = mkldnn::memory::format::nCdhw8c; // doesn't work with "any"

            if (src != nullptr && src->getBuffer() != nullptr && pool_src_md != nullptr) {
                *pool_src_md = mkldnn::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
                *user_src_md = mkldnn::memory::desc({ pool_src_tz }, type, format);
                user_src_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[isNCDHW ? 0 : 0];
                user_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[isNCDHW ? 1 : 4];
                user_src_md->data.layout_desc.blocking.strides[0][2] = src->stridesOf()[isNCDHW ? 2 : 1];
                user_src_md->data.layout_desc.blocking.strides[0][3] = src->stridesOf()[isNCDHW ? 3 : 2];
                user_src_md->data.layout_desc.blocking.strides[0][4] = src->stridesOf()[isNCDHW ? 4 : 3];
            }

            if (diff_src != nullptr && diff_src->getBuffer() != nullptr && pool_diff_src_md != nullptr) {
                *pool_diff_src_md = mkldnn::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
                *user_diff_src_md = mkldnn::memory::desc({ pool_src_tz }, type, format);
                user_diff_src_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[isNCDHW ? 0 : 0];
                user_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[isNCDHW ? 1 : 4];
                user_diff_src_md->data.layout_desc.blocking.strides[0][2] = diff_src->stridesOf()[isNCDHW ? 2 : 1];
                user_diff_src_md->data.layout_desc.blocking.strides[0][3] = diff_src->stridesOf()[isNCDHW ? 3 : 2];
                user_diff_src_md->data.layout_desc.blocking.strides[0][4] = diff_src->stridesOf()[isNCDHW ? 4 : 3];
            }

            if (dst != nullptr && dst->getBuffer() != nullptr && pool_dst_md != nullptr) {
                *pool_dst_md = mkldnn::memory::desc({ pool_dst_tz }, type, supposed_to_be_any_format);
                *user_dst_md = mkldnn::memory::desc({ pool_dst_tz }, type, format);
                user_dst_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_dst_md->data.layout_desc.blocking.strides[0][0] = dst->stridesOf()[isNCDHW ? 0 : 0];
                user_dst_md->data.layout_desc.blocking.strides[0][1] = dst->stridesOf()[isNCDHW ? 1 : 4];
                user_dst_md->data.layout_desc.blocking.strides[0][2] = dst->stridesOf()[isNCDHW ? 2 : 1];
                user_dst_md->data.layout_desc.blocking.strides[0][3] = dst->stridesOf()[isNCDHW ? 3 : 2];
                user_dst_md->data.layout_desc.blocking.strides[0][4] = dst->stridesOf()[isNCDHW ? 4 : 3];
            }
        };



        void getMKLDNNMemoryDescConv2d(
                int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, bool isNCHW,
                int bS, int iC, int iH, int iW, int oC, int oH, int oW, const NDArray* src, const NDArray* diff_src,
                const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
                mkldnn::memory::desc* conv_src_md, mkldnn::memory::desc* conv_diff_src_md, mkldnn::memory::desc* conv_weights_md,
                mkldnn::memory::desc* conv_diff_weights_md, mkldnn::memory::desc* conv_bias_md, mkldnn::memory::desc* conv_dst_md,
                mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_weights_md,
                mkldnn::memory::desc* user_diff_weights_md, mkldnn::memory::desc* user_bias_md, mkldnn::memory::desc* user_dst_md,
                mkldnn::memory::dims& conv_strides, mkldnn::memory::dims& conv_padding, mkldnn::memory::dims& conv_padding_r) {
            mkldnn::memory::dims conv_src_tz = { bS, iC, iH, iW };
            mkldnn::memory::dims conv_weights_tz = { oC, iC, kH, kW };
            mkldnn::memory::dims conv_bias_tz = { oC };
            mkldnn::memory::dims conv_dst_tz = { bS, oC, oH, oW };

            conv_strides = { sH, sW };
            conv_padding = { pH, pW };
            conv_padding_r = { (oH - 1) * sH - iH + kH - pH,
                               (oW - 1) * sW - iW + kW - pW };

            auto type = mkldnn::memory::data_type::f32;
            auto format = isNCHW ? mkldnn::memory::format::nchw : mkldnn::memory::format::nhwc;
            auto formatw = mkldnn::memory::format::hwio;

            if (src != nullptr && conv_src_md != nullptr) {
                *conv_src_md = mkldnn::memory::desc({ conv_src_tz }, type, mkldnn::memory::format::any);
                *user_src_md = mkldnn::memory::desc({ conv_src_tz }, type, format);
                user_src_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[isNCHW ? 0 : 0];
                user_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[isNCHW ? 1 : 3];
                user_src_md->data.layout_desc.blocking.strides[0][2] = src->stridesOf()[isNCHW ? 2 : 1];
                user_src_md->data.layout_desc.blocking.strides[0][3] = src->stridesOf()[isNCHW ? 3 : 2];
            }

            if (diff_src != nullptr && conv_diff_src_md != nullptr) {
                *conv_diff_src_md = mkldnn::memory::desc({ conv_src_tz }, type, mkldnn::memory::format::any);
                *user_diff_src_md = mkldnn::memory::desc({ conv_src_tz }, type, format);
                user_diff_src_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[isNCHW ? 0 : 0];
                user_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[isNCHW ? 1 : 3];
                user_diff_src_md->data.layout_desc.blocking.strides[0][2] = diff_src->stridesOf()[isNCHW ? 2 : 1];
                user_diff_src_md->data.layout_desc.blocking.strides[0][3] = diff_src->stridesOf()[isNCHW ? 3 : 2];
            }

            if (weights != nullptr && conv_weights_md != nullptr) {
                *conv_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, mkldnn::memory::format::any);
                *user_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, formatw);
                user_weights_md->data.format = mkldnn_blocked; // overrides "formatw = hwio"
                user_weights_md->data.layout_desc.blocking.strides[0][0] = weights->stridesOf()[3];
                user_weights_md->data.layout_desc.blocking.strides[0][1] = weights->stridesOf()[2];
                user_weights_md->data.layout_desc.blocking.strides[0][2] = weights->stridesOf()[0];
                user_weights_md->data.layout_desc.blocking.strides[0][3] = weights->stridesOf()[1];
            }

            if (diff_weights != nullptr && conv_diff_weights_md != nullptr) {
                *conv_diff_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, mkldnn::memory::format::any);
                *user_diff_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, formatw);
                user_diff_weights_md->data.format = mkldnn_blocked; // overrides "formatw = hwio"
                user_diff_weights_md->data.layout_desc.blocking.strides[0][0] = diff_weights->stridesOf()[3];
                user_diff_weights_md->data.layout_desc.blocking.strides[0][1] = diff_weights->stridesOf()[2];
                user_diff_weights_md->data.layout_desc.blocking.strides[0][2] = diff_weights->stridesOf()[0];
                user_diff_weights_md->data.layout_desc.blocking.strides[0][3] = diff_weights->stridesOf()[1];
            }

            if (bias != nullptr && conv_bias_md != nullptr) {
                *conv_bias_md = mkldnn::memory::desc({ conv_bias_tz }, type, mkldnn::memory::format::any);
                *user_bias_md = mkldnn::memory::desc({ conv_bias_tz }, type, mkldnn::memory::format::x);
            }

            if (dst != nullptr && conv_dst_md != nullptr) {
                *conv_dst_md = mkldnn::memory::desc({ conv_dst_tz }, type, mkldnn::memory::format::any);
                *user_dst_md = mkldnn::memory::desc({ conv_dst_tz }, type, format);
                user_dst_md->data.format = mkldnn_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_dst_md->data.layout_desc.blocking.strides[0][0] = dst->stridesOf()[isNCHW ? 0 : 0];
                user_dst_md->data.layout_desc.blocking.strides[0][1] = dst->stridesOf()[isNCHW ? 1 : 3];
                user_dst_md->data.layout_desc.blocking.strides[0][2] = dst->stridesOf()[isNCHW ? 2 : 1];
                user_dst_md->data.layout_desc.blocking.strides[0][3] = dst->stridesOf()[isNCHW ? 3 : 2];
            }
        }

        void getMKLDNNMemoryDescConv3d(
                int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, bool isSameMode, bool isNCDHW,
                int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW, const NDArray* src, const NDArray* diff_src,
                const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
                mkldnn::memory::desc* conv_src_md, mkldnn::memory::desc* conv_diff_src_md, mkldnn::memory::desc* conv_weights_md,
                mkldnn::memory::desc* conv_diff_weights_md, mkldnn::memory::desc* conv_bias_md, mkldnn::memory::desc* conv_dst_md,
                mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_weights_md,
                mkldnn::memory::desc* user_diff_weights_md, mkldnn::memory::desc* user_bias_md, mkldnn::memory::desc* user_dst_md,
                mkldnn::memory::dims& conv_strides, mkldnn::memory::dims& conv_padding, mkldnn::memory::dims& conv_padding_r) {
            mkldnn::memory::dims conv_src_tz = { bS, iC, iD, iH, iW };
            mkldnn::memory::dims conv_weights_tz = { oC, iC, kD, kH, kW };
            mkldnn::memory::dims conv_bias_tz = { oC };
            mkldnn::memory::dims conv_dst_tz = { bS, oC, oD, oH, oW };

            conv_strides = { sD, sH, sW };
            conv_padding = { pD, pH, pW };
            conv_padding_r = { (oD - 1) * sD - iD + kD - pD,
                               (oH - 1) * sH - iH + kH - pH,
                               (oW - 1) * sW - iW + kW - pW };

            auto type = mkldnn::memory::data_type::f32;
            auto format = isNCDHW ? mkldnn::memory::format::ncdhw : mkldnn::memory::format::ndhwc;
            auto formatw = mkldnn::memory::format::dhwio;

            if (src != nullptr && conv_src_md != nullptr) {
                *conv_src_md = mkldnn::memory::desc({ conv_src_tz }, type, mkldnn::memory::format::any);
                *user_src_md = mkldnn::memory::desc({ conv_src_tz }, type, format);
                user_src_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[isNCDHW ? 0 : 0];
                user_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[isNCDHW ? 1 : 4];
                user_src_md->data.layout_desc.blocking.strides[0][2] = src->stridesOf()[isNCDHW ? 2 : 1];
                user_src_md->data.layout_desc.blocking.strides[0][3] = src->stridesOf()[isNCDHW ? 3 : 2];
                user_src_md->data.layout_desc.blocking.strides[0][4] = src->stridesOf()[isNCDHW ? 4 : 3];
            }

            if (diff_src != nullptr && conv_diff_src_md != nullptr) {
                *conv_diff_src_md = mkldnn::memory::desc({ conv_src_tz }, type, mkldnn::memory::format::any);
                *user_diff_src_md = mkldnn::memory::desc({ conv_src_tz }, type, format);
                user_diff_src_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[isNCDHW ? 0 : 0];
                user_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[isNCDHW ? 1 : 4];
                user_diff_src_md->data.layout_desc.blocking.strides[0][2] = diff_src->stridesOf()[isNCDHW ? 2 : 1];
                user_diff_src_md->data.layout_desc.blocking.strides[0][3] = diff_src->stridesOf()[isNCDHW ? 3 : 2];
                user_diff_src_md->data.layout_desc.blocking.strides[0][4] = diff_src->stridesOf()[isNCDHW ? 4 : 3];
            }

            if (weights != nullptr && conv_weights_md != nullptr) {
                *conv_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, mkldnn::memory::format::any);
                *user_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, formatw);
                user_weights_md->data.format = mkldnn_blocked; // overrides "formatw = dhwio"
                user_weights_md->data.layout_desc.blocking.strides[0][0] = weights->stridesOf()[4];
                user_weights_md->data.layout_desc.blocking.strides[0][1] = weights->stridesOf()[3];
                user_weights_md->data.layout_desc.blocking.strides[0][2] = weights->stridesOf()[0];
                user_weights_md->data.layout_desc.blocking.strides[0][3] = weights->stridesOf()[1];
                user_weights_md->data.layout_desc.blocking.strides[0][4] = weights->stridesOf()[2];
            }

            if (diff_weights != nullptr && conv_diff_weights_md != nullptr) {
                *conv_diff_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, mkldnn::memory::format::any);
                *user_diff_weights_md = mkldnn::memory::desc({ conv_weights_tz }, type, formatw);
                user_diff_weights_md->data.format = mkldnn_blocked; // overrides "formatw = dhwio"
                user_diff_weights_md->data.layout_desc.blocking.strides[0][0] = diff_weights->stridesOf()[4];
                user_diff_weights_md->data.layout_desc.blocking.strides[0][1] = diff_weights->stridesOf()[3];
                user_diff_weights_md->data.layout_desc.blocking.strides[0][2] = diff_weights->stridesOf()[0];
                user_diff_weights_md->data.layout_desc.blocking.strides[0][3] = diff_weights->stridesOf()[1];
                user_diff_weights_md->data.layout_desc.blocking.strides[0][4] = diff_weights->stridesOf()[2];
            }

            if (bias != nullptr && conv_bias_md != nullptr) {
                *conv_bias_md = mkldnn::memory::desc({ conv_bias_tz }, type, mkldnn::memory::format::any);
                *user_bias_md = mkldnn::memory::desc({ conv_bias_tz }, type, mkldnn::memory::format::x);
            }

            if (dst != nullptr && conv_dst_md != nullptr) {
                *conv_dst_md = mkldnn::memory::desc({ conv_dst_tz }, type, mkldnn::memory::format::any);
                *user_dst_md = mkldnn::memory::desc({ conv_dst_tz }, type, format);
                user_dst_md->data.format = mkldnn_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_dst_md->data.layout_desc.blocking.strides[0][0] = dst->stridesOf()[isNCDHW ? 0 : 0];
                user_dst_md->data.layout_desc.blocking.strides[0][1] = dst->stridesOf()[isNCDHW ? 1 : 4];
                user_dst_md->data.layout_desc.blocking.strides[0][2] = dst->stridesOf()[isNCDHW ? 2 : 1];
                user_dst_md->data.layout_desc.blocking.strides[0][3] = dst->stridesOf()[isNCDHW ? 3 : 2];
                user_dst_md->data.layout_desc.blocking.strides[0][4] = dst->stridesOf()[isNCDHW ? 4 : 3];
            }
        };

    }
}