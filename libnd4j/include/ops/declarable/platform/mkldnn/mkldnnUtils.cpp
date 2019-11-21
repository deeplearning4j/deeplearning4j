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

#include <dnnl_types.h>
#include "mkldnnUtils.h"

using namespace dnnl;

namespace nd4j {
    namespace mkldnnUtils {
        void getMKLDNNMemoryDescPool2d(
                int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int poolingMode, int extraParam0, bool isNCHW,
                int bS, int iC, int iH, int iW, int oC, int oH, int oW,
                const NDArray* src, const NDArray* diff_src, const NDArray* dst, dnnl::algorithm& algorithm,
                dnnl::memory::desc* pool_src_md, dnnl::memory::desc* pool_diff_src_md, dnnl::memory::desc* pool_dst_md,
                dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_dst_md,
                dnnl::memory::dims& pool_strides, dnnl::memory::dims& pool_kernel, dnnl::memory::dims& pool_padding, dnnl::memory::dims& pool_padding_r) {
            dnnl::memory::dims pool_src_tz = { bS, iC, iH, iW };
            dnnl::memory::dims pool_dst_tz = { bS, oC, oH, oW };

            pool_strides = { sH, sW };
            pool_kernel = { kH, kW };
            pool_padding = { pH, pW };
            pool_padding_r = { (oH - 1) * sH - iH + kH - pH,
                               (oW - 1) * sW - iW + kW - pW };

            algorithm = poolingMode == 0 ? algorithm::pooling_max
                                         : extraParam0 == 0 ? algorithm::pooling_avg_exclude_padding
                                                            : algorithm::pooling_avg_include_padding;
            auto type = dnnl::memory::data_type::f32;
            auto format = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
            auto supposed_to_be_any_format = dnnl::memory::format_tag::nChw8c; // doesn't work with "any"

            if (src != nullptr && src->getBuffer() != nullptr && pool_src_md != nullptr) {
                *pool_src_md = dnnl::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
                *user_src_md = dnnl::memory::desc({ pool_src_tz }, type, format);
                user_src_md->data.format_kind = dnnl_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_src_md->data.format_desc.blocking.strides[0] = src->stridesOf()[isNCHW ? 0 : 0];
                user_src_md->data.format_desc.blocking.strides[1] = src->stridesOf()[isNCHW ? 1 : 3];
                user_src_md->data.format_desc.blocking.strides[2] = src->stridesOf()[isNCHW ? 2 : 1];
                user_src_md->data.format_desc.blocking.strides[3] = src->stridesOf()[isNCHW ? 3 : 2];
            }

            if (diff_src != nullptr && diff_src->getBuffer() != nullptr && pool_diff_src_md != nullptr) {
                *pool_diff_src_md = dnnl::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
                *user_diff_src_md = dnnl::memory::desc({ pool_src_tz }, type, format);
                user_diff_src_md->data.format_kind = dnnl_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_diff_src_md->data.format_desc.blocking.strides[0] = diff_src->stridesOf()[isNCHW ? 0 : 0];
                user_diff_src_md->data.format_desc.blocking.strides[1] = diff_src->stridesOf()[isNCHW ? 1 : 3];
                user_diff_src_md->data.format_desc.blocking.strides[2] = diff_src->stridesOf()[isNCHW ? 2 : 1];
                user_diff_src_md->data.format_desc.blocking.strides[3] = diff_src->stridesOf()[isNCHW ? 3 : 2];
            }

            if (dst != nullptr && dst->getBuffer() != nullptr && pool_dst_md != nullptr) {
                *pool_dst_md = dnnl::memory::desc({ pool_dst_tz }, type, supposed_to_be_any_format);
                *user_dst_md = dnnl::memory::desc({ pool_dst_tz }, type, format);
                user_dst_md->data.format_kind = dnnl_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_dst_md->data.format_desc.blocking.strides[0] = dst->stridesOf()[isNCHW ? 0 : 0];
                user_dst_md->data.format_desc.blocking.strides[1] = dst->stridesOf()[isNCHW ? 1 : 3];
                user_dst_md->data.format_desc.blocking.strides[2] = dst->stridesOf()[isNCHW ? 2 : 1];
                user_dst_md->data.format_desc.blocking.strides[3] = dst->stridesOf()[isNCHW ? 3 : 2];
            }
        };


        void getMKLDNNMemoryDescPool3d(
                int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, int poolingMode, int extraParam0, bool isNCDHW,
                int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW,
                const NDArray* src, const NDArray* diff_src, const NDArray* dst, dnnl::algorithm& algorithm,
                dnnl::memory::desc* pool_src_md, dnnl::memory::desc* pool_diff_src_md, dnnl::memory::desc* pool_dst_md,
                dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_dst_md,
                dnnl::memory::dims& pool_strides, dnnl::memory::dims& pool_kernel, dnnl::memory::dims& pool_padding, dnnl::memory::dims& pool_padding_r) {
            dnnl::memory::dims pool_src_tz = { bS, iC, iD, iH, iW };
            dnnl::memory::dims pool_dst_tz = { bS, oC, oD, oH, oW };

            pool_strides = { sD, sH, sW };
            pool_kernel = { kD, kH, kW };
            pool_padding = { pD, pH, pW };
            pool_padding_r = { (oD - 1) * sD - iD + kD - pD,
                               (oH - 1) * sH - iH + kH - pH,
                               (oW - 1) * sW - iW + kW - pW };

            algorithm = poolingMode == 0 ? algorithm::pooling_max
                                         : extraParam0 == 0 ? algorithm::pooling_avg_exclude_padding
                                                            : algorithm::pooling_avg_include_padding;
            auto type = dnnl::memory::data_type::f32;
            auto format = isNCDHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
            auto supposed_to_be_any_format = dnnl::memory::format_tag::nCdhw8c; // doesn't work with "any"

            if (src != nullptr && src->getBuffer() != nullptr && pool_src_md != nullptr) {
                *pool_src_md = dnnl::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
                *user_src_md = dnnl::memory::desc({ pool_src_tz }, type, format);
                user_src_md->data.format_kind = dnnl_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_src_md->data.format_desc.blocking.strides[0] = src->stridesOf()[isNCDHW ? 0 : 0];
                user_src_md->data.format_desc.blocking.strides[1] = src->stridesOf()[isNCDHW ? 1 : 4];
                user_src_md->data.format_desc.blocking.strides[2] = src->stridesOf()[isNCDHW ? 2 : 1];
                user_src_md->data.format_desc.blocking.strides[3] = src->stridesOf()[isNCDHW ? 3 : 2];
                user_src_md->data.format_desc.blocking.strides[4] = src->stridesOf()[isNCDHW ? 4 : 3];
            }

            if (diff_src != nullptr && diff_src->getBuffer() != nullptr && pool_diff_src_md != nullptr) {
                *pool_diff_src_md = dnnl::memory::desc({ pool_src_tz }, type, supposed_to_be_any_format);
                *user_diff_src_md = dnnl::memory::desc({ pool_src_tz }, type, format);
                user_diff_src_md->data.format_kind = dnnl_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_diff_src_md->data.format_desc.blocking.strides[0] = diff_src->stridesOf()[isNCDHW ? 0 : 0];
                user_diff_src_md->data.format_desc.blocking.strides[1] = diff_src->stridesOf()[isNCDHW ? 1 : 4];
                user_diff_src_md->data.format_desc.blocking.strides[2] = diff_src->stridesOf()[isNCDHW ? 2 : 1];
                user_diff_src_md->data.format_desc.blocking.strides[3] = diff_src->stridesOf()[isNCDHW ? 3 : 2];
                user_diff_src_md->data.format_desc.blocking.strides[4] = diff_src->stridesOf()[isNCDHW ? 4 : 3];
            }

            if (dst != nullptr && dst->getBuffer() != nullptr && pool_dst_md != nullptr) {
                *pool_dst_md = dnnl::memory::desc({ pool_dst_tz }, type, supposed_to_be_any_format);
                *user_dst_md = dnnl::memory::desc({ pool_dst_tz }, type, format);
                user_dst_md->data.format_kind = dnnl_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_dst_md->data.format_desc.blocking.strides[0] = dst->stridesOf()[isNCDHW ? 0 : 0];
                user_dst_md->data.format_desc.blocking.strides[1] = dst->stridesOf()[isNCDHW ? 1 : 4];
                user_dst_md->data.format_desc.blocking.strides[2] = dst->stridesOf()[isNCDHW ? 2 : 1];
                user_dst_md->data.format_desc.blocking.strides[3] = dst->stridesOf()[isNCDHW ? 3 : 2];
                user_dst_md->data.format_desc.blocking.strides[4] = dst->stridesOf()[isNCDHW ? 4 : 3];
            }
        };



        void getMKLDNNMemoryDescConv2d(
                int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, bool isSameMode, bool isNCHW,
                int bS, int iC, int iH, int iW, int oC, int oH, int oW, const NDArray* src, const NDArray* diff_src,
                const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
                dnnl::memory::desc* conv_src_md, dnnl::memory::desc* conv_diff_src_md, dnnl::memory::desc* conv_weights_md,
                dnnl::memory::desc* conv_diff_weights_md, dnnl::memory::desc* conv_bias_md, dnnl::memory::desc* conv_dst_md,
                dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_weights_md,
                dnnl::memory::desc* user_diff_weights_md, dnnl::memory::desc* user_bias_md, dnnl::memory::desc* user_dst_md,
                dnnl::memory::dims& conv_strides, dnnl::memory::dims& conv_padding, dnnl::memory::dims& conv_padding_r, dnnl::memory::dims& conv_dilation) {
            dnnl::memory::dims conv_src_tz = { bS, iC, iH, iW };
            dnnl::memory::dims conv_weights_tz = { oC, iC, kH, kW };
            dnnl::memory::dims conv_bias_tz = { oC };
            dnnl::memory::dims conv_dst_tz = { bS, oC, oH, oW };

            conv_strides   = { sH, sW };
            conv_padding   = { pH, pW };
            conv_padding_r = { (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pW };
            conv_dilation  = { dH-1, dW-1};

            auto type = dnnl::memory::data_type::f32;
            auto format = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
            auto formatw = dnnl::memory::format_tag::hwio;

            if (src != nullptr && conv_src_md != nullptr) {
                *conv_src_md = dnnl::memory::desc({ conv_src_tz }, type, dnnl::memory::format_tag::any);
                *user_src_md = dnnl::memory::desc({ conv_src_tz }, type, format);
                user_src_md->data.format_kind = dnnl_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_src_md->data.format_desc.blocking.strides[0] = src->stridesOf()[isNCHW ? 0 : 0];
                user_src_md->data.format_desc.blocking.strides[1] = src->stridesOf()[isNCHW ? 1 : 3];
                user_src_md->data.format_desc.blocking.strides[2] = src->stridesOf()[isNCHW ? 2 : 1];
                user_src_md->data.format_desc.blocking.strides[3] = src->stridesOf()[isNCHW ? 3 : 2];
            }

            if (diff_src != nullptr && conv_diff_src_md != nullptr) {
                *conv_diff_src_md = dnnl::memory::desc({ conv_src_tz }, type, dnnl::memory::format_tag::any);
                *user_diff_src_md = dnnl::memory::desc({ conv_src_tz }, type, format);
                user_diff_src_md->data.format_kind = dnnl_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_diff_src_md->data.format_desc.blocking.strides[0] = diff_src->stridesOf()[isNCHW ? 0 : 0];
                user_diff_src_md->data.format_desc.blocking.strides[1] = diff_src->stridesOf()[isNCHW ? 1 : 3];
                user_diff_src_md->data.format_desc.blocking.strides[2] = diff_src->stridesOf()[isNCHW ? 2 : 1];
                user_diff_src_md->data.format_desc.blocking.strides[3] = diff_src->stridesOf()[isNCHW ? 3 : 2];
            }

            if (weights != nullptr && conv_weights_md != nullptr) {
                *conv_weights_md = dnnl::memory::desc({ conv_weights_tz }, type, dnnl::memory::format_tag::any);
                *user_weights_md = dnnl::memory::desc({ conv_weights_tz }, type, formatw);
                user_weights_md->data.format_kind = dnnl_blocked; // overrides "formatw = hwio"
                user_weights_md->data.format_desc.blocking.strides[0] = weights->stridesOf()[3];
                user_weights_md->data.format_desc.blocking.strides[1] = weights->stridesOf()[2];
                user_weights_md->data.format_desc.blocking.strides[2] = weights->stridesOf()[0];
                user_weights_md->data.format_desc.blocking.strides[3] = weights->stridesOf()[1];
            }

            if (diff_weights != nullptr && conv_diff_weights_md != nullptr) {
                *conv_diff_weights_md = dnnl::memory::desc({ conv_weights_tz }, type, dnnl::memory::format_tag::any);
                *user_diff_weights_md = dnnl::memory::desc({ conv_weights_tz }, type, formatw);
                user_diff_weights_md->data.format_kind = dnnl_blocked; // overrides "formatw = hwio"
                user_diff_weights_md->data.format_desc.blocking.strides[0] = diff_weights->stridesOf()[3];
                user_diff_weights_md->data.format_desc.blocking.strides[1] = diff_weights->stridesOf()[2];
                user_diff_weights_md->data.format_desc.blocking.strides[2] = diff_weights->stridesOf()[0];
                user_diff_weights_md->data.format_desc.blocking.strides[3] = diff_weights->stridesOf()[1];
            }

            if (bias != nullptr && conv_bias_md != nullptr) {
                *conv_bias_md = dnnl::memory::desc({ conv_bias_tz }, type, dnnl::memory::format_tag::any);
                *user_bias_md = dnnl::memory::desc({ conv_bias_tz }, type, dnnl::memory::format_tag::x);
            }

            if (dst != nullptr && conv_dst_md != nullptr) {
                *conv_dst_md = dnnl::memory::desc({ conv_dst_tz }, type, dnnl::memory::format_tag::any);
                *user_dst_md = dnnl::memory::desc({ conv_dst_tz }, type, format);
                user_dst_md->data.format_kind = dnnl_blocked; // overrides "format = isNCHW ? nchw : nhwc"
                user_dst_md->data.format_desc.blocking.strides[0] = dst->stridesOf()[isNCHW ? 0 : 0];
                user_dst_md->data.format_desc.blocking.strides[1] = dst->stridesOf()[isNCHW ? 1 : 3];
                user_dst_md->data.format_desc.blocking.strides[2] = dst->stridesOf()[isNCHW ? 2 : 1];
                user_dst_md->data.format_desc.blocking.strides[3] = dst->stridesOf()[isNCHW ? 3 : 2];
            }
        }

        void getMKLDNNMemoryDescConv3d(
                int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, bool isSameMode, bool isNCDHW,
                int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW, const NDArray* src, const NDArray* diff_src,
                const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
                dnnl::memory::desc* conv_src_md, dnnl::memory::desc* conv_diff_src_md, dnnl::memory::desc* conv_weights_md,
                dnnl::memory::desc* conv_diff_weights_md, dnnl::memory::desc* conv_bias_md, dnnl::memory::desc* conv_dst_md,
                dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_weights_md,
                dnnl::memory::desc* user_diff_weights_md, dnnl::memory::desc* user_bias_md, dnnl::memory::desc* user_dst_md,
                dnnl::memory::dims& conv_strides, dnnl::memory::dims& conv_padding, dnnl::memory::dims& conv_padding_r, dnnl::memory::dims& conv_dilation) {
            dnnl::memory::dims conv_src_tz = { bS, iC, iD, iH, iW };
            dnnl::memory::dims conv_weights_tz = { oC, iC, kD, kH, kW };
            dnnl::memory::dims conv_bias_tz = { oC };
            dnnl::memory::dims conv_dst_tz = { bS, oC, oD, oH, oW };

            conv_strides   = { sD, sH, sW };
            conv_padding   = { pD, pH, pW };
            conv_padding_r = { (oD - 1) * sD - iD + kD - pD, (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pW };
            conv_dilation  = { dD-1, dH-1, dW-1};

            auto type = dnnl::memory::data_type::f32;
            auto format = isNCDHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
            auto formatw = dnnl::memory::format_tag::dhwio;

            if (src != nullptr && conv_src_md != nullptr) {
                *conv_src_md = dnnl::memory::desc({ conv_src_tz }, type, dnnl::memory::format_tag::any);
                *user_src_md = dnnl::memory::desc({ conv_src_tz }, type, format);
                user_src_md->data.format_kind = dnnl_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_src_md->data.format_desc.blocking.strides[0] = src->stridesOf()[isNCDHW ? 0 : 0];
                user_src_md->data.format_desc.blocking.strides[1] = src->stridesOf()[isNCDHW ? 1 : 4];
                user_src_md->data.format_desc.blocking.strides[2] = src->stridesOf()[isNCDHW ? 2 : 1];
                user_src_md->data.format_desc.blocking.strides[3] = src->stridesOf()[isNCDHW ? 3 : 2];
                user_src_md->data.format_desc.blocking.strides[4] = src->stridesOf()[isNCDHW ? 4 : 3];
            }

            if (diff_src != nullptr && conv_diff_src_md != nullptr) {
                *conv_diff_src_md = dnnl::memory::desc({ conv_src_tz }, type, dnnl::memory::format_tag::any);
                *user_diff_src_md = dnnl::memory::desc({ conv_src_tz }, type, format);
                user_diff_src_md->data.format_kind = dnnl_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_diff_src_md->data.format_desc.blocking.strides[0] = diff_src->stridesOf()[isNCDHW ? 0 : 0];
                user_diff_src_md->data.format_desc.blocking.strides[1] = diff_src->stridesOf()[isNCDHW ? 1 : 4];
                user_diff_src_md->data.format_desc.blocking.strides[2] = diff_src->stridesOf()[isNCDHW ? 2 : 1];
                user_diff_src_md->data.format_desc.blocking.strides[3] = diff_src->stridesOf()[isNCDHW ? 3 : 2];
                user_diff_src_md->data.format_desc.blocking.strides[4] = diff_src->stridesOf()[isNCDHW ? 4 : 3];
            }

            if (weights != nullptr && conv_weights_md != nullptr) {
                *conv_weights_md = dnnl::memory::desc({ conv_weights_tz }, type, dnnl::memory::format_tag::any);
                *user_weights_md = dnnl::memory::desc({ conv_weights_tz }, type, formatw);
                user_weights_md->data.format_kind = dnnl_blocked; // overrides "formatw = dhwio"
                user_weights_md->data.format_desc.blocking.strides[0] = weights->stridesOf()[4];
                user_weights_md->data.format_desc.blocking.strides[1] = weights->stridesOf()[3];
                user_weights_md->data.format_desc.blocking.strides[2] = weights->stridesOf()[0];
                user_weights_md->data.format_desc.blocking.strides[3] = weights->stridesOf()[1];
                user_weights_md->data.format_desc.blocking.strides[4] = weights->stridesOf()[2];
            }

            if (diff_weights != nullptr && conv_diff_weights_md != nullptr) {
                *conv_diff_weights_md = dnnl::memory::desc({ conv_weights_tz }, type, dnnl::memory::format_tag::any);
                *user_diff_weights_md = dnnl::memory::desc({ conv_weights_tz }, type, formatw);
                user_diff_weights_md->data.format_kind = dnnl_blocked; // overrides "formatw = dhwio"
                user_diff_weights_md->data.format_desc.blocking.strides[0] = diff_weights->stridesOf()[4];
                user_diff_weights_md->data.format_desc.blocking.strides[1] = diff_weights->stridesOf()[3];
                user_diff_weights_md->data.format_desc.blocking.strides[2] = diff_weights->stridesOf()[0];
                user_diff_weights_md->data.format_desc.blocking.strides[3] = diff_weights->stridesOf()[1];
                user_diff_weights_md->data.format_desc.blocking.strides[4] = diff_weights->stridesOf()[2];
            }

            if (bias != nullptr && conv_bias_md != nullptr) {
                *conv_bias_md = dnnl::memory::desc({ conv_bias_tz }, type, dnnl::memory::format_tag::any);
                *user_bias_md = dnnl::memory::desc({ conv_bias_tz }, type, dnnl::memory::format_tag::x);
            }

            if (dst != nullptr && conv_dst_md != nullptr) {
                *conv_dst_md = dnnl::memory::desc({ conv_dst_tz }, type, dnnl::memory::format_tag::any);
                *user_dst_md = dnnl::memory::desc({ conv_dst_tz }, type, format);
                user_dst_md->data.format_kind = dnnl_blocked; // overrides "format = isNCDHW ? ncdhw : ndhwc"
                user_dst_md->data.format_desc.blocking.strides[0] = dst->stridesOf()[isNCDHW ? 0 : 0];
                user_dst_md->data.format_desc.blocking.strides[1] = dst->stridesOf()[isNCDHW ? 1 : 4];
                user_dst_md->data.format_desc.blocking.strides[2] = dst->stridesOf()[isNCDHW ? 2 : 1];
                user_dst_md->data.format_desc.blocking.strides[3] = dst->stridesOf()[isNCDHW ? 3 : 2];
                user_dst_md->data.format_desc.blocking.strides[4] = dst->stridesOf()[isNCDHW ? 4 : 3];
            }
        };


        // void getMKLDNNMemoryDescBatchNorm(const NDArray* src, const NDArray* diff_src, const NDArray* dst,
        //                                   dnnl::memory::desc* batchnorm_src_md, dnnl::memory::desc* batchnorm_diff_src_md, dnnl::memory::desc* batchnorm_dst_md,
        //                                   dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_dst_md, int axis) {
        //     const Nd4jLong* shape = src->getShapeInfo();
        //     Nd4jLong rank = shape[0];
        //     Nd4jLong dim1 = axis; // MKL-DNN supports only 1 axis, which has to be the "channel" one
        //     Nd4jLong dim2 = axis >= 2 ? 1 : 2;
        //     Nd4jLong dim3 = axis >= 3 ? 2 : 3;
        //     dnnl::memory::dims batchnorm_src_tz = { (int)shape[1], (int)shape[dim1 + 1], rank > 2 ? (int)shape[dim2 + 1] : 1, rank > 3 ? (int)shape[dim3 + 1] : 1};

        //     auto type = dnnl::memory::data_type::f32;
        //     auto format = dnnl::memory::format_tag::nchw;
        //     auto supposed_to_be_any_format = dnnl::memory::format_tag::nChw8c; // doesn't work with "any"

        //     if (src != nullptr && src->getBuffer() != nullptr && batchnorm_src_md != nullptr) {
        //         *batchnorm_src_md = dnnl::memory::desc({ batchnorm_src_tz }, type, supposed_to_be_any_format);
        //         *user_src_md = dnnl::memory::desc({ batchnorm_src_tz }, type, format);
        //         user_src_md->data.format_kind = dnnl_blocked; // overrides format
        //         user_src_md->data.format_desc.blocking.strides[0] = src->stridesOf()[0];
        //         user_src_md->data.format_desc.blocking.strides[1] = src->stridesOf()[dim1];
        //         user_src_md->data.format_desc.blocking.strides[2] = rank > 2 ? src->stridesOf()[dim2] : 1;
        //         user_src_md->data.format_desc.blocking.strides[3] = rank > 3 ? src->stridesOf()[dim3] : 1;
        //     }

        //     if (diff_src != nullptr && diff_src->getBuffer() != nullptr && batchnorm_diff_src_md != nullptr) {
        //         *batchnorm_diff_src_md = dnnl::memory::desc({ batchnorm_src_tz }, type, supposed_to_be_any_format);
        //         *user_diff_src_md = dnnl::memory::desc({ batchnorm_src_tz }, type, format);
        //         user_diff_src_md->data.format_kind = dnnl_blocked; // overrides format
        //         user_diff_src_md->data.format_desc.blocking.strides[0] = diff_src->stridesOf()[0];
        //         user_diff_src_md->data.format_desc.blocking.strides[1] = diff_src->stridesOf()[dim1];
        //         user_diff_src_md->data.format_desc.blocking.strides[2] = rank > 2 ? diff_src->stridesOf()[dim2] : 1;
        //         user_diff_src_md->data.format_desc.blocking.strides[3] = rank > 3 ? diff_src->stridesOf()[dim3] : 1;
        //     }

        //     if (dst != nullptr && dst->getBuffer() != nullptr && batchnorm_dst_md != nullptr) {
        //         *batchnorm_dst_md = dnnl::memory::desc({ batchnorm_src_tz }, type, supposed_to_be_any_format);
        //         *user_dst_md = dnnl::memory::desc({ batchnorm_src_tz }, type, format);
        //         user_dst_md->data.format_kind = dnnl_blocked; // overrides format
        //         user_dst_md->data.format_desc.blocking.strides[0] = dst->stridesOf()[0];
        //         user_dst_md->data.format_desc.blocking.strides[1] = dst->stridesOf()[dim1];
        //         user_dst_md->data.format_desc.blocking.strides[2] = rank > 2 ? dst->stridesOf()[dim2] : 1;
        //         user_dst_md->data.format_desc.blocking.strides[3] = rank > 3 ? dst->stridesOf()[dim3] : 1;
        //     }
        // };


        void getMKLDNNMemoryDescLrn(const NDArray* src, const NDArray* diff_src, const NDArray* dst,
                                    dnnl::memory::desc* lrn_src_md, dnnl::memory::desc* lrn_diff_src_md, dnnl::memory::desc* lrn_dst_md,
                                    dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_dst_md, int axis) {
            const Nd4jLong* shape = src->getShapeInfo();
            long rank = shape[0];
            long dim1 = axis; // MKL-DNN supports only 1 axis, which has to be the "channel" one
            long dim2 = axis >= 2 ? 1 : 2;
            long dim3 = axis >= 3 ? 2 : 3;
            dnnl::memory::dims lrn_src_tz = { (int)shape[1], (int)shape[dim1 + 1], rank > 2 ? (int)shape[dim2 + 1] : 1, rank > 3 ? (int)shape[dim3 + 1] : 1};

            auto type = dnnl::memory::data_type::f32;
            auto format = axis == 1 ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
            auto supposed_to_be_any_format = format; // doesn't work with "any"

            if (src != nullptr && src->getBuffer() != nullptr && lrn_src_md != nullptr) {
                *lrn_src_md = dnnl::memory::desc({ lrn_src_tz }, type, supposed_to_be_any_format);
                *user_src_md = dnnl::memory::desc({ lrn_src_tz }, type, format);
                user_src_md->data.format_kind = dnnl_blocked;
                user_src_md->data.format_desc.blocking.strides[0] = src->stridesOf()[0];
                user_src_md->data.format_desc.blocking.strides[1] = src->stridesOf()[dim1];
                user_src_md->data.format_desc.blocking.strides[2] = rank > 2 ? src->stridesOf()[dim2] : 1;
                user_src_md->data.format_desc.blocking.strides[3] = rank > 3 ? src->stridesOf()[dim3] : 1;
            }

            if (diff_src != nullptr && diff_src->getBuffer() != nullptr && lrn_diff_src_md != nullptr) {
                *lrn_diff_src_md = dnnl::memory::desc({ lrn_src_tz }, type, supposed_to_be_any_format);
                *user_diff_src_md = dnnl::memory::desc({ lrn_src_tz }, type, format);
                user_diff_src_md->data.format_kind = dnnl_blocked;
                user_diff_src_md->data.format_desc.blocking.strides[0] = diff_src->stridesOf()[0];
                user_diff_src_md->data.format_desc.blocking.strides[1] = diff_src->stridesOf()[dim1];
                user_diff_src_md->data.format_desc.blocking.strides[2] = rank > 2 ? diff_src->stridesOf()[dim2] : 1;
                user_diff_src_md->data.format_desc.blocking.strides[3] = rank > 3 ? diff_src->stridesOf()[dim3] : 1;
            }

            if (dst != nullptr && dst->getBuffer() != nullptr && lrn_dst_md != nullptr) {
                *lrn_dst_md = dnnl::memory::desc({ lrn_src_tz }, type, supposed_to_be_any_format);
                *user_dst_md = dnnl::memory::desc({ lrn_src_tz }, type, format);
                user_dst_md->data.format_kind = dnnl_blocked;
                user_dst_md->data.format_desc.blocking.strides[0] = dst->stridesOf()[0];
                user_dst_md->data.format_desc.blocking.strides[1] = dst->stridesOf()[dim1];
                user_dst_md->data.format_desc.blocking.strides[2] = rank > 2 ? dst->stridesOf()[dim2] : 1;
                user_dst_md->data.format_desc.blocking.strides[3] = rank > 3 ? dst->stridesOf()[dim3] : 1;
            }
        }

        dnnl::engine& getEngine(void *ptr) {
            auto eng = reinterpret_cast<dnnl::engine*>(ptr);
            return *eng;
        }
    }
}