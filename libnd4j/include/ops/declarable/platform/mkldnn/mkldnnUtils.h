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
 // @author Yurii Shyrma (iuriish@yahoo.com)
 //

#ifndef DEV_TESTS_MKLDNNUTILS_H
#define DEV_TESTS_MKLDNNUTILS_H


#include <legacy/NativeOps.h>
#include <array/NDArray.h>
#include <dnnl.hpp>
#include <helpers/MKLDNNStream.h>
#include <graph/Context.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

using namespace samediff;


namespace sd {
    namespace ops {
        namespace platforms {
            /**
             * Here we actually declare our platform helpers
             */
            DECLARE_PLATFORM(conv2d, ENGINE_CPU);

            DECLARE_PLATFORM(conv2d_bp, ENGINE_CPU);

            DECLARE_PLATFORM(avgpool2d, ENGINE_CPU);

            DECLARE_PLATFORM(avgpool2d_bp, ENGINE_CPU);

            DECLARE_PLATFORM(maxpool2d, ENGINE_CPU);

            DECLARE_PLATFORM(maxpool2d_bp, ENGINE_CPU);

            DECLARE_PLATFORM(conv3dnew, ENGINE_CPU);

            DECLARE_PLATFORM(conv3dnew_bp, ENGINE_CPU);

            DECLARE_PLATFORM(maxpool3dnew, ENGINE_CPU);

            DECLARE_PLATFORM(maxpool3dnew_bp, ENGINE_CPU);

            DECLARE_PLATFORM(avgpool3dnew, ENGINE_CPU);

            DECLARE_PLATFORM(avgpool3dnew_bp, ENGINE_CPU);

            DECLARE_PLATFORM(lrn, ENGINE_CPU);

            DECLARE_PLATFORM(batchnorm, ENGINE_CPU);

            DECLARE_PLATFORM(batchnorm_bp, ENGINE_CPU);

            DECLARE_PLATFORM(lstmLayer, ENGINE_CPU);

            DECLARE_PLATFORM(deconv2d, ENGINE_CPU);

            DECLARE_PLATFORM(deconv2d_tf, ENGINE_CPU);

            DECLARE_PLATFORM(deconv3d, ENGINE_CPU);

            DECLARE_PLATFORM(deconv2d_bp, ENGINE_CPU);

            DECLARE_PLATFORM(deconv3d_bp, ENGINE_CPU);

            DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CPU);

            DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_CPU);

            DECLARE_PLATFORM(matmul, ENGINE_CPU);

            DECLARE_PLATFORM(softmax, ENGINE_CPU);

            DECLARE_PLATFORM(softmax_bp, ENGINE_CPU);

            DECLARE_PLATFORM(tanh, ENGINE_CPU);

            DECLARE_PLATFORM(tanh_bp, ENGINE_CPU);

        }
    }

    namespace mkldnnUtils {

        void poolingMKLDNN(const NDArray* input, NDArray* output, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int isNCHW, const dnnl::algorithm mode);

        void poolingBpMKLDNN(const NDArray* input, const NDArray* gradO, NDArray* gradI, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW, const int isNCHW, const dnnl::algorithm mode);

        void getMKLDNNMemoryDescLrn(const NDArray* src, const NDArray* diff_src, const NDArray* dst,
            dnnl::memory::desc* lrn_src_md, dnnl::memory::desc* lrn_diff_src_md, dnnl::memory::desc* lrn_dst_md,
            dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_dst_md, int axis);

        dnnl::engine& getEngine(void* ptr);

        /**
        * This function creates memory dimentions
        * @param const pointer to array
        * @param const array rank
        * @param reference to memory dimentions
        */
        void getDims(const NDArray* array, const int rank, dnnl::memory::dims& mklDims);
        /**
         * This function generate memory format tag based on rank
         * @param const array rank
         * @return memory format
         */
        dnnl::memory::format_tag   getFormat(const int rank);
        /**
         * This function generate memory format tag based on rank
         * @param const pointer to dataset
         * @param const dataset rank
         * @param reference to memory descriptor
         * @return memory format
         */
        void   setBlockStrides(const NDArray* array, const int rank, dnnl::memory::desc& mklMd);
        //////////////////////////////////////////////////////////////////////
        /**
        * This function load and reorder user memory to mkl
        * @param const pointer to dataset
        * @param reference to mkl engine
        * @param reference to mkl stream
        * @param reference to args container for dnnl
        * @param reference to user memory description
        * @param primitive memory descriptor
        * @param dnnl arg activation enumerator
        */
        void loadDataToMklStream(const NDArray* array, dnnl::engine& engine, dnnl::stream& stream,
             std::unordered_map<int, dnnl::memory>& args, dnnl::memory::desc& user_md, dnnl::memory::desc primitive_md, int DNNL_ARG);

        /**
         * Utility methods for MKLDNN
         */
         /*        void getMKLDNNMemoryDescConv2d(
                         int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, const int paddingMode, bool isNCHW,
                         int bS, int iC, int iH, int iW, int oC, int oH, int oW, const NDArray* src, const NDArray* diff_src,
                         const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
                         dnnl::memory::desc* conv_src_md, dnnl::memory::desc* conv_diff_src_md, dnnl::memory::desc* conv_weights_md,
                         dnnl::memory::desc* conv_diff_weights_md, dnnl::memory::desc* conv_bias_md, dnnl::memory::desc* conv_dst_md,
                         dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_weights_md,
                         dnnl::memory::desc* user_diff_weights_md, dnnl::memory::desc* user_bias_md, dnnl::memory::desc* user_dst_md,
                         dnnl::memory::dims& conv_strides, dnnl::memory::dims& conv_padding, dnnl::memory::dims& conv_padding_r, dnnl::memory::dims& conv_dilation);

                 void getMKLDNNMemoryDescConv3d(
                         int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, bool isSameMode, bool isNCDHW,
                         int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW, const NDArray* src, const NDArray* diff_src,
                         const NDArray* weights, const NDArray* diff_weights, const NDArray* bias, const NDArray* dst,
                         dnnl::memory::desc* conv_src_md, dnnl::memory::desc* conv_diff_src_md, dnnl::memory::desc* conv_weights_md,
                         dnnl::memory::desc* conv_diff_weights_md, dnnl::memory::desc* conv_bias_md, dnnl::memory::desc* conv_dst_md,
                         dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_weights_md,
                         dnnl::memory::desc* user_diff_weights_md, dnnl::memory::desc* user_bias_md, dnnl::memory::desc* user_dst_md,
                         dnnl::memory::dims& conv_strides, dnnl::memory::dims& conv_padding, dnnl::memory::dims& conv_padding_r, dnnl::memory::dims& conv_dilation);

                 void getMKLDNNMemoryDescPool2d(
                         int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int poolingMode, int extraParam0, bool isNCHW,
                         int bS, int iC, int iH, int iW, int oC, int oH, int oW,
                         const NDArray* src, const NDArray* diff_src, const NDArray* dst, dnnl::algorithm& algorithm,
                         dnnl::memory::desc* pool_src_md, dnnl::memory::desc* pool_diff_src_md, dnnl::memory::desc* pool_dst_md,
                         dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_dst_md,
                         dnnl::memory::dims& pool_strides, dnnl::memory::dims& pool_kernel, dnnl::memory::dims& pool_padding, dnnl::memory::dims& pool_padding_r);

                 void getMKLDNNMemoryDescPool3d(
                         int kD, int kH, int kW, int sD, int sH, int sW, int pD, int pH, int pW, int dD, int dH, int dW, int poolingMode, int extraParam0, bool isNCDHW,
                         int bS, int iC, int iD, int iH, int iW, int oC, int oD, int oH, int oW,
                         const NDArray* src, const NDArray* diff_src, const NDArray* dst, dnnl::algorithm& algorithm,
                         dnnl::memory::desc* pool_src_md, dnnl::memory::desc* pool_diff_src_md, dnnl::memory::desc* pool_dst_md,
                         dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_dst_md,
                         dnnl::memory::dims& pool_strides, dnnl::memory::dims& pool_kernel, dnnl::memory::dims& pool_padding, dnnl::memory::dims& pool_padding_r);

                 void getMKLDNNMemoryDescBatchNorm(const NDArray* src, const NDArray* diff_src, const NDArray* dst,
                                                   dnnl::memory::desc* batchnorm_src_md, dnnl::memory::desc* batchnorm_diff_src_md, dnnl::memory::desc* batchnorm_dst_md,
                                                   dnnl::memory::desc* user_src_md, dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_dst_md, int axis);
         */
    }
}



#endif //DEV_TESTS_MKLDNNUTILS_H
