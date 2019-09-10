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
// @author raver119@gmail.com
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <platform_boilerplate.h>

#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

using namespace mkldnn;

namespace nd4j {
    namespace ops {
        namespace platforms {
            PLATFORM_IMPL(avgpool2d_bp) {
                auto input = INPUT_VARIABLE(
                        0);                          // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
                auto gradO = INPUT_VARIABLE(
                        1);                          // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
                auto gradI = OUTPUT_VARIABLE(
                        0);                         // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

                int kH = INT_ARG(0);                                                        // filter(kernel) height
                int kW = INT_ARG(1);                                                        // filter(kernel) width
                int sH = INT_ARG(2);                                                        // strides height
                int sW = INT_ARG(3);                                                        // strides width
                int pH = INT_ARG(4);                                                        // paddings height
                int pW = INT_ARG(5);                                                        // paddings width
                int dH = INT_ARG(6);                                                        // dilations height
                int dW = INT_ARG(7);                                                        // dilations width
                int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
                int extraParam0 = INT_ARG(9);
                int isNCHW =
                        block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;         // INT_ARG(10): 0-NCHW, 1-NHWC

                REQUIRE_TRUE(input->rankOf() == 4, 0,
                             "AVGPOOL2D_BP op: input should have rank of 4, but got %i instead", input->rankOf());
                REQUIRE_TRUE(dH != 0 && dW != 0, 0,
                             "AVGPOOL2D_BP op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

                int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
                int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
                ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                                           indIiH, indWiC, indWoC, indWkH, indOoH);

                std::string expectedGradOShape = ShapeUtils::shapeAsString(
                        ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, oH, oW, 0, indIOioC, indIiH, indIiH + 1}));
                std::string expectedGradIShape = ShapeUtils::shapeAsString(
                        ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, iH, iW, 0, indIOioC, indIiH, indIiH + 1}));
                REQUIRE_TRUE(expectedGradOShape == ShapeUtils::shapeAsString(gradO), 0,
                             "AVGPOOL2D_BP op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s instead !",
                             expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradO).c_str());
                REQUIRE_TRUE(expectedGradIShape == ShapeUtils::shapeAsString(gradI), 0,
                             "AVGPOOL2D_BP op: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !",
                             expectedGradIShape.c_str(), ShapeUtils::shapeAsString(gradI).c_str());


                if (!isNCHW) {
                    input = new NDArray(input->permute(
                            {0, 3, 1, 2}));                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
                    gradI = new NDArray(gradI->permute(
                            {0, 3, 1, 2}));                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
                    gradO = new NDArray(gradO->permute(
                            {0, 3, 1, 2}));                                   // [bS, oH, oW, iC] -> [bS, iC, oH, oW]
                }

                if (isSameMode)                       // SAME
                    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

                auto poolingMode = PoolingType::AVG_POOL;

                mkldnn_memory_desc_t empty;
                mkldnn::memory::desc pool_src_md(empty), pool_diff_src_md(empty), pool_dst_md(empty);
                mkldnn::memory::desc user_src_md(empty), user_diff_src_md(empty), user_dst_md(empty);
                mkldnn::memory::dims pool_strides, pool_kernel, pool_padding, pool_padding_r;
                mkldnn::algorithm algorithm;
                mkldnnUtils::getMKLDNNMemoryDescPool2d(kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0,
                                                       true,
                                                       bS, iC, iH, iW, oC, oH, oW, input, gradI, gradO, algorithm,
                                                       &pool_src_md, &pool_diff_src_md, &pool_dst_md, &user_src_md,
                                                       &user_diff_src_md, &user_dst_md,
                                                       pool_strides, pool_kernel, pool_padding, pool_padding_r);
                auto pool_desc = pooling_forward::desc(prop_kind::forward, algorithm,
                                                       input->buffer() != nullptr ? pool_src_md : pool_diff_src_md,
                                                       pool_dst_md, pool_strides, pool_kernel, pool_padding,
                                                       pool_padding_r);
                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());
                auto pool_prim_desc = pooling_forward::primitive_desc(pool_desc, engine);
                auto poolB_desc = pooling_backward::desc(algorithm, pool_diff_src_md, pool_dst_md, pool_strides,
                                                         pool_kernel, pool_padding, pool_padding_r);
                auto poolB_prim_desc = pooling_backward::primitive_desc(poolB_desc, engine, pool_prim_desc);
                auto userB_src_memory = mkldnn::memory(user_src_md, engine, gradI->buffer());
                auto userB_dst_memory = mkldnn::memory(user_dst_md, engine, gradO->buffer());
                auto poolB_src_memory = userB_src_memory;
                mkldnn::stream stream(engine);
                if (poolB_prim_desc.diff_src_desc() != userB_src_memory.get_desc()) {
                    poolB_src_memory = mkldnn::memory(poolB_prim_desc.diff_src_desc(), engine);
                }
                auto poolB_dst_memory = userB_dst_memory;
                if (poolB_prim_desc.diff_dst_desc() != userB_dst_memory.get_desc()) {
                    poolB_dst_memory = mkldnn::memory(poolB_prim_desc.diff_dst_desc(), engine);
                    reorder(userB_dst_memory, poolB_dst_memory).execute(stream, userB_dst_memory, poolB_dst_memory);
                }
                pooling_backward(poolB_prim_desc).execute(stream, {{MKLDNN_ARG_DIFF_DST, poolB_dst_memory},
                                                                   {MKLDNN_ARG_DIFF_SRC, poolB_src_memory}});
                if (poolB_prim_desc.diff_src_desc() != userB_src_memory.get_desc()) {
                    reorder(poolB_src_memory, userB_src_memory).execute(stream, poolB_src_memory, userB_src_memory);
                }
                stream.wait();

                return Status::OK();
            }

            PLATFORM_CHECK(avgpool2d_bp) {
                auto input = INPUT_VARIABLE(0);
                auto output = OUTPUT_VARIABLE(0);

                return block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, output});
            }
        }
    }
}