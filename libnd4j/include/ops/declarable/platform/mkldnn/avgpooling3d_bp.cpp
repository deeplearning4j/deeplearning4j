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
            PLATFORM_IMPL(avgpool3dnew_bp) {
                auto input = INPUT_VARIABLE(
                        0);                          // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
                auto gradO = INPUT_VARIABLE(
                        1);                          // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
                auto gradI = OUTPUT_VARIABLE(
                        0);                         // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon

                const int kD = INT_ARG(0);                                                  // filter(kernel) depth
                const int kH = INT_ARG(1);                                                  // filter(kernel) height
                const int kW = INT_ARG(2);                                                  // filter(kernel) width
                const int sD = INT_ARG(3);                                                  // strides depth
                const int sH = INT_ARG(4);                                                  // strides height
                const int sW = INT_ARG(5);                                                  // strides width
                int pD = INT_ARG(6);                                                  // paddings depth
                int pH = INT_ARG(7);                                                  // paddings height
                int pW = INT_ARG(8);                                                  // paddings width
                const int dD = INT_ARG(9);                                                  // dilations depth
                const int dH = INT_ARG(10);                                                 // dilations height
                const int dW = INT_ARG(11);                                                 // dilations width
                const int isSameMode = INT_ARG(12);                                         // 1-SAME,  0-VALID
                int extraParam0 = INT_ARG(13);                                           // unnecessary for max case, required only for avg and pnorm cases
                int isNCDHW = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;       // 1-NDHWC, 0-NCDHW

                REQUIRE_TRUE(input->rankOf() == 5, 0,
                             "MAXPOOL3D_BP op: input should have rank of 5, but got %i instead", input->rankOf());
                REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0,
                             "MAXPOOL3DNEW op: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

                int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
                int indIOioC, indIOioD, indWoC, indWiC, indWkD;       // corresponding indexes
                ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                                           indIOioC, indIOioD, indWiC, indWoC, indWkD);

                std::string expectedGradOShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx(
                        {bS, iC, oD, oH, oW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2}));
                std::string expectedGradIShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx(
                        {bS, iC, iD, iH, iW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2}));
                REQUIRE_TRUE(expectedGradOShape == ShapeUtils::shapeAsString(gradO), 0,
                             "MAXPOOL3D_BP op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s instead !",
                             expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradO).c_str());
                REQUIRE_TRUE(expectedGradIShape == ShapeUtils::shapeAsString(gradI), 0,
                             "MAXPOOL3D_BP op: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !",
                             expectedGradIShape.c_str(), ShapeUtils::shapeAsString(gradI).c_str());

                if (!isNCDHW) {
                    input = new NDArray(input->permute(
                            {0, 4, 1, 2, 3}));                   // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
                    gradI = new NDArray(gradI->permute(
                            {0, 4, 1, 2, 3}));                   // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
                    gradO = new NDArray(gradO->permute(
                            {0, 4, 1, 2, 3}));                   // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]
                }

                if (isSameMode)                       // SAME
                    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);



                auto poolingMode = PoolingType::AVG_POOL;

                mkldnn_memory_desc_t empty;
                mkldnn::memory::desc pool_src_md(empty), pool_diff_src_md(empty), pool_dst_md(empty);
                mkldnn::memory::desc user_src_md(empty), user_diff_src_md(empty), user_dst_md(empty);
                mkldnn::memory::dims pool_strides, pool_kernel, pool_padding, pool_padding_r;
                mkldnn::algorithm algorithm;
                mkldnnUtils::getMKLDNNMemoryDescPool3d(kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode,
                                                       extraParam0, true,
                                                       bS, iC, iD, iH, iW, oC, oD, oH, oW, input, gradI, gradO,
                                                       algorithm,
                                                       &pool_src_md, &pool_diff_src_md, &pool_dst_md, &user_src_md,
                                                       &user_diff_src_md, &user_dst_md,
                                                       pool_strides, pool_kernel, pool_padding, pool_padding_r);
                if (input->buffer() == nullptr) {
                    pool_src_md = pool_diff_src_md;
                    user_src_md = user_diff_src_md;
                }
                auto pool_desc = pooling_forward::desc(prop_kind::forward, algorithm, pool_src_md, pool_dst_md,
                                                       pool_strides, pool_kernel, pool_padding, pool_padding_r);
                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());
                mkldnn::stream stream(engine);
                auto pool_prim_desc = pooling_forward::primitive_desc(pool_desc, engine);
                auto poolB_desc = pooling_backward::desc(algorithm, pool_diff_src_md, pool_dst_md, pool_strides,
                                                         pool_kernel, pool_padding, pool_padding_r);
                auto poolB_prim_desc = pooling_backward::primitive_desc(poolB_desc, engine, pool_prim_desc);
                auto userB_src_memory = mkldnn::memory(user_diff_src_md, engine, gradI->buffer());
                auto userB_dst_memory = mkldnn::memory(user_dst_md, engine, gradO->buffer());
                auto poolB_src_memory = userB_src_memory;
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

            PLATFORM_CHECK(avgpool3dnew_bp) {
                auto input = INPUT_VARIABLE(0);
                auto output = OUTPUT_VARIABLE(0);

                return block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, output});
            }
        }
    }
}