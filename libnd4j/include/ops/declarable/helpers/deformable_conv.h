/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//
// Reference: "Deformable Convolutional Networks" (Dai et al., 2017)
//            "Deformable ConvNets v2" (Zhu et al., 2019)
//

#ifndef LIBND4J_HELPERS_DEFORMABLE_CONV_H
#define LIBND4J_HELPERS_DEFORMABLE_CONV_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * @brief Deformable Convolution 2D forward pass
 *
 * Implements deformable convolution where learned offsets are added to
 * the regular sampling grid, allowing the convolution to adapt to
 * geometric transformations in the input.
 *
 * @param context Launch context for execution
 * @param input Input tensor [batch, channels, height, width] (NCHW)
 * @param weights Convolution weights [out_channels, in_channels/groups, kH, kW]
 * @param offset Learned offsets [batch, 2*kH*kW*offset_groups, out_h, out_w]
 * @param bias Bias tensor [out_channels] (can be nullptr)
 * @param mask Modulation mask [batch, kH*kW*offset_groups, out_h, out_w] (can be nullptr, for v2)
 * @param output Output tensor [batch, out_channels, out_h, out_w]
 * @param kH Kernel height
 * @param kW Kernel width
 * @param sH Stride height
 * @param sW Stride width
 * @param pH Padding height
 * @param pW Padding width
 * @param dH Dilation height
 * @param dW Dilation width
 * @param groups Number of convolution groups
 * @param offsetGroups Number of offset/deformable groups
 */
SD_LIB_HIDDEN void deformableConv2d(sd::LaunchContext* context,
                                     NDArray* input,
                                     NDArray* weights,
                                     NDArray* offset,
                                     NDArray* bias,
                                     NDArray* mask,
                                     NDArray* output,
                                     int kH, int kW,
                                     int sH, int sW,
                                     int pH, int pW,
                                     int dH, int dW,
                                     int groups, int offsetGroups);

/**
 * @brief Bilinear interpolation helper for deformable convolution
 *
 * Samples a value from the input at a fractional position using bilinear interpolation.
 *
 * @param input Input tensor
 * @param batch Batch index
 * @param channel Channel index
 * @param y Y coordinate (fractional)
 * @param x X coordinate (fractional)
 * @param height Input height
 * @param width Input width
 * @return Interpolated value
 */
template <typename T>
SD_HOST_DEVICE T bilinearInterpolate(const T* input,
                                      sd::LongType batch,
                                      sd::LongType channel,
                                      double y, double x,
                                      sd::LongType height, sd::LongType width,
                                      sd::LongType batchStride,
                                      sd::LongType channelStride,
                                      sd::LongType heightStride,
                                      sd::LongType widthStride);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_DEFORMABLE_CONV_H
