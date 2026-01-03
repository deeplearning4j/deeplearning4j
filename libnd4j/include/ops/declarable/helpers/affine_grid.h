/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Eclipse Deeplearning4j Development Team
//

#ifndef LIBND4J_HELPERS_AFFINE_GRID_H
#define LIBND4J_HELPERS_AFFINE_GRID_H

#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Generate affine sampling grid from transformation matrices
 *
 * @param context Launch context
 * @param theta Affine transformation matrix [N, 2, 3] for 2D or [N, 3, 4] for 3D
 * @param size Target size tensor [N, C, H, W] or [N, C, D, H, W]
 * @param alignCorners Whether to align corners
 * @param output Output grid [N, H, W, 2] or [N, D, H, W, 3]
 */
SD_LIB_EXPORT void affineGrid(LaunchContext* context, NDArray* theta, NDArray* size,
                               bool alignCorners, NDArray* output);

/**
 * Grid sample with interpolation
 *
 * @param context Launch context
 * @param input Input tensor [N, C, H, W] or [N, C, D, H, W]
 * @param grid Sampling grid [N, H_out, W_out, 2] or [N, D_out, H_out, W_out, 3]
 * @param mode Interpolation mode (0=bilinear, 1=nearest, 2=bicubic)
 * @param paddingMode Padding mode (0=zeros, 1=border, 2=reflection)
 * @param alignCorners Whether to align corners
 * @param output Output tensor
 */
SD_LIB_EXPORT void gridSample(LaunchContext* context, NDArray* input, NDArray* grid,
                               int mode, int paddingMode, bool alignCorners, NDArray* output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_AFFINE_GRID_H
