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
// @author Adam Gibson
//

#ifndef LIBND4J_HELPERS_SIGNAL_H
#define LIBND4J_HELPERS_SIGNAL_H

#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Discrete Fourier Transform
 *
 * @param context Launch context
 * @param input Complex input tensor [..., N, 2]
 * @param axis Axis along which to compute DFT
 * @param inverse If true, compute inverse DFT
 * @param onesided If true, return only positive frequencies
 * @param output Complex output tensor [..., M, 2]
 */
SD_LIB_EXPORT void dft(LaunchContext* context, NDArray* input, int axis,
                        bool inverse, bool onesided, NDArray* output);

/**
 * Short-Time Fourier Transform
 *
 * @param context Launch context
 * @param signal Input signal [batch, signal_length]
 * @param frameStep Number of samples between frames
 * @param window Window function (can be nullptr)
 * @param frameLength Length of each frame
 * @param onesided If true, return only positive frequencies
 * @param output Complex output [batch, num_frames, num_bins, 2]
 */
SD_LIB_EXPORT void stft(LaunchContext* context, NDArray* signal, int frameStep,
                         NDArray* window, int frameLength, bool onesided, NDArray* output);

/**
 * Generate Hann window
 *
 * @param context Launch context
 * @param size Window size
 * @param periodic If true, generate periodic window
 * @param output Output window tensor
 */
SD_LIB_EXPORT void hannWindow(LaunchContext* context, int size, bool periodic, NDArray* output);

/**
 * Generate Hamming window
 *
 * @param context Launch context
 * @param size Window size
 * @param periodic If true, generate periodic window
 * @param output Output window tensor
 */
SD_LIB_EXPORT void hammingWindow(LaunchContext* context, int size, bool periodic, NDArray* output);

/**
 * Generate Blackman window
 *
 * @param context Launch context
 * @param size Window size
 * @param periodic If true, generate periodic window
 * @param output Output window tensor
 */
SD_LIB_EXPORT void blackmanWindow(LaunchContext* context, int size, bool periodic, NDArray* output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_SIGNAL_H
