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

#ifndef LIBND4J_HEADERS_SIGNAL_H
#define LIBND4J_HEADERS_SIGNAL_H

#include <ops/declarable/headers/common.h>

namespace sd {
namespace ops {

/**
 * Discrete Fourier Transform (DFT)
 *
 * Computes the Discrete Fourier Transform of a complex input tensor.
 *
 * Input arrays:
 *   0 - input: Complex tensor with shape [..., N, 2] where last dim is [real, imag]
 *   1 - n: Optional, DFT length (if different from input size)
 *
 * Int arguments:
 *   0 - axis: The axis along which to compute DFT (default: -2, second to last)
 *   1 - inverse: If 1, compute inverse DFT (default: 0)
 *   2 - onesided: If 1, return only positive frequencies (default: 0)
 *
 * Output:
 *   0 - Complex tensor with DFT result, shape [..., M, 2]
 *       where M = N if not onesided, M = N/2+1 if onesided
 */
#if NOT_EXCLUDED(OP_dft)
DECLARE_CUSTOM_OP(dft, 1, 1, false, 0, -2);
#endif

/**
 * Short-Time Fourier Transform (STFT)
 *
 * Computes STFT by applying DFT to windowed overlapping segments.
 *
 * Input arrays:
 *   0 - signal: Input signal tensor [batch, signal_length] or [batch, signal_length, 1]
 *   1 - frame_step: Scalar, number of samples between frames
 *   2 - window: Optional window function [frame_length]
 *   3 - frame_length: Optional scalar, length of each frame
 *
 * Int arguments:
 *   0 - onesided: If 1, return only positive frequencies (default: 1)
 *
 * Output:
 *   0 - Complex tensor [batch, num_frames, num_bins, 2]
 */
#if NOT_EXCLUDED(OP_stft)
DECLARE_CUSTOM_OP(stft, 2, 1, false, 0, -2);
#endif

/**
 * Hann Window
 *
 * Generates a Hann window function.
 *
 * Input arrays:
 *   0 - size: Scalar, window length
 *
 * Int arguments:
 *   0 - periodic: If 1, return periodic window (default: 1)
 *
 * Output:
 *   0 - Window tensor [size]
 */
#if NOT_EXCLUDED(OP_hann_window)
DECLARE_CUSTOM_OP(hann_window, 1, 1, false, 0, -2);
#endif

/**
 * Hamming Window
 *
 * Generates a Hamming window function.
 *
 * Input arrays:
 *   0 - size: Scalar, window length
 *
 * Int arguments:
 *   0 - periodic: If 1, return periodic window (default: 1)
 *
 * Output:
 *   0 - Window tensor [size]
 */
#if NOT_EXCLUDED(OP_hamming_window)
DECLARE_CUSTOM_OP(hamming_window, 1, 1, false, 0, -2);
#endif

/**
 * Blackman Window
 *
 * Generates a Blackman window function.
 *
 * Input arrays:
 *   0 - size: Scalar, window length
 *
 * Int arguments:
 *   0 - periodic: If 1, return periodic window (default: 1)
 *
 * Output:
 *   0 - Window tensor [size]
 */
#if NOT_EXCLUDED(OP_blackman_window)
DECLARE_CUSTOM_OP(blackman_window, 1, 1, false, 0, -2);
#endif

}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HEADERS_SIGNAL_H
