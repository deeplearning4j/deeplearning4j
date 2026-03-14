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

#ifndef LIBND4J_KV_CACHE_QUANTIZE_H
#define LIBND4J_KV_CACHE_QUANTIZE_H

#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {

// Quantization format enum
enum class KVQuantFormat : int {
    INT8 = 0,
    FP8_E4M3 = 1,
    FP8_E5M2 = 2,
    INT4 = 3
};

SD_LIB_HIDDEN void kvCacheQuantizeCpu(
    NDArray* input,       // float KV data [...]
    NDArray* quantized,   // output quantized data [...]
    NDArray* scales,      // output per-channel scales [...]
    int quantFormat);     // KVQuantFormat

SD_LIB_HIDDEN void kvCacheDequantizeCpu(
    NDArray* quantized,   // quantized data [...]
    NDArray* scales,      // per-channel scales [...]
    NDArray* output,      // float output [...]
    int quantFormat);

#if defined(SD_CUDA)
SD_LIB_HIDDEN void kvCacheQuantizeCuda(
    NDArray* input, NDArray* quantized, NDArray* scales,
    int quantFormat, LaunchContext* context);

SD_LIB_HIDDEN void kvCacheDequantizeCuda(
    NDArray* quantized, NDArray* scales, NDArray* output,
    int quantFormat, LaunchContext* context);
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
