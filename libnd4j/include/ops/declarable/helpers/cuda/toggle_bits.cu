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
// @author raver119@gmail.com
//
#include <helpers/BitwiseUtils.h>
#include <ops/declarable/helpers/toggle_bits.h>


namespace sd {
namespace ops {
namespace helpers {

template <typename T>
SD_KERNEL void toggleBitsKernel(const void* inputBuf, void* outputBuf, sd::LongType length) {
  auto input = reinterpret_cast<const T*>(inputBuf);
  auto output = reinterpret_cast<T*>(outputBuf);
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (sd::LongType i = tid; i < length; i += blockDim.x * gridDim.x) {
    output[i] = ~input[i];
  }
}

template <typename T>
void toggle_bits__(LaunchContext* context, NDArray *in, NDArray *out) {
  auto stream = context->getCudaStream();
  auto length = in->lengthOf();
  int threads = 256;
  int blocks = (length + threads - 1) / threads;
  if (blocks > 1024) blocks = 1024;
  toggleBitsKernel<T><<<blocks, threads, 0, *stream>>>(
      in->specialBuffer(), out->specialBuffer(), length);
}
BUILD_SINGLE_TEMPLATE( void toggle_bits__, (LaunchContext* context, NDArray* in, NDArray* out), SD_INTEGER_TYPES);

void __toggle_bits(LaunchContext *context, NDArray *in, NDArray *out) {
  BUILD_SINGLE_SELECTOR(in->dataType(), toggle_bits__, (context, in, out), SD_INTEGER_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
