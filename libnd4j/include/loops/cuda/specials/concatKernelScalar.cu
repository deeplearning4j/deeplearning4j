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
// @author Yurii Shyrma, created on 15.11.2018
//
#include <loops/special_kernels.h>

namespace sd {

///////////////////////////////////////////////////////////////////////
template <typename T>
SD_DEVICE void concatKernelScalar(int numArrays, Pointer *data, void *vz) {
  auto z = static_cast<T *>(vz);
  LongType tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto input = reinterpret_cast<T **>(data);

  for (int i = tid; i < numArrays; i += blockDim.x * gridDim.x) z[i] = input[i][0];
}

///////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void execConcatKernelScalar(int numArrays, Pointer *data, void *vz) {
  concatKernelScalar<T>(numArrays, data, vz);
}

///////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void concatKernelScalarGeneric(dim3 &launchDims, cudaStream_t *stream, int numArrays, Pointer *data,
                                       void *vz) {
  execConcatKernelScalar<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(numArrays, data, vz);
  DebugHelper::checkErrorCode(stream, "concatScalar(...) failed");
}

BUILD_SINGLE_TEMPLATE(template void concatKernelScalarGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, int numArrays, sd::Pointer *data, void *vz),
                      SD_COMMON_TYPES);
}  // namespace sd
