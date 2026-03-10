/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
* the License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#ifndef NATIVEOPSHELPERS_DATABUFFERS_CUDA_H
#define NATIVEOPSHELPERS_DATABUFFERS_CUDA_H

#include <system/common.h>

/**
 * Query the actual CUDA device where the given GPU pointer resides.
 *
 * @param specialPtr  The device (special) pointer to query.
 * @param outDeviceId On success, set to the CUDA device id owning the pointer.
 * @return true if the device was successfully determined, false otherwise
 *         (e.g., pointer is null, not a device pointer, or query failed).
 */
#ifdef SD_CUDA
bool dbDeviceId_cudaQuery(const void* specialPtr, int& outDeviceId);
#else
// CPU stub — always returns false so the caller falls through to metadata-based id.
inline bool dbDeviceId_cudaQuery(const void* /*specialPtr*/, int& /*outDeviceId*/) {
  return false;
}
#endif

#endif // NATIVEOPSHELPERS_DATABUFFERS_CUDA_H
