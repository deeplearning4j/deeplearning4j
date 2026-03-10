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

#pragma once

#include <system/common.h>
#include <utility>

namespace sd {

#ifdef SD_CUDA

/**
 * Set the current CUDA device. Returns true on success.
 * @param deviceId the device to activate
 * @param deviceCount total number of CUDA devices
 */
bool Environment_setCudaCurrentDevice_cuda(int deviceId, int deviceCount);

/**
 * Configure shared memory for tensor core usage on the given device.
 * @param enabled whether tensor cores should be enabled
 * @param deviceId current CUDA device id
 * @param deviceCount total number of CUDA devices
 * @param computeMajor the compute capability major version of the device
 */
bool Environment_setCudaTensorCoreEnabled_cuda(bool enabled, int deviceId, int deviceCount, int computeMajor);

/**
 * Set CUDA device flags for blocking sync mode.
 * @param mode 0 = spin, 1 = blocking sync
 */
void Environment_setCudaBlockingSync_cuda(int mode);

/**
 * Set CUDA device scheduling policy.
 * @param schedule 0=auto, 1=spin, 2=yield, 3=blocking
 */
void Environment_setCudaDeviceSchedule_cuda(int schedule);

#else

// CPU stubs -- these are never called in CPU builds, but allow the header
// to be included unconditionally without #ifdef guards at call sites.
static inline bool Environment_setCudaCurrentDevice_cuda(int, int) { return false; }
static inline bool Environment_setCudaTensorCoreEnabled_cuda(bool, int, int, int) { return false; }
static inline void Environment_setCudaBlockingSync_cuda(int) {}
static inline void Environment_setCudaDeviceSchedule_cuda(int) {}

#endif

}  // namespace sd
