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

// CPU-specific implementations for NativeOps DataBuffer helpers.
// CUDA implementations are in legacy/cuda/ or NativeOps.cu.

#include <execution/LaunchContext.h>
#include <legacy/NativeOps.h>
#include <array/DataBuffer.h>

#include <algorithm>
#include <cstring>

int dbDeviceId(OpaqueDataBuffer *dataBuffer) {
  if (dataBuffer == nullptr)
    THROW_EXCEPTION("dbDeviceId: dataBuffer is null");
  return dataBuffer->deviceId();
}

void clearLastError() {
  sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(0);
  sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage("");
}

// CPU implementation - simple sequential fallback
void batchSyncToSpecialAsync(OpaqueDataBuffer **buffers, int bufferCount, int streamCount) {
  // On CPU, just iterate through and sync each buffer sequentially
  // The 'async' and 'streamCount' parameters are ignored as CPU doesn't have streams
  for (int i = 0; i < bufferCount; i++) {
    if (buffers[i] != nullptr) {
      dbSyncToSpecial(buffers[i]);
    }
  }
}

// CPU fallback: cross-device copy is GPU-only. On CPU there's only one "device"
// so this is a simple memcpy between DataBuffer primary buffers.
void dbAsyncCrossDeviceCopy(OpaqueDataBuffer *dstBuffer, OpaqueDataBuffer *srcBuffer, void *dstStream) {
  if (dstBuffer == nullptr || srcBuffer == nullptr)
    THROW_EXCEPTION("dbAsyncCrossDeviceCopy: null buffer");
  auto* dst = dstBuffer->dataBuffer();
  auto* src = srcBuffer->dataBuffer();
  if (dst == nullptr || src == nullptr)
    THROW_EXCEPTION("dbAsyncCrossDeviceCopy: null inner DataBuffer");
  size_t bytes = std::min(dst->getLenInBytes(), src->getLenInBytes());
  if (bytes == 0) return;
  // On CPU, just memcpy between primary buffers
  if (dst->primary() != nullptr && src->primary() != nullptr) {
    std::memcpy(dst->primary(), src->primary(), bytes);
    dst->writePrimary();
  }
}
