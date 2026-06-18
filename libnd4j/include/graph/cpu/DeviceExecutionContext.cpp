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

#include <graph/DeviceExecutionContext.h>

// CPU-build implementation of DeviceExecutionContext.
// CUDA-specific version is in DeviceExecutionContext_cuda.cu.

namespace sd {
namespace graph {

DeviceExecutionContext DeviceExecutionContext::fromThreadLocals() {
  DeviceExecutionContext ctx;
  ctx.deviceId = 0;
  ctx.stream = nullptr;
  return ctx;
}

void DeviceExecutionContext::applyToThreadLocals() const {
  // No-op on CPU — there are no thread-local device/stream to set
}

void* DeviceExecutionContext::workspaceAlloc(size_t bytes, size_t align) {
  if (captureWorkspace == nullptr || workspaceSize == 0) return nullptr;

  // Align the offset
  size_t alignedOffset = (workspaceOffset + align - 1) & ~(align - 1);
  if (alignedOffset + bytes > workspaceSize) return nullptr;

  void* ptr = static_cast<char*>(captureWorkspace) + alignedOffset;
  workspaceOffset = alignedOffset + bytes;
  return ptr;
}

}  // namespace graph
}  // namespace sd
