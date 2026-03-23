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

#ifdef SD_METAL

#import <Metal/Metal.h>

#include <graph/metal/MetalReplayHandle.h>
#include <graph/DspDiagnostics.h>

namespace sd {
namespace graph {

// ============================================================================
// ARC bridge helpers
// ============================================================================
//
// Under ARC (Automatic Reference Counting), Objective-C object lifetimes
// are managed by the compiler. To store id<MTL...> objects in C++ void*
// members we use:
//   - __bridge_retained void*  : Transfer ownership to C++ (retain count +1)
//   - __bridge_transfer id<..> : Transfer ownership back to ARC (retain count -1)
//   - __bridge id<..>          : Temporary cast, no ownership change
//
// cleanup() uses __bridge_transfer to release all retained objects.
// ============================================================================

MetalReplayHandle::MetalReplayHandle(int deviceId)
    : deviceId_(deviceId) {
  if (!initMetal()) {
    state_ = ReplayState::ERROR;
  }
}

MetalReplayHandle::~MetalReplayHandle() {
  freeHostPointers();
  cleanup();
}

// ── Metal initialization ────────────────────────────────────────────────────

bool MetalReplayHandle::initMetal() {
  // Get the default Metal device (system GPU)
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    sd_printf("MetalReplayHandle: MTLCreateSystemDefaultDevice() returned nil\n", "");
    return false;
  }

  // Create command queue
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (queue == nil) {
    sd_printf("MetalReplayHandle: Failed to create MTLCommandQueue\n", "");
    return false;
  }

  // Retain into opaque void* members
  device_ = (__bridge_retained void*)device;
  commandQueue_ = (__bridge_retained void*)queue;

  DSP_DIAG(MEMORY, "MetalReplayHandle: initialized device '%s', maxCommands=%d",
           [device.name UTF8String], maxCommands_);

  // Create the indirect command buffer
  return createICB();
}

bool MetalReplayHandle::createICB() {
  if (device_ == nullptr) return false;

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;

  // Release previous ICB if any
  if (icb_ != nullptr) {
    (void)(__bridge_transfer id<MTLIndirectCommandBuffer>)icb_;
    icb_ = nullptr;
  }

  // Configure ICB descriptor for compute dispatches
  MTLIndirectCommandBufferDescriptor* desc =
      [[MTLIndirectCommandBufferDescriptor alloc] init];
  desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
  desc.inheritBuffers = NO;
  desc.inheritPipelineState = NO;
  desc.maxKernelBufferBindCount = 8;  // Max buffer bindings per command

  // Create the indirect command buffer
  id<MTLIndirectCommandBuffer> icb =
      [device newIndirectCommandBufferWithDescriptor:desc
                                     maxCommandCount:(NSUInteger)maxCommands_
                                             options:MTLResourceStorageModePrivate];
  if (icb == nil) {
    // Fall back to shared storage mode (some older devices)
    icb = [device newIndirectCommandBufferWithDescriptor:desc
                                        maxCommandCount:(NSUInteger)maxCommands_
                                                options:MTLResourceStorageModeShared];
  }

  if (icb == nil) {
    sd_printf("MetalReplayHandle: Failed to create MTLIndirectCommandBuffer "
              "(maxCommands=%d)\n", maxCommands_);
    return false;
  }

  icb_ = (__bridge_retained void*)icb;

  DSP_DIAG(MEMORY, "MetalReplayHandle: created ICB with capacity %d", maxCommands_);
  return true;
}

void MetalReplayHandle::cleanup() {
  // Release ICB
  if (icb_ != nullptr) {
    (void)(__bridge_transfer id<MTLIndirectCommandBuffer>)icb_;
    icb_ = nullptr;
  }

  // Release command queue
  if (commandQueue_ != nullptr) {
    (void)(__bridge_transfer id<MTLCommandQueue>)commandQueue_;
    commandQueue_ = nullptr;
  }

  // Release device
  if (device_ != nullptr) {
    (void)(__bridge_transfer id<MTLDevice>)device_;
    device_ = nullptr;
  }

  // Release workspace buffer
  if (workspaceBuffer_ != nullptr) {
    (void)(__bridge_transfer id<MTLBuffer>)workspaceBuffer_;
    workspaceBuffer_ = nullptr;
    captureWorkspacePtr_ = nullptr;
    captureWorkspaceBytes_ = 0;
  }

  numCommands_ = 0;
  replayCount_ = 0;
  state_ = ReplayState::EMPTY;
}

// ── Capture lifecycle ───────────────────────────────────────────────────────

bool MetalReplayHandle::beginCapture(void* /*stream*/) {
  if (icb_ == nullptr) {
    sd_printf("MetalReplayHandle::beginCapture: ICB not initialized\n", "");
    return false;
  }

  if (state_ == ReplayState::CAPTURING) {
    sd_printf("MetalReplayHandle::beginCapture: already capturing\n", "");
    return false;
  }

  id<MTLIndirectCommandBuffer> icb =
      (__bridge id<MTLIndirectCommandBuffer>)icb_;

  // Reset the ICB to clear all previously encoded commands
  [icb resetWithRange:NSMakeRange(0, (NSUInteger)maxCommands_)];

  numCommands_ = 0;
  state_ = ReplayState::CAPTURING;

  DSP_DIAG(EXECUTION, "MetalReplayHandle: beginCapture (maxCommands=%d)", maxCommands_);
  return true;
}

bool MetalReplayHandle::endCapture(void* /*stream*/) {
  if (state_ != ReplayState::CAPTURING) {
    sd_printf("MetalReplayHandle::endCapture: not currently capturing (state=%d)\n",
              static_cast<int>(state_));
    return false;
  }

  state_ = ReplayState::CAPTURED;

  DSP_DIAG(EXECUTION, "MetalReplayHandle: endCapture (numCommands=%d)", numCommands_);
  return true;
}

bool MetalReplayHandle::finalize() {
  if (state_ != ReplayState::CAPTURED) {
    sd_printf("MetalReplayHandle::finalize: not in CAPTURED state (state=%d)\n",
              static_cast<int>(state_));
    return false;
  }

  // Metal ICBs are ready for execution immediately after encoding.
  // This method exists for API symmetry with CUDA (which needs instantiation).
  state_ = ReplayState::READY;

  DSP_DIAG(EXECUTION, "MetalReplayHandle: finalized (numCommands=%d)", numCommands_);
  return true;
}

bool MetalReplayHandle::replay(void* /*stream*/) {
  if (state_ != ReplayState::READY) {
    sd_printf("MetalReplayHandle::replay: not in READY state (state=%d)\n",
              static_cast<int>(state_));
    return false;
  }

  if (numCommands_ == 0) {
    // Nothing to execute -- succeed silently
    replayCount_++;
    return true;
  }

  if (commandQueue_ == nullptr || icb_ == nullptr) {
    sd_printf("MetalReplayHandle::replay: Metal objects not initialized\n", "");
    return false;
  }

  id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue_;
  id<MTLIndirectCommandBuffer> icb =
      (__bridge id<MTLIndirectCommandBuffer>)icb_;

  // Create a command buffer for this replay
  id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
  if (cmdBuf == nil) {
    sd_printf("MetalReplayHandle::replay: failed to create command buffer\n", "");
    return false;
  }
  cmdBuf.label = @"MetalReplayHandle::replay";

  // Create compute encoder and execute the ICB
  id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
  if (encoder == nil) {
    sd_printf("MetalReplayHandle::replay: failed to create compute encoder\n", "");
    return false;
  }

  [encoder executeCommandsInBuffer:icb
                         withRange:NSMakeRange(0, (NSUInteger)numCommands_)];
  [encoder endEncoding];

  // Commit and wait for completion (synchronous replay)
  [cmdBuf commit];
  [cmdBuf waitUntilCompleted];

  // Check for errors
  if (cmdBuf.status == MTLCommandBufferStatusError) {
    NSError* error = cmdBuf.error;
    sd_printf("MetalReplayHandle::replay: command buffer error: %s\n",
              error ? [[error localizedDescription] UTF8String] : "unknown");
    return false;
  }

  replayCount_++;

  DSP_DIAG(EXECUTION, "MetalReplayHandle: replay #%d completed (%d commands)",
           replayCount_, numCommands_);
  return true;
}

// ── State and statistics ────────────────────────────────────────────────────

ReplayState MetalReplayHandle::getState() const {
  return state_;
}

ReplayStatistics MetalReplayHandle::getStatistics() const {
  ReplayStatistics stats;
  stats.numOperations = numCommands_;
  stats.numMemoryOps = 0;  // Metal ICBs track compute dispatches only
  stats.estimatedMemory = captureWorkspaceBytes_;
  stats.captureTimeMs = 0.0;  // Not tracked yet
  stats.lastReplayTimeMs = 0.0;  // Not tracked yet
  stats.replayCount = replayCount_;
  return stats;
}

// ── Metal-specific: record compute commands ─────────────────────────────────

bool MetalReplayHandle::recordComputeCommand(void* pipelineState,
                                              void* buffer,
                                              uint32_t bufferIndex,
                                              uint32_t threadsPerGroup,
                                              uint32_t threadgroupsPerGrid) {
  if (state_ != ReplayState::CAPTURING) {
    sd_printf("MetalReplayHandle::recordComputeCommand: not capturing\n", "");
    return false;
  }

  if (numCommands_ >= maxCommands_) {
    sd_printf("MetalReplayHandle::recordComputeCommand: ICB full (%d/%d)\n",
              numCommands_, maxCommands_);
    return false;
  }

  if (pipelineState == nullptr) {
    sd_printf("MetalReplayHandle::recordComputeCommand: null pipeline state\n", "");
    return false;
  }

  id<MTLIndirectCommandBuffer> icb =
      (__bridge id<MTLIndirectCommandBuffer>)icb_;
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)pipelineState;

  // Get the indirect compute command at the current index
  id<MTLIndirectComputeCommand> cmd =
      [icb indirectComputeCommandAtIndex:(NSUInteger)numCommands_];

  // Configure the command
  [cmd setComputePipelineState:pipeline];

  // Bind the buffer if provided
  if (buffer != nullptr) {
    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
    [cmd setKernelBuffer:mtlBuffer offset:0 atIndex:bufferIndex];
  }

  // Set threadgroup size and dispatch
  MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);
  MTLSize threadgroupCount = MTLSizeMake(threadgroupsPerGrid, 1, 1);

  [cmd concurrentDispatchThreadgroups:threadgroupCount
                threadsPerThreadgroup:threadsPerThreadgroup];

  numCommands_++;

  DSP_DIAG(EXECUTION, "MetalReplayHandle: recorded command %d (threads=%u, groups=%u)",
           numCommands_ - 1, threadsPerGroup, threadgroupsPerGrid);
  return true;
}

void MetalReplayHandle::setMaxCommands(int max) {
  if (max <= 0) return;
  if (max == maxCommands_) return;

  if (state_ == ReplayState::CAPTURING) {
    sd_printf("MetalReplayHandle::setMaxCommands: cannot resize during capture\n", "");
    return;
  }

  maxCommands_ = max;

  // Recreate the ICB with new capacity if Metal is initialized
  if (device_ != nullptr) {
    createICB();
    state_ = ReplayState::EMPTY;
  }
}

// ── Workspace management ────────────────────────────────────────────────────

bool MetalReplayHandle::allocateWorkspace(size_t bytes, int /*deviceId*/,
                                           void* /*registryPtr*/, int /*segIdx*/) {
  if (captureWorkspacePtr_ != nullptr) return true;  // Already allocated

  if (device_ == nullptr) {
    sd_printf("MetalReplayHandle::allocateWorkspace: device not initialized\n", "");
    return false;
  }

  id<MTLDevice> device = (__bridge id<MTLDevice>)device_;

  // Allocate GPU-private buffer for workspace
  id<MTLBuffer> buf = [device newBufferWithLength:(NSUInteger)bytes
                                          options:MTLResourceStorageModePrivate];
  if (buf == nil) {
    // Fall back to shared storage
    buf = [device newBufferWithLength:(NSUInteger)bytes
                              options:MTLResourceStorageModeShared];
  }

  if (buf == nil) {
    sd_printf("MetalReplayHandle::allocateWorkspace: failed to allocate %zuMB\n",
              bytes / (1024 * 1024));
    return false;
  }

  workspaceBuffer_ = (__bridge_retained void*)buf;
  captureWorkspacePtr_ = workspaceBuffer_;  // Opaque handle for base class
  captureWorkspaceBytes_ = bytes;

  DSP_DIAG(MEMORY, "MetalReplayHandle: allocated %zuMB workspace", bytes / (1024 * 1024));
  return true;
}

void MetalReplayHandle::releaseWorkspace(void* /*registryPtr*/, int /*segIdx*/) {
  if (workspaceBuffer_ == nullptr) return;

  (void)(__bridge_transfer id<MTLBuffer>)workspaceBuffer_;
  workspaceBuffer_ = nullptr;
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;

  DSP_DIAG(MEMORY, "MetalReplayHandle: released workspace");
}

void MetalReplayHandle::freeHostPointers() {
  // Metal does not use pinned host memory like CUDA (cudaHostAlloc).
  // Shared-mode buffers (MTLResourceStorageModeShared) are accessible
  // from both CPU and GPU without explicit pinning. Simply free the
  // captured host pointer vector entries.
  for (auto* ptr : capturedHostPtrs_) {
    if (ptr != nullptr) {
      free(ptr);
    }
  }
  capturedHostPtrs_.clear();
}

}  // namespace graph
}  // namespace sd

#endif  // SD_METAL
