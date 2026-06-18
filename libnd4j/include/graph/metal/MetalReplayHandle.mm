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

// Author: Adam Gibson

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
// Under ARC (Automatic Reference Counting), Objective-C object lifetimes are
// managed by the compiler.  To store id<MTL...> objects in C++ void* members:
//
//   __bridge_retained void*    Transfer ownership to C++ (retain count +1).
//                              Caller is responsible for eventual release.
//   __bridge_transfer id<...>  Transfer ownership back to ARC (retain count -1).
//                              Assigning to a local id<...> causes ARC to
//                              release when the local goes out of scope.
//   __bridge id<...>           Non-owning temporary cast; no refcount change.
//
// cleanup() transfers all retained pointers back to ARC for release.
// All Metal API calls are wrapped in @autoreleasepool to ensure any
// autoreleased objects returned by Metal are drained promptly.
// ============================================================================

MetalReplayHandle::MetalReplayHandle(int deviceId)
    : deviceId_(deviceId) {
  @autoreleasepool {
    if (!initMetal()) {
      state_ = ReplayState::ERRORED;
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle: initialization failed for deviceId=%d",
               deviceId_);
    }
  }
}

MetalReplayHandle::~MetalReplayHandle() {
  freeHostPointers();
  cleanup();
}

// ── Metal initialization ────────────────────────────────────────────────────

bool MetalReplayHandle::initMetal() {
  @autoreleasepool {
    // Acquire the system default Metal device (the primary GPU).
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle: MTLCreateSystemDefaultDevice() returned nil "
               "for deviceId=%d",
               deviceId_);
      return false;
    }

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle: acquired device '%s' for deviceId=%d",
             [device.name UTF8String], deviceId_);

    // Create the command queue used for all replay submissions.
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle: newCommandQueue failed for deviceId=%d",
               deviceId_);
      return false;
    }

    // Transfer ownership from ARC to C++ void* storage.
    device_ = (__bridge_retained void*)device;
    commandQueue_ = (__bridge_retained void*)queue;

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle: device and command queue created, "
             "maxCommands=%d deviceId=%d",
             maxCommands_, deviceId_);

    return createICB();
  }
}

bool MetalReplayHandle::createICB() {
  @autoreleasepool {
    if (device_ == nullptr) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::createICB: device not initialized");
      return false;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;

    // Release any previously allocated ICB before creating a new one.
    if (icb_ != nullptr) {
      (void)(__bridge_transfer id<MTLIndirectCommandBuffer>)icb_;
      icb_ = nullptr;
    }

    // Configure the ICB descriptor for compute dispatches.
    MTLIndirectCommandBufferDescriptor* desc =
        [[MTLIndirectCommandBufferDescriptor alloc] init];
    desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    desc.inheritBuffers = NO;
    desc.inheritPipelineState = NO;
    // Allow up to 8 buffer bindings per indirect compute command.
    desc.maxKernelBufferBindCount = 8;

    // Create the ICB with GPU-private storage (fastest for replay).
    id<MTLIndirectCommandBuffer> icb =
        [device newIndirectCommandBufferWithDescriptor:desc
                                      maxCommandCount:(NSUInteger)maxCommands_
                                              options:MTLResourceStorageModePrivate];

    // Some older Apple GPUs do not support Private-mode ICBs; fall back to
    // Shared storage which is accessible from both CPU and GPU.
    if (icb == nil) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::createICB: Private storage failed, "
               "retrying with Shared storage (maxCommands=%d)",
               maxCommands_);
      icb = [device newIndirectCommandBufferWithDescriptor:desc
                                          maxCommandCount:(NSUInteger)maxCommands_
                                                  options:MTLResourceStorageModeShared];
    }

    if (icb == nil) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::createICB: failed to create "
               "MTLIndirectCommandBuffer (maxCommands=%d)",
               maxCommands_);
      return false;
    }

    icb_ = (__bridge_retained void*)icb;

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle::createICB: ICB created with capacity %d",
             maxCommands_);
    return true;
  }
}

void MetalReplayHandle::cleanup() {
  @autoreleasepool {
    // Transfer ownership back to ARC for each Metal object.
    // ARC will release (dealloc) them when the local id<...> goes out of scope.

    if (icb_ != nullptr) {
      (void)(__bridge_transfer id<MTLIndirectCommandBuffer>)icb_;
      icb_ = nullptr;
    }

    if (commandQueue_ != nullptr) {
      (void)(__bridge_transfer id<MTLCommandQueue>)commandQueue_;
      commandQueue_ = nullptr;
    }

    if (device_ != nullptr) {
      (void)(__bridge_transfer id<MTLDevice>)device_;
      device_ = nullptr;
    }

    if (workspaceBuffer_ != nullptr) {
      (void)(__bridge_transfer id<MTLBuffer>)workspaceBuffer_;
      workspaceBuffer_ = nullptr;
      captureWorkspacePtr_ = nullptr;
      captureWorkspaceBytes_ = 0;
    }

    numCommands_ = 0;
    replayCount_ = 0;
    state_ = ReplayState::EMPTY;

    DSP_DIAG(GRAPH_REPLAY, "MetalReplayHandle::cleanup: all Metal objects released");
  }
}

// ── Capture lifecycle ───────────────────────────────────────────────────────

bool MetalReplayHandle::beginCapture(void* /*stream*/) {
  @autoreleasepool {
    if (icb_ == nullptr) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::beginCapture: ICB not initialized "
               "deviceId=%d",
               deviceId_);
      return false;
    }

    if (state_ == ReplayState::CAPTURING) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::beginCapture: already capturing deviceId=%d",
               deviceId_);
      return false;
    }

    // Only EMPTY and READY states may begin a new capture.
    if (state_ != ReplayState::EMPTY && state_ != ReplayState::READY) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::beginCapture: invalid state=%d deviceId=%d",
               static_cast<int>(state_), deviceId_);
      return false;
    }

    id<MTLIndirectCommandBuffer> icb =
        (__bridge id<MTLIndirectCommandBuffer>)icb_;

    // Reset all ICB slots to prepare for a fresh encoding pass.
    [icb resetWithRange:NSMakeRange(0, (NSUInteger)maxCommands_)];

    numCommands_ = 0;
    state_ = ReplayState::CAPTURING;

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle::beginCapture: started, maxCommands=%d "
             "deviceId=%d",
             maxCommands_, deviceId_);
    return true;
  }
}

bool MetalReplayHandle::endCapture(void* /*stream*/) {
  @autoreleasepool {
    if (state_ != ReplayState::CAPTURING) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::endCapture: not capturing (state=%d) "
               "deviceId=%d",
               static_cast<int>(state_), deviceId_);
      return false;
    }

    state_ = ReplayState::CAPTURED;

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle::endCapture: captured %d commands deviceId=%d",
             numCommands_, deviceId_);
    return true;
  }
}

bool MetalReplayHandle::finalize() {
  @autoreleasepool {
    if (state_ != ReplayState::CAPTURED) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::finalize: not in CAPTURED state (state=%d) "
               "deviceId=%d",
               static_cast<int>(state_), deviceId_);
      return false;
    }

    // Metal ICBs are ready for replay immediately after encoding completes.
    // finalize() exists for API symmetry with the CUDA backend which must
    // instantiate the captured graph into a cudaGraphExec_t.
    state_ = ReplayState::READY;

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle::finalize: READY (%d commands) deviceId=%d",
             numCommands_, deviceId_);
    return true;
  }
}

bool MetalReplayHandle::replay(void* /*stream*/) {
  @autoreleasepool {
    if (state_ != ReplayState::READY) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::replay: not READY (state=%d) deviceId=%d",
               static_cast<int>(state_), deviceId_);
      return false;
    }

    // Zero-command ICB is a valid no-op capture; succeed silently.
    if (numCommands_ == 0) {
      replayCount_++;
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::replay: zero commands, no-op replay #%d "
               "deviceId=%d",
               replayCount_, deviceId_);
      return true;
    }

    if (commandQueue_ == nullptr || icb_ == nullptr) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::replay: Metal objects not initialized "
               "deviceId=%d",
               deviceId_);
      return false;
    }

    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)commandQueue_;
    id<MTLIndirectCommandBuffer> icb =
        (__bridge id<MTLIndirectCommandBuffer>)icb_;

    // Create a one-shot command buffer for this replay submission.
    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    if (cmdBuf == nil) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::replay: failed to create MTLCommandBuffer "
               "deviceId=%d",
               deviceId_);
      return false;
    }
    cmdBuf.label = @"MetalReplayHandle::replay";

    // Create a compute encoder and execute all ICB commands in one call.
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    if (encoder == nil) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::replay: failed to create compute encoder "
               "deviceId=%d",
               deviceId_);
      return false;
    }

    [encoder executeCommandsInBuffer:icb
                           withRange:NSMakeRange(0, (NSUInteger)numCommands_)];
    [encoder endEncoding];

    // Commit and wait for GPU completion (synchronous replay for correctness).
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // Check for GPU-side errors reported via the command buffer status.
    if (cmdBuf.status == MTLCommandBufferStatusError) {
      NSError* error = cmdBuf.error;
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::replay: command buffer error: %s deviceId=%d",
               error ? [[error localizedDescription] UTF8String] : "unknown",
               deviceId_);
      return false;
    }

    replayCount_++;

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle::replay: completed replay #%d (%d commands) "
             "deviceId=%d",
             replayCount_, numCommands_, deviceId_);
    return true;
  }
}

// ── State and statistics ────────────────────────────────────────────────────

ReplayState MetalReplayHandle::getState() const {
  return state_;
}

ReplayStatistics MetalReplayHandle::getStatistics() const {
  ReplayStatistics stats;
  stats.numOperations    = numCommands_;
  stats.numMemoryOps     = 0;              // Metal ICBs encode compute only
  stats.estimatedMemory  = captureWorkspaceBytes_;
  stats.captureTimeMs    = 0.0;            // Not instrumented yet
  stats.lastReplayTimeMs = 0.0;            // Not instrumented yet
  stats.replayCount      = replayCount_;
  return stats;
}

int MetalReplayHandle::getMaxCommands() const { return maxCommands_; }
int MetalReplayHandle::getNumCommands() const { return numCommands_; }
int MetalReplayHandle::getDeviceId()    const { return deviceId_; }

// ── Metal-specific: record compute commands ─────────────────────────────────

bool MetalReplayHandle::recordComputeCommand(void* pipelineState,
                                              void* buffer,
                                              uint32_t bufferIndex,
                                              uint32_t threadsPerGroup,
                                              uint32_t threadgroupsPerGrid) {
  @autoreleasepool {
    if (state_ != ReplayState::CAPTURING) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::recordComputeCommand: not capturing "
               "(state=%d) deviceId=%d",
               static_cast<int>(state_), deviceId_);
      return false;
    }

    if (numCommands_ >= maxCommands_) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::recordComputeCommand: ICB full (%d/%d) "
               "deviceId=%d",
               numCommands_, maxCommands_, deviceId_);
      return false;
    }

    if (pipelineState == nullptr) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::recordComputeCommand: null pipelineState "
               "deviceId=%d",
               deviceId_);
      return false;
    }

    id<MTLIndirectCommandBuffer> icb =
        (__bridge id<MTLIndirectCommandBuffer>)icb_;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipelineState;

    // Retrieve the indirect compute command slot at the current index.
    id<MTLIndirectComputeCommand> cmd =
        [icb indirectComputeCommandAtIndex:(NSUInteger)numCommands_];

    // Bind the pipeline.
    [cmd setComputePipelineState:pipeline];

    // Optionally bind a data buffer at the specified index.
    if (buffer != nullptr) {
      id<MTLBuffer> mtlBuf = (__bridge id<MTLBuffer>)buffer;
      [cmd setKernelBuffer:mtlBuf offset:0 atIndex:bufferIndex];
    }

    // Encode a 1-D concurrent dispatch.
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);
    MTLSize threadgroupCount      = MTLSizeMake(threadgroupsPerGrid, 1, 1);
    [cmd concurrentDispatchThreadgroups:threadgroupCount
                  threadsPerThreadgroup:threadsPerThreadgroup];

    numCommands_++;

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle::recordComputeCommand: slot=%d "
             "threadsPerGroup=%u groups=%u deviceId=%d",
             numCommands_ - 1, threadsPerGroup, threadgroupsPerGrid, deviceId_);
    return true;
  }
}

void MetalReplayHandle::setMaxCommands(int max) {
  @autoreleasepool {
    if (max <= 0) return;
    if (max == maxCommands_) return;

    if (state_ == ReplayState::CAPTURING) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::setMaxCommands: cannot resize during "
               "capture deviceId=%d",
               deviceId_);
      return;
    }

    maxCommands_ = max;

    // If Metal is already initialized, recreate the ICB with the new capacity.
    if (device_ != nullptr) {
      createICB();
      // Any previously captured content is now invalid.
      if (state_ == ReplayState::READY || state_ == ReplayState::CAPTURED) {
        state_ = ReplayState::EMPTY;
      }
    }

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle::setMaxCommands: set to %d deviceId=%d",
             maxCommands_, deviceId_);
  }
}

// ── Workspace management ────────────────────────────────────────────────────

bool MetalReplayHandle::allocateWorkspace(size_t bytes,
                                           int /*deviceId*/,
                                           void* /*registryPtr*/,
                                           int /*segIdx*/) {
  @autoreleasepool {
    if (captureWorkspacePtr_ != nullptr) return true;  // Already allocated

    if (device_ == nullptr) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::allocateWorkspace: device not initialized "
               "deviceId=%d",
               deviceId_);
      return false;
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)device_;

    // Prefer GPU-private storage for best performance.
    id<MTLBuffer> buf =
        [device newBufferWithLength:(NSUInteger)bytes
                            options:MTLResourceStorageModePrivate];

    // Fall back to shared storage on devices that do not support private-mode
    // buffers (e.g., some older Apple Silicon SKUs in certain configurations).
    if (buf == nil) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::allocateWorkspace: Private storage failed, "
               "retrying with Shared (%zuMB) deviceId=%d",
               bytes / (1024 * 1024), deviceId_);
      buf = [device newBufferWithLength:(NSUInteger)bytes
                                options:MTLResourceStorageModeShared];
    }

    if (buf == nil) {
      DSP_DIAG(GRAPH_REPLAY,
               "MetalReplayHandle::allocateWorkspace: allocation failed "
               "(%zuMB) deviceId=%d",
               bytes / (1024 * 1024), deviceId_);
      return false;
    }

    workspaceBuffer_ = (__bridge_retained void*)buf;
    // Expose as an opaque pointer for the base-class getWorkspacePtr() accessor.
    captureWorkspacePtr_   = workspaceBuffer_;
    captureWorkspaceBytes_ = bytes;

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle::allocateWorkspace: allocated %zuMB deviceId=%d",
             bytes / (1024 * 1024), deviceId_);
    return true;
  }
}

void MetalReplayHandle::releaseWorkspace(void* /*registryPtr*/,
                                          int /*segIdx*/) {
  @autoreleasepool {
    if (workspaceBuffer_ == nullptr) return;

    // Transfer ownership back to ARC; the MTLBuffer is released when the
    // local variable goes out of the autoreleasepool scope.
    (void)(__bridge_transfer id<MTLBuffer>)workspaceBuffer_;
    workspaceBuffer_       = nullptr;
    captureWorkspacePtr_   = nullptr;
    captureWorkspaceBytes_ = 0;

    DSP_DIAG(GRAPH_REPLAY,
             "MetalReplayHandle::releaseWorkspace: released deviceId=%d",
             deviceId_);
  }
}

void MetalReplayHandle::freeHostPointers() {
  // Metal does not use pinned host memory (cudaHostAlloc equivalent).
  // Shared-mode MTLBuffers provide unified CPU/GPU access without explicit
  // pinning.  Any plain heap pointers stored in capturedHostPtrs_ are freed
  // here for symmetry with the CUDA backend.
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
