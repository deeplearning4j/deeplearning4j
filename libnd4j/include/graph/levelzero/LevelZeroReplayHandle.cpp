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

#ifdef HAVE_LEVELZERO

#include <graph/levelzero/LevelZeroReplayHandle.h>
#include <graph/DspDiagnostics.h>

#include <cstring>

namespace sd {
namespace graph {

// ── Helper macro for Level Zero error checking ──────────────────────────────

#define L0_CHECK(call)                                                         \
  do {                                                                         \
    ze_result_t _l0_result = (call);                                           \
    if (_l0_result != ZE_RESULT_SUCCESS) {                                     \
      DSP_DIAG(EXECUTION, "LevelZeroReplayHandle: %s failed with 0x%x at "    \
               "%s:%d", #call, static_cast<int>(_l0_result),                   \
               __FILE__, __LINE__);                                            \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define L0_CHECK_VOID(call)                                                    \
  do {                                                                         \
    ze_result_t _l0_result = (call);                                           \
    if (_l0_result != ZE_RESULT_SUCCESS) {                                     \
      DSP_DIAG(EXECUTION, "LevelZeroReplayHandle: %s failed with 0x%x at "    \
               "%s:%d", #call, static_cast<int>(_l0_result),                   \
               __FILE__, __LINE__);                                            \
    }                                                                          \
  } while (0)

// ── Constructor / Destructor ────────────────────────────────────────────────

LevelZeroReplayHandle::LevelZeroReplayHandle(int deviceId)
    : state_(ReplayState::EMPTY),
      deviceId_(deviceId) {
}

LevelZeroReplayHandle::~LevelZeroReplayHandle() {
  // Release workspace if still allocated (safety net for raw allocations)
  if (captureWorkspacePtr_ != nullptr && context_ != nullptr) {
    zeMemFree(context_, captureWorkspacePtr_);
    captureWorkspacePtr_ = nullptr;
    captureWorkspaceBytes_ = 0;
  }

  // Free pinned host pointers
  freeHostPointers();

  // Destroy all Level Zero handles
  cleanup();
}

// ── Device Initialization ───────────────────────────────────────────────────

bool LevelZeroReplayHandle::initDevice() {
  if (context_ != nullptr) return true;  // Already initialized

  // Initialize the Level Zero driver
  L0_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));

  // Get the first driver
  uint32_t driverCount = 1;
  L0_CHECK(zeDriverGet(&driverCount, &driver_));
  if (driverCount == 0 || driver_ == nullptr) {
    DSP_DIAG(EXECUTION, "LevelZeroReplayHandle: no Level Zero drivers found");
    return false;
  }

  // Get the target device
  uint32_t deviceCount = 0;
  L0_CHECK(zeDeviceGet(driver_, &deviceCount, nullptr));
  if (deviceCount == 0) {
    DSP_DIAG(EXECUTION, "LevelZeroReplayHandle: no Level Zero devices found");
    return false;
  }

  if (static_cast<uint32_t>(deviceId_) >= deviceCount) {
    DSP_DIAG(EXECUTION, "LevelZeroReplayHandle: deviceId %d >= deviceCount %u",
             deviceId_, deviceCount);
    return false;
  }

  std::vector<ze_device_handle_t> devices(deviceCount);
  L0_CHECK(zeDeviceGet(driver_, &deviceCount, devices.data()));
  device_ = devices[deviceId_];

  // Create context
  ze_context_desc_t contextDesc = {};
  contextDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  L0_CHECK(zeContextCreate(driver_, &contextDesc, &context_));

  // Create command queue (ordinal 0, index 0, synchronous mode for simplicity)
  ze_command_queue_desc_t queueDesc = {};
  queueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  queueDesc.ordinal = 0;
  queueDesc.index = 0;
  queueDesc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  queueDesc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  L0_CHECK(zeCommandQueueCreate(context_, device_, &queueDesc, &cmdQueue_));

  // Create event pool (1 event for completion synchronization)
  ze_event_pool_desc_t poolDesc = {};
  poolDesc.stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
  poolDesc.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  poolDesc.count = 1;
  L0_CHECK(zeEventPoolCreate(context_, &poolDesc, 1, &device_, &eventPool_));

  // Create completion event
  ze_event_desc_t eventDesc = {};
  eventDesc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  eventDesc.index = 0;
  eventDesc.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
  eventDesc.wait = ZE_EVENT_SCOPE_FLAG_HOST;
  L0_CHECK(zeEventCreate(eventPool_, &eventDesc, &completionEvent_));

  DSP_DIAG(EXECUTION, "LevelZeroReplayHandle: initialized device %d, context %p, "
           "queue %p", deviceId_, context_, cmdQueue_);
  return true;
}

// ── Capture Lifecycle ───────────────────────────────────────────────────────

bool LevelZeroReplayHandle::beginCapture(void* stream) {
  if (state_ != ReplayState::EMPTY && state_ != ReplayState::ERROR) {
    DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::beginCapture: invalid state %d",
             static_cast<int>(state_));
    return false;
  }

  // Lazy initialization of device/context/queue
  if (!initDevice()) {
    state_ = ReplayState::ERROR;
    return false;
  }

  // Destroy any previously captured command list
  if (cmdList_ != nullptr) {
    zeCommandListDestroy(cmdList_);
    cmdList_ = nullptr;
  }

  // Create a mutable command list.
  // The ze_mutable_command_list_exp_desc_t in pNext enables mutation support.
  ze_mutable_command_list_exp_desc_t mutableDesc = {};
  mutableDesc.stype = ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC;
  mutableDesc.flags = 0;

  ze_command_list_desc_t cmdListDesc = {};
  cmdListDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  cmdListDesc.pNext = &mutableDesc;
  cmdListDesc.commandQueueGroupOrdinal = 0;
  cmdListDesc.flags = 0;

  L0_CHECK(zeCommandListCreate(context_, device_, &cmdListDesc, &cmdList_));

  numCommands_ = 0;
  state_ = ReplayState::CAPTURING;

  DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::beginCapture: mutable command list "
           "created on device %d", deviceId_);
  return true;
}

bool LevelZeroReplayHandle::endCapture(void* stream) {
  if (state_ != ReplayState::CAPTURING) {
    DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::endCapture: not capturing (state %d)",
             static_cast<int>(state_));
    return false;
  }

  if (cmdList_ == nullptr) {
    state_ = ReplayState::ERROR;
    return false;
  }

  // Append the completion event signal as the last command
  L0_CHECK(zeCommandListAppendSignalEvent(cmdList_, completionEvent_));

  // Close seals the command list — no more appends until mutation
  L0_CHECK(zeCommandListClose(cmdList_));

  state_ = ReplayState::CAPTURED;
  DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::endCapture: command list closed with "
           "%d commands", numCommands_);
  return true;
}

bool LevelZeroReplayHandle::finalize() {
  if (state_ == ReplayState::READY) return true;  // Already finalized

  if (state_ != ReplayState::CAPTURED) {
    DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::finalize: not captured (state %d)",
             static_cast<int>(state_));
    return false;
  }

  // For Level Zero, zeCommandListClose() already seals the command list.
  // No additional finalization step is needed (unlike CUDA which requires
  // cudaGraphInstantiate after cudaStreamEndCapture). Just transition state.
  state_ = ReplayState::READY;

  DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::finalize: ready for replay "
           "(%d commands)", numCommands_);
  return true;
}

bool LevelZeroReplayHandle::replay(void* stream) {
  if (state_ != ReplayState::READY) {
    DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::replay: not ready (state %d)",
             static_cast<int>(state_));
    return false;
  }

  if (cmdList_ == nullptr || cmdQueue_ == nullptr || completionEvent_ == nullptr) {
    DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::replay: null handle(s)");
    state_ = ReplayState::ERROR;
    return false;
  }

  // Reset the completion event before execution
  L0_CHECK(zeEventHostReset(completionEvent_));

  // Submit the command list to the queue
  L0_CHECK(zeCommandQueueExecuteCommandLists(cmdQueue_, 1, &cmdList_, nullptr));

  // Wait for completion on the host
  L0_CHECK(zeEventHostSynchronize(completionEvent_, UINT64_MAX));

  replayCount_++;

  DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::replay: execution %d complete",
           replayCount_);
  return true;
}

// ── State & Statistics ──────────────────────────────────────────────────────

ReplayState LevelZeroReplayHandle::getState() const {
  return state_;
}

ReplayStatistics LevelZeroReplayHandle::getStatistics() const {
  ReplayStatistics stats;
  stats.numOperations = numCommands_;
  stats.numMemoryOps = 0;  // Not separately tracked for L0
  stats.estimatedMemory = captureWorkspaceBytes_;
  stats.captureTimeMs = 0.0;
  stats.lastReplayTimeMs = 0.0;
  stats.replayCount = replayCount_;
  return stats;
}

// ── Kernel Recording & Mutation ─────────────────────────────────────────────

bool LevelZeroReplayHandle::recordKernelLaunch(ze_kernel_handle_t kernel,
                                                const ze_group_count_t* launchArgs,
                                                uint64_t* outCommandId) {
  if (state_ != ReplayState::CAPTURING) {
    DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::recordKernelLaunch: not capturing");
    return false;
  }

  if (cmdList_ == nullptr || kernel == nullptr || launchArgs == nullptr || outCommandId == nullptr) {
    return false;
  }

  // Get the next mutable command ID before appending
  ze_mutable_command_id_exp_desc_t cmdIdDesc = {};
  cmdIdDesc.stype = ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC;
  cmdIdDesc.flags = ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS |
                    ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT;

  L0_CHECK(zeCommandListGetNextCommandIdExp(cmdList_, &cmdIdDesc, outCommandId));

  // Append the kernel launch (no wait events, completion event is appended in endCapture)
  L0_CHECK(zeCommandListAppendLaunchKernel(cmdList_, kernel, launchArgs,
                                            nullptr, 0, nullptr));

  numCommands_++;
  DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::recordKernelLaunch: command %d, "
           "id=%" PRIu64 ", groups=[%u,%u,%u]",
           numCommands_, *outCommandId,
           launchArgs->groupCountX, launchArgs->groupCountY, launchArgs->groupCountZ);
  return true;
}

bool LevelZeroReplayHandle::updateKernelArgs(uint64_t commandId,
                                              ze_kernel_handle_t kernel,
                                              const ze_group_count_t* launchArgs) {
  if (state_ != ReplayState::READY) {
    DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::updateKernelArgs: not ready (state %d)",
             static_cast<int>(state_));
    return false;
  }

  if (cmdList_ == nullptr) {
    return false;
  }

  // Build the mutable command update descriptor
  ze_mutable_kernel_argument_exp_desc_t kernelArgDesc = {};
  kernelArgDesc.stype = ZE_STRUCTURE_TYPE_MUTABLE_KERNEL_ARGUMENT_EXP_DESC;
  kernelArgDesc.commandId = commandId;

  ze_mutable_group_count_exp_desc_t groupCountDesc = {};
  groupCountDesc.stype = ZE_STRUCTURE_TYPE_MUTABLE_GROUP_COUNT_EXP_DESC;
  groupCountDesc.commandId = commandId;
  groupCountDesc.pGroupCount = launchArgs;

  // Chain the descriptors: kernel args -> group count (if provided)
  ze_mutable_commands_exp_desc_t mutableCmdsDesc = {};
  mutableCmdsDesc.stype = ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC;
  mutableCmdsDesc.flags = 0;

  if (kernel != nullptr) {
    // Update kernel arguments
    mutableCmdsDesc.numCommandUpdates = 1;
    mutableCmdsDesc.pNext = &kernelArgDesc;

    if (launchArgs != nullptr) {
      // Also update group count — chain descriptors
      kernelArgDesc.pNext = &groupCountDesc;
    }
  } else if (launchArgs != nullptr) {
    // Only update group count
    mutableCmdsDesc.numCommandUpdates = 1;
    mutableCmdsDesc.pNext = &groupCountDesc;
  } else {
    // Nothing to update
    return true;
  }

  // Apply the mutations
  L0_CHECK(zeCommandListUpdateMutableCommandsExp(cmdList_, &mutableCmdsDesc));

  // Re-close the command list to seal it for the next execution
  L0_CHECK(zeCommandListClose(cmdList_));

  DSP_DIAG(EXECUTION, "LevelZeroReplayHandle::updateKernelArgs: updated command "
           "%" PRIu64, commandId);
  return true;
}

// ── Workspace Management ────────────────────────────────────────────────────

bool LevelZeroReplayHandle::allocateWorkspace(size_t bytes, int deviceId,
                                               void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ != nullptr) return true;  // Already allocated

  // Ensure device is initialized
  if (!initDevice()) return false;

  // Allocate device memory
  ze_device_mem_alloc_desc_t allocDesc = {};
  allocDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  allocDesc.ordinal = 0;

  void* ptr = nullptr;
  ze_result_t result = zeMemAllocDevice(context_, &allocDesc, bytes, 64, device_, &ptr);
  if (result != ZE_RESULT_SUCCESS || ptr == nullptr) {
    DSP_DIAG(MEMORY, "LevelZeroReplayHandle: workspace alloc failed (0x%x) for "
             "%zu bytes", static_cast<int>(result), bytes);
    return false;
  }

  captureWorkspacePtr_ = ptr;
  captureWorkspaceBytes_ = bytes;
  deviceAllocations_.push_back(ptr);

  DSP_DIAG(MEMORY, "LevelZeroReplayHandle: allocated %zuMB workspace on device %d",
           bytes / (1024 * 1024), deviceId);
  return true;
}

void LevelZeroReplayHandle::releaseWorkspace(void* registryPtr, int segIdx) {
  if (captureWorkspacePtr_ == nullptr) return;

  if (context_ != nullptr) {
    L0_CHECK_VOID(zeMemFree(context_, captureWorkspacePtr_));

    // Remove from device allocations tracking
    auto it = std::find(deviceAllocations_.begin(), deviceAllocations_.end(),
                        captureWorkspacePtr_);
    if (it != deviceAllocations_.end()) {
      deviceAllocations_.erase(it);
    }

    DSP_DIAG(MEMORY, "LevelZeroReplayHandle: released workspace");
  }

  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
}

void LevelZeroReplayHandle::freeHostPointers() {
  if (context_ == nullptr) {
    // Without a context we cannot free L0 host allocations.
    // Clear the tracking vectors — these are leaked if context was already
    // destroyed, but cleanup() destroys context last for this reason.
    capturedHostPtrs_.clear();
    hostAllocations_.clear();
    return;
  }

  for (auto* ptr : capturedHostPtrs_) {
    if (ptr != nullptr) {
      L0_CHECK_VOID(zeMemFree(context_, ptr));
    }
  }
  capturedHostPtrs_.clear();

  for (auto* ptr : hostAllocations_) {
    if (ptr != nullptr) {
      L0_CHECK_VOID(zeMemFree(context_, ptr));
    }
  }
  hostAllocations_.clear();
}

// ── Cleanup ─────────────────────────────────────────────────────────────────

void LevelZeroReplayHandle::cleanup() {
  // Destroy in reverse creation order

  // Free device allocations (workspace was already handled, but safety net)
  if (context_ != nullptr) {
    for (auto* ptr : deviceAllocations_) {
      if (ptr != nullptr) {
        zeMemFree(context_, ptr);
      }
    }
    deviceAllocations_.clear();
  }

  if (completionEvent_ != nullptr) {
    zeEventDestroy(completionEvent_);
    completionEvent_ = nullptr;
  }

  if (eventPool_ != nullptr) {
    zeEventPoolDestroy(eventPool_);
    eventPool_ = nullptr;
  }

  if (cmdList_ != nullptr) {
    zeCommandListDestroy(cmdList_);
    cmdList_ = nullptr;
  }

  if (cmdQueue_ != nullptr) {
    zeCommandQueueDestroy(cmdQueue_);
    cmdQueue_ = nullptr;
  }

  // Context is destroyed last — all allocations and objects must be freed first
  if (context_ != nullptr) {
    zeContextDestroy(context_);
    context_ = nullptr;
  }

  driver_ = nullptr;
  device_ = nullptr;
  state_ = ReplayState::EMPTY;
  replayCount_ = 0;
  numCommands_ = 0;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_LEVELZERO
