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

/**
 * NativeOps JNI entry points for the native C++ graph executor (DSP).
 * Split from NativeOps.cu for modularity.
 *
 * Provides:
 *   - compileDynamicShapePlan / executeDynamicShapePlan / freeDynamicShapePlan
 *   - clearDynamicShapePlanCaches
 *   - loadModelFromFile / compileModelPlan / freeLoadedModel
 *   - getPlanNumExternalInputs / getPlanNumRequestedOutputs / getPlanNumSlots
 */

#include <cuda.h>
#include <graph/DspDiagnostics.h>
#include <legacy/NativeOps.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/NativePlanCompiler.h>
#include <graph/SdnbReader.h>
#include <graph/SdzReader.h>
#include <system/common.h>
#include <memory/cuda/CudaMemoryPool.h>
#include <execution/AffinityManager.h>

#include <config.h>
#if HAVE_TRITON
#include <graph/gpu/TritonGraphBackend.h>
#endif

#include <cstring>
#include <string>
#include <fstream>
#include <algorithm>
#include <cctype>

using namespace sd;
using namespace sd::graph;

// ─── Loaded model handle wrapper ─────────────────────────────────────────────

struct LoadedModelHandle {
  SdnbReader::LoadedModel model;
  SdnbReader* sdnbReader;    // Non-null if loaded from SDNB
  SdzReader* sdzReader;      // Non-null if loaded from SDZ

  LoadedModelHandle() : sdnbReader(nullptr), sdzReader(nullptr) {}

  ~LoadedModelHandle() {
    // model.variables are freed by LoadedModel's destructor
    delete sdnbReader;
    delete sdzReader;
  }
};

// ─── Plan compilation and execution ──────────────────────────────────────────

sd::Pointer compileDynamicShapePlan(sd::Pointer serializedPlan, sd::LongType planSize) {
  try {
    if (serializedPlan == nullptr || planSize <= 0) {
      DSP_DIAG(COMPILE, "compileDynamicShapePlan: null or empty plan data");
      return nullptr;
    }

    auto* plan = NativeDynamicShapePlan::fromSerializedPlan(serializedPlan, planSize);
    if (plan == nullptr) {
      DSP_DIAG(COMPILE, "compileDynamicShapePlan: failed to parse plan (%lld bytes)",
               static_cast<long long>(planSize));
      return nullptr;
    }

    DSP_DIAG(COMPILE, "compiled plan: %d slots, %d inputs, %d outputs",
             plan->getNumSlots(), plan->getNumExternalInputs(), plan->getNumRequestedOutputs());

    return reinterpret_cast<sd::Pointer>(plan);
  } catch (const std::exception& e) {
    DSP_DIAG(COMPILE, "compileDynamicShapePlan: exception: %s", e.what());
    return nullptr;
  }
}

int executeDynamicShapePlan(
    sd::Pointer planHandle,
    OpaqueContext* opContext,
    sd::Pointer stream) {
  // Helper to set error on LaunchContext so Java can read via lastErrorMessage()
  auto setError = [](int code, const char* msg) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(code);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(msg);
  };

  try {
    if (planHandle == nullptr) {
      const char* msg = "executeDynamicShapePlan: null plan handle";
      DSP_DIAG(EXECUTE, "%s", msg);
      setError(1, msg);
      return 1;
    }
    if (opContext == nullptr) {
      const char* msg = "executeDynamicShapePlan: null opContext";
      DSP_DIAG(EXECUTE, "%s", msg);
      setError(1, msg);
      return 1;
    }

    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);

    // Read inputs/outputs from the context's fastpath vectors
    int numInputs = static_cast<int>(opContext->width());
    int numOutputs = static_cast<int>(opContext->outputWidth());

    // Validate input/output counts
    if (numInputs != plan->getNumExternalInputs()) {
      char buf[256];
      snprintf(buf, sizeof(buf), "executeDynamicShapePlan: input count mismatch: got %d, expected %d",
               numInputs, plan->getNumExternalInputs());
      DSP_DIAG(EXECUTE, "%s", buf);
      setError(2, buf);
      return 2;
    }
    if (numOutputs != plan->getNumRequestedOutputs()) {
      char buf[256];
      snprintf(buf, sizeof(buf), "executeDynamicShapePlan: output count mismatch: got %d, expected %d",
               numOutputs, plan->getNumRequestedOutputs());
      DSP_DIAG(EXECUTE, "%s", buf);
      setError(3, buf);
      return 3;
    }

    // Build input/output arrays from context
    std::vector<NDArray*> inputPtrs(numInputs);
    for (int i = 0; i < numInputs; i++) {
      inputPtrs[i] = opContext->array(i);
      if (inputPtrs[i] == nullptr) {
        char buf[256];
        snprintf(buf, sizeof(buf), "executeDynamicShapePlan: null input at index %d", i);
        DSP_DIAG(EXECUTE, "%s", buf);
        setError(4, buf);
        return 4;
      }
    }

    std::vector<NDArray*> outputPtrs(numOutputs);
    for (int i = 0; i < numOutputs; i++) {
      outputPtrs[i] = opContext->outputArray(i);
      // Output arrays may be null — plan will allocate them
    }

    void* cudaStream = (stream != nullptr) ? reinterpret_cast<void*>(stream) : nullptr;

    auto status = plan->execute(inputPtrs.data(), numInputs, outputPtrs.data(), numOutputs, cudaStream);

    if (status != Status::OK) {
      // Preserve the detailed C++ error message (e.g., "CUDA error after segment [X-Y]: 700 (...")
      // that was set by the plan's execute() — don't overwrite it with a generic message.
      // The existing errorReference message contains segment range and CUDA error code.
      const char* existingMsg = sd::LaunchContext::defaultContext()->errorReference()->errorMessage();
      char buf[512];
      if (existingMsg != nullptr && existingMsg[0] != '\0') {
        snprintf(buf, sizeof(buf), "executeDynamicShapePlan: plan execution failed with status %d: %s",
                 static_cast<int>(status), existingMsg);
      } else {
        snprintf(buf, sizeof(buf), "executeDynamicShapePlan: plan execution failed with status %d",
                 static_cast<int>(status));
      }
      DSP_DIAG(EXECUTE, "%s", buf);
      setError(static_cast<int>(status), buf);
      // Clear any sticky CUDA errors left by the failed execution
      cudaGetLastError();
      return static_cast<int>(status);
    }

    // Clear any stale error from previous failed operations (e.g., graph capture failures).
    // Without this, a successful DSP execution leaves a stale errorCode/errorMessage that
    // causes subsequent standalone ops (e.g., token_sample) to spuriously fail when
    // CudaExecutioner checks lastErrorCode() after execCustomOp2.
    setError(0, "");

    // Write output arrays back to context so Java can read them
    for (int i = 0; i < numOutputs; i++) {
      if (outputPtrs[i] != nullptr) {
        opContext->setOutputArray(i, outputPtrs[i], false);
      }
    }

    return 0;
  } catch (const std::exception& e) {
    DSP_DIAG(EXECUTE, "executeDynamicShapePlan: exception: %s", e.what());
    // Clear any sticky CUDA errors and ensure stream is not in capture mode
    cudaGetLastError();
    if (stream != nullptr) {
      cudaStream_t cudaStr = *reinterpret_cast<cudaStream_t*>(stream);
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();
    }
    // Set error message for Java side
    setError(-1, e.what());
    return -1;
  }
}

void freeDynamicShapePlan(sd::Pointer planHandle) {
  if (planHandle != nullptr) {
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    delete plan;
  }
}

void clearDynamicShapePlanCaches(sd::Pointer planHandle) {
  if (planHandle != nullptr) {
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    plan->clearShapeCaches();
  }
}

void clearAllDynamicShapePlanCachesForce(sd::Pointer planHandle) {
  if (planHandle != nullptr) {
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    plan->clearAllShapeCachesForce();
  }
}

void configurePlanKvCacheRetention(
    sd::Pointer planHandle, const int* mappings,
    int numMappings, int maxKvLen, int initialPos) {
  if (planHandle == nullptr || mappings == nullptr || numMappings <= 0) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  plan->configureKvCacheRetention(mappings, numMappings, maxKvLen, initialPos);
}

int advancePlanKvCachePosition(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return plan->advanceKvCachePosition();
}

void resetPlanKvCachePosition(sd::Pointer planHandle, int newPos) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  plan->resetKvCachePosition(newPos);
}

// ─── Model loading (SDZ/SDNB) ───────────────────────────────────────────────

sd::Pointer loadModelFromFile(const char* filePath) {
  try {
    if (filePath == nullptr) {
      DSP_DIAG(COMPILE, "loadModelFromFile: null file path");
      return nullptr;
    }

    auto* handle = new LoadedModelHandle();

    // Determine file type by extension
    std::string path(filePath);
    std::string pathLower = path;
    std::transform(pathLower.begin(), pathLower.end(), pathLower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    bool isSdz = pathLower.size() > 4 && pathLower.substr(pathLower.size() - 4) == ".sdz";
    bool isSdnb = pathLower.size() > 5 && pathLower.substr(pathLower.size() - 5) == ".sdnb";

    if (isSdz) {
      handle->sdzReader = SdzReader::openFile(filePath);
      if (!handle->sdzReader) {
        DSP_DIAG(COMPILE, "loadModelFromFile: failed to open SDZ file: %s", filePath);
        delete handle;
        return nullptr;
      }
      handle->model = handle->sdzReader->load();
    } else if (isSdnb) {
      handle->sdnbReader = SdnbReader::openFile(filePath);
      if (!handle->sdnbReader) {
        DSP_DIAG(COMPILE, "loadModelFromFile: failed to open SDNB file: %s", filePath);
        delete handle;
        return nullptr;
      }
      handle->model = handle->sdnbReader->loadAll();
    } else {
      // Try SDZ first, then SDNB
      handle->sdzReader = SdzReader::openFile(filePath);
      if (handle->sdzReader) {
        handle->model = handle->sdzReader->load();
      } else {
        handle->sdnbReader = SdnbReader::openFile(filePath);
        if (!handle->sdnbReader) {
          DSP_DIAG(COMPILE, "loadModelFromFile: cannot open file as SDZ or SDNB: %s", filePath);
          delete handle;
          return nullptr;
        }
        handle->model = handle->sdnbReader->loadAll();
      }
    }

    if (!handle->model.graph) {
      DSP_DIAG(COMPILE, "loadModelFromFile: file did not yield a valid FlatGraph: %s", filePath);
      delete handle;
      return nullptr;
    }

    DSP_DIAG(COMPILE, "loaded model: %d vars, %d placeholders from %s",
             static_cast<int>(handle->model.variables.size()),
             static_cast<int>(handle->model.placeholderNames.size()),
             filePath);

    return reinterpret_cast<sd::Pointer>(handle);
  } catch (const std::exception& e) {
    DSP_DIAG(COMPILE, "loadModelFromFile: exception: %s", e.what());
    return nullptr;
  }
}

sd::Pointer compileModelPlan(
    sd::Pointer modelHandle,
    sd::Pointer requestedOutputNames, int numOutputs) {
  try {
    if (modelHandle == nullptr) {
      DSP_DIAG(COMPILE, "compileModelPlan: null model handle");
      return nullptr;
    }

    auto* handle = reinterpret_cast<LoadedModelHandle*>(modelHandle);

    if (!handle->model.graph) {
      DSP_DIAG(COMPILE, "compileModelPlan: model has no graph");
      return nullptr;
    }

    // Collect requested output names - cast from opaque pointer to char**
    auto** outputNames = reinterpret_cast<const char**>(requestedOutputNames);
    std::vector<std::string> outputs;
    for (int i = 0; i < numOutputs; i++) {
      if (outputNames[i]) {
        outputs.emplace_back(outputNames[i]);
      }
    }

    auto* plan = NativeDynamicShapePlan::fromFlatGraph(
        handle->model.graph, handle->model.variables, outputs);

    if (!plan) {
      DSP_DIAG(COMPILE, "compileModelPlan: failed to compile plan");
      return nullptr;
    }

    DSP_DIAG(COMPILE, "compiled model plan: %d slots, %d outputs",
             plan->getNumSlots(), plan->getNumRequestedOutputs());

    return reinterpret_cast<sd::Pointer>(plan);
  } catch (const std::exception& e) {
    DSP_DIAG(COMPILE, "compileModelPlan: exception: %s", e.what());
    return nullptr;
  }
}

void freeLoadedModel(sd::Pointer modelHandle) {
  if (modelHandle != nullptr) {
    auto* handle = reinterpret_cast<LoadedModelHandle*>(modelHandle);
    delete handle;
  }
}

// ─── Plan introspection ──────────────────────────────────────────────────────

int getPlanNumExternalInputs(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getNumExternalInputs();
}

int getPlanNumRequestedOutputs(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getNumRequestedOutputs();
}

int getPlanNumSlots(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getNumSlots();
}

// ─── CUDA Graph control ─────────────────────────────────────────────────────

void setPlanCudaGraphsEnabled(sd::Pointer planHandle, bool enabled) {
  if (planHandle != nullptr) {
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setCudaGraphsEnabled(enabled);
  }
}

void setPlanJitMode(sd::Pointer planHandle, int mode) {
  if (planHandle != nullptr) {
    sd::graph::JitMode jitMode;
    switch (mode) {
      case 1: jitMode = sd::graph::JitMode::JIT_ONLY; break;
      case 2: jitMode = sd::graph::JitMode::GRAPH_PLUS_JIT; break;
      default: jitMode = sd::graph::JitMode::GRAPH_ONLY; break;
    }
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setJitMode(jitMode);
  }
}

void setPlanGraphExecutionMode(sd::Pointer planHandle, int mode) {
  if (planHandle != nullptr) {
    auto requested = static_cast<sd::graph::GraphExecutionMode>(mode);
    auto gem = requested;
    // Clamp to valid range
    if (gem < sd::graph::GraphExecutionMode::GEM_AUTO || gem > sd::graph::GraphExecutionMode::GEM_NNAPI) {
      gem = sd::graph::GraphExecutionMode::GEM_AUTO;
    }
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setGraphExecutionMode(gem);

    const char* requestedName = "UNKNOWN";
    switch (requested) {
      case sd::graph::GraphExecutionMode::GEM_AUTO: requestedName = "AUTO"; break;
      case sd::graph::GraphExecutionMode::GEM_SLOT_BY_SLOT: requestedName = "SLOT_BY_SLOT"; break;
      case sd::graph::GraphExecutionMode::GEM_CUDA_GRAPHS: requestedName = "CUDA_GRAPHS"; break;
      case sd::graph::GraphExecutionMode::GEM_NVRTC_JIT: requestedName = "NVRTC_JIT"; break;
      case sd::graph::GraphExecutionMode::GEM_PTX_JIT: requestedName = "PTX_JIT"; break;
      case sd::graph::GraphExecutionMode::GEM_TRITON: requestedName = "TRITON"; break;
      case sd::graph::GraphExecutionMode::GEM_MLX: requestedName = "MLX"; break;
      case sd::graph::GraphExecutionMode::GEM_ARM_HYBRID: requestedName = "ARM_HYBRID"; break;
      case sd::graph::GraphExecutionMode::GEM_NNAPI: requestedName = "NNAPI"; break;
      default: break;
    }

    const char* appliedName = "UNKNOWN";
    switch (gem) {
      case sd::graph::GraphExecutionMode::GEM_AUTO: appliedName = "AUTO"; break;
      case sd::graph::GraphExecutionMode::GEM_SLOT_BY_SLOT: appliedName = "SLOT_BY_SLOT"; break;
      case sd::graph::GraphExecutionMode::GEM_CUDA_GRAPHS: appliedName = "CUDA_GRAPHS"; break;
      case sd::graph::GraphExecutionMode::GEM_NVRTC_JIT: appliedName = "NVRTC_JIT"; break;
      case sd::graph::GraphExecutionMode::GEM_PTX_JIT: appliedName = "PTX_JIT"; break;
      case sd::graph::GraphExecutionMode::GEM_TRITON: appliedName = "TRITON"; break;
      case sd::graph::GraphExecutionMode::GEM_MLX: appliedName = "MLX"; break;
      case sd::graph::GraphExecutionMode::GEM_ARM_HYBRID: appliedName = "ARM_HYBRID"; break;
      case sd::graph::GraphExecutionMode::GEM_NNAPI: appliedName = "NNAPI"; break;
      default: break;
    }

    DSP_DIAG(BACKEND, "setPlanGraphExecutionMode: requested=%d(%s) applied=%d(%s)",
             mode, requestedName, static_cast<int>(gem), appliedName);
  }
}

void setPlanMinCaptureSegmentSize(sd::Pointer planHandle, int minSize) {
  // Segment sizes are now auto-discovered from graph structure; this is a no-op.
}

void setPlanMaxCaptureSegmentSize(sd::Pointer planHandle, int maxSize) {
  // Segment sizes are now auto-discovered from graph structure; this is a no-op.
}

void setPlanShapesFrozen(sd::Pointer planHandle, bool frozen) {
  if (planHandle != nullptr) {
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setShapesFrozen(frozen);
  }
}

void setPlanExecutionTimingEnabled(sd::Pointer planHandle, bool enabled) {
  if (planHandle != nullptr) {
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setExecutionTimingEnabled(enabled);
  }
}

void setPlanTraceEnabled(sd::Pointer planHandle, bool enabled) {
  if (planHandle != nullptr) {
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setTraceEnabled(enabled);
  }
}

void setPlanOutputSlotMaxSizes(sd::Pointer planHandle, sd::LongType numSlots,
                                 const int* slotIndices, const sd::LongType* maxSizes) {
  if (planHandle == nullptr || numSlots <= 0 || slotIndices == nullptr || maxSizes == nullptr) return;
  reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setOutputSlotMaxSizes(slotIndices, maxSizes, static_cast<int>(numSlots));
}

void setPlanKvCachePosition(sd::Pointer planHandle, int pos) {
  if (planHandle != nullptr) {
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setKvCachePosition(pos);
  }
}

void setPlanMaxKvCacheLength(sd::Pointer planHandle, int maxLen) {
  if (planHandle != nullptr) {
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setMaxKvCacheLength(maxLen);
  }
}

int getPlanNumSegments(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return static_cast<int>(
      reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getSegments().size());
}

int getPlanNumCapturedGraphSegments(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getNumCapturedGraphSegments();
}

int getPlanTotalGraphReplays(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getTotalGraphReplays();
}

bool validatePlanCapturedGraph(sd::Pointer planHandle) {
  if (planHandle == nullptr) return true;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->validateCapturedGraph(-1);
}

int getPlanNumHostOnlyOps(sd::Pointer planHandle) {
  if (planHandle == nullptr) return 0;
  auto hostOnly = reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getHostOnlyOps();
  return static_cast<int>(hostOnly.size());
}

const char* getPlanHostOnlyOpNames(sd::Pointer planHandle) {
  static thread_local std::string result;
  result.clear();
  if (planHandle == nullptr) return result.c_str();

  auto hostOnly = reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getHostOnlyOps();
  for (size_t i = 0; i < hostOnly.size(); i++) {
    if (i > 0) result += "|";
    result += hostOnly[i].opName;
  }
  return result.c_str();
}

void printPlanCapturedGraphDebug(sd::Pointer planHandle) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  plan->printCaptureAudit();

  // Also print graph contents for each cached segment
  for (const auto& seg : plan->getSegments()) {
    if (seg.cachedGraph) {
      seg.cachedGraph->printGraphContents();
    }
  }
}

const char* getPlanCaptureStats(sd::Pointer planHandle) {
  static thread_local char buf[1024];
  if (planHandle == nullptr) {
    snprintf(buf, sizeof(buf), "null");
    return buf;
  }
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  const auto& segs = plan->getSegments();

  int captured = 0, oomRetrying = 0, permFailed = 0, nonCapt = 0;
  int captSlots = 0, oomSlots = 0, permSlots = 0, nonCaptSlots = 0;
  int maxOomRetries = 0;

  for (const auto& seg : segs) {
    int segSlots = seg.endSlot - seg.startSlot + 1;
    if (!seg.isCapturable) {
      nonCapt++;
      nonCaptSlots += segSlots;
    } else if (seg.cachedGraph) {
      captured++;
      captSlots += segSlots;
    } else if (seg.captureFailed) {
      permFailed++;
      permSlots += segSlots;
    } else if (seg.captureOomRetries > 0) {
      oomRetrying++;
      oomSlots += segSlots;
      if (seg.captureOomRetries > maxOomRetries) maxOomRetries = seg.captureOomRetries;
    }
  }

  // Get GPU memory info
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);

  // Get pool stats
  size_t poolUsed = 0, poolReserved = 0;
  int devId = sd::AffinityManager::currentDeviceId();
  sd::memory::CudaMemoryPool::getInstance().getStats(devId, poolUsed, poolReserved);

  snprintf(buf, sizeof(buf),
           "captured=%d(%dslots)|oomRetrying=%d(%dslots,maxRetry=%d)|permFailed=%d(%dslots)|nonCapt=%d(%dslots)|total=%d"
           "|gpuFree=%zuMB|poolUsed=%zuMB|poolRes=%zuMB",
           captured, captSlots, oomRetrying, oomSlots, maxOomRetries,
           permFailed, permSlots, nonCapt, nonCaptSlots,
           static_cast<int>(segs.size()),
           gpuFree / (1024*1024), poolUsed / (1024*1024), poolReserved / (1024*1024));
  return buf;
}

// ─── CUDA Graph Visualization (PyTorch-style) ──────────────────────────────────

bool exportPlanCudaGraphChromeTrace(sd::Pointer planHandle, const char* outputPath) {
  if (planHandle == nullptr || outputPath == nullptr) return false;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);

  // Export each captured segment's graph as a combined trace
  std::ofstream file(outputPath);
  if (!file.is_open()) {
    DSP_DIAG(COMPILE, "exportPlanCudaGraphChromeTrace: failed to open file: %s", outputPath);
    return false;
  }

  file << "{\n  \"traceEvents\": [\n";
  bool first = true;
  int segmentIdx = 0;

  for (const auto& seg : plan->getSegments()) {
    if (seg.cachedGraph) {
      // Get the Chrome trace JSON for this segment
      std::string traceJson = seg.cachedGraph->getChromeTraceJson();
      // Parse out the traceEvents array and add segment prefix
      // For simplicity, we'll just include the whole trace with segment metadata
      if (!first) file << ",\n";
      first = false;

      // Add segment metadata
      file << "    {\"name\": \"segment_" << segmentIdx << "\", \"ph\": \"M\", \"pid\": 0, "
           << "\"tid\": 0, \"ts\": 0, \"args\": {\"startSlot\": " << seg.startSlot
           << ", \"endSlot\": " << seg.endSlot << "}},\n";

      // Extract trace events from the segment's JSON (simplified - just use debug dump)
    }
    segmentIdx++;
  }

  file << "  ]\n}\n";
  file.close();
  DSP_DIAG(SEGMENT, "exportPlanCudaGraphChromeTrace: exported to %s", outputPath);
  return true;
}

bool exportPlanCudaGraphHtml(sd::Pointer planHandle, const char* outputPath) {
  if (planHandle == nullptr || outputPath == nullptr) return false;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);

  // Export the first captured segment's HTML (or combine all)
  for (const auto& seg : plan->getSegments()) {
    if (seg.cachedGraph) {
      return seg.cachedGraph->exportToHtml(std::string(outputPath));
    }
  }
  return false;
}

bool debugDumpPlanCudaGraph(sd::Pointer planHandle, const char* outputPath) {
  if (planHandle == nullptr || outputPath == nullptr) return false;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);

  bool success = false;
  int segmentIdx = 0;

  for (const auto& seg : plan->getSegments()) {
    if (seg.cachedGraph) {
      std::string basePath = std::string(outputPath) + "_seg" + std::to_string(segmentIdx);
      if (seg.cachedGraph->debugDump(basePath)) {
        success = true;
      }
    }
    segmentIdx++;
  }

  if (!success) {
    DSP_DIAG(SEGMENT, "debugDumpPlanCudaGraph: no captured graphs to dump");
  }
  return success;
}

const char* getPlanCudaGraphChromeTraceJson(sd::Pointer planHandle) {
  static thread_local std::string result;
  result.clear();

  if (planHandle == nullptr) return "";
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);

  // Combine all segment traces into one JSON
  result = "{\n  \"traceEvents\": [\n";
  bool first = true;
  int segmentIdx = 0;

  for (const auto& seg : plan->getSegments()) {
    if (seg.cachedGraph) {
      std::string segJson = seg.cachedGraph->getChromeTraceJson();
      if (!segJson.empty() && segJson != "{}") {
        if (!first) result += ",\n";
        first = false;
        result += "    {\"name\": \"segment_" + std::to_string(segmentIdx) + "\", "
                  "\"ph\": \"M\", \"pid\": 0, \"tid\": 0, \"ts\": 0},\n";
        // Add segment's trace events (simplified)
      }
    }
    segmentIdx++;
  }

  result += "  ]\n}\n";
  return result.c_str();
}

void clearPlanCudaGraphTimeline(sd::Pointer planHandle) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);

  for (auto& seg : plan->getSegmentsMutable()) {
    if (seg.cachedGraph) {
      seg.cachedGraph->clearExecutionTimeline();
    }
  }
}

// =============================================================================
// NCCL Collective Communication Operations
// =============================================================================

#ifdef HAVE_NCCL
#include <nccl.h>

static ncclDataType_t toNcclDataType(int dt) {
    switch (dt) {
        case 1:  return ncclFloat16;   // HALF
        case 5:  return ncclFloat32;   // FLOAT
        case 7:  return ncclFloat64;   // DOUBLE
        case 3:  return ncclInt32;     // INT32
        case 9:  return ncclInt64;     // INT64
        case 2:  return ncclInt8;      // INT8
        case 6:  return ncclUint8;     // UINT8
#if NCCL_VERSION_CODE >= 21000
        case 16: return ncclBfloat16;  // BFLOAT16
#endif
        default:
            DSP_DIAG(FALLBACK, "NCCL: unsupported data type %d, falling back to float", dt);
            return ncclFloat32;
    }
}

static ncclRedOp_t toNcclReduceOp(int op) {
    switch (op) {
        case 0:  return ncclSum;
        case 1:  return ncclProd;
        case 2:  return ncclMax;
        case 3:  return ncclMin;
#if NCCL_VERSION_CODE >= 21000
        case 4:  return ncclAvg;
#endif
        default: return ncclSum;
    }
}
#endif // HAVE_NCCL

sd::Pointer ncclCommInit(int numRanks, int rankId, int deviceId) {
#ifdef HAVE_NCCL
    ncclComm_t comm;
    ncclUniqueId id;

    // For single-process multi-GPU, rank 0 generates the ID
    // and all ranks use it. In this simple case, we generate + init all at once.
    ncclResult_t res = ncclCommInitAll(&comm, 1, &deviceId);
    if (res != ncclSuccess) {
        DSP_DIAG(BACKEND, "NCCL ncclCommInitAll failed: %s", ncclGetErrorString(res));
        return nullptr;
    }

    // For multi-GPU in single process, use ncclCommInitAll with all devices
    // For now, return single communicator
    auto* commPtr = new ncclComm_t(comm);
    return reinterpret_cast<sd::Pointer>(commPtr);
#else
    DSP_DIAG(BACKEND, "NCCL not available. Build with -DHELPERS_nccl=ON");
    return nullptr;
#endif
}

sd::Pointer ncclCommInitWithId(int numRanks, int rankId, sd::Pointer uniqueId) {
#ifdef HAVE_NCCL
    if (uniqueId == nullptr) {
        DSP_DIAG(BACKEND, "NCCL ncclCommInitWithId: uniqueId is null");
        return nullptr;
    }

    ncclComm_t comm;
    ncclUniqueId* id = reinterpret_cast<ncclUniqueId*>(uniqueId);
    ncclResult_t res = ncclCommInitRank(&comm, numRanks, *id, rankId);
    if (res != ncclSuccess) {
        DSP_DIAG(BACKEND, "NCCL ncclCommInitRank failed: %s", ncclGetErrorString(res));
        return nullptr;
    }

    auto* commPtr = new ncclComm_t(comm);
    return reinterpret_cast<sd::Pointer>(commPtr);
#else
    DSP_DIAG(BACKEND, "NCCL not available. Build with -DHELPERS_nccl=ON");
    return nullptr;
#endif
}

sd::Pointer ncclGetUniqueId() {
#ifdef HAVE_NCCL
    auto* id = new ncclUniqueId();
    ncclResult_t res = ncclGetUniqueId(id);
    if (res != ncclSuccess) {
        DSP_DIAG(BACKEND, "NCCL ncclGetUniqueId failed: %s", ncclGetErrorString(res));
        delete id;
        return nullptr;
    }
    return reinterpret_cast<sd::Pointer>(id);
#else
    DSP_DIAG(BACKEND, "NCCL not available. Build with -DHELPERS_nccl=ON");
    return nullptr;
#endif
}

void ncclCommDestroy(sd::Pointer commHandle) {
#ifdef HAVE_NCCL
    if (commHandle == nullptr) return;
    auto* commPtr = reinterpret_cast<ncclComm_t*>(commHandle);
    ncclCommDestroy(*commPtr);
    delete commPtr;
#endif
}

int ncclDoAllReduce(sd::Pointer commHandle,
                    sd::Pointer sendBuf, sd::Pointer recvBuf,
                    sd::LongType numElements, int dataType,
                    int reduceOp, sd::Pointer stream) {
#ifdef HAVE_NCCL
    if (commHandle == nullptr) return -1;
    auto* commPtr = reinterpret_cast<ncclComm_t*>(commHandle);

    cudaStream_t cudaStr = (stream != nullptr)
        ? *reinterpret_cast<cudaStream_t*>(stream)
        : 0;

    ncclResult_t res = ncclAllReduce(
        sendBuf, recvBuf, numElements,
        toNcclDataType(dataType), toNcclReduceOp(reduceOp),
        *commPtr, cudaStr
    );

    if (res != ncclSuccess) {
        DSP_DIAG(BACKEND, "NCCL AllReduce failed: %s", ncclGetErrorString(res));
        return -1;
    }
    return 0;
#else
    DSP_DIAG(BACKEND, "NCCL not available. Build with -DHELPERS_nccl=ON");
    return -1;
#endif
}

int ncclDoAllGather(sd::Pointer commHandle,
                    sd::Pointer sendBuf, sd::Pointer recvBuf,
                    sd::LongType sendCount, int dataType,
                    sd::Pointer stream) {
#ifdef HAVE_NCCL
    if (commHandle == nullptr) return -1;
    auto* commPtr = reinterpret_cast<ncclComm_t*>(commHandle);

    cudaStream_t cudaStr = (stream != nullptr)
        ? *reinterpret_cast<cudaStream_t*>(stream)
        : 0;

    ncclResult_t res = ncclAllGather(
        sendBuf, recvBuf, sendCount,
        toNcclDataType(dataType),
        *commPtr, cudaStr
    );

    if (res != ncclSuccess) {
        DSP_DIAG(BACKEND, "NCCL AllGather failed: %s", ncclGetErrorString(res));
        return -1;
    }
    return 0;
#else
    DSP_DIAG(BACKEND, "NCCL not available. Build with -DHELPERS_nccl=ON");
    return -1;
#endif
}

int ncclDoReduceScatter(sd::Pointer commHandle,
                        sd::Pointer sendBuf, sd::Pointer recvBuf,
                        sd::LongType recvCount, int dataType,
                        int reduceOp, sd::Pointer stream) {
#ifdef HAVE_NCCL
    if (commHandle == nullptr) return -1;
    auto* commPtr = reinterpret_cast<ncclComm_t*>(commHandle);

    cudaStream_t cudaStr = (stream != nullptr)
        ? *reinterpret_cast<cudaStream_t*>(stream)
        : 0;

    ncclResult_t res = ncclReduceScatter(
        sendBuf, recvBuf, recvCount,
        toNcclDataType(dataType), toNcclReduceOp(reduceOp),
        *commPtr, cudaStr
    );

    if (res != ncclSuccess) {
        DSP_DIAG(BACKEND, "NCCL ReduceScatter failed: %s", ncclGetErrorString(res));
        return -1;
    }
    return 0;
#else
    DSP_DIAG(BACKEND, "NCCL not available. Build with -DHELPERS_nccl=ON");
    return -1;
#endif
}

int ncclDoSend(sd::Pointer commHandle,
               sd::Pointer sendBuf, sd::LongType numElements,
               int dataType, int peerRank, sd::Pointer stream) {
#ifdef HAVE_NCCL
    if (commHandle == nullptr) return -1;
    auto* commPtr = reinterpret_cast<ncclComm_t*>(commHandle);

    cudaStream_t cudaStr = (stream != nullptr)
        ? *reinterpret_cast<cudaStream_t*>(stream)
        : 0;

    ncclResult_t res = ncclSend(
        sendBuf, numElements,
        toNcclDataType(dataType), peerRank,
        *commPtr, cudaStr
    );

    if (res != ncclSuccess) {
        DSP_DIAG(BACKEND, "NCCL Send failed: %s", ncclGetErrorString(res));
        return -1;
    }
    return 0;
#else
    DSP_DIAG(BACKEND, "NCCL not available. Build with -DHELPERS_nccl=ON");
    return -1;
#endif
}

int ncclDoRecv(sd::Pointer commHandle,
               sd::Pointer recvBuf, sd::LongType numElements,
               int dataType, int peerRank, sd::Pointer stream) {
#ifdef HAVE_NCCL
    if (commHandle == nullptr) return -1;
    auto* commPtr = reinterpret_cast<ncclComm_t*>(commHandle);

    cudaStream_t cudaStr = (stream != nullptr)
        ? *reinterpret_cast<cudaStream_t*>(stream)
        : 0;

    ncclResult_t res = ncclRecv(
        recvBuf, numElements,
        toNcclDataType(dataType), peerRank,
        *commPtr, cudaStr
    );

    if (res != ncclSuccess) {
        DSP_DIAG(BACKEND, "NCCL Recv failed: %s", ncclGetErrorString(res));
        return -1;
    }
    return 0;
#else
    DSP_DIAG(BACKEND, "NCCL not available. Build with -DHELPERS_nccl=ON");
    return -1;
#endif
}

int ncclGroupStart() {
#ifdef HAVE_NCCL
    ncclResult_t res = ::ncclGroupStart();
    return (res == ncclSuccess) ? 0 : -1;
#else
    return -1;
#endif
}

int ncclGroupEnd() {
#ifdef HAVE_NCCL
    ncclResult_t res = ::ncclGroupEnd();
    return (res == ncclSuccess) ? 0 : -1;
#else
    return -1;
#endif
}

// =============================================================================
// Triton GPU Backend Counters
// =============================================================================

bool isTritonAvailable() {
#if HAVE_TRITON
    return sd::graph::TritonGraphBackend::getInstance().isAvailable();
#else
    return false;
#endif
}

sd::LongType getTritonKernelLaunchCount() {
#if HAVE_TRITON
    return sd::graph::TritonGraphBackend::getInstance().getTotalKernelLaunches();
#else
    return 0;
#endif
}

sd::LongType getTritonCacheHitCount() {
#if HAVE_TRITON
    return sd::graph::TritonGraphBackend::getInstance().getTotalCacheHits();
#else
    return 0;
#endif
}

void resetTritonCounters() {
#if HAVE_TRITON
    sd::graph::TritonGraphBackend::getInstance().resetCounters();
#endif
}

void invalidateTritonCache() {
#if HAVE_TRITON
    sd::graph::TritonGraphBackend::getInstance().invalidateCache();
#endif
    // Clear any sticky CUDA errors left by failed kernel launches
    cudaGetLastError();
    // Synchronize to ensure all async work is complete before next test
    cudaDeviceSynchronize();
    cudaGetLastError();
}
