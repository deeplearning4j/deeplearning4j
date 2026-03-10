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
 * NativeOps JNI entry points for the native C++ graph executor (DSP) — CPU backend.
 * Split from NativeOps.cpp for modularity.
 *
 * Provides:
 *   - compileDynamicShapePlan / executeDynamicShapePlan / freeDynamicShapePlan
 *   - clearDynamicShapePlanCaches
 *   - loadModelFromFile / compileModelPlan / freeLoadedModel
 *   - getPlanNumExternalInputs / getPlanNumRequestedOutputs / getPlanNumSlots
 */

#include <graph/DspDiagnostics.h>
#include <legacy/NativeOps.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/NativePlanCompiler.h>
#include <graph/SdnbReader.h>
#include <graph/SdzReader.h>
#include <system/common.h>

#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cctype>
#include <string>

using namespace sd;
using namespace sd::graph;

// ─── Loaded model handle wrapper ─────────────────────────────────────────────

struct LoadedModelHandle {
  SdnbReader::LoadedModel model;
  SdnbReader* sdnbReader;
  SdzReader* sdzReader;

  LoadedModelHandle() : sdnbReader(nullptr), sdzReader(nullptr) {}

  ~LoadedModelHandle() {
    // LoadedModel destructor handles variable cleanup
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
  // Keep Java-visible error reporting in sync with CUDA NativeOps_dsp behavior.
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

    int numInputs = static_cast<int>(opContext->width());
    int numOutputs = static_cast<int>(opContext->outputWidth());

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
    }

    // Pass through the execution stream from Java. CUDA-backed DSP execution relies on
    // a consistent stream for Triton launches, KV scatter, and downstream consumers.
    // CPU backends ignore the pointer inside NativeDynamicShapePlan::execute().
    auto status = plan->execute(inputPtrs.data(), numInputs, outputPtrs.data(), numOutputs, stream);

    if (status != Status::OK) {
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
      return static_cast<int>(status);
    }

    for (int i = 0; i < numOutputs; i++) {
      if (outputPtrs[i] != nullptr) {
        opContext->setOutputArray(i, outputPtrs[i], false);
      }
    }

    return 0;
  } catch (const std::exception& e) {
    DSP_DIAG(EXECUTE, "executeDynamicShapePlan: exception: %s", e.what());
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

// ─── Model loading ───────────────────────────────────────────────────────────

sd::Pointer loadModelFromFile(const char* filePath) {
  try {
    if (filePath == nullptr) {
      DSP_DIAG(COMPILE, "loadModelFromFile: null file path");
      return nullptr;
    }

    auto* handle = new LoadedModelHandle();

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
    delete reinterpret_cast<LoadedModelHandle*>(modelHandle);
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

// ─── CUDA Graph control (delegates to plan, CPU has no CUDA Graphs) ─────────

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

    if (gem < sd::graph::GraphExecutionMode::GEM_AUTO || gem > sd::graph::GraphExecutionMode::GEM_NNAPI) {
      gem = sd::graph::GraphExecutionMode::GEM_AUTO;
    }

#ifndef SD_CUDA
    // On non-CUDA builds, map CUDA-specific JIT modes to the closest equivalent.
    // Keep GEM_CUDA_GRAPHS as a distinct "graph replay" request so the plan
    // can route to CPU graph backends (oneDNN/ACL) without forcing AUTO mode.
    if (gem == sd::graph::GraphExecutionMode::GEM_NVRTC_JIT ||
        gem == sd::graph::GraphExecutionMode::GEM_PTX_JIT) {
      gem = sd::graph::GraphExecutionMode::GEM_TRITON;
    }
#endif

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
  // CPU backend: no CUDA graphs, always valid
  return true;
}

int getPlanNumHostOnlyOps(sd::Pointer planHandle) {
  return 0;  // No CUDA graph capture on CPU
}

const char* getPlanHostOnlyOpNames(sd::Pointer planHandle) {
  static const char* empty = "";
  return empty;
}

void printPlanCapturedGraphDebug(sd::Pointer planHandle) {
  // No-op on CPU backend
}

const char* getPlanCaptureStats(sd::Pointer planHandle) {
  static thread_local char buf[64];
  snprintf(buf, sizeof(buf), "cpu-backend");
  return buf;
}

// ─── CUDA Graph Visualization stubs (CPU backend) ─────────────────────────────

bool exportPlanCudaGraphChromeTrace(sd::Pointer planHandle, const char* outputPath) {
  return false;  // Not supported on CPU backend
}

bool exportPlanCudaGraphHtml(sd::Pointer planHandle, const char* outputPath) {
  return false;  // Not supported on CPU backend
}

bool debugDumpPlanCudaGraph(sd::Pointer planHandle, const char* outputPath) {
  return false;  // Not supported on CPU backend
}

const char* getPlanCudaGraphChromeTraceJson(sd::Pointer planHandle) {
  return "";  // Not supported on CPU backend
}

void clearPlanCudaGraphTimeline(sd::Pointer planHandle) {
  // No-op on CPU backend
}

// =============================================================================
// NCCL stubs for CPU backend (NCCL requires CUDA)
// =============================================================================

sd::Pointer ncclCommInit(int numRanks, int rankId, int deviceId) {
    return nullptr;
}

sd::Pointer ncclCommInitWithId(int numRanks, int rankId, sd::Pointer uniqueId) {
    return nullptr;
}

sd::Pointer ncclGetUniqueId() {
    return nullptr;
}

void ncclCommDestroy(sd::Pointer commHandle) {
}

int ncclDoAllReduce(sd::Pointer commHandle,
                    sd::Pointer sendBuf, sd::Pointer recvBuf,
                    sd::LongType numElements, int dataType,
                    int reduceOp, sd::Pointer stream) {
    return -1;
}

int ncclDoAllGather(sd::Pointer commHandle,
                    sd::Pointer sendBuf, sd::Pointer recvBuf,
                    sd::LongType sendCount, int dataType,
                    sd::Pointer stream) {
    return -1;
}

int ncclDoReduceScatter(sd::Pointer commHandle,
                        sd::Pointer sendBuf, sd::Pointer recvBuf,
                        sd::LongType recvCount, int dataType,
                        int reduceOp, sd::Pointer stream) {
    return -1;
}

int ncclDoSend(sd::Pointer commHandle,
               sd::Pointer sendBuf, sd::LongType numElements,
               int dataType, int peerRank, sd::Pointer stream) {
    return -1;
}

int ncclDoRecv(sd::Pointer commHandle,
               sd::Pointer recvBuf, sd::LongType numElements,
               int dataType, int peerRank, sd::Pointer stream) {
    return -1;
}

int ncclGroupStart() {
    return -1;
}

int ncclGroupEnd() {
    return -1;
}

// Triton counters — CPU backend always returns 0
bool isTritonAvailable() { return false; }
sd::LongType getTritonKernelLaunchCount() { return 0; }
sd::LongType getTritonCacheHitCount() { return 0; }
void resetTritonCounters() {}
void invalidateTritonCache() {}

// ─── DSP Diagnostics JNI bridge ──────────────────────────────────────────────

void dspDiagSetCategories(int mask) {
    sd::graph::DspDiagnostics::getInstance().setCategories(static_cast<uint32_t>(mask));
}

void dspDiagEnableCategories(int mask) {
    sd::graph::DspDiagnostics::getInstance().enableCategories(static_cast<uint32_t>(mask));
}

void dspDiagDisableCategories(int mask) {
    sd::graph::DspDiagnostics::getInstance().disableCategories(static_cast<uint32_t>(mask));
}

int dspDiagGetEnabledMask() {
    return static_cast<int>(sd::graph::DspDiagnostics::getInstance().getEnabledMask());
}

void dspDiagSetLevel(int level) {
    sd::graph::DspDiagnostics::getInstance().setLevel(
        static_cast<sd::graph::DspDiagLevel>(level));
}

void dspDiagSetJsonPath(const char* path) {
    if (path != nullptr) {
        sd::graph::DspDiagnostics::getInstance().setJsonPath(path);
    }
}

void dspDiagRecordJavaEvent(int category, int slotId, int segmentId,
                             const char* opName, sd::LongType timingUs,
                             const char* message) {
    if (sd::graph::DspDiagnostics::getInstance().isEnabled(static_cast<uint32_t>(category))) {
        sd::graph::DspDiagnostics::getInstance().recordEvent(
            static_cast<uint32_t>(category), slotId, segmentId, -1,
            opName, static_cast<int64_t>(timingUs), "%s", message != nullptr ? message : "");
    }
}

const char* dspDiagGetPlanReport() {
    // Thread-local static to hold the report string across JNI boundary
    thread_local std::string reportBuf;
    reportBuf = sd::graph::DspDiagnostics::getInstance().generatePlanReport();
    return reportBuf.c_str();
}

const char* dspDiagGetJsonReport() {
    thread_local std::string jsonBuf;
    jsonBuf = sd::graph::DspDiagnostics::getInstance().generateJsonReport();
    return jsonBuf.c_str();
}

void dspDiagClear() {
    sd::graph::DspDiagnostics::getInstance().clear();
}
