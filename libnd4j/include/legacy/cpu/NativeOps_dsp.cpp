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

#include <legacy/NativeOps.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/NativePlanCompiler.h>
#include <graph/SdnbReader.h>
#include <graph/SdzReader.h>
#include <system/common.h>

#include <cstring>
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
      sd_printf("compileDynamicShapePlan: null or empty plan data\n", "");
      return nullptr;
    }

    auto* plan = NativeDynamicShapePlan::fromSerializedPlan(serializedPlan, planSize);
    if (plan == nullptr) {
      sd_printf("compileDynamicShapePlan: failed to parse plan (%lld bytes)\n",
                static_cast<long long>(planSize));
      return nullptr;
    }

    sd_printf("compileDynamicShapePlan: compiled plan with %d slots, %d external inputs, %d outputs\n",
              plan->getNumSlots(), plan->getNumExternalInputs(), plan->getNumRequestedOutputs());

    return reinterpret_cast<sd::Pointer>(plan);
  } catch (const std::exception& e) {
    sd_printf("compileDynamicShapePlan: exception: %s\n", e.what());
    return nullptr;
  }
}

int executeDynamicShapePlan(
    sd::Pointer planHandle,
    OpaqueContext* opContext,
    sd::Pointer stream) {
  try {
    if (planHandle == nullptr) {
      sd_printf("executeDynamicShapePlan: null plan handle\n", "");
      return 1;
    }
    if (opContext == nullptr) {
      sd_printf("executeDynamicShapePlan: null opContext\n", "");
      return 1;
    }

    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);

    int numInputs = static_cast<int>(opContext->width());
    int numOutputs = static_cast<int>(opContext->outputWidth());

    if (numInputs != plan->getNumExternalInputs()) {
      sd_printf("executeDynamicShapePlan: input count mismatch: got %d, expected %d\n",
                numInputs, plan->getNumExternalInputs());
      return 2;
    }
    if (numOutputs != plan->getNumRequestedOutputs()) {
      sd_printf("executeDynamicShapePlan: output count mismatch: got %d, expected %d\n",
                numOutputs, plan->getNumRequestedOutputs());
      return 3;
    }

    std::vector<NDArray*> inputPtrs(numInputs);
    for (int i = 0; i < numInputs; i++) {
      inputPtrs[i] = opContext->array(i);
      if (inputPtrs[i] == nullptr) {
        sd_printf("executeDynamicShapePlan: null input at index %d\n", i);
        return 4;
      }
    }

    std::vector<NDArray*> outputPtrs(numOutputs);
    for (int i = 0; i < numOutputs; i++) {
      outputPtrs[i] = opContext->outputArray(i);
    }

    // CPU backend: stream is always nullptr
    auto status = plan->execute(inputPtrs.data(), numInputs, outputPtrs.data(), numOutputs, nullptr);

    for (int i = 0; i < numOutputs; i++) {
      if (outputPtrs[i] != nullptr) {
        opContext->setOutputArray(i, outputPtrs[i], false);
      }
    }

    return (status == Status::OK) ? 0 : static_cast<int>(status);
  } catch (const std::exception& e) {
    sd_printf("executeDynamicShapePlan: exception: %s\n", e.what());
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
      sd_printf("loadModelFromFile: null file path\n", "");
      return nullptr;
    }

    auto* handle = new LoadedModelHandle();

    std::string path(filePath);
    bool isSdz = path.size() > 4 && path.substr(path.size() - 4) == ".sdz";
    bool isSdnb = path.size() > 5 && path.substr(path.size() - 5) == ".sdnb";

    if (isSdz) {
      handle->sdzReader = SdzReader::openFile(filePath);
      if (!handle->sdzReader) {
        delete handle;
        return nullptr;
      }
      handle->model = handle->sdzReader->load();
    } else if (isSdnb) {
      handle->sdnbReader = SdnbReader::openFile(filePath);
      if (!handle->sdnbReader) {
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
          delete handle;
          return nullptr;
        }
        handle->model = handle->sdnbReader->loadAll();
      }
    }

    return reinterpret_cast<sd::Pointer>(handle);
  } catch (const std::exception& e) {
    sd_printf("loadModelFromFile: exception: %s\n", e.what());
    return nullptr;
  }
}

sd::Pointer compileModelPlan(
    sd::Pointer modelHandle,
    sd::Pointer requestedOutputNames, int numOutputs) {
  try {
    if (modelHandle == nullptr) return nullptr;

    auto* handle = reinterpret_cast<LoadedModelHandle*>(modelHandle);
    if (!handle->model.graph) return nullptr;

    auto** outputNames = reinterpret_cast<const char**>(requestedOutputNames);
    std::vector<std::string> outputs;
    for (int i = 0; i < numOutputs; i++) {
      if (outputNames[i]) {
        outputs.emplace_back(outputNames[i]);
      }
    }

    auto* plan = NativeDynamicShapePlan::fromFlatGraph(
        handle->model.graph, handle->model.variables, outputs);

    return reinterpret_cast<sd::Pointer>(plan);
  } catch (const std::exception& e) {
    sd_printf("compileModelPlan: exception: %s\n", e.what());
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
  // JIT mode only supported on CUDA — no-op on CPU
}

void setPlanGraphExecutionMode(sd::Pointer planHandle, int mode) {
  // Graph execution mode only supported on CUDA — no-op on CPU
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
