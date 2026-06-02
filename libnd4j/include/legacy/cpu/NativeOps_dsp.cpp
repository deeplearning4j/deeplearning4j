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

#include <dsp/NativeOpsDsp.h>
#include <graph/DspBufferPool.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspVerifyUtils.h>
#include <legacy/NativeOps.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/NativePlanCache.h>
#include <graph/NativePlanCompiler.h>
#include <graph/SdnbReader.h>
#include <graph/SdzReader.h>
#include <graph/ReplayCacheManager.h>
#include <system/common.h>
#include <system/Environment.h>

#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cctype>
#include <string>
#include <sstream>
#include <thread>

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
      // Validate DataBuffer integrity before passing to plan->execute().
      // Return error code 5 (STALE_BUFFER) with the bad input index encoded
      // in the error message so Java can re-resolve only that input and retry.
      auto* db = inputPtrs[i]->dataBuffer();
      if (db != nullptr) {
        if (db->isClosed() || db->isDestroyed() || !db->isValid()) {
          char buf[256];
          snprintf(buf, sizeof(buf),
                   "executeDynamicShapePlan: stale buffer at input %d (closed=%d destroyed=%d valid=%d)",
                   i, db->isClosed() ? 1 : 0, db->isDestroyed() ? 1 : 0, db->isValid() ? 1 : 0);
          DSP_DIAG(EXECUTE, "%s", buf);
          setError(5, buf);
          return 5;
        }
        DSP_DIAG(EXECUTE, "executeDSP: input[%d] ndarray=%p db=%p closed=%d const=%d special=%p primary=%p destroyed=%d valid=%d lenBytes=%lld",
                 i, (void*)inputPtrs[i], (void*)db, db->isClosed() ? 1 : 0,
                 db->isConstant ? 1 : 0, db->special(), db->primary(),
                 db->isDestroyed() ? 1 : 0, db->isValid() ? 1 : 0,
                 (long long)db->getLenInBytes());
      }

      // DIAGNOSTIC: Dump raw bytes and INT64 values for INT64 inputs (to find source of negative gather indices)
      if (inputPtrs[i]->dataType() == sd::DataType::INT64 && db != nullptr) {
        void* primary = db->primary();
        sd::LongType len = inputPtrs[i]->lengthOf();
        if (primary != nullptr && len > 0 && len <= 32) {
          int8_t* rawBytes = reinterpret_cast<int8_t*>(primary);
          sd::LongType offset = inputPtrs[i]->offset();
          sd_printf("DSP_INT64_INPUT: idx=%d len=%lld offset=%lld dtype=%d addr=%p\n",
                    i, (long long)len, (long long)offset,
                    (int)inputPtrs[i]->dataType(), primary);
          // Print raw bytes
          sd_printf("  rawBytes[0..%lld]: ", (long long)(len * 8 - 1));
          for (sd::LongType b = 0; b < len * 8; b++) {
            sd_printf("%02x ", (unsigned char)rawBytes[b]);
          }
          sd_printf("\n");
          // Print as INT64 values
          int64_t* asInt64 = reinterpret_cast<int64_t*>(primary);
          sd_printf("  asInt64[]: ");
          for (sd::LongType e = 0; e < len; e++) {
            sd_printf("%lld ", (long long)asInt64[e + offset]);
          }
          sd_printf("\n");
          // Print as INT32 values (first 4 per element)
          int32_t* asInt32 = reinterpret_cast<int32_t*>(primary);
          sd_printf("  asInt32[]: ");
          for (sd::LongType e = 0; e < len * 2; e++) {
            sd_printf("%d ", asInt32[e]);
          }
          sd_printf("\n");
        }
      }
    }

    // Initialize outputPtrs to nullptr — NOT from opContext->outputArray(i).
    //
    // Rationale: the Java side places dummy Nd4j.empty(FLOAT) arrays (shape=[])
    // into the opContext output slots before calling here.  If we pre-populate
    // outputPtrs from those dummies and plan->execute() then writes nullptr for
    // an output whose slotIdx==-1 (CONSTANT/VARIABLE/pruned-op-output), the
    // post-execute loop sees nullptr and skips setOutputArray — leaving the dummy
    // in _fastpath_out.  Java then reads shape [] from getOutputArrayNative and
    // crashes downstream (e.g. "Illegal set of indices for array: []").
    //
    // By starting with nullptr, plan->execute() can only fill slots it actually
    // produced.  Any slot that stays nullptr after execute is either a
    // CONSTANT/VARIABLE (Java handles those before calling getOutputArrayNative)
    // or a genuine plan bug (Java throws a clear RuntimeException at index i).
    std::vector<NDArray*> outputPtrs(numOutputs, nullptr);

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

// ─── NativePlanCache JNI entry points ────────────────────────────────────────

sd::Pointer createNativePlanCache() {
  try {
    return reinterpret_cast<sd::Pointer>(new sd::graph::NativePlanCache());
  } catch (const std::exception& e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

void freeNativePlanCache(sd::Pointer cacheHandle) {
  if (!cacheHandle) return;
  delete reinterpret_cast<sd::graph::NativePlanCache*>(cacheHandle);
}

void clearNativePlanCacheHandle(sd::Pointer cacheHandle) {
  if (!cacheHandle) return;
  reinterpret_cast<sd::graph::NativePlanCache*>(cacheHandle)->clear();
}

sd::Pointer dispatchNativePlan(sd::Pointer cacheHandle,
                               sd::Pointer planBytes,
                               sd::LongType planBytesLen,
                               sd::Pointer outputNames,
                               sd::LongType numOutputs,
                               sd::Pointer phShapeInfoPtrs,
                               sd::LongType numPlaceholders,
                               int graphExecutionMode) {
  try {
    if (!cacheHandle) throw std::runtime_error("dispatchNativePlan: cacheHandle is null");

    auto* cache = reinterpret_cast<sd::graph::NativePlanCache*>(cacheHandle);

    // Build the output-set hash: FNV-1a over sorted names (order-independent).
    // outputNames is a char** (array of C-string pointers), matching compileDynamicShapePlan's
    // requestedOutputNames convention (passed from Java as PointerPointer of BytePointers).
    auto** namesArr = reinterpret_cast<const char**>(outputNames);
    std::vector<std::string> names;
    names.reserve(numOutputs);
    for (sd::LongType i = 0; i < numOutputs; i++) {
      if (namesArr[i]) names.emplace_back(namesArr[i]);
    }
    std::sort(names.begin(), names.end());

    uint64_t h = 14695981039346656037ULL;  // FNV-1a offset basis
    for (auto& n : names) {
      for (char c : n) { h ^= static_cast<uint8_t>(c); h *= 1099511628211ULL; }
      h ^= 0; h *= 1099511628211ULL;  // NUL separator
    }

    sd::graph::NativePlanCache::Key key;
    key.outputSetHash = h;
    auto** ptrs = reinterpret_cast<sd::LongType**>(phShapeInfoPtrs);
    key.phShapeContentHash = sd::graph::NativePlanCache::hashShapeInfoContents(ptrs, numPlaceholders);
    key.phCount = numPlaceholders;
    key.graphExecutionMode = graphExecutionMode;
    // Thread isolation: hash std::this_thread::get_id() into a uint64_t so each
    // thread gets its own plan instance with independent execution state.
    key.threadId = std::hash<std::thread::id>{}(std::this_thread::get_id());

    // Factory: deserialize and build the plan on cold miss.
    // Mode is passed to fromSerializedPlan so it's set BEFORE buildSegments() —
    // segments are born with the correct selectedBackend (one flow, no reclassification).
    auto factory = [&]() -> sd::graph::NativeDynamicShapePlan* {
      return NativeDynamicShapePlan::fromSerializedPlan(
          planBytes, planBytesLen,
          static_cast<sd::graph::GraphExecutionMode>(graphExecutionMode));
    };

    auto* plan = cache->getOrInsert(key, factory);
    return reinterpret_cast<sd::Pointer>(plan);
  } catch (const std::exception& e) {
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
    return nullptr;
  }
}

void unpinNativePlan(sd::Pointer cacheHandle, sd::Pointer planHandle) {
  if (!cacheHandle || !planHandle) return;
  auto* cache = reinterpret_cast<sd::graph::NativePlanCache*>(cacheHandle);
  auto* plan  = reinterpret_cast<sd::graph::NativeDynamicShapePlan*>(planHandle);
  cache->unpinPlan(plan);
}

void setPlanCacheShutdownInProgress(bool inProgress) {
  sd::graph::NativePlanCache::setShutdownInProgress(inProgress);
  if (inProgress) {
    sd::graph::DspBufferPool::destroyAll();
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

int releaseGpuIntermediates(sd::Pointer planHandle) {
  if (planHandle == nullptr) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return plan->releaseGpuIntermediates();
}

// ─── Replay diagnostics (Phase 2) ──────────────────────────────────────────

unsigned long long getPlanReplaySignatureHash(sd::Pointer planHandle, int segIdx) {
  if (planHandle == nullptr || segIdx < 0) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  const auto& segments = plan->getSegments();
  if (segIdx >= static_cast<int>(segments.size())) return 0;
  return segments[segIdx].exec.replaySignatureHash;
}

int getPlanReplayUnitCount(sd::Pointer planHandle, int segIdx) {
  if (planHandle == nullptr || segIdx < 0) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  const auto& segments = plan->getSegments();
  if (segIdx >= static_cast<int>(segments.size())) return 0;
  return segments[segIdx].exec.replayUnitCount;
}

int getSegmentExecutionCount(sd::Pointer planHandle, int segIdx) {
  if (planHandle == nullptr || segIdx < 0) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  const auto& segments = plan->getSegments();
  if (segIdx >= static_cast<int>(segments.size())) return -1;
  return segments[segIdx].exec.executionCount;
}

int getPlanSegmentCount(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return static_cast<int>(plan->getSegments().size());
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

void setPlanShapeOnlyMode(sd::Pointer planHandle, bool enabled) {
  if (planHandle != nullptr) {
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setShapeOnlyMode(enabled);
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

// =============================================================================
// Per-Segment Replay State (CPU backend)
// =============================================================================

int getPlanSegmentReplayState(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return -1;
  auto& seg = segs[segmentIdx];
  // Composite capture uses merged/composite replay handles instead of the
  // monolithic replayHandle (which is an EMPTY sentinel for workspace only).
  // Report READY when any composite handle is ready.
  if (plan->hasCompositeHandles(seg)) {
    return static_cast<int>(sd::graph::ReplayState::READY);
  }
  if (!seg.exec.replayHandle) return -1;
  return static_cast<int>(seg.exec.replayHandle->getState());
}

int getPlanSegmentReplayCount(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return 0;
  auto& seg = segs[segmentIdx];
  // Aggregate replay counts from composite handles (merged + per-island).
  // Composite capture stores graphs in mergedReplayHandles/compositeReplayHandles,
  // NOT in the monolithic replayHandle (which is an EMPTY sentinel).
  auto& sched = seg.exec.compositeReplaySchedule;
  int compositeReplays = 0;
  for (auto& h : sched.mergedReplayHandles) {
    if (h != nullptr && h->isReady()) {
      compositeReplays += h->getStatistics().replayCount;
    }
  }
  for (auto& h : sched.compositeReplayHandles) {
    if (h != nullptr && h->isReady()) {
      compositeReplays += h->getStatistics().replayCount;
    }
  }
  if (compositeReplays > 0) return compositeReplays;
  // Fallback to monolithic handle
  if (!seg.exec.replayHandle) return 0;
  return seg.exec.replayHandle->getStatistics().replayCount;
}

const char* getPlanSegmentBackendName(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return "";
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return "";
  auto& seg = segs[segmentIdx];
  // Use compiledByBackend first — it's set for both composite and monolithic paths
  if (!seg.exec.compiledByBackend.empty()) return seg.exec.compiledByBackend.c_str();
  if (!seg.exec.replayHandle) return "";
  return seg.exec.replayHandle->backendName();
}

const char* getPlanSegmentStatisticsJson(sd::Pointer planHandle, int segmentIdx) {
  static thread_local char buf[1024];
  if (planHandle == nullptr) { buf[0] = '\0'; return buf; }
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) { buf[0] = '\0'; return buf; }
  auto& seg = segs[segmentIdx];
  int numOps = seg.def.endSlot - seg.def.startSlot + 1;
  // Composite-aware: prefer compiledByBackend, aggregate composite replay counts
  const char* backend = !seg.exec.compiledByBackend.empty()
      ? seg.exec.compiledByBackend.c_str()
      : (seg.exec.replayHandle ? seg.exec.replayHandle->backendName() : "");
  int replayCount = 0;
  int replayState = -1;
  int mergedGroups = static_cast<int>(seg.exec.compositeReplaySchedule.mergedReplayHandles.size());
  int compositeIslands = 0;
  // Check composite handles first
  bool hasComposite = plan->hasCompositeHandles(seg);
  auto& sched = seg.exec.compositeReplaySchedule;
  if (hasComposite) {
    replayState = static_cast<int>(sd::graph::ReplayState::READY);
    for (auto& h : sched.mergedReplayHandles) {
      if (h && h->isReady()) replayCount += h->getStatistics().replayCount;
    }
    for (auto& h : sched.compositeReplayHandles) {
      if (h && h->isReady()) { replayCount += h->getStatistics().replayCount; compositeIslands++; }
    }
  } else if (seg.exec.replayHandle) {
    replayCount = seg.exec.replayHandle->getStatistics().replayCount;
    replayState = static_cast<int>(seg.exec.replayHandle->getState());
  }
  snprintf(buf, sizeof(buf),
           "{\"numOperations\":%d,\"replayCount\":%d,\"replayState\":%d,"
           "\"backendName\":\"%s\",\"executionCount\":%d,"
           "\"capturable\":%s,\"compilationFailed\":%s,\"compiledByBackend\":\"%s\","
           "\"mergedGroups\":%d,\"compositeIslands\":%d}",
           numOps, replayCount, replayState, backend, seg.exec.executionCount,
           seg.def.isCapturable ? "true" : "false",
           seg.exec.compilationFailed ? "true" : "false",
           seg.exec.compiledByBackend.c_str(),
           mergedGroups, compositeIslands);
  return buf;
}

int getPlanSegmentExecutionCount(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return 0;
  return segs[segmentIdx].exec.executionCount;
}

int getPlanSegmentExecutionPhase(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return -1;
  return segs[segmentIdx].exec.getExecutionPhaseCode();
}

int getPlanPhase(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return plan->getPlanPhaseCode();
}

int getPlanPointersStable(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return plan->arePointersStable() ? 1 : 0;
}

int getPlanFrozenExecutionCount(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  if (!plan->isShapesFrozen()) return -1;
  auto& segs = plan->getSegments();
  int maxExecCount = 0;
  for (auto& seg : segs) {
    if (seg.exec.executionCount > maxExecCount)
      maxExecCount = seg.exec.executionCount;
  }
  return maxExecCount;
}

int getPlanSlotState(sd::Pointer planHandle, int slotIdx) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return plan->getSlotStateCode(slotIdx);
}

const char* getPlanSlotOpName(sd::Pointer planHandle, int slotIdx) {
  if (planHandle == nullptr) return "";
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  if (slotIdx < 0 || slotIdx >= plan->getNumSlots()) return "";
  return plan->getSlots()[slotIdx].ident.opName.c_str();
}

int getPlanSlotFlags(sd::Pointer planHandle, int slotIdx) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  if (slotIdx < 0 || slotIdx >= plan->getNumSlots()) return -1;
  auto& slot = plan->getSlots()[slotIdx];
  int flags = 0;
  if (slot.isViewCapableOp())                    flags |= (1 << 0);
  if (slot.isDataDependent())                    flags |= (1 << 1);
  if (slot.flags.outputShapeDependsOnInputValues) flags |= (1 << 2);
  if (slot.isIdentityOp())                       flags |= (1 << 3);
  if (slot.flags.inPlaceFused)                  flags |= (1 << 4);
  if (slot.fusedChain.isFusedChainHead)              flags |= (1 << 5);
  if (slot.fusedChain.isFusedChainTail)              flags |= (1 << 6);
  if (slot.needsZeroedOutput())                  flags |= (1 << 7);
  if (slot.flags.needsIntLongSync)              flags |= (1 << 8);
  if (slot.shapeCache.shapeStatic)                   flags |= (1 << 9);
  if (slot.slotPhase.isSealed() && slot.slotPhase.isConstant) flags |= (1 << 10);
  return flags;
}

int getPlanSlotIOCounts(sd::Pointer planHandle, int slotIdx,
                         int* numInputsOut, int* numOutputsOut) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  if (slotIdx < 0 || slotIdx >= plan->getNumSlots()) return -1;
  auto& slot = plan->getSlots()[slotIdx];
  if (numInputsOut) *numInputsOut = slot.wiring.numInputs;
  if (numOutputsOut) *numOutputsOut = slot.wiring.numOutputs;
  return 0;
}

bool isPlanSegmentCapturable(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return false;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return false;
  return segs[segmentIdx].def.isCapturable;
}

bool isPlanSegmentCaptureFailed(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return false;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return false;
  return segs[segmentIdx].exec.compilationFailed;
}

// =============================================================================
// Per-Segment Pointer Tracking (CPU backend)
// =============================================================================

const char* getPlanSegmentTrackedPointers(sd::Pointer planHandle, int segmentIdx) {
  static thread_local std::string result;
  if (planHandle == nullptr) { result = "[]"; return result.c_str(); }
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) { result = "[]"; return result.c_str(); }
  auto& seg = segs[segmentIdx];
  if (!seg.exec.replayHandle) { result = "[]"; return result.c_str(); }

  auto& addrs = seg.exec.replayHandle->getCapturedExternalAddresses();
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < addrs.size(); ++i) {
    if (i > 0) ss << ",";
    ss << "{\"inputIdx\":" << i
       << ",\"capturedAddr\":\"0x" << std::hex << addrs[i] << std::dec << "\""
       << ",\"match\":true}";
  }
  ss << "]";
  result = ss.str();
  return result.c_str();
}

int getPlanSegmentNumCaptureBuffers(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return 0;
  auto& seg = segs[segmentIdx];
  if (!seg.exec.replayHandle) return 0;
  return 0;
}

const char* getPlanSegmentCaptureBuffersJson(sd::Pointer planHandle, int segmentIdx) {
  static thread_local std::string result;
  if (planHandle == nullptr) { result = "[]"; return result.c_str(); }
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) { result = "[]"; return result.c_str(); }
  auto& seg = segs[segmentIdx];
  if (!seg.exec.replayHandle) { result = "[]"; return result.c_str(); }

  result = "[]";
  return result.c_str();
}

int getPlanSegmentNumHostPointers(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return 0;
  auto& seg = segs[segmentIdx];
  // Aggregate from composite handles first
  int total = 0;
  for (auto& h : seg.exec.compositeReplaySchedule.mergedReplayHandles) {
    if (h) total += static_cast<int>(h->getCapturedHostPtrs().size());
  }
  for (auto& h : seg.exec.compositeReplaySchedule.compositeReplayHandles) {
    if (h) total += static_cast<int>(h->getCapturedHostPtrs().size());
  }
  if (total > 0) return total;
  if (!seg.exec.replayHandle) return 0;
  return static_cast<int>(seg.exec.replayHandle->getCapturedHostPtrs().size());
}

// =============================================================================
// Replay Cache Management (CPU backend)
// =============================================================================

bool isReplayCacheEnabled() {
  return sd::graph::ReplayCacheManager::getInstance().isEnabled();
}

int getReplayCacheHits() {
  return sd::graph::ReplayCacheManager::getInstance().getCacheHits();
}

int getReplayCacheMisses() {
  return sd::graph::ReplayCacheManager::getInstance().getCacheMisses();
}

void clearReplayCache() {
  sd::graph::ReplayCacheManager::getInstance().clearAll();
}

const char* getReplayCacheDir() {
  static thread_local std::string dir;
  dir = sd::graph::ReplayCacheManager::getInstance().getCacheDir();
  return dir.c_str();
}

const char* getReplayCacheDeviceStatsJson() {
  static thread_local std::string result;
  result = sd::graph::ReplayCacheManager::getInstance().getDeviceCacheStatsJson();
  return result.c_str();
}

int getReplayCacheDeviceEntryCount(int deviceType, int deviceIndex) {
  using namespace sd::graph;
  auto key = ReplayCacheDeviceKey(static_cast<sd::modelparallel::DeviceType>(deviceType), deviceIndex, "");
  return ReplayCacheManager::getInstance().getDeviceCacheEntryCount(key);
}

void clearReplayCacheForDevice(int deviceType, int deviceIndex) {
  using namespace sd::graph;
  auto key = ReplayCacheDeviceKey(static_cast<sd::modelparallel::DeviceType>(deviceType), deviceIndex, "");
  ReplayCacheManager::getInstance().clearDevice(key);
}

bool migrateReplayCache(int fromType, int fromIdx, int toType, int toIdx) {
  using namespace sd::graph;
  auto from = ReplayCacheDeviceKey::fromDeviceManager(
      static_cast<sd::modelparallel::DeviceType>(fromType), fromIdx);
  auto to = ReplayCacheDeviceKey::fromDeviceManager(
      static_cast<sd::modelparallel::DeviceType>(toType), toIdx);
  return ReplayCacheManager::getInstance().migrateDeviceCache(from, to);
}

int pruneStaleReplayCacheDevices() {
  return sd::graph::ReplayCacheManager::getInstance().pruneStaleDevices();
}

int loadReplayCacheForDevice(sd::Pointer planHandle, int deviceType, int deviceIndex) {
  using namespace sd::graph;
  auto key = ReplayCacheDeviceKey::fromDeviceManager(
      static_cast<sd::modelparallel::DeviceType>(deviceType), deviceIndex);
  return ReplayCacheManager::getInstance().loadAllForDevice(key);
}

const char* getReplayCachedDevicesJson() {
  static thread_local std::string result;
  result = sd::graph::ReplayCacheManager::getInstance().getCachedDevicesJson();
  return result.c_str();
}

// =============================================================================
// Backend Plan Management (CPU backend)
// =============================================================================

const char* getPlanAvailableBackends(sd::Pointer planHandle) {
  static thread_local std::string result;
  result = "[{\"name\":\"CPU\",\"type\":\"CPU\",\"available\":true,\"priority\":0}]";
  return result.c_str();
}

const char* getPlanSegmentCompiledBackend(sd::Pointer planHandle, int segIdx) {
  if (planHandle == nullptr) return "";
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segIdx < 0 || segIdx >= static_cast<int>(segs.size())) return "";
  static thread_local std::string result;
  result = segs[segIdx].exec.compiledByBackend;
  return result.c_str();
}

const char* getPlanSegmentCompilationAudit(sd::Pointer planHandle, int segIdx) {
  static thread_local std::string result;
  result = "{}";
  return result.c_str();
}

void invalidatePlanSegmentCache(sd::Pointer planHandle, int segIdx) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegmentsMutable();
  if (segIdx < 0 || segIdx >= static_cast<int>(segs.size())) return;
  auto& seg = segs[segIdx];
  seg.exec.replayHandle.reset();
  seg.exec.outcome = SegmentExecOutcome::PENDING;
  seg.exec.cachedShapeKey = 0;
  seg.exec.executionCount = 0;
  seg.exec.compilationFailed = false;
  seg.exec.compiledByBackend.clear();
}

void invalidatePlanBackendCaches(sd::Pointer planHandle, const char* backendName) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  std::string name = backendName ? backendName : "";
  for (auto& seg : plan->getSegmentsMutable()) {
    if (seg.exec.compiledByBackend == name || name.empty()) {
      seg.exec.replayHandle.reset();
      seg.exec.outcome = SegmentExecOutcome::PENDING;
      seg.exec.cachedShapeKey = 0;
      seg.exec.executionCount = 0;
      seg.exec.compiledByBackend.clear();
    }
  }
}

const char* getPlanBackendCacheStats(sd::Pointer planHandle) {
  static thread_local std::string result;
  result = "{\"backends\":[{\"name\":\"CPU\",\"compiledSegments\":0}]}";
  if (planHandle != nullptr) {
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    int compiled = 0;
    for (const auto& seg : plan->getSegments()) {
      if (!seg.exec.compiledByBackend.empty() && seg.exec.compiledByBackend != "slot-by-slot") compiled++;
    }
    std::ostringstream ss;
    ss << "{\"backends\":[{\"name\":\"CPU\",\"compiledSegments\":" << compiled << "}]}";
    result = ss.str();
  }
  return result.c_str();
}

void setPlanSegmentBackendOverride(sd::Pointer planHandle, int segIdx, const char* backendName) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegmentsMutable();
  if (segIdx < 0 || segIdx >= static_cast<int>(segs.size())) return;
  segs[segIdx].def.backendOverride = backendName ? backendName : "";
}

void setPlanBackendPriority(sd::Pointer planHandle, const char* priorityList) {
  // CPU backend has only one backend, priority doesn't apply
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

// Triton availability — check compile-time flag
bool isTritonAvailable() {
#if HAVE_TRITON
  return true;
#else
  return false;
#endif
}
sd::LongType getTritonKernelLaunchCount() { return 0; }
sd::LongType getTritonCacheHitCount() { return 0; }
void resetTritonCounters() {}
void invalidateTritonCache() {}

// Triton cache bundle — CPU backend has no Triton, stubs return error
int exportTritonCacheBundle(const char* outputPath) { return -1; }
int importTritonCacheBundle(const char* bundlePath, bool validateArch) { return -1; }
const char* inspectTritonCacheBundle(const char* bundlePath) {
  static const char* err = "{\"error\": \"Triton not available on CPU backend\"}";
  return err;
}

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

// ─── Freeze config + segment summary ─────────────────────────────────────────

void setDspFreezeMergeSegments(bool enable) {
    sd::Environment::getInstance().setDspFreezeMergeSegments(enable);
}

void setDspFreezeRecompile(bool enable) {
    sd::Environment::getInstance().setDspFreezeRecompile(enable);
}

bool getDspFreezeMergeSegments() {
    return sd::Environment::getInstance().dspFreezeMergeSegments();
}

bool getDspFreezeRecompile() {
    return sd::Environment::getInstance().dspFreezeRecompile();
}

const char* getPlanSegmentsSummaryJson(sd::Pointer planHandle) {
    static thread_local std::string result;
    if (planHandle == nullptr) { result = "[]"; return result.c_str(); }
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    auto& segs = plan->getSegments();
    auto* slots = plan->getSlots();

    std::string json = "[";
    for (int i = 0; i < (int)segs.size(); i++) {
        auto& seg = segs[i];
        int numOps = seg.def.endSlot - seg.def.startSlot + 1;
        std::unordered_map<std::string, int> opCounts;
        if (slots != nullptr) {
            for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
                opCounts[slots[s].ident.opName]++;
            }
        }
        if (i > 0) json += ",";
        json += "{";
        json += "\"index\":" + std::to_string(i);
        json += ",\"startSlot\":" + std::to_string(seg.def.startSlot);
        json += ",\"endSlot\":" + std::to_string(seg.def.endSlot);
        json += ",\"numOps\":" + std::to_string(numOps);
        json += ",\"executionCount\":" + std::to_string(seg.exec.executionCount);
        json += ",\"isCapturable\":" + std::string(seg.def.isCapturable ? "true" : "false");
        json += ",\"compilationFailed\":" + std::string(seg.exec.compilationFailed ? "true" : "false");
        json += ",\"hasReplayHandle\":" + std::string(seg.exec.replayHandle ? "true" : "false");
        json += ",\"shapeKey\":" + std::to_string(seg.def.shapeKeyState.compiledShapeKey);
        json += ",\"shapeKeyStatus\":\"" + std::string(seg.def.shapeKeyState.statusName()) + "\"";
        json += ",\"ops\":{";
        bool first = true;
        for (auto& kv : opCounts) {
            if (!first) json += ",";
            json += "\"" + kv.first + "\":" + std::to_string(kv.second);
            first = false;
        }
        json += "}";
        json += "}";
    }
    json += "]";
    result = json;
    return result.c_str();
}

// ─── Output validation ──────────────────────────────────────────────────────

int dspValidateOutputs(sd::Pointer planHandle, int* flagsOut) {
    if (planHandle == nullptr || flagsOut == nullptr) return -1;
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    int numOutputs = plan->getNumRequestedOutputs();
    if (numOutputs <= 0) return 0;

    // Collect the last-execution output arrays from the plan's output slot indices
    auto* def = plan->getPlanDefinition();
    if (def == nullptr) return -1;
    auto* outputSlots = plan->getOutputSlots();
    if (outputSlots == nullptr) return -1;

    std::vector<NDArray*> outputs(numOutputs);
    for (int i = 0; i < numOutputs; i++) {
        int slotIdx = def->requestedOutputSlotIndices()[i];
        outputs[i] = (slotIdx >= 0 && slotIdx < plan->getTotalOutputSlots()) ? outputSlots[slotIdx] : nullptr;
    }
    return sd::graph::dspValidateOutputs(outputs.data(), numOutputs, flagsOut);
}

int dspDetectStaleOutputs(sd::Pointer planHandle, float* prevNorms, bool* staleOut, float epsilon) {
    if (planHandle == nullptr || prevNorms == nullptr || staleOut == nullptr) return -1;
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    int numOutputs = plan->getNumRequestedOutputs();
    if (numOutputs <= 0) return 0;

    auto* def = plan->getPlanDefinition();
    if (def == nullptr) return -1;
    auto* outputSlots = plan->getOutputSlots();
    if (outputSlots == nullptr) return -1;

    std::vector<NDArray*> outputs(numOutputs);
    for (int i = 0; i < numOutputs; i++) {
        int slotIdx = def->requestedOutputSlotIndices()[i];
        outputs[i] = (slotIdx >= 0 && slotIdx < plan->getTotalOutputSlots()) ? outputSlots[slotIdx] : nullptr;
    }
    return sd::graph::dspDetectStaleOutputs(outputs.data(), numOutputs, prevNorms, staleOut, epsilon);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FrozenPlan Hierarchy API (Step 6)
// ═══════════════════════════════════════════════════════════════════════════════

int executeFrozenPlan(sd::Pointer planHandle, OpaqueContext* opContext, sd::Pointer stream) {
    // Step 1 compatibility: delegates to existing executeDynamicShapePlan
    return executeDynamicShapePlan(planHandle, opContext, stream);
}

int isFrozenPlanSealed(sd::Pointer planHandle) {
    if (planHandle == nullptr) return -1;
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    // Sealed = plan phase >= REPLAYING (phase code 2)
    return plan->getPlanPhaseCode() >= 2 ? 1 : 0;
}

int getFrozenPlanBuildPassCount(sd::Pointer planHandle) {
    if (planHandle == nullptr) return -1;
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    // Map from PlanPhase: SLOT_BY_SLOT=0, SHAPES_FROZEN=1, REPLAYING=2
    // buildPassCount semantics: 0=warmup, 1=compile, 2=sealed
    int phase = plan->getPlanPhaseCode();
    if (phase >= 2) return 2;  // REPLAYING = sealed
    return phase;
}

int getSegmentExecutorPhase(sd::Pointer planHandle, int segIdx) {
    if (planHandle == nullptr) return -1;
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    const auto& segments = plan->getSegments();
    if (segIdx < 0 || segIdx >= static_cast<int>(segments.size())) return -1;

    // Map SegmentLifecycleState to SegmentExecutor::Phase
    // NEEDS_WARMUP/NEEDS_COMPILE/CAPTURE_PENDING/CAPTURED = BUILDING (0)
    // REPLAYING = SEALED (1)
    // FAILED/OOM_DEFERRED = FAILED (2)
    int phaseCode = segments[segIdx].exec.getExecutionPhaseCode();
    switch (phaseCode) {
        case 0: return 0;  // WARMUP → BUILDING
        case 1: return 0;  // COMPILING → BUILDING
        case 2: return 0;  // COMPILED → BUILDING
        case 3: return 1;  // REPLAYING → SEALED
        case 4: return 2;  // SLOT_BY_SLOT (failed) → FAILED
        default: return -1;
    }
}

// ─── External input variable management ─────────────────────────────────────

void markPlanExternalInputVariable(sd::Pointer planHandle, int extIdx) {
  if (planHandle == nullptr) return;
  reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->markExternalInputVariable(extIdx);
}

int getPlanNumCachedVariableExtIndices(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getNumCachedVariableExtIndices();
}

int getPlanCachedVariableExtIndex(sd::Pointer planHandle, int i) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getCachedVariableExtIndex(i);
}

void markPlanExternalInputPlaceholder(sd::Pointer planHandle, int extIdx) {
  if (planHandle == nullptr) return;
  reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->markExternalInputPlaceholder(extIdx);
}

bool getPlanIsExternalInputVariable(sd::Pointer planHandle, int extIdx) {
  if (planHandle == nullptr) return false;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->isExternalInputVariable(extIdx);
}

bool getPlanIsExternalInputPlaceholder(sd::Pointer planHandle, int extIdx) {
  if (planHandle == nullptr) return false;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->isExternalInputPlaceholder(extIdx);
}

int getPlanNumVariableExternalInputs(sd::Pointer planHandle) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getNumVariableExternalInputs();
}

int getPlanExecuteCount(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getExecuteCount();
}

// =============================================================================
// DSP Diagnostics (additional counters)
// =============================================================================

int dspDiagGetStepCount() {
  return sd::graph::DspDiagnostics::getInstance().getStepsExecuted();
}

long long dspDiagGetTotalEventCount() {
  return sd::graph::DspDiagnostics::getInstance().getTotalEventCount();
}

long long dspDiagGetCategoryEventCount(int categoryIndex) {
  return sd::graph::DspDiagnostics::getInstance().getCategoryEventCount(categoryIndex);
}

// =============================================================================
// Staging Buffer Introspection
// =============================================================================

int getPlanNumStagingBuffers(sd::Pointer planHandle) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getNumStagingBuffers();
}

long long getPlanStagingBufferAddress(sd::Pointer planHandle, int extIdx) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getStagingBufferAddress(extIdx);
}

long long getPlanEffectiveExternalAddress(sd::Pointer planHandle, int extIdx) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getEffectiveExternalAddress(extIdx);
}

long long getPlanLastExternalInputAddress(sd::Pointer planHandle, int extIdx) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getLastExternalInputAddress(extIdx);
}

OpaqueNDArray getPlanStagingBufferArray(sd::Pointer planHandle, int extIdx) {
  if (planHandle == nullptr) return nullptr;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getStagingBufferArray(extIdx);
}

int copyPlanStagingToBuffer(sd::Pointer planHandle, int extIdx, OpaqueDataBuffer* dstBuffer) {
  if (planHandle == nullptr) return -1;
  if (dstBuffer == nullptr) return -3;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->copyStagingToBuffer(extIdx, dstBuffer->dataBuffer());
}

// =============================================================================
// Slot Output Introspection
// =============================================================================

OpaqueNDArray getPlanSlotOutputArray(sd::Pointer planHandle, int slotIdx) {
  if (planHandle == nullptr) return nullptr;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getSlotOutputArray(slotIdx);
}

int getTotalPlanOutputSlots(sd::Pointer planHandle) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getTotalOutputSlots();
}

int getPlanSlotGeneration(sd::Pointer planHandle, int slotIdx) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getSlotGeneration(slotIdx);
}

// =============================================================================
// Replay Mode & Arg Generation
// =============================================================================

int getPlanSegmentReplayMode(sd::Pointer planHandle, int segIdx) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getSegmentReplayMode(segIdx);
}

long long getPlanSegmentArgGeneration(sd::Pointer planHandle, int segIdx) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getSegmentArgGeneration(segIdx);
}

long long getPlanSegmentCapturedArgGeneration(sd::Pointer planHandle, int segIdx) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getSegmentCapturedArgGeneration(segIdx);
}

int getPlanSegmentNeedsArgRefresh(sd::Pointer planHandle, int segIdx) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getSegmentNeedsArgRefresh(segIdx);
}

long long getPlanSegmentCapturedInputAddrKey(sd::Pointer planHandle, int segIdx) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getSegmentCapturedInputAddrKey(segIdx);
}

// =============================================================================
// Per-Execution Stats
// =============================================================================

int getLastExecSegmentsWarmup(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getLastExecSegmentsWarmup();
}

int getLastExecSegmentsCaptured(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getLastExecSegmentsCaptured();
}

int getLastExecSegmentsReplayed(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getLastExecSegmentsReplayed();
}

int getLastExecSegmentsSlotBySlot(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getLastExecSegmentsSlotBySlot();
}

int getLastExecSegmentsFailed(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getLastExecSegmentsFailed();
}

int getLastExecSegmentsTotal(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getLastExecSegmentsTotal();
}

int getLastExecSyncLevel(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getLastExecSyncLevel();
}

int getLastExecStreamSyncCount(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getLastExecStreamSyncCount();
}

int getLastExecConsecutiveUnchangedCount(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getLastExecConsecutiveUnchangedCount();
}

// =============================================================================
// Cross-Stream Testing API (CPU stubs — no CUDA streams on CPU)
// =============================================================================

sd::Pointer dspCreateTestStream() {
  return nullptr;  // No streams on CPU
}

void dspDestroyTestStream(sd::Pointer streamPtr) {
  // No-op on CPU
}

int dspWriteDeviceBufferOnDefaultStream(sd::Pointer planHandle, int extIdx, sd::Pointer srcHost, long long numBytes) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->writeDeviceBufferOnDefaultStream(
      extIdx, reinterpret_cast<void*>(srcHost), numBytes);
}

int dspWriteDeviceBufferOnExplicitStream(sd::Pointer planHandle, int extIdx, sd::Pointer srcHost, long long numBytes, sd::Pointer streamPtr) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->writeDeviceBufferOnExplicitStream(
      extIdx, reinterpret_cast<void*>(srcHost), numBytes, reinterpret_cast<void*>(streamPtr));
}

int dspSyncStream(sd::Pointer streamPtr) {
  return 0;  // No-op on CPU — always "synced"
}

int dspIsExtInputDeviceAuthoritative(sd::Pointer planHandle, int extIdx) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->isExtInputDeviceAuthoritative(extIdx);
}

sd::Pointer dspGetExecutionStream(sd::Pointer planHandle) {
  return nullptr;  // No streams on CPU
}

sd::Pointer dspGetDefaultStream() {
  return nullptr;  // No streams on CPU
}

// ── Buffer coloring introspection ────────────────────────────────────────────

bool getPlanBufferColoringApplied(sd::Pointer planHandle) {
  if (planHandle == nullptr) return false;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->bufferColorMap().isApplied();
}

int getPlanBufferColoringNumColors(sd::Pointer planHandle) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->bufferColorMap().numColors();
}

sd::LongType getPlanBufferColoringBytesSaved(sd::Pointer planHandle) {
  if (planHandle == nullptr) return 0;
  return static_cast<sd::LongType>(
      reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->bufferColorMap().estimatedBytesSaved());
}

int getPlanSlotColor(sd::Pointer planHandle, int slotIdx) {
  if (planHandle == nullptr) return -1;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->bufferColorMap().colorOf(slotIdx);
}

// ── Buffer pool introspection ────────────────────────────────────────────────

sd::LongType getBufferPoolPooledBytes(int deviceId) {
  return static_cast<sd::LongType>(DspBufferPool::forDevice(deviceId).pooledBytes());
}

int getBufferPoolPooledCount(int deviceId) {
  return DspBufferPool::forDevice(deviceId).pooledCount();
}

sd::LongType getBufferPoolTotalAcquired(int deviceId) {
  return static_cast<sd::LongType>(DspBufferPool::forDevice(deviceId).totalAcquired());
}

sd::LongType getBufferPoolTotalReused(int deviceId) {
  return static_cast<sd::LongType>(DspBufferPool::forDevice(deviceId).totalReused());
}
