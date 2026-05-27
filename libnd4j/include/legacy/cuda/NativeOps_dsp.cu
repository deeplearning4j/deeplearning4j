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
#include <dsp/NativeOpsDsp.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspVerifyUtils.h>
#include <helpers/ShapeUtils.h>
#include <array/DataTypeUtils.h>
#include <legacy/NativeOps.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/NativePlanCache.h>
#include <graph/Context.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
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
#include <thread>
#include <algorithm>
#include <sstream>
#include <map>
#include <graph/ReplayCacheManager.h>
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
      // Validate DataBuffer integrity — return error code 5 (STALE_BUFFER) with the
      // bad input index so Java can re-resolve only that input and retry.
      auto* db = inputPtrs[i]->dataBuffer();
      if (db != nullptr && (db->isClosed() || db->isDestroyed() || !db->isValid())) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "executeDynamicShapePlan: stale buffer at input %d (closed=%d destroyed=%d valid=%d)",
                 i, db->isClosed() ? 1 : 0, db->isDestroyed() ? 1 : 0, db->isValid() ? 1 : 0);
        DSP_DIAG(EXECUTE, "%s", buf);
        setError(5, buf);
        return 5;
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

    // Validate external input GPU pointers before execution.
    // Detect corrupted/freed _specialBuffer pointers that would cause SIGSEGV
    // in cudaMemcpyAsync during slot execution.
    {
      int badCount = 0;
      for (int i = 0; i < numInputs; i++) {
        auto* arr = inputPtrs[i];
        if (arr == nullptr) continue;
        auto* db = arr->dataBuffer();
        if (db == nullptr || db->isClosed()) continue;
        void* specialBuf = db->special();
        if (specialBuf == nullptr) continue;  // null is OK — will be allocated on demand

        cudaPointerAttributes attrs;
        auto attrRes = cudaPointerGetAttributes(&attrs, specialBuf);
        if (attrRes != cudaSuccess) {
          cudaGetLastError();  // clear sticky error
          if (badCount < 5) {
            DSP_DIAG(MEMORY, "input[%d] INVALID _specialBuffer=%p (err=%d: %s) "
                     "rank=%d len=%lld dtype=%d pAct=%d sAct=%d",
                     i, specialBuf, static_cast<int>(attrRes), cudaGetErrorString(attrRes),
                     arr->rankOf(), (long long)arr->lengthOf(),
                     static_cast<int>(arr->dataType()),
                     db->isPrimaryActual() ? 1 : 0, db->isSpecialActual() ? 1 : 0);
            cudaGetLastError();  // clear error from GetErrorString
          }
          badCount++;
          // Null out the invalid pointer so syncToSpecial can reallocate
          db->setSpecialBuffer(nullptr, 0);
        }
      }
      if (badCount > 0) {
        DSP_DIAG(MEMORY, "%d of %d inputs had invalid GPU pointers (nulled for recovery)",
                 badCount, numInputs);
      }
    }

    void* cudaStream = (stream != nullptr) ? reinterpret_cast<void*>(stream) : nullptr;

    // Trim the CUDA memory pool to reclaim reserved-but-unused memory.
    // Only trim during early executions (warmup/capture) when large transient
    // allocations create pool bloat. In frozen steady-state replay, no new
    // allocations occur and trimPool's internal cudaStreamSynchronize on dirty
    // free streams would needlessly block the CPU.
    if (!plan->isShapesFrozen() || !plan->isCompilationSealed()) {
      int devId = 0;
      cudaGetDevice(&devId);
      memory::CudaMemoryPool::getInstance().trimPool(devId);
      cudaGetLastError();  // clear any trim error
    }

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

    // ── Non-blocking post-execute CUDA error check ─────────────────────────
    // Check for sticky CUDA errors (e.g., error 700 from kernels that wrote to
    // invalid GPU memory) WITHOUT blocking the CPU. platformEndExecution() already
    // handles stream ordering via event-based sync (cudaEventRecord +
    // cudaStreamWaitEvent) — a full cudaDeviceSynchronize here would serialize the
    // entire GPU pipeline and destroy throughput (measured: 9.6 vs 87 tok/s).
    //
    // cudaPeekAtLastError() returns any sticky error from a prior async kernel
    // failure without blocking. If a kernel truly wrote to invalid memory, the
    // error will be caught here on this step or the next.
    {
      cudaError_t peekErr = cudaPeekAtLastError();
      if (peekErr != cudaSuccess) {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "POST-EXECUTE CUDA ERROR: cudaPeekAtLastError after plan->execute() "
                 "returned error %d (%s). A kernel during execution accessed invalid GPU memory. "
                 "numInputs=%d numOutputs=%d",
                 static_cast<int>(peekErr), cudaGetErrorString(peekErr),
                 numInputs, numOutputs);
        DSP_DIAG(EXECUTE, "%s", buf);
        cudaGetLastError(); // clear sticky error
        setError(static_cast<int>(peekErr), buf);
        return static_cast<int>(peekErr);
      }
    }

    // Write output arrays back to context so Java can read them
    for (int i = 0; i < numOutputs; i++) {
      if (outputPtrs[i] != nullptr) {
        opContext->setOutputArray(i, outputPtrs[i], false);
        auto* arr = outputPtrs[i];
        if (Environment::getInstance().isDebugAndVerbose()) {
          auto shape = ShapeUtils::shapeAsString(arr);
          auto type = DataTypeUtils::asString(arr->dataType());
          auto shapeInfoStr = ShapeUtils::shapeInfoAsString(arr->shapeInfo());
          DSP_DIAG(EXECUTE, "PLAN_OUTPUT[%d] shapeInfo=%s dtype=%s len=%lld isView=%d "
                   "ptr=%p specialPtr=%p asyncValues=true",
                   i, shapeInfoStr.c_str(), type.c_str(),
                   (long long)arr->lengthOf(), arr->isView() ? 1 : 0,
                   (void*)arr, arr->dataBuffer() != nullptr ? arr->dataBuffer()->special() : nullptr);
        }
      }
    }

    return 0;
  } catch (const std::exception& e) {
    DSP_DIAG(EXECUTE, "executeDynamicShapePlan: exception: %s", e.what());
    // Clear any sticky CUDA errors and ensure stream is not in capture mode
    cudaGetLastError();
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
    if (plan) {
      DSP_DIAG(COMPILE, "dispatchNativePlan: plan=%p slots=%d contentHash=0x%016llx cacheSize=%zu",
               (void*)plan, plan->getNumSlots(),
               (unsigned long long)key.phShapeContentHash, cache->size());
      if (DSP_DIAG_ENABLED(COMPILE)) {
        for (sd::LongType pi = 0; pi < numPlaceholders; pi++) {
          const sd::LongType* si = ptrs[pi];
          if (si == nullptr) {
            DSP_DIAG(COMPILE, "  ph[%lld]: NULL", (long long)pi);
            continue;
          }
          auto infoStr = ShapeUtils::shapeInfoAsString(si);
          DSP_DIAG(COMPILE, "  ph[%lld]: %s", (long long)pi, infoStr.c_str());
        }
      }
    }
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
  int freed = plan->releaseGpuIntermediates();

  // Trim the CUDA memory pool after freeing intermediates.
  // releaseGpuIntermediates routes frees through stream-ordered CUDA APIs.
  int deviceId = 0;
  cudaGetDevice(&deviceId);
  memory::CudaMemoryPool::getInstance().trimPool(deviceId);

  return freed;
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

// isPlanCompilationSealed / getPlanMidExecutionCompileCount / resetPlanMidExecutionCompileCount
// live in legacy/impl/NativeOps_dsp_shared.cpp so they are shared between CPU and CUDA builds.

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

void setPlanOutputSlotMaxSizes(sd::Pointer planHandle, sd::LongType numSlots,
                                 const int* slotIndices, const sd::LongType* maxSizes) {
  if (planHandle == nullptr || numSlots <= 0 || slotIndices == nullptr || maxSizes == nullptr) return;
  reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setOutputSlotMaxSizes(slotIndices, maxSizes, static_cast<int>(numSlots));
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

// Helper: extract native CudaGraphHandle from a replayHandle (returns nullptr if not CUDA)
static sd::cuda::CudaGraphHandle* getNativeCudaGraph(const GraphSegment& seg) {
  if (!seg.exec.replayHandle || !seg.exec.replayHandle->isReady()) return nullptr;
  auto* cudaReplay = dynamic_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
  if (!cudaReplay) return nullptr;
  auto* nativeHandle = cudaReplay->getNativeHandle();
  return nativeHandle;
}

void printPlanCapturedGraphDebug(sd::Pointer planHandle) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  plan->printCaptureAudit();

  // Also print graph contents for each cached segment
  for (const auto& seg : plan->getSegments()) {
    auto* nativeGraph = getNativeCudaGraph(seg);
    if (nativeGraph) {
      nativeGraph->printGraphContents();
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
    int segSlots = seg.def.endSlot - seg.def.startSlot + 1;
    if (!seg.def.isCapturable) {
      nonCapt++;
      nonCaptSlots += segSlots;
    } else if (seg.exec.replayHandle && seg.exec.replayHandle->isReady()) {
      captured++;
      captSlots += segSlots;
    } else if (seg.exec.compilationFailed) {
      permFailed++;
      permSlots += segSlots;
    } else if (seg.exec.captureOomRetries > 0) {
      oomRetrying++;
      oomSlots += segSlots;
      if (seg.exec.captureOomRetries > maxOomRetries) maxOomRetries = seg.exec.captureOomRetries;
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

// =============================================================================
// Per-Segment Replay State (CUDA backend)
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
  if (hasComposite) {
    replayState = static_cast<int>(sd::graph::ReplayState::READY);
    for (auto& h : seg.exec.compositeReplaySchedule.mergedReplayHandles) {
      if (h && h->isReady()) replayCount += h->getStatistics().replayCount;
    }
    for (auto& h : seg.exec.compositeReplaySchedule.compositeReplayHandles) {
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
  // Return the executeCount_ which tracks executions since shapes were frozen
  // (exposed indirectly via the segments — but we need a direct accessor)
  // Use the plan phase to infer: frozenExecCount is internal, but we can
  // iterate segments to count. For simplicity, return the total graph replays
  // as a proxy — or better, just expose via getPlanPhaseCode context.
  // Actually we have postFreezeExecCount_ but no getter yet — use segments.
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
// Per-Segment Pointer Tracking (CUDA backend)
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
  // Aggregate host ptrs from composite handles
  int total = 0;
  for (auto& h : seg.exec.compositeReplaySchedule.mergedReplayHandles) {
    if (h) total += static_cast<int>(h->getCapturedHostPtrs().size());
  }
  for (auto& h : seg.exec.compositeReplaySchedule.compositeReplayHandles) {
    if (h) total += static_cast<int>(h->getCapturedHostPtrs().size());
  }
  if (total > 0) return total;
  // Fallback to monolithic handle
  if (!seg.exec.replayHandle) return 0;
  return static_cast<int>(seg.exec.replayHandle->getCapturedHostPtrs().size());
}

// =============================================================================
// Replay Cache Management (CUDA backend)
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
// Backend Plan Management (CUDA backend)
// =============================================================================

const char* getPlanAvailableBackends(sd::Pointer planHandle) {
  static thread_local std::string result;
  // List CUDA-available backends
  result = "[{\"name\":\"CUDA\",\"type\":\"GPU\",\"available\":true,\"priority\":0}";
#ifdef SD_HAVE_TRITON
  result += ",{\"name\":\"Triton\",\"type\":\"GPU\",\"available\":true,\"priority\":1}";
#endif
  result += ",{\"name\":\"NVRTC\",\"type\":\"GPU\",\"available\":true,\"priority\":2}";
  result += ",{\"name\":\"PTX\",\"type\":\"GPU\",\"available\":true,\"priority\":3}";
  result += "]";
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
  if (planHandle == nullptr) { result = "{}"; return result.c_str(); }
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segIdx < 0 || segIdx >= static_cast<int>(segs.size())) { result = "{}"; return result.c_str(); }
  // Query the GPU backend for compilation audit if available
  result = plan->getSegmentCompilationAudit(segIdx);
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
  if (planHandle == nullptr) { result = "{}"; return result.c_str(); }
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);

  std::map<std::string, int> backendCounts;
  for (const auto& seg : plan->getSegments()) {
    if (!seg.exec.compiledByBackend.empty()) {
      backendCounts[seg.exec.compiledByBackend]++;
    }
  }

  std::ostringstream ss;
  ss << "{\"backends\":[";
  bool first = true;
  for (const auto& entry : backendCounts) {
    if (!first) ss << ",";
    first = false;
    ss << "{\"name\":\"" << entry.first << "\",\"compiledSegments\":" << entry.second << "}";
  }
  ss << "]}";
  result = ss.str();
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
  if (planHandle == nullptr || priorityList == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  // Parse comma-separated priority list
  std::string list(priorityList);
  std::vector<std::string> priority;
  std::istringstream iss(list);
  std::string token;
  while (std::getline(iss, token, ',')) {
    // Trim whitespace
    size_t start = token.find_first_not_of(" \t");
    size_t end = token.find_last_not_of(" \t");
    if (start != std::string::npos) {
      priority.push_back(token.substr(start, end - start + 1));
    }
  }
  plan->setBackendPriority(priority);
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
    auto* nativeGraph = getNativeCudaGraph(seg);
    if (nativeGraph) {
      // Get the Chrome trace JSON for this segment
      std::string traceJson = nativeGraph->getChromeTraceJson();
      // Parse out the traceEvents array and add segment prefix
      // For simplicity, we'll just include the whole trace with segment metadata
      if (!first) file << ",\n";
      first = false;

      // Add segment metadata
      file << "    {\"name\": \"segment_" << segmentIdx << "\", \"ph\": \"M\", \"pid\": 0, "
           << "\"tid\": 0, \"ts\": 0, \"args\": {\"startSlot\": " << seg.def.startSlot
           << ", \"endSlot\": " << seg.def.endSlot << "}},\n";

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
    auto* nativeGraph = getNativeCudaGraph(seg);
    if (nativeGraph) {
      return nativeGraph->exportToHtml(std::string(outputPath));
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
    auto* nativeGraph = getNativeCudaGraph(seg);
    if (nativeGraph) {
      std::string basePath = std::string(outputPath) + "_seg" + std::to_string(segmentIdx);
      if (nativeGraph->debugDump(basePath)) {
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
    auto* nativeGraph = getNativeCudaGraph(seg);
    if (nativeGraph) {
      std::string segJson = nativeGraph->getChromeTraceJson();
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
    auto* nativeGraph = getNativeCudaGraph(seg);
    if (nativeGraph) {
      nativeGraph->clearExecutionTimeline();
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
    cudaGetLastError();
}

// ─── Triton cache bundle export/import ──────────────────────────────────────

#if HAVE_TRITON
// Defined in TritonCacheBundle.cpp
namespace sd::graph {
  int exportTritonCacheBundleImpl(const char* outputPath);
  int importTritonCacheBundleImpl(const char* bundlePath, bool validateArch);
  const char* inspectTritonCacheBundleImpl(const char* bundlePath);
}
#endif

int exportTritonCacheBundle(const char* outputPath) {
#if HAVE_TRITON
    return sd::graph::exportTritonCacheBundleImpl(outputPath);
#else
    return -1;
#endif
}

int importTritonCacheBundle(const char* bundlePath, bool validateArch) {
#if HAVE_TRITON
    return sd::graph::importTritonCacheBundleImpl(bundlePath, validateArch);
#else
    return -1;
#endif
}

const char* inspectTritonCacheBundle(const char* bundlePath) {
#if HAVE_TRITON
    return sd::graph::inspectTritonCacheBundleImpl(bundlePath);
#else
    static const char* err = "{\"error\": \"Triton not available\"}";
    return err;
#endif
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
        // Count op types in this segment
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
        // Op histogram
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

// ─── Native KV scatter configuration ────────────────────────────────────────

/**
 * Configure plan-managed KV scatter for CUDA-graph-compatible decode loops.
 *
 * After this call, the plan executes a batched KV scatter after each execute(),
 * updating the static KV buffers at the current position and incrementing the
 * position scalar. This eliminates the Java-side scatterNewEntries() round-trip.
 *
 * @param planHandle          The native plan handle
 * @param presentSlotIndices  Array of output slot indices for present KV tensors
 * @param staticKvBuffers     Array of sd::Pointer (NDArray*) for static KV buffers
 * @param numPairs            Number of (present, static) pairs
 * @param dtypeInt            DataType code of the KV tensors
 * @param heads               Number of attention heads
 * @param srcSeqLen           Present sequence length (typically 1 for decode)
 * @param dstSeqLen           Static buffer sequence length (= maxKvLen)
 * @param dim                 Head dimension
 * @param kvPositionPtr       Pointer to device-side int64 position scalar (plan updates it)
 */
void configurePlanKvScatter(sd::Pointer planHandle,
                             const int* presentSlotIndices,
                             const sd::Pointer* staticKvBufferPtrs,
                             sd::LongType numPairs,
                             int dtypeInt,
                             sd::LongType heads,
                             sd::LongType srcSeqLen,
                             sd::LongType dstSeqLen,
                             sd::LongType dim,
                             sd::LongType* kvPositionPtr) {
    if (planHandle == nullptr || presentSlotIndices == nullptr ||
        staticKvBufferPtrs == nullptr || numPairs <= 0 || kvPositionPtr == nullptr) {
        return;
    }

    // Collect NDArray* from opaque Pointer array
    std::vector<NDArray*> staticBufs(numPairs);
    for (sd::LongType i = 0; i < numPairs; i++) {
        staticBufs[i] = reinterpret_cast<NDArray*>(staticKvBufferPtrs[i]);
    }

    auto dtype = static_cast<sd::DataType>(dtypeInt);
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    plan->configureKvScatter(presentSlotIndices, staticBufs.data(),
                              static_cast<int>(numPairs),
                              dtype, heads, srcSeqLen, dstSeqLen, dim,
                              kvPositionPtr);
}

/**
 * Reset the KV cache position managed by the plan (e.g., after prefill).
 */
void resetPlanKvCachePosition(sd::Pointer planHandle, sd::LongType position) {
    if (planHandle == nullptr) return;
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->resetKvCachePosition(position);
}

/**
 * Get the current KV cache position managed by the plan.
 * Returns -1 if KV scatter is not configured.
 */
sd::LongType getPlanKvCachePosition(sd::Pointer planHandle) {
    if (planHandle == nullptr) return -1LL;
    return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getKvCachePosition();
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
    return plan->getPlanPhaseCode() >= 3 ? 1 : 0;
}

int getFrozenPlanBuildPassCount(sd::Pointer planHandle) {
    if (planHandle == nullptr) return -1;
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    int phase = plan->getPlanPhaseCode();
    if (phase >= 3) return 2;  // REPLAYING = sealed
    return phase;
}

int getSegmentExecutorPhase(sd::Pointer planHandle, int segIdx) {
    if (planHandle == nullptr) return -1;
    auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
    const auto& segments = plan->getSegments();
    if (segIdx < 0 || segIdx >= static_cast<int>(segments.size())) return -1;

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
// Cross-Stream Testing API
// =============================================================================

sd::Pointer dspCreateTestStream() {
  cudaStream_t stream;
  auto err = cudaStreamCreate(&stream);
  if (err != cudaSuccess) return nullptr;
  return reinterpret_cast<sd::Pointer>(stream);
}

void dspDestroyTestStream(sd::Pointer streamPtr) {
  if (streamPtr != nullptr) {
    cudaStreamDestroy(reinterpret_cast<cudaStream_t>(streamPtr));
  }
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
  cudaStream_t src = streamPtr == nullptr ? nullptr : reinterpret_cast<cudaStream_t>(streamPtr);
  cudaEvent_t event = nullptr;
  auto err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  err = cudaEventRecord(event, src);
  if (err == cudaSuccess) {
    err = cudaStreamWaitEvent(nullptr, event, 0);
  }
  cudaEventDestroy(event);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  return 0;
}

int dspIsExtInputDeviceAuthoritative(sd::Pointer planHandle, int extIdx) {
  if (planHandle == nullptr) return 0;
  return reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->isExtInputDeviceAuthoritative(extIdx);
}

sd::Pointer dspGetExecutionStream(sd::Pointer planHandle) {
  if (planHandle == nullptr) return nullptr;
  return static_cast<sd::Pointer>(reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->getExecutionStream());
}

sd::Pointer dspGetDefaultStream() {
  auto lc = sd::LaunchContext::defaultContext();
  return lc != nullptr ? reinterpret_cast<sd::Pointer>(lc->getCudaStream()) : nullptr;
}

// ─── Async cross-device buffer copy ─────────────────────────────────────────
// tl_dspExecutionStream is declared extern in DataBuffer.h, defined in DataBuffer.cu

// Replaces the sync host-relay path (D2H + memcpySync H2D) with a single
// cudaMemcpyPeerAsync call. The CUDA driver handles P2P vs host-staging
// internally — no manual host relay needed.
//
// This is the ONLY correct way to do cross-device copies in DSP context:
// - cudaMemcpyPeerAsync is ordered on dstStream — the plan executes on the
//   same stream, so data is visible without any sync
// - No pipeline drain (no cudaStreamSynchronize, no cudaMemcpy blocking)
// - No host relay (no syncToPrimary, no isPrimaryActual checks, no TLS guards)
// - Works with cudaMallocAsync stream-ordered allocations
void dbAsyncCrossDeviceCopy(OpaqueDataBuffer *dstBuffer, OpaqueDataBuffer *srcBuffer, void *dstStream) {
  if (dstBuffer == nullptr || srcBuffer == nullptr) {
    THROW_EXCEPTION("dbAsyncCrossDeviceCopy: null buffer");
  }

  auto* dst = dstBuffer->dataBuffer();
  auto* src = srcBuffer->dataBuffer();
  if (dst == nullptr || src == nullptr) {
    THROW_EXCEPTION("dbAsyncCrossDeviceCopy: null inner DataBuffer");
  }

  void* dstPtr = dst->special();
  void* srcPtr = src->special();
  if (dstPtr == nullptr || srcPtr == nullptr) {
    THROW_EXCEPTION("dbAsyncCrossDeviceCopy: null special (device) buffer");
  }

  size_t bytes = std::min(dst->getLenInBytes(), src->getLenInBytes());
  if (bytes == 0) return;

  // Resolve device IDs from CUDA pointer attributes — metadata can be stale
  int srcDevice = -1, dstDevice = -1;
  {
    cudaPointerAttributes srcAttrs, dstAttrs;
    auto srcRes = cudaPointerGetAttributes(&srcAttrs, srcPtr);
    auto dstRes = cudaPointerGetAttributes(&dstAttrs, dstPtr);
    if (srcRes != cudaSuccess || dstRes != cudaSuccess) {
      cudaGetLastError();
      THROW_EXCEPTION("dbAsyncCrossDeviceCopy: cudaPointerGetAttributes failed");
    }
    srcDevice = srcAttrs.device;
    dstDevice = dstAttrs.device;
  }

  // Resolve the stream to use for the copy
  cudaStream_t stream = nullptr;
  if (dstStream != nullptr) {
    stream = *reinterpret_cast<cudaStream_t*>(dstStream);
  } else {
    // Fall back to the DSP execution stream or default LaunchContext stream
    if (sd::tl_dspExecutionStream != nullptr) {
      stream = sd::tl_dspExecutionStream;
    } else {
      auto* lc = sd::LaunchContext::defaultContext();
      if (lc != nullptr && lc->getCudaStream() != nullptr) {
        stream = *lc->getCudaStream();
      }
    }
  }

  cudaError_t copyRes;
  if (srcDevice == dstDevice) {
    // Same device — async D2D on the stream
    int currentDevice = -1;
    cudaGetDevice(&currentDevice);
    if (currentDevice != dstDevice) cudaSetDevice(dstDevice);
    copyRes = cudaMemcpyAsync(dstPtr, srcPtr, bytes, cudaMemcpyDeviceToDevice, stream);
    if (currentDevice != dstDevice) cudaSetDevice(currentDevice);
  } else {
    // Cross-device — cudaMemcpyPeerAsync handles P2P or host staging internally.
    // The copy is ordered on `stream` which must belong to dstDevice.
    // Ensure we're on the destination device for the stream to be valid.
    int currentDevice = -1;
    cudaGetDevice(&currentDevice);
    if (currentDevice != dstDevice) cudaSetDevice(dstDevice);
    copyRes = cudaMemcpyPeerAsync(dstPtr, dstDevice, srcPtr, srcDevice, bytes, stream);
    if (currentDevice != dstDevice) cudaSetDevice(currentDevice);
  }

  if (copyRes != cudaSuccess) {
    std::string err = "dbAsyncCrossDeviceCopy failed: " + std::string(cudaGetErrorString(copyRes))
        + " bytes=" + std::to_string(bytes)
        + " src(dev=" + std::to_string(srcDevice) + " ptr=" + std::to_string((uintptr_t)srcPtr) + ")"
        + " dst(dev=" + std::to_string(dstDevice) + " ptr=" + std::to_string((uintptr_t)dstPtr) + ")";
    THROW_EXCEPTION(err.c_str());
  }

  // Mark destination as device-actual. No host sync needed — the data
  // flows directly device-to-device (or via driver-managed host staging).
  dst->writeSpecial();

  DSP_DIAG(TRANSFER, "ASYNC_CROSS_DEVICE: %zu bytes src(dev=%d) -> dst(dev=%d) stream=%p",
           bytes, srcDevice, dstDevice, (void*)stream);
}

// FrozenPlan API v1 - cache invalidation marker
