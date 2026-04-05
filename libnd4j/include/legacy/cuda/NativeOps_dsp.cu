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
    }

    std::vector<NDArray*> outputPtrs(numOutputs);
    for (int i = 0; i < numOutputs; i++) {
      outputPtrs[i] = opContext->outputArray(i);
      // Output arrays may be null — plan will allocate them
    }

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

    // Trim the CUDA memory pool before execution to reclaim reserved-but-unused memory.
    // The pool reserves memory for future allocations but doesn't release it back to CUDA.
    // After the first execution, there can be hundreds of MB of reserved-but-unused memory
    // that prevents even basic CUDA operations (stream creation, module loading) from succeeding.
    {
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

    // ── Definitive post-execute CUDA error check ──────────────────────────
    // Synchronize the device and check for latent CUDA errors (e.g., error 700
    // from kernels that wrote to invalid GPU memory). This catches errors that
    // the per-slot diagnostic in executeSegmentSlotBySlot might miss if the error
    // manifests on a different stream or after all slots complete.
    // Without this, the latent error surfaces as "cudaMallocAsync failed" when Java
    // allocates output arrays, making the root cause impossible to identify.
    {
      cudaError_t postExecErr = cudaDeviceSynchronize();
      if (postExecErr != cudaSuccess) {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "POST-EXECUTE CUDA ERROR: cudaDeviceSynchronize after plan->execute() "
                 "returned error %d (%s). A kernel during execution accessed invalid GPU memory. "
                 "numInputs=%d numOutputs=%d",
                 static_cast<int>(postExecErr), cudaGetErrorString(postExecErr),
                 numInputs, numOutputs);
        sd_printf("%s\n", buf);
        cudaGetLastError(); // clear sticky error
        setError(static_cast<int>(postExecErr), buf);
        return static_cast<int>(postExecErr);
      }
      // Also check cudaPeekAtLastError for non-synchronous errors
      cudaError_t peekErr = cudaPeekAtLastError();
      if (peekErr != cudaSuccess) {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "POST-EXECUTE CUDA PEEK ERROR: cudaPeekAtLastError after plan->execute() "
                 "returned error %d (%s). numInputs=%d numOutputs=%d",
                 static_cast<int>(peekErr), cudaGetErrorString(peekErr),
                 numInputs, numOutputs);
        sd_printf("%s\n", buf);
        cudaGetLastError(); // clear sticky error
        setError(static_cast<int>(peekErr), buf);
        return static_cast<int>(peekErr);
      }
    }

    // Write output arrays back to context so Java can read them
    for (int i = 0; i < numOutputs; i++) {
      if (outputPtrs[i] != nullptr) {
        opContext->setOutputArray(i, outputPtrs[i], false);
        // Debug: print output shape
        auto* arr = outputPtrs[i];
        fprintf(stdout, "DSP_DEBUG: output[%d] rank=%d shape=[", i, arr->rankOf());
        for (int d = 0; d < arr->rankOf(); d++) {
          fprintf(stdout, "%lld%s", (long long)arr->sizeAt(d), d < arr->rankOf()-1 ? "," : "");
        }
        fprintf(stdout, "] len=%lld\n", (long long)arr->lengthOf());
        fflush(stdout);
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

int releaseGpuIntermediates(sd::Pointer planHandle) {
  if (planHandle == nullptr) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return plan->releaseGpuIntermediates();
}

int releaseGpuIntermediates(sd::Pointer planHandle, bool preserveDecodeState) {
  if (planHandle == nullptr) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return plan->releaseGpuIntermediates(preserveDecodeState);
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

void configurePlanDecodeInputs(
    sd::Pointer planHandle,
    int inputIdsExtIdx, int positionIdsExtIdx,
    int attentionMaskExtIdx, int maxKvLen) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  plan->configureDecodeInputs(inputIdsExtIdx, positionIdsExtIdx,
                               attentionMaskExtIdx, maxKvLen);
}

void setPlanNextDecodeToken(sd::Pointer planHandle, sd::LongType tokenId, int cachePos) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  plan->setNextDecodeToken(tokenId, cachePos);
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

// Helper: extract native CudaGraphHandle from a replayHandle (returns nullptr if not CUDA)
static sd::cuda::CudaGraphHandle* getNativeCudaGraph(const GraphSegment& seg) {
  if (!seg.exec.replayHandle || !seg.exec.replayHandle->isReady()) return nullptr;
  auto* cudaReplay = dynamic_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
  if (!cudaReplay) return nullptr;
  auto nativeHandle = cudaReplay->getNativeHandle();
  return nativeHandle ? nativeHandle.get() : nullptr;
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
    int segSlots = seg.endSlot - seg.startSlot + 1;
    if (!seg.isCapturable) {
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
  if (!seg.exec.replayHandle) return -1;
  return static_cast<int>(seg.exec.replayHandle->getState());
}

int getPlanSegmentReplayCount(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return 0;
  auto& seg = segs[segmentIdx];
  if (!seg.exec.replayHandle) return 0;
  return seg.exec.replayHandle->getStatistics().replayCount;
}

const char* getPlanSegmentBackendName(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return "";
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return "";
  auto& seg = segs[segmentIdx];
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
  int numOps = seg.endSlot - seg.startSlot + 1;
  const char* backend = (seg.exec.replayHandle) ? seg.exec.replayHandle->backendName() : "";
  int replayCount = (seg.exec.replayHandle) ? seg.exec.replayHandle->getStatistics().replayCount : 0;
  int replayState = (seg.exec.replayHandle) ? static_cast<int>(seg.exec.replayHandle->getState()) : -1;
  snprintf(buf, sizeof(buf),
           "{\"numOperations\":%d,\"replayCount\":%d,\"replayState\":%d,"
           "\"backendName\":\"%s\",\"executionCount\":%d,"
           "\"capturable\":%s,\"compilationFailed\":%s,\"compiledByBackend\":\"%s\"}",
           numOps, replayCount, replayState, backend, seg.exec.executionCount,
           seg.isCapturable ? "true" : "false",
           seg.exec.compilationFailed ? "true" : "false",
           seg.exec.compiledByBackend.c_str());
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
  return static_cast<int>(segs[segmentIdx].exec.currentPhase);
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
  // Actually we have frozenExecutionCount_ but no getter yet — use segments.
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
  return plan->getSlots()[slotIdx].opName.c_str();
}

int getPlanSlotFlags(sd::Pointer planHandle, int slotIdx) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  if (slotIdx < 0 || slotIdx >= plan->getNumSlots()) return -1;
  auto& slot = plan->getSlots()[slotIdx];
  int flags = 0;
  if (slot.isViewCapableOp)               flags |= (1 << 0);
  if (slot.isDataDependent)               flags |= (1 << 1);
  if (slot.outputShapeDependsOnInputValues) flags |= (1 << 2);
  if (slot.isIdentityOp)                  flags |= (1 << 3);
  if (slot.inPlaceFused)                  flags |= (1 << 4);
  if (slot.isFusedChainHead)              flags |= (1 << 5);
  if (slot.isFusedChainTail)              flags |= (1 << 6);
  if (slot.needsZeroedOutput)             flags |= (1 << 7);
  if (slot.needsIntLongSync)              flags |= (1 << 8);
  if (slot.shapeStatic)                   flags |= (1 << 9);
  if (slot.state_ >= NativeSlot::SlotState::FROZEN_CONSTANT) flags |= (1 << 10);
  return flags;
}

int getPlanSlotIOCounts(sd::Pointer planHandle, int slotIdx,
                         int* numInputsOut, int* numOutputsOut) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  if (slotIdx < 0 || slotIdx >= plan->getNumSlots()) return -1;
  auto& slot = plan->getSlots()[slotIdx];
  if (numInputsOut) *numInputsOut = slot.numInputs;
  if (numOutputsOut) *numOutputsOut = slot.numOutputs;
  return 0;
}

bool isPlanSegmentCapturable(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return false;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return false;
  return segs[segmentIdx].isCapturable;
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
  return static_cast<int>(seg.exec.replayHandle->getCaptureBuffers().size());
}

const char* getPlanSegmentCaptureBuffersJson(sd::Pointer planHandle, int segmentIdx) {
  static thread_local std::string result;
  if (planHandle == nullptr) { result = "[]"; return result.c_str(); }
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) { result = "[]"; return result.c_str(); }
  auto& seg = segs[segmentIdx];
  if (!seg.exec.replayHandle) { result = "[]"; return result.c_str(); }

  auto& bufs = seg.exec.replayHandle->getCaptureBuffers();
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < bufs.size(); ++i) {
    if (i > 0) ss << ",";
    ss << "{\"extInputIdx\":" << bufs[i].externalInputIndex
       << ",\"size\":" << bufs[i].capturedSize
       << ",\"directRef\":" << (bufs[i].directReference ? "true" : "false")
       << ",\"neverSkip\":" << (bufs[i].neverSkipCopy ? "true" : "false") << "}";
  }
  ss << "]";
  result = ss.str();
  return result.c_str();
}

int getPlanSegmentNumHostPointers(sd::Pointer planHandle, int segmentIdx) {
  if (planHandle == nullptr) return 0;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  auto& segs = plan->getSegments();
  if (segmentIdx < 0 || segmentIdx >= static_cast<int>(segs.size())) return 0;
  auto& seg = segs[segmentIdx];
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
  segs[segIdx].backendOverride = backendName ? backendName : "";
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
    // Synchronize to ensure all async work is complete before next test
    cudaDeviceSynchronize();
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
        int numOps = seg.endSlot - seg.startSlot + 1;
        // Count op types in this segment
        std::unordered_map<std::string, int> opCounts;
        if (slots != nullptr) {
            for (int s = seg.startSlot; s <= seg.endSlot; s++) {
                opCounts[slots[s].opName]++;
            }
        }
        if (i > 0) json += ",";
        json += "{";
        json += "\"index\":" + std::to_string(i);
        json += ",\"startSlot\":" + std::to_string(seg.startSlot);
        json += ",\"endSlot\":" + std::to_string(seg.endSlot);
        json += ",\"numOps\":" + std::to_string(numOps);
        json += ",\"executionCount\":" + std::to_string(seg.exec.executionCount);
        json += ",\"isCapturable\":" + std::string(seg.isCapturable ? "true" : "false");
        json += ",\"compilationFailed\":" + std::string(seg.exec.compilationFailed ? "true" : "false");
        json += ",\"hasReplayHandle\":" + std::string(seg.exec.replayHandle ? "true" : "false");
        json += ",\"shapeKey\":" + std::to_string(seg.shapeKey);
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
