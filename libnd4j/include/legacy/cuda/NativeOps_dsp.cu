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

    // Read inputs/outputs from the context's fastpath vectors
    int numInputs = static_cast<int>(opContext->width());
    int numOutputs = static_cast<int>(opContext->outputWidth());

    // Validate input/output counts
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

    // Build input/output arrays from context
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
      // Output arrays may be null — plan will allocate them
    }

    void* cudaStream = (stream != nullptr) ? reinterpret_cast<void*>(stream) : nullptr;

    auto status = plan->execute(inputPtrs.data(), numInputs, outputPtrs.data(), numOutputs, cudaStream);

    // Write output arrays back to context so Java can read them
    for (int i = 0; i < numOutputs; i++) {
      if (outputPtrs[i] != nullptr) {
        opContext->setOutputArray(i, outputPtrs[i], false);
      }
    }

    return (status == Status::OK) ? 0 : static_cast<int>(status);
  } catch (const std::exception& e) {
    sd_printf("executeDynamicShapePlan: exception: %s\n", e.what());
    // Clear any sticky CUDA errors and ensure stream is not in capture mode
    cudaGetLastError();
    if (stream != nullptr) {
      cudaStream_t cudaStr = *reinterpret_cast<cudaStream_t*>(stream);
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();
    }
    // Set error message for Java side
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(e.what());
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

// ─── Model loading (SDZ/SDNB) ───────────────────────────────────────────────

sd::Pointer loadModelFromFile(const char* filePath) {
  try {
    if (filePath == nullptr) {
      sd_printf("loadModelFromFile: null file path\n", "");
      return nullptr;
    }

    auto* handle = new LoadedModelHandle();

    // Determine file type by extension
    std::string path(filePath);
    bool isSdz = path.size() > 4 && path.substr(path.size() - 4) == ".sdz";
    bool isSdnb = path.size() > 5 && path.substr(path.size() - 5) == ".sdnb";

    if (isSdz) {
      handle->sdzReader = SdzReader::openFile(filePath);
      if (!handle->sdzReader) {
        sd_printf("loadModelFromFile: failed to open SDZ file: %s\n", filePath);
        delete handle;
        return nullptr;
      }
      handle->model = handle->sdzReader->load();
    } else if (isSdnb) {
      handle->sdnbReader = SdnbReader::openFile(filePath);
      if (!handle->sdnbReader) {
        sd_printf("loadModelFromFile: failed to open SDNB file: %s\n", filePath);
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
          sd_printf("loadModelFromFile: cannot open file as SDZ or SDNB: %s\n", filePath);
          delete handle;
          return nullptr;
        }
        handle->model = handle->sdnbReader->loadAll();
      }
    }

    sd_printf("loadModelFromFile: loaded %d variables, %d placeholders from %s\n",
              static_cast<int>(handle->model.variables.size()),
              static_cast<int>(handle->model.placeholderNames.size()),
              filePath);

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
    if (modelHandle == nullptr) {
      sd_printf("compileModelPlan: null model handle\n", "");
      return nullptr;
    }

    auto* handle = reinterpret_cast<LoadedModelHandle*>(modelHandle);

    if (!handle->model.graph) {
      sd_printf("compileModelPlan: model has no graph\n", "");
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
      sd_printf("compileModelPlan: failed to compile plan\n", "");
      return nullptr;
    }

    sd_printf("compileModelPlan: compiled %d slots, %d outputs\n",
              plan->getNumSlots(), plan->getNumRequestedOutputs());

    return reinterpret_cast<sd::Pointer>(plan);
  } catch (const std::exception& e) {
    sd_printf("compileModelPlan: exception: %s\n", e.what());
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

void setPlanMinCaptureSegmentSize(sd::Pointer planHandle, int minSize) {
  if (planHandle != nullptr) {
    reinterpret_cast<NativeDynamicShapePlan*>(planHandle)->setMinCaptureSegmentSize(minSize);
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
