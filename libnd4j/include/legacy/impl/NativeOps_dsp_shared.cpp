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

//
// Platform-agnostic NativeOps DSP entry points.
//
// These functions are trivial accessors on NativeDynamicShapePlan that have
// identical implementations on CPU and CUDA builds. Keeping them in
// legacy/impl/ means there is exactly one definition shared between both
// backends (BuildCPU.cmake and MainBuildFlow.cmake both glob
// ./include/legacy/impl/*.cpp into the source list).
//

#include <dsp/NativeOpsDsp.h>
#include <execution/LaunchContext.h>
#include <graph/DspDiagnostics.h>
#include <legacy/NativeOps.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/SdnbReader.h>
#include <graph/SdzReader.h>

#include <algorithm>
#include <cctype>
#include <exception>
#include <string>
#include <vector>

using sd::graph::NativeDynamicShapePlan;

namespace {

struct LoadedModelHandle {
  sd::graph::SdnbReader::LoadedModel model;
  sd::graph::SdnbReader* sdnbReader = nullptr;
  sd::graph::SdzReader* sdzReader = nullptr;

  ~LoadedModelHandle() {
    delete sdnbReader;
    delete sdzReader;
  }
};

void setDspNativeError(int code, const std::string& message) {
  auto* error = sd::LaunchContext::defaultContext()->errorReference();
  error->setErrorCode(code);
  error->setErrorMessage(message.c_str());
}

}  // namespace

void dspDiagFlushJson() {
  sd::graph::DspDiagnostics::getInstance().flushJsonReport();
}


int isPlanCompilationSealed(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return plan->isCompilationSealed() ? 1 : 0;
}

long long getPlanMidExecutionCompileCount(sd::Pointer planHandle) {
  if (planHandle == nullptr) return -1;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  return static_cast<long long>(plan->getMidExecutionCompileCount());
}

void resetPlanMidExecutionCompileCount(sd::Pointer planHandle) {
  if (planHandle == nullptr) return;
  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  plan->resetMidExecutionCompileCount();
}

const char* getPlanLifecycleSnapshot(sd::Pointer planHandle) {
  static thread_local std::string result;
  if (planHandle == nullptr) {
    result = "valid=false";
    return result.c_str();
  }

  auto* plan = reinterpret_cast<NativeDynamicShapePlan*>(planHandle);
  const auto& lifecycle = plan->planLifecycle();
  const auto& segments = plan->getSegments();
  int buildingSegments = 0;
  int sealedSegments = 0;
  int failedSegments = 0;
  for (const auto& segment : segments) {
    switch (segment.exec.graphNodePhase()) {
      case sd::graph::GraphNodePhase::BUILDING:
        ++buildingSegments;
        break;
      case sd::graph::GraphNodePhase::SEALED:
        ++sealedSegments;
        break;
      case sd::graph::GraphNodePhase::FAILED:
        ++failedSegments;
        break;
    }
  }

  result = "valid=true;planPhase=" + std::to_string(lifecycle.toLegacyCode()) +
           ";graphNodePhase=" + std::to_string(static_cast<int>(lifecycle.phase)) +
           ";buildStage=" + std::to_string(static_cast<int>(lifecycle.buildStage)) +
           ";executionCount=" + std::to_string(plan->getExecuteCount()) +
           ";postFreezeExecutionCount=" + std::to_string(lifecycle.postFreezeExecCount) +
           ";pointersStableCount=" + std::to_string(lifecycle.pointersStableCount) +
           ";compilationDone=" + (lifecycle.compilationDone ? "true" : "false") +
           ";segmentCount=" + std::to_string(static_cast<int>(segments.size())) +
           ";buildingSegments=" + std::to_string(buildingSegments) +
           ";sealedSegments=" + std::to_string(sealedSegments) +
           ";failedSegments=" + std::to_string(failedSegments);
  return result.c_str();
}

sd::Pointer loadModelFromFile(const char* filePath) {
  return loadModelFromFileWithOptions(filePath, false);
}

sd::Pointer loadModelFromFileWithOptions(const char* filePath,
                                         bool requireFileBacked) {
  LoadedModelHandle* handle = nullptr;
  try {
    if (filePath == nullptr) {
      DSP_DIAG(COMPILE, "loadModelFromFile: null file path");
      setDspNativeError(1, "loadModelFromFile: null file path");
      return nullptr;
    }

    handle = new LoadedModelHandle();
    std::string path(filePath);
    std::string pathLower = path;
    std::transform(pathLower.begin(), pathLower.end(), pathLower.begin(),
                   [](unsigned char c) {
                     return static_cast<char>(std::tolower(c));
                   });
    const bool isSdz = pathLower.size() > 4 &&
                       pathLower.substr(pathLower.size() - 4) == ".sdz";
    const bool isSdnb = pathLower.size() > 5 &&
                        pathLower.substr(pathLower.size() - 5) == ".sdnb";

    if (isSdz) {
      handle->sdzReader = sd::graph::SdzReader::openFile(filePath);
      if (handle->sdzReader == nullptr) {
        DSP_DIAG(COMPILE, "loadModelFromFile: failed to open SDZ file: %s", filePath);
        delete handle;
        return nullptr;
      }
      handle->model = handle->sdzReader->load(true, requireFileBacked);
    } else if (isSdnb) {
      handle->sdnbReader = sd::graph::SdnbReader::openFile(filePath);
      if (handle->sdnbReader == nullptr) {
        DSP_DIAG(COMPILE, "loadModelFromFile: failed to open SDNB file: %s", filePath);
        delete handle;
        return nullptr;
      }
      handle->model =
          handle->sdnbReader->loadAllOwned(true, requireFileBacked);
    } else {
      handle->sdzReader = sd::graph::SdzReader::openFile(filePath);
      if (handle->sdzReader != nullptr) {
        handle->model = handle->sdzReader->load(true, requireFileBacked);
      } else {
        handle->sdnbReader = sd::graph::SdnbReader::openFile(filePath);
        if (handle->sdnbReader == nullptr) {
          DSP_DIAG(COMPILE,
                   "loadModelFromFile: cannot open file as SDZ or SDNB: %s",
                   filePath);
          delete handle;
          return nullptr;
        }
        handle->model =
            handle->sdnbReader->loadAllOwned(true, requireFileBacked);
      }
    }

    if (!handle->model.graph) {
      DSP_DIAG(COMPILE,
               "loadModelFromFile: file did not yield a valid FlatGraph: %s",
               filePath);
      setDspNativeError(
          2, std::string("loadModelFromFile: file did not yield a valid "
                         "file-backed FlatGraph: ") + filePath);
      delete handle;
      return nullptr;
    }

    DSP_DIAG(COMPILE, "loaded model: %d vars, %d placeholders from %s",
             static_cast<int>(handle->model.variables.size()),
             static_cast<int>(handle->model.placeholderNames.size()), filePath);
    setDspNativeError(0, "");
    return reinterpret_cast<sd::Pointer>(handle);
  } catch (const std::exception& e) {
    DSP_DIAG(COMPILE, "loadModelFromFile: exception: %s", e.what());
    setDspNativeError(3, e.what());
    delete handle;
    return nullptr;
  }
}

OpaqueNDArray getLoadedModelVariable(sd::Pointer modelHandle,
                                     const char* variableName) {
  if (modelHandle == nullptr || variableName == nullptr) return nullptr;
  auto* handle = reinterpret_cast<LoadedModelHandle*>(modelHandle);
  auto it = handle->model.variables.find(variableName);
  return it == handle->model.variables.end() ? nullptr : it->second;
}

int getLoadedModelVariableShape(sd::Pointer modelHandle,
                                const char* variableName,
                                sd::LongType* dimensions,
                                int maxRank) {
  if (modelHandle == nullptr || variableName == nullptr || maxRank < 0) return -1;
  auto* handle = reinterpret_cast<LoadedModelHandle*>(modelHandle);
  const auto* graph = handle->model.graph;
  const auto* variables = graph == nullptr ? nullptr : graph->variables();
  if (variables == nullptr) return -1;

  for (unsigned int index = 0; index < variables->size(); ++index) {
    const auto* variable = variables->Get(index);
    if (variable == nullptr || variable->name() == nullptr ||
        variable->name()->str() != variableName) {
      continue;
    }
    const auto* shape = variable->shape();
    if (shape == nullptr || shape->size() == 0) return -1;
    const int rank = static_cast<int>(shape->size());
    if (dimensions != nullptr) {
      if (maxRank < rank) return -1;
      for (int axis = 0; axis < rank; ++axis) {
        dimensions[axis] = static_cast<sd::LongType>(shape->Get(axis));
      }
    }
    return rank;
  }
  return -1;
}

sd::Pointer compileModelPlan(sd::Pointer modelHandle,
                             sd::Pointer requestedOutputNames,
                             int numOutputs) {
  return compileModelPlanWithRuntimeOptions(
      modelHandle, requestedOutputNames, numOutputs,
      static_cast<int>(sd::graph::GraphExecutionMode::GEM_AUTO), true, "", "",
      "");
}

sd::Pointer compileModelPlanWithRuntimeOptions(
    sd::Pointer modelHandle, sd::Pointer requestedOutputNames, int numOutputs,
    int graphExecutionMode, bool runtimeCompilationAllowed,
    const char* runtimeArtifactDirectory,
    const char* deviceCompilationCacheDirectory,
    const char* deviceCompilationCacheModelKey) {
  try {
    if (modelHandle == nullptr) {
      DSP_DIAG(COMPILE, "compileModelPlan: null model handle");
      setDspNativeError(1, "compileModelPlan: null model handle");
      return nullptr;
    }

    auto* handle = reinterpret_cast<LoadedModelHandle*>(modelHandle);
    if (!handle->model.graph) {
      DSP_DIAG(COMPILE, "compileModelPlan: model has no graph");
      setDspNativeError(2, "compileModelPlan: model has no graph");
      return nullptr;
    }
    if (numOutputs < 0 ||
        (numOutputs > 0 && requestedOutputNames == nullptr)) {
      setDspNativeError(3, "compileModelPlan: invalid requested outputs");
      return nullptr;
    }

    auto** outputNames = reinterpret_cast<const char**>(requestedOutputNames);
    std::vector<std::string> outputs;
    for (int i = 0; i < numOutputs; ++i) {
      if (outputNames[i] != nullptr) outputs.emplace_back(outputNames[i]);
    }

    auto mode = static_cast<sd::graph::GraphExecutionMode>(graphExecutionMode);
    if (mode < sd::graph::GraphExecutionMode::GEM_AUTO ||
        mode > sd::graph::GraphExecutionMode::GEM_ONEDNN) {
      mode = sd::graph::GraphExecutionMode::GEM_AUTO;
    }

    sd::graph::NativePlanCompileOptions compileOptions;
    compileOptions.runtimeCompilationAllowed = runtimeCompilationAllowed;
    compileOptions.runtimeArtifactDirectory =
        runtimeArtifactDirectory == nullptr ? "" : runtimeArtifactDirectory;
    compileOptions.deviceCompilationCacheDirectory =
        deviceCompilationCacheDirectory == nullptr
            ? ""
            : deviceCompilationCacheDirectory;
    compileOptions.deviceCompilationCacheModelKey =
        deviceCompilationCacheModelKey == nullptr
            ? ""
            : deviceCompilationCacheModelKey;

    std::string compileError;
    auto* plan = NativeDynamicShapePlan::fromFlatGraph(
        handle->model.graph, handle->model.variables, outputs, mode,
        &compileError, compileOptions);
    if (plan == nullptr) {
      DSP_DIAG(COMPILE, "compileModelPlan: failed to compile plan");
      setDspNativeError(
          4, compileError.empty() ? "compileModelPlan: failed to compile plan"
                                  : compileError);
      return nullptr;
    }

    DSP_DIAG(COMPILE, "compiled model plan: %d slots, %d outputs",
             plan->getNumSlots(), plan->getNumRequestedOutputs());
    setDspNativeError(0, "");
    return reinterpret_cast<sd::Pointer>(plan);
  } catch (const std::exception& e) {
    DSP_DIAG(COMPILE, "compileModelPlan: exception: %s", e.what());
    setDspNativeError(5, e.what());
    return nullptr;
  }
}

void freeLoadedModel(sd::Pointer modelHandle) {
  delete reinterpret_cast<LoadedModelHandle*>(modelHandle);
}

void setPlanGraphExecutionMode(sd::Pointer planHandle, int mode) {
  if (planHandle == nullptr) return;

  const auto requested = static_cast<sd::graph::GraphExecutionMode>(mode);
  auto applied = requested;
  if (applied < sd::graph::GraphExecutionMode::GEM_AUTO ||
      applied > sd::graph::GraphExecutionMode::GEM_ONEDNN) {
    applied = sd::graph::GraphExecutionMode::GEM_AUTO;
  }

  reinterpret_cast<NativeDynamicShapePlan*>(planHandle)
      ->setGraphExecutionMode(applied);
  DSP_DIAG(BACKEND,
           "setPlanGraphExecutionMode: requested=%d(%s) applied=%d(%s)",
           mode,
           sd::graph::ModeContract::modeName(static_cast<int>(requested)),
           static_cast<int>(applied),
           sd::graph::ModeContract::modeName(static_cast<int>(applied)));
}

