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

