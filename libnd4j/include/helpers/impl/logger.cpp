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
// Created by raver119 on 31.10.2017.
//
#include <helpers/logger.h>
#include <system/Environment.h>

namespace sd {

// --- Free-function wrappers for env_functions.h ---
// These allow logger.h and op_boilerplate.h to avoid including Environment.h
// (and all 6 subsystem config headers) in every translation unit.
// --- Core flags ---
bool env_isDebug()   { return Environment::getInstance().isDebug(); }
bool env_isVerbose() { return Environment::getInstance().isVerbose(); }
bool env_isCPU()     { return Environment::getInstance().isCPU(); }
int  env_elementwiseThreshold() { return Environment::getInstance().elementwiseThreshold(); }
int  env_tadThreshold()         { return Environment::getInstance().tadThreshold(); }

// --- Compound checks ---
bool env_isDebugAndVerbose()         { return Environment::getInstance().isDebugAndVerbose(); }
bool env_isLogNativeNDArrayCreation(){ return Environment::getInstance().isLogNativeNDArrayCreation(); }
bool env_isProfiling()               { return Environment::getInstance().isProfiling(); }

// --- Threading ---
int  env_maxThreads()       { return Environment::getInstance().maxThreads(); }
int  env_maxMasterThreads() { return Environment::getInstance().maxMasterThreads(); }

// --- Feature flags ---
bool env_isExperimentalBuild()     { return Environment::getInstance().isExperimentalBuild(); }
bool env_helpersAllowed()          { return Environment::getInstance().helpersAllowed(); }
bool env_isEnableBlas()            { return Environment::getInstance().isEnableBlas(); }
bool env_blasFallback()            { return Environment::getInstance().blasFallback(); }
bool env_isDeletePrimary()         { return Environment::getInstance().isDeletePrimary(); }
bool env_isFuncTracePrintAllocate(){ return Environment::getInstance().isFuncTracePrintAllocate(); }
bool env_precisionBoostAllowed()   { return Environment::getInstance().precisionBoostAllowed(); }

// --- Memory ---
LongType env_maxPrimaryMemory()  { return Environment::getInstance().maxPrimaryMemory(); }
LongType env_maxSpecialMemory()  { return Environment::getInstance().maxSpecialMemory(); }

// --- Default types ---
int env_defaultFloatDataType()   { return static_cast<int>(Environment::getInstance().defaultFloatDataType()); }

// --- Backend / platform flags ---
bool env_isUseMPS()     { return Environment::getInstance().isUseMPS(); }
bool env_isUseONEDNN()  { return Environment::getInstance().isUseONEDNN(); }

// --- DSP config ---
bool env_dspReplayCacheEnabled() { return Environment::getInstance().dsp().replayCacheEnabled(); }
std::string env_dspReplayCacheDir() { return Environment::getInstance().dsp().replayCacheDir(); }
int env_dspCaptureHostWorkspaceMb() { return Environment::getInstance().dsp().captureHostWorkspaceMb(); }
int env_dspCaptureWorkspaceMb()     { return Environment::getInstance().dsp().captureWorkspaceMb(); }
int env_dspCublasWorkspaceMb()      { return Environment::getInstance().dspCublasWorkspaceMb(); }

// --- Triton / CUDA graph config ---
bool env_tritonGraphAutoFree() { return Environment::getInstance().tritonGraphAutoFree(); }

// --- Triton fusion config ---
bool env_tritonFuseAttentionNeighborhoods() { return Environment::getInstance().tritonFuseAttentionNeighborhoods(); }

// --- CUDA device capabilities ---
int env_deviceCapabilityMajor(int deviceId) {
  auto& caps = Environment::getInstance().capabilities();
  if (deviceId >= 0 && deviceId < static_cast<int>(caps.size()))
    return caps[deviceId].first();
  return 0;
}

SD_HOST void Logger::info(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
  fflush(stdout);
}

SD_HOST void Logger::infoEmpty(const char *format) {
 if(format != nullptr)
  printf("%s",format);
  fflush(stdout);
}


SD_HOST void Logger::printv(const char *format, const std::vector<int> &vec) {
  printf("%s: {", format);
  for (size_t e = 0; e < vec.size(); e++) {
    auto v = vec[e];
    printf("%i", v);
    if (e < vec.size() - 1) printf(", ");
  }
  printf("}\n");
  fflush(stdout);
}

SD_HOST void Logger::printv(const char *format, const std::vector<LongType> &vec) {
  printf("%s: {", format);
  for (size_t e = 0; e < vec.size(); e++) {
    auto v = vec[e];
    printf("%lld", (long long)v);
    if (e < vec.size() - 1) printf(", ");
  }
  printf("}\n");
  fflush(stdout);
}

SD_HOST_DEVICE Status Logger::logStatusMsg(Status code, const char *msg) {
  if (msg != nullptr) sd_printf("%s\n", msg);
  return code;
}

SD_HOST_DEVICE Status Logger::logKernelFailureMsg(const char *msg) { return logStatusMsg(Status::KERNEL_FAILURE, msg); }
}  // namespace sd
