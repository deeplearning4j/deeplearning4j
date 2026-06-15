/* ******************************************************************************
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
// Lightweight free-function accessors for the most commonly queried
// Environment settings.  These are declared here (with no Environment.h
// dependency) so that high-fan-out headers like logger.h, Threads.h,
// and op_boilerplate.h can avoid pulling in all 6 subsystem config
// headers through Environment.h.
//
// Implementations live in helpers/impl/logger.cpp (which includes
// Environment.h for itself).
//

#ifndef LIBND4J_ENV_FUNCTIONS_H
#define LIBND4J_ENV_FUNCTIONS_H

#include <system/common.h>
#include <string>

namespace sd {

// --- Core flags (already existed) ---
SD_LIB_EXPORT bool env_isDebug();
SD_LIB_EXPORT bool env_isVerbose();
SD_LIB_EXPORT bool env_isCPU();
SD_LIB_EXPORT int  env_elementwiseThreshold();
SD_LIB_EXPORT int  env_tadThreshold();

// --- Compound checks ---
SD_LIB_EXPORT bool env_isDebugAndVerbose();
SD_LIB_EXPORT bool env_isLogNativeNDArrayCreation();
SD_LIB_EXPORT bool env_isProfiling();

// --- Threading ---
SD_LIB_EXPORT int  env_maxThreads();
SD_LIB_EXPORT int  env_maxMasterThreads();

// --- Feature flags ---
SD_LIB_EXPORT bool env_isExperimentalBuild();
SD_LIB_EXPORT bool env_helpersAllowed();
SD_LIB_EXPORT bool env_isEnableBlas();
SD_LIB_EXPORT bool env_blasFallback();
SD_LIB_EXPORT bool env_isDeletePrimary();
SD_LIB_EXPORT bool env_isFuncTracePrintAllocate();
SD_LIB_EXPORT bool env_precisionBoostAllowed();

// --- Memory ---
SD_LIB_EXPORT sd::LongType env_maxPrimaryMemory();
SD_LIB_EXPORT sd::LongType env_maxSpecialMemory();

// --- Default types ---
SD_LIB_EXPORT int env_defaultFloatDataType();

// --- Backend / platform flags ---
SD_LIB_EXPORT bool env_isUseMPS();
SD_LIB_EXPORT bool env_isUseONEDNN();

// --- DSP config ---
SD_LIB_EXPORT bool env_dspReplayCacheEnabled();
SD_LIB_EXPORT std::string env_dspReplayCacheDir();
SD_LIB_EXPORT int env_dspCaptureHostWorkspaceMb();
SD_LIB_EXPORT int env_dspCaptureWorkspaceMb();
SD_LIB_EXPORT int env_dspCublasWorkspaceMb();

// --- Triton / CUDA graph config ---
SD_LIB_EXPORT bool env_tritonGraphAutoFree();

// --- CUDA device capabilities ---
SD_LIB_EXPORT int env_deviceCapabilityMajor(int deviceId);

// --- Triton fusion config ---
SD_LIB_EXPORT bool env_tritonFuseAttentionNeighborhoods();

}  // namespace sd

#endif  // LIBND4J_ENV_FUNCTIONS_H
