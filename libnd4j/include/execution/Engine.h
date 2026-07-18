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
// @author raver119@gmail.com
//

#ifndef SD_ENGINE_H
#define SD_ENGINE_H

namespace samediff {

/**
 * Engine types for platform-specific operation dispatch.
 *
 * ENGINE_CPU: Standard CPU execution
 * ENGINE_CUDA: NVIDIA CUDA GPU execution
 * ENGINE_TPU: Google Cloud TPU execution via PJRT
 * ENGINE_ZLUDA_AMD: ZLUDA transpiler on AMD GPUs (ROCm/HIP backend)
 * ENGINE_ZLUDA_INTEL: ZLUDA transpiler on Intel GPUs (Level Zero backend)
 */
enum Engine {
  ENGINE_CPU = 0,
  ENGINE_CUDA = 1,
  ENGINE_TPU = 2,
  ENGINE_ZLUDA_AMD = 3,
  ENGINE_ZLUDA_INTEL = 4,
  ENGINE_METAL = 5,        // Apple Metal GPU backend
  ENGINE_VULKAN = 6,       // Vulkan GPU backend
  ENGINE_OPENCL = 7,       // OpenCL GPU backend
  ENGINE_VLM = 9,          // Vision-Language Model backend
  ENGINE_MPS = 10,         // Apple Metal Performance Shaders
  ENGINE_ACCELERATE = 11,  // Apple Accelerate framework
  ENGINE_ONEDNN = 12,      // Intel oneDNN (MKL-DNN) backend
  ENGINE_ARM = 13,         // ARM Compute Library backend
  ENGINE_ANY = 99          // Let system choose best available
};

/**
 * Model parallelism capability flags
 * Used to indicate what parallelism features an engine supports
 */
enum ParallelCapabilityFlags {
  PARALLEL_NONE = 0,
  PARALLEL_TENSOR = 1 << 0,      // Tensor parallelism (split tensors across devices)
  PARALLEL_PIPELINE = 1 << 1,    // Pipeline parallelism (different layers on different devices)
  PARALLEL_DATA = 1 << 2,        // Data parallelism (same model, different data)
  PARALLEL_EXPERT = 1 << 3,      // Expert parallelism (for MoE models)
  PARALLEL_SEQUENCE = 1 << 4,    // Sequence parallelism (split sequence dimension)
  PARALLEL_P2P = 1 << 5,         // Peer-to-peer device communication
  PARALLEL_ASYNC = 1 << 6,       // Asynchronous execution
  PARALLEL_UNIFIED_MEM = 1 << 7, // Unified memory support
  PARALLEL_QUANTIZED = 1 << 8,   // Quantized operation support
  PARALLEL_MIXED_PRECISION = 1 << 9  // Mixed precision support
};

/**
 * Check if the engine is a ZLUDA variant.
 * @param e The engine to check
 * @return true if engine is ENGINE_ZLUDA_AMD or ENGINE_ZLUDA_INTEL
 */
inline bool isZludaEngine(Engine e) {
  return e == ENGINE_ZLUDA_AMD || e == ENGINE_ZLUDA_INTEL;
}

/**
 * Get the effective CUDA-compatible engine for ZLUDA.
 * ZLUDA engines use CUDA code, so for compatibility checks
 * they should be treated as CUDA.
 * @param e The engine to check
 * @return ENGINE_CUDA if e is a ZLUDA engine, otherwise returns e unchanged
 */
inline Engine getEffectiveCudaEngine(Engine e) {
  if (isZludaEngine(e)) return ENGINE_CUDA;
  return e;
}

/**
 * Get the name of the engine as a string.
 * @param e The engine
 * @return Human-readable name of the engine
 */
inline const char* getEngineName(Engine e) {
  switch (e) {
    case ENGINE_CPU: return "CPU";
    case ENGINE_CUDA: return "CUDA";
    case ENGINE_TPU: return "TPU";
    case ENGINE_ZLUDA_AMD: return "ZLUDA_AMD";
    case ENGINE_ZLUDA_INTEL: return "ZLUDA_INTEL";
    case ENGINE_METAL: return "Metal";
    case ENGINE_VULKAN: return "Vulkan";
    case ENGINE_OPENCL: return "OpenCL";
    case ENGINE_VLM: return "VLM";
    case ENGINE_MPS: return "MPS";
    case ENGINE_ACCELERATE: return "Accelerate";
    case ENGINE_ONEDNN: return "oneDNN";
    case ENGINE_ARM: return "ARM";
    case ENGINE_ANY: return "ANY";
    default: return "UNKNOWN";
  }
}

/**
 * Get the parallel capabilities of an engine
 * @param e The engine
 * @return Bitmask of ParallelCapabilityFlags
 */
inline int getEngineParallelCapabilities(Engine e) {
  switch (e) {
    case ENGINE_CUDA:
    case ENGINE_ZLUDA_AMD:
    case ENGINE_ZLUDA_INTEL:
      return PARALLEL_TENSOR | PARALLEL_PIPELINE | PARALLEL_DATA |
             PARALLEL_P2P | PARALLEL_ASYNC | PARALLEL_MIXED_PRECISION |
             PARALLEL_QUANTIZED;
    case ENGINE_VLM:
      return PARALLEL_TENSOR | PARALLEL_PIPELINE | PARALLEL_DATA |
             PARALLEL_QUANTIZED | PARALLEL_MIXED_PRECISION;
    case ENGINE_METAL:
    case ENGINE_MPS:
      return PARALLEL_TENSOR | PARALLEL_ASYNC | PARALLEL_UNIFIED_MEM |
             PARALLEL_MIXED_PRECISION;
    case ENGINE_ONEDNN:
      return PARALLEL_DATA | PARALLEL_MIXED_PRECISION;
    case ENGINE_TPU:
      return PARALLEL_TENSOR | PARALLEL_DATA | PARALLEL_PIPELINE;
    case ENGINE_VULKAN:
    case ENGINE_OPENCL:
      return PARALLEL_TENSOR | PARALLEL_DATA | PARALLEL_ASYNC;
    case ENGINE_ARM:
      return PARALLEL_DATA | PARALLEL_QUANTIZED;
    case ENGINE_CPU:
    default:
      return PARALLEL_DATA;
  }
}

/**
 * Check if an engine supports a specific parallel capability
 * @param e The engine
 * @param capability The capability flag to check
 * @return true if the engine supports the capability
 */
inline bool engineSupportsParallel(Engine e, ParallelCapabilityFlags capability) {
  return (getEngineParallelCapabilities(e) & capability) != 0;
}

/**
 * Check if an engine supports model parallelism (tensor or pipeline)
 * @param e The engine
 * @return true if the engine supports some form of model parallelism
 */
inline bool engineSupportsModelParallel(Engine e) {
  int caps = getEngineParallelCapabilities(e);
  return (caps & (PARALLEL_TENSOR | PARALLEL_PIPELINE)) != 0;
}

/**
 * Check if an engine is GPU-based
 * @param e The engine
 * @return true if the engine runs on GPU hardware
 */
inline bool isGpuEngine(Engine e) {
  return e == ENGINE_CUDA || e == ENGINE_ZLUDA_AMD || e == ENGINE_ZLUDA_INTEL ||
         e == ENGINE_METAL || e == ENGINE_MPS || e == ENGINE_VULKAN ||
         e == ENGINE_OPENCL;
}

/**
 * Check if an engine is suitable for LLM/VLM workloads
 * @param e The engine
 * @return true if the engine is optimized for LLM/VLM inference
 */
inline bool isLlmEngine(Engine e) {
  return e == ENGINE_VLM;
}

/**
 * Get the preferred engine for multi-GPU model parallelism
 * @return The best available engine for model parallelism
 */
static inline Engine getPreferredParallelEngine() {
#if defined(DEFAULT_ENGINE)
  return DEFAULT_ENGINE;
#elif defined(SD_CUDA)
  return ENGINE_CUDA;
#elif defined(SD_VULKAN)
  return ENGINE_VULKAN;
#elif defined(SD_APPLE_MPS)
  return ENGINE_METAL;
#else
  return ENGINE_CPU;
#endif
}

static inline Engine resolveConfiguredEngine(Engine engine) {
  return engine == ENGINE_ANY ? getPreferredParallelEngine() : engine;
}

}  // namespace samediff

#endif  // SD_ENGINE_H
