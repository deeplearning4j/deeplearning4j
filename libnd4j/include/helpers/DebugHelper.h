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
// Created by raver119 on 20/04/18.
//

#ifndef LIBND4J_DEBUGHELPER_H
#define LIBND4J_DEBUGHELPER_H

#include <system/op_boilerplate.h>

#include <string>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#endif
#include <helpers/DebugInfo.h>
namespace sd {

#ifdef __CUDACC__
// Used to skip cudaStreamSynchronize during CUDA graph capture.
// Defined in DataBuffer.cu within namespace sd.
extern SD_TLS_EXPORT thread_local bool tl_graphExecutionActive;
#endif
class NDArray;
class SD_LIB_EXPORT DebugHelper {
 public:
  // cuda-specific debug functions
#ifdef __CUDACC__
  static SD_INLINE void checkErrorCode(cudaStream_t* stream, int opType = 0) {
    // During CUDA graph capture, cudaStreamSynchronize is illegal (error 900).
    // Kernels aren't actually launched during capture — they're recorded into the graph.
    // Skip the sync entirely when graph capture is active.
    if (tl_graphExecutionActive) return;

    cudaError_t res = cudaStreamSynchronize(*stream);

    if (res != 0) {
      std::string op = "Kernel OpNum [" + std::to_string(opType) + "] cudaStreamSynchronize error [" +
                       std::to_string(res) + "] = " + std::string(cudaGetErrorString(res));
      cudaError_t sticky = cudaGetLastError();
      if (sticky != 0 && sticky != res) {
        op += "; also sticky error [" + std::to_string(sticky) + "] = " + std::string(cudaGetErrorString(sticky));
      }
      THROW_EXCEPTION(op.c_str());
    }

    // Clear any stale sticky errors from unrelated API calls (e.g. failed cudaMallocAsync).
    // After a successful cudaStreamSynchronize, kernel errors are already caught above.
    // cudaGetLastError() here would only pick up unrelated errors, causing false failures.
    cudaGetLastError();
  }



  static SD_INLINE void checkGlobalErrorCode(const char* failMessage = nullptr) {
    // During CUDA graph capture, kernels are recorded but not executed.
    // cudaGetLastError() may return stale errors from Triton compilation or other
    // non-stream CUDA API calls, causing false failures and heap corruption.
    if (tl_graphExecutionActive) return;

    cudaError_t res2 = cudaGetLastError();
    if (res2 != 0) {
      if (failMessage == nullptr) {
        std::string op = "CUDA call ended with error code [" + std::to_string(res2) +
                         "] = " + std::string(cudaGetErrorString(res2));
        THROW_EXCEPTION(op.c_str());
      } else {
        std::string op = std::string(failMessage) + std::string("Error code [") + std::to_string(res2) +
                         "] = " + std::string(cudaGetErrorString(res2));
        THROW_EXCEPTION(op.c_str());
      }
    }
  }

  static SD_INLINE void checkErrorCode(cudaStream_t* stream, const char* failMessage = nullptr) {
    // During CUDA graph capture, cudaStreamSynchronize is illegal (error 900).
    if (tl_graphExecutionActive) return;

    cudaError_t res = cudaStreamSynchronize(*stream);
    if (res != 0) {
      std::string msg = failMessage ? std::string(failMessage) : std::string("CUDA call");
      msg += " cudaStreamSynchronize error code [" + std::to_string(res) + "] = " + std::string(cudaGetErrorString(res));
      cudaError_t sticky = cudaGetLastError();
      if (sticky != 0 && sticky != res) {
        msg += "; also sticky error [" + std::to_string(sticky) + "] = " + std::string(cudaGetErrorString(sticky));
      }
      THROW_EXCEPTION(msg.c_str());
    }

    // Clear any stale sticky errors from unrelated API calls (e.g. failed cudaMallocAsync).
    // After a successful cudaStreamSynchronize, kernel errors are already caught above.
    cudaGetLastError();
  }
#endif
  static DebugInfo debugStatistics(NDArray * input);
  static void retrieveDebugStatistics(DebugInfo* statistics, NDArray* input);
};
}  // namespace sd

#endif  // LIBND4J_DEBUGHELPER_H
