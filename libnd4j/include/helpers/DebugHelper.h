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
#include <array/DataBuffer.h>
namespace sd {
class NDArray;
class SD_LIB_EXPORT DebugHelper {
 public:
  // cuda-specific debug functions
#ifdef __CUDACC__
  // Ground-truth CUDA-graph capture detection from the stream itself. During capture a host
  // cudaStreamSynchronize is illegal (error 900 "operation not permitted when stream is
  // capturing") and unnecessary — kernels are recorded into the graph, not executed, and any
  // error surfaces at replay. tl_graphExecutionActive is a fast thread-local hint, but it is
  // FRAGILE: every capture path must remember to set it, and composite merged-capture does
  // not cover every captured slot — so the flag-only guard let a captured op (reshape,
  // equals, ...) reach cudaStreamSynchronize and abort the whole capture. Querying the stream
  // is always correct regardless of which capture path we are in, so it is the real guard;
  // the flag is kept only as a cheap short-circuit.
  static SD_INLINE bool streamIsCapturing(cudaStream_t* stream) {
    if (stream == nullptr) return false;
    cudaStreamCaptureStatus capStatus = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(*stream, &capStatus) != cudaSuccess) {
      cudaGetLastError();  // clear benign query error (e.g. non-capturing legacy stream)
      return false;
    }
    return capStatus != cudaStreamCaptureStatusNone;
  }

  // Single source of truth for the active CUDA-graph capture stream. Reads ALL capture
  // thread-locals in ONE place so no consumer re-derives the condition:
  //   - tl_graphExecutionActive + tl_graphCaptureStream : per-merged-group (inner) capture
  //   - tl_compositeCaptureStream : the whole composite-capture region (outer scope, set once)
  //   - streamIsCapturing(candidate) : CUDA ground-truth backstop if a flag was missed
  // Returns the stream to route async work onto during capture, or nullptr when not
  // capturing. Consumers route transfers onto it and skip host syncs when it is non-null,
  // instead of each re-deriving the three conditions inline.
  static SD_INLINE cudaStream_t currentCaptureStream(cudaStream_t candidate = nullptr) {
    if (tl_graphExecutionActive && tl_graphCaptureStream != nullptr)
      return reinterpret_cast<cudaStream_t>(tl_graphCaptureStream);
    if (tl_compositeCaptureStream != nullptr)
      return reinterpret_cast<cudaStream_t>(tl_compositeCaptureStream);
    if (candidate != nullptr && streamIsCapturing(&candidate)) return candidate;
    return nullptr;
  }

  // Bool authority: is a host stream sync / D2H illegal right now (i.e. are we capturing)?
  // True for ANY capture context — the per-group capture flag, the composite-capture outer
  // region, or the stream itself reporting capture. Checks tl_graphExecutionActive DIRECTLY
  // (not via currentCaptureStream) so a set flag with no recorded stream still suppresses the
  // illegal sync. currentCaptureStream() is the matching STREAM authority (which stream to
  // route async work onto); captureSafeStream() wraps it with a fallback. Consumers MUST call
  // one of these three — never re-derive the tl_* conditions inline (drift = days of debugging).
  static SD_INLINE bool inGraphCapture(cudaStream_t* stream) {
    if (tl_graphExecutionActive) return true;
    if (tl_compositeCaptureStream != nullptr) return true;
    if (stream != nullptr && streamIsCapturing(stream)) return true;
    return false;
  }

  // Stream authority for async transfers: the stream work must run on so it is recorded into
  // the active CUDA graph — the capture stream while capturing, else the provided `fallback`
  // (the op's own LaunchContext stream). Single replacement for the per-file
  // captureSafeStreamOrDefault() copies that used to live in DataBuffer.cu / ConstantHelper.cu
  // / PointersManager.cu, each independently re-deriving this same three-tier priority.
  static SD_INLINE cudaStream_t captureSafeStream(cudaStream_t fallback) {
    cudaStream_t cap = currentCaptureStream(fallback);
    return cap != nullptr ? cap : fallback;
  }

  static SD_INLINE void checkErrorCode(cudaStream_t* stream, int opType = 0) {
    // Skip the host sync during CUDA graph capture — illegal (error 900) + unnecessary.
    // One source of truth: inGraphCapture() (the capture thread-locals + the stream as
    // ground truth), not a fragile inline flag check. See currentCaptureStream().
    if (inGraphCapture(stream)) return;

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
    // Skip during capture: cudaGetLastError may return stale errors from recorded-but-not-
    // executed kernels. One source of truth: inGraphCapture() (no stream available here).
    if (inGraphCapture(nullptr)) return;

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
    // Skip the host sync during CUDA graph capture — illegal (error 900) + unnecessary.
    // One source of truth: inGraphCapture(). See currentCaptureStream().
    if (inGraphCapture(stream)) return;

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
