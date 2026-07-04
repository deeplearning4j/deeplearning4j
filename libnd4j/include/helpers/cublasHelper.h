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

#ifndef DEV_TESTS_CUBLASHELPER_H
#define DEV_TESTS_CUBLASHELPER_H

#include <system/common.h>

#include <mutex>
#include <vector>

namespace sd {
class SD_LIB_EXPORT CublasHelper {
 private:
  static std::mutex _mutex;

  std::vector<void*> _cache;
  std::vector<void*> _solvers;
  std::vector<void*> _cudnn;
  std::vector<void*> _sparse;

  CublasHelper();

 public:
  ~CublasHelper();
  static CublasHelper& getInstance();

  void* cudnn();
  void* solver();
  void* sparseHandle();
  void* sparseHandle(int deviceId);

  void* handle();
  void* handle(int deviceId);

  // Apply or remove TF32 math mode on all cached cuBLAS handles.
  // Call after setCublasTf32Enabled() to update existing handles.
  void applyTf32Mode(bool enable);

  // ── Process-global deterministic-math window ──────────────────────────
  // cuBLAS handles are THREAD-LOCAL, so a math mode set by the plan-execution
  // thread never reaches GEMMs dispatched from other threads (ordered-range
  // executors, pool threads). While the window is open (depth > 0), handle()
  // applies CUBLAS_PEDANTIC_MATH to WHICHEVER thread's handle is acquired,
  // and lazily converges back to TF32/DEFAULT after it closes. Depth-counted
  // and exception-balanced by the callers (plan begin/end pairs).
  static void enterDeterministicWindow();
  static void exitDeterministicWindow();
  static bool inDeterministicWindow();

  // cuBLAS Lt (Library) API for optimized GEMM
  // Thread-local, created on-demand per device
  void* ltHandle();
  void* ltHandle(int deviceId);
};
}  // namespace sd

#endif  // DEV_TESTS_CUBLASHELPER_H
