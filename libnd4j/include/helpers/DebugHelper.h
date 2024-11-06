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

#include <helpers/StringUtils.h>
#include <system/Environment.h>
#include <system/op_boilerplate.h>

#include <string>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#endif
#include <helpers/DebugInfo.h>
namespace sd {
class NDArray;
class SD_LIB_EXPORT DebugHelper {
 public:
  // cuda-specific debug functions
#ifdef __CUDACC__
  static SD_INLINE void checkErrorCode(cudaStream_t* stream, int opType = 0) {
    cudaError_t res = cudaStreamSynchronize(*stream);

    if (res != 0) {
      std::string op = "Kernel OpNum failed: [";
      op += StringUtils::valueToString<int>(opType);
      op += "]";

      THROW_EXCEPTION(op.c_str());
    }

    cudaError_t res2 = cudaGetLastError();
    if(res2 != 0) {
        std::string op = "Kernel OpNum failed: [";
        op += StringUtils::valueToString<int>(opType);
        op += "]";

        THROW_EXCEPTION(op.c_str());
    }
  }



  static SD_INLINE void checkGlobalErrorCode(const char* failMessage = nullptr) {
    cudaError_t res2 = cudaGetLastError();
    if (res2 != 0) {
      if (failMessage == nullptr) {
        std::string op = "CUDA call ended with error code [" + StringUtils::valueToString<int>(res2) + std::string("]");
        THROW_EXCEPTION(op.c_str());
      } else {
        std::string op = std::string(failMessage) + std::string("Error code [") + StringUtils::valueToString<int>(res2) +
                         std::string("]");
        THROW_EXCEPTION(op.c_str());
      }
    }
  }

  static SD_INLINE void checkErrorCode(cudaStream_t* stream, const char* failMessage = nullptr) {
    cudaError_t res = cudaStreamSynchronize(*stream);
    if (res != 0) {
      if (failMessage == nullptr) {
        std::string op = "CUDA call ended with error code [" + StringUtils::valueToString<int>(res) + std::string("]");
        THROW_EXCEPTION(op.c_str());
      } else {
        std::string op = std::string(failMessage) + std::string("Error code [") + StringUtils::valueToString<int>(res) +
                         std::string("]");
        THROW_EXCEPTION(op.c_str());
      }
    }



    cudaError_t res2 = cudaGetLastError();
    if (res2 != 0) {
      if (failMessage == nullptr) {
        std::string op = "CUDA call ended with error code [" + StringUtils::valueToString<int>(res2) + std::string("]");
        THROW_EXCEPTION(op.c_str());
      } else {
        std::string op = std::string(failMessage) + std::string("Error code [") + StringUtils::valueToString<int>(res2) +
                         std::string("]");
        THROW_EXCEPTION(op.c_str());
      }
    }
  }
#endif
  static DebugInfo debugStatistics(NDArray * input);
  static void retrieveDebugStatistics(DebugInfo* statistics, NDArray* input);
};
}  // namespace sd

#endif  // LIBND4J_DEBUGHELPER_H
