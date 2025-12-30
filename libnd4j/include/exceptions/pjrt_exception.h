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

#ifndef SD_PJRT_EXCEPTION_H
#define SD_PJRT_EXCEPTION_H

#include <system/common.h>

#include <stdexcept>
#include <string>

namespace sd {

/**
 * Exception class for PJRT/TPU related errors.
 * Used to wrap PJRT API error codes and messages.
 */
class SD_LIB_EXPORT pjrt_exception : public std::runtime_error {
 public:
  explicit pjrt_exception(std::string message);
  ~pjrt_exception() = default;

  /**
   * Build a pjrt_exception with an error code.
   * @param message Human-readable error message
   * @param errorCode PJRT error code
   * @return pjrt_exception instance
   */
  static pjrt_exception build(const std::string& message, int errorCode);

  /**
   * Build a pjrt_exception with a function name prefix.
   * @param message Human-readable error message
   * @param errorCode PJRT error code
   * @param funcName Name of the function where error occurred
   * @return pjrt_exception instance
   */
  static pjrt_exception build(const std::string& message, int errorCode, const char* funcName);

  /**
   * Get the PJRT error code associated with this exception.
   * @return Error code
   */
  int errorCode() const { return _errorCode; }

 private:
  int _errorCode = 0;
};

}  // namespace sd

#endif  // SD_PJRT_EXCEPTION_H
