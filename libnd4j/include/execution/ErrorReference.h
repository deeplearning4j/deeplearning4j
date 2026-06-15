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

#ifndef DEV_TESTS_ERRORREFERENCE_H
#define DEV_TESTS_ERRORREFERENCE_H
#include <system/common.h>

#include <string>

namespace sd {
class SD_LIB_EXPORT ErrorReference {
 private:
  int _errorCode = 0;
  // Fixed-size buffer to avoid heap allocation. Buffer overruns from C++ ops
  // can corrupt adjacent heap metadata, causing free() crashes when
  // setErrorMessage tries to delete the old std::string. A fixed char array
  // has no heap pointer to corrupt.
  static constexpr size_t ERROR_MSG_SIZE = 256;
  char _errorMessage[ERROR_MSG_SIZE] = {0};

 public:
  ErrorReference() = default;
  ~ErrorReference() = default;

  int errorCode();
  const char *errorMessage();

  void setErrorCode(int errorCode);
  void setErrorMessage(std::string message);
  void setErrorMessage(const char *message);
};
}  // namespace sd

#endif  // DEV_TESTS_ERRORREFERENCE_H
