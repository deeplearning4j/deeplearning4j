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

#include <execution/ErrorReference.h>
#include <cstring>

namespace sd {
int ErrorReference::errorCode() { return _errorCode; }

const char* ErrorReference::errorMessage() {
  // since we're fetching error message - error code will be assumed consumed & nullified
  _errorCode = 0;
  return _errorMessage;
}

void ErrorReference::setErrorCode(int errorCode) { _errorCode = errorCode; }

void ErrorReference::setErrorMessage(std::string message) {
  std::strncpy(_errorMessage, message.c_str(), ERROR_MSG_SIZE - 1);
  _errorMessage[ERROR_MSG_SIZE - 1] = '\0';
}

void ErrorReference::setErrorMessage(const char* message) {
  std::strncpy(_errorMessage, message, ERROR_MSG_SIZE - 1);
  _errorMessage[ERROR_MSG_SIZE - 1] = '\0';
}
}  // namespace sd
