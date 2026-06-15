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

#include <exceptions/pjrt_exception.h>

#include <sstream>

namespace sd {

pjrt_exception::pjrt_exception(std::string message) : std::runtime_error(message) {}

pjrt_exception pjrt_exception::build(const std::string& message, int errorCode) {
  std::ostringstream oss;
  oss << "PJRT error " << errorCode << ": " << message;
  pjrt_exception ex(oss.str());
  ex._errorCode = errorCode;
  return ex;
}

pjrt_exception pjrt_exception::build(const std::string& message, int errorCode, const char* funcName) {
  std::ostringstream oss;
  oss << "PJRT error " << errorCode << " in " << funcName << ": " << message;
  pjrt_exception ex(oss.str());
  ex._errorCode = errorCode;
  return ex;
}

}  // namespace sd
