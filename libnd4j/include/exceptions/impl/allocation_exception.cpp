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
#include <exceptions/allocation_exception.h>
#include <helpers/StringUtils.h>

namespace sd {
allocation_exception::allocation_exception(std::string message) : std::runtime_error(message) {
  //
}

allocation_exception allocation_exception::build(std::string message, LongType numBytes) {
  auto bytes = StringUtils::valueToString<LongType>(numBytes);
  message += "; Requested bytes: [" + bytes + "]";
  return allocation_exception(message);
}

allocation_exception allocation_exception::build(std::string message, LongType limit, LongType numBytes) {
  auto bytes = StringUtils::valueToString<LongType>(numBytes);
  auto lim = StringUtils::valueToString<LongType>(limit);
  message += "; Limit bytes: [" + lim + "]; Requested bytes: [" + bytes + "]";
  return allocation_exception(message);
}
}  // namespace sd
