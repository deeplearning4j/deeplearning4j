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
#include <helpers/ArrayUtils.h>

namespace sd {
namespace ArrayUtils {
void toIntPtr(std::initializer_list<int> list, int* target) {
  std::vector<int> vec(list);
  toIntPtr(vec, target);
}

void toIntPtr(std::vector<int>& list, int* target) { memcpy(target, list.data(), list.size() * sizeof(int)); }

void toLongPtr(std::initializer_list<sd::LongType> list, sd::LongType* target) {
  std::vector<sd::LongType> vec(list);
  toLongPtr(vec, target);
}

void toLongPtr(std::vector<sd::LongType>& list, sd::LongType* target) {
  memcpy(target, list.data(), list.size() * sizeof(sd::LongType));
}

std::vector<sd::LongType> toLongVector(std::vector<int> vec) {
  std::vector<sd::LongType> result(vec.size());
  sd::LongType vecSize = vec.size();

  for (sd::LongType e = 0; e < vecSize; e++) result[e] = vec[e];

  return result;
}

std::vector<sd::LongType> toLongVector(std::vector<sd::LongType> vec) { return vec; }
}  // namespace ArrayUtils
}  // namespace sd
