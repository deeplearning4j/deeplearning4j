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

#ifndef LIBND4J_ARRAYUTILS_H
#define LIBND4J_ARRAYUTILS_H
#include <system/common.h>

#include <cstring>
#include <initializer_list>
#include <vector>

namespace sd {
namespace ArrayUtils {
void toIntPtr(std::initializer_list<int> list, int* target);
void toIntPtr(std::vector<int>& list, int* target);

void toLongPtr(std::initializer_list<LongType> list, LongType* target);
void toLongPtr(std::vector<LongType>& list, LongType* target);

std::vector<LongType> toLongVector(std::vector<int> vec);
std::vector<LongType> toLongVector(std::vector<LongType> vec);
}  // namespace ArrayUtils
}  // namespace sd

#endif  // LIBND4J_ARRAYUTILS_H
