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
// Created by raver119 on 22.11.2017.
//

#ifndef LIBND4J_FLATUTILS_H
#define LIBND4J_FLATUTILS_H
#include <array/NDArray.h>
#include <graph/scheme/array_generated.h>
#include <graph/scheme/node_generated.h>

#include <utility>

namespace sd {
namespace graph {
class SD_LIB_EXPORT FlatUtils {
 public:
  static std::pair<int, int> fromIntPair(IntPair* pair);

  static std::pair<LongType, LongType> fromLongPair(LongPair* pair);

  static NDArray* fromFlatArray(const FlatArray* flatArray);

  static flatbuffers::Offset<FlatArray> toFlatArray(flatbuffers::FlatBufferBuilder& builder, NDArray& array);
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_FLATUTILS_H
