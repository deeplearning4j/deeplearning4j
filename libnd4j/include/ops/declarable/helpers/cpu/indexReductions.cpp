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
// @author AbdelRauf
//
#include <ops/declarable/helpers/reductions.h>
#include <system/selective_rendering.h>
namespace sd {
namespace ops {
namespace helpers {
//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
void argMax_(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions);

template <typename X, typename Z>
void argMin_(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions);

template <typename X, typename Z>
void argAbsMax_(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions);

template <typename X, typename Z>
void argAbsMin_(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions);

//////////////////////////////////////////////////////////////////////////
void argMax(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
auto inputDType = input.dataType();
auto outputDType = output.dataType();
  BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), argMax_, (input, output, dimensions), SD_COMMON_TYPES,
                        SD_INDEXING_TYPES);
}

void argMin(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  auto inputDType = input.dataType();
  auto outputDType = output.dataType();
  BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), argMin_, (input, output, dimensions), SD_COMMON_TYPES,
                        SD_INDEXING_TYPES);
}

void argAbsMax(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  auto inputDType = input.dataType();
  auto outputDType = output.dataType();
  BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), argAbsMax_, (input, output, dimensions), SD_COMMON_TYPES,
                        SD_INDEXING_TYPES);
}

void argAbsMin(NDArray& input, NDArray& output, const std::vector<LongType>& dimensions) {
  auto inputDType = input.dataType();
  auto outputDType = output.dataType();
  BUILD_DOUBLE_SELECTOR(input.dataType(), output.dataType(), argAbsMin_, (input, output, dimensions), SD_COMMON_TYPES,
                        SD_INDEXING_TYPES);

}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
