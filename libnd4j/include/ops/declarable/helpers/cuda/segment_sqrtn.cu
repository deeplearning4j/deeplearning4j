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
//  @author GS <sgazeos@gmail.com>
//
// Thin TU: unsorted_segment_sqrt_n is the shared segment_ops implementation instantiated with the sqrt_n policy (the
// sum of a segment divided once by the square root of its row count, computed in the accumulator type: DOUBLE for a
// DOUBLE input, FLOAT otherwise) and the matching backprop policy. There is no sorted sqrt_n op.
//
#include <ops/declarable/helpers/cuda/segment_ops.cuh>

namespace sd {
namespace ops {
namespace helpers {
namespace segment_ops {
template <typename X, typename Z>
static void unsortedSqrtN_(LaunchContext* c, NDArray* in, NDArray* idx, LongType n, NDArray* out) {
  unsortedForward_<segment_sem::SegSqrtN, X, Z>(c, in, idx, n, out);
}

template <typename T>
static Status bpSqrtN_(LaunchContext* c, NDArray* in, NDArray* idx, NDArray* go, NDArray* out) {
  return backprop_<segment_sem::SegSqrtN, segment_sem::GradSqrtN, T>(c, in, idx, go, out);
}
}  // namespace segment_ops

void unsortedSegmentSqrtNFunctor(LaunchContext* context, NDArray* input, NDArray* indices, LongType numOfClasses,
                                 NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices});
  BUILD_DOUBLE_SELECTOR(input->dataType(), output->dataType(), segment_ops::unsortedSqrtN_,
                        (context, input, indices, numOfClasses, output), SD_NUMERIC_TYPES, SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices});
}

Status unsortedSegmentSqrtNFunctorBP(LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut,
                                     LongType numOfClasses, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, indices, gradOut});
  BUILD_SINGLE_SELECTOR(output->dataType(), segment_ops::bpSqrtN_, (context, input, indices, gradOut, output),
                        SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({output}, {input, indices, gradOut});
  return Status::OK;
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
