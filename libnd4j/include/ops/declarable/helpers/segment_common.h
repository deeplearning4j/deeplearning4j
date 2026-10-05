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
//  @author sgazeos@gmail.com
//  @brief helpers common fuctions for segment_* ops (segment_max, segment_min, etc.)
//  @brief helpers common fuctions for unsorted_segment_* ops (unsorted_segment_max, etc.)
//
//  CUDA only: device-side preparation of the segment ids shared by every segment kernel (implemented in
//  helpers/cuda/segment.cu). All of them enqueue on the context's stream and expect the arrays to be prepared for
//  special (device) use by the caller.
//
#ifndef __SEGMENT_COMMON_HELPERS__
#define __SEGMENT_COMMON_HELPERS__
#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// The segment ids of `indices` (any integer dtype, rank and strides) as one dense sequence of signed 64 bit values in
// `dense`, device memory of indices->lengthOf() elements.
SD_LIB_HIDDEN void segmentReadIds(LaunchContext* context, NDArray* indices, LongType* dense);

// For SORTED dense ids: the first row and the end (one past the last row) of every class; both are zero for a class no
// id names. begin / end are device memory of numClasses elements. Ids outside [0, numClasses) are ignored.
SD_LIB_HIDDEN void segmentBuildRanges(LaunchContext* context, const LongType* ids, LongType n, LongType numClasses,
                                      LongType* begin, LongType* end);

// The number of ids naming each class (counts: device memory of numClasses elements, written here). Ids outside
// [0, numClasses) are ignored.
SD_LIB_HIDDEN void segmentCountIds(LaunchContext* context, const LongType* ids, LongType n, LongType numClasses,
                                   unsigned long long* counts);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
