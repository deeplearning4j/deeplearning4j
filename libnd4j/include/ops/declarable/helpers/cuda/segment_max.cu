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
// Thin TU: segment_max / unsorted_segment_max are the shared segment_ops implementation instantiated with the max
// policy (NaN propagating, sorted empty segments 0, unsorted empty segments the lowest value of the type) and the
// compare-match backprop policy (the whole gradient flows to every element equal to the maximum).
//
#include <ops/declarable/helpers/cuda/segment_ops.cuh>

SEGMENT_OP_SAME_TYPE(Max, SegMax)
SEGMENT_OP_BACKPROP_NUMERIC(Max, SegMax, GradCompare)
