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
// Thin TU: segment_mean / unsorted_segment_mean are the shared segment_ops implementation instantiated with the mean
// policy (floating output of any numeric input, the sum divided once by the count) and the segment-length-scaled
// backprop policy.
//
#include <ops/declarable/helpers/cuda/segment_ops.cuh>

SEGMENT_OP_FLOAT_OUT(Mean, SegMean)
SEGMENT_OP_BACKPROP_FLOAT(Mean, SegMean, GradMean)
