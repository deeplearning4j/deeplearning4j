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
// Thin TU: segment_prod / unsorted_segment_prod are the shared segment_ops implementation instantiated with the
// product policy (modular integer products, empty segments 1) and the product-rule backprop policy. The backprop
// reads the product of the nonzero elements of every segment and the number of its zeros (SegProdNonZero /
// SegZeroCount), so an element equal to zero gets the product of the others instead of 0 / 0.
//
#include <ops/declarable/helpers/cuda/segment_ops.cuh>

SEGMENT_OP_SAME_TYPE(Prod, SegProd)
SEGMENT_OP_BACKPROP_FLOAT(Prod, SegProdNonZero, GradProd)
