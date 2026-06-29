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
// dense_to_coo_bp: backward pass for dense_to_coo.
//
// Forward recap:
//   Input:   [0] dense [rows, cols]  float
//   TArgs:   [0] threshold
//   Outputs: [0] indices [nnz, 2]   INT64
//            [1] values  [nnz]      float
//
// Backward:
//   Inputs:  [0] indices    [nnz, 2]  INT64  (forward output[0])
//            [1] gradValues [nnz]     float  (upstream gradient for values)
//   IArgs:   [0] rows, [1] cols
//   Output:  [0] dDense [rows, cols]  float
//
//   Math: dDense is ZERO-INITIALIZED then for each k in [0, nnz):
//     dDense[ indices[k,0], indices[k,1] ] = gradValues[k]
//   (Scatter with no atomics — dense_to_coo emits unique index pairs.)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dense_to_coo_bp)

#include <ops/declarable/headers/sparse.h>
#include <ops/declarable/helpers/sparse_coo_bp.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(dense_to_coo_bp, 2, 1, false, 0, 2) {
  auto indices    = INPUT_VARIABLE(0);  // [nnz, 2] INT64
  auto gradValues = INPUT_VARIABLE(1);  // [nnz]    float

  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  REQUIRE_TRUE(indices->rankOf() == 2, 0,
               "dense_to_coo_bp: indices must be 2D [nnz, 2], got rank %d",
               indices->rankOf());
  REQUIRE_TRUE(indices->sizeAt(1) == 2, 0,
               "dense_to_coo_bp: indices second dim must be 2, got %lld",
               (long long)indices->sizeAt(1));
  REQUIRE_TRUE(gradValues->rankOf() == 1, 0,
               "dense_to_coo_bp: gradValues must be 1D [nnz], got rank %d",
               gradValues->rankOf());
  REQUIRE_TRUE(indices->sizeAt(0) == gradValues->lengthOf(), 0,
               "dense_to_coo_bp: indices rows (%lld) must match gradValues length (%lld)",
               (long long)indices->sizeAt(0), (long long)gradValues->lengthOf());

  auto dDense = OUTPUT_NULLIFIED(0);  // [rows, cols] float — written by scatter

  sd::ops::helpers::denseToCoo_bp(block.launchContext(),
                                   indices, gradValues, dDense, rows, cols);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(dense_to_coo_bp) {
  // dDense has shape [rows, cols] with the same float dtype as gradValues (input 1).
  auto gradValues = INPUT_VARIABLE(1);
  const LongType rows = INT_ARG(0);
  const LongType cols = INT_ARG(1);

  auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
      gradValues->dataType(), 'c', {rows, cols});

  return SHAPELIST(outShape);
}

DECLARE_TYPES(dense_to_coo_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_INTS})    // indices
      ->setAllowedInputTypes(1, {ALL_FLOATS})  // gradValues
      ->setAllowedOutputTypes(0, {ALL_FLOATS}); // dDense
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_dense_to_coo_bp)
