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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_permute)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/headers/shape.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// here iArgs is int vector of ordered set of dimensions to be permuted
CUSTOM_OP_IMPL(permute, 1, 1, true, 0, -2) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  if (x->isEmpty()) {
    REQUIRE_TRUE(z->isEmpty(), 0, "PERMUTE OP: when input is empty, output must also be empty");
    return Status::OK;  // No op
  }

  // Handle scalar input - permute is a no-op for scalars
  if (x->rankOf() == 0) {
    z->assign(x);
    return Status::OK;
  }

  if (block.width() == 1 && block.getIArguments()->size() == 0) {
    NDArray *t = x->transpose();
    z->assign(t);
    // FIXED: transpose() returns a view - only delete if not a view
    if (t != nullptr && !t->isView()) {
      delete t;
    }
    return Status::OK;
  }

  std::vector<LongType> permutationVector = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<LongType>() : *block.getIArguments();

  // Handle empty permutation vector - just copy input to output
  if (permutationVector.empty()) {
    z->assign(x);
    return Status::OK;
  }

  // Handle dynamic shape mismatch: if permutation vector size doesn't match input rank,
  // try to adapt by keeping only valid indices
  if (permutationVector.size() != static_cast<size_t>(x->rankOf())) {
    sd_printf("PERMUTE OP: permutation vector size (%lld) != input rank (%d), adapting permutation\n",
              (long long)permutationVector.size(), x->rankOf());

    // Find the valid indices and create a new permutation for the actual rank
    std::vector<LongType> validIndices;
    for (size_t i = 0; i < permutationVector.size(); ++i) {
      if (permutationVector[i] < x->rankOf()) {
        validIndices.push_back(permutationVector[i]);
      }
    }

    // If we have exactly the right number of valid indices, use them
    if (validIndices.size() == static_cast<size_t>(x->rankOf())) {
      permutationVector = validIndices;
    } else {
      // Fall back to identity permutation (no change)
      permutationVector.clear();
      for (int i = 0; i < x->rankOf(); ++i) {
        permutationVector.push_back(i);
      }
    }
  }

  // Fast path: check if permutation is identity [0, 1, 2, ...]
  bool isIdentity = true;
  for (size_t i = 0; i < permutationVector.size(); ++i) {
    if (permutationVector[i] != static_cast<LongType>(i)) {
      isIdentity = false;
      break;
    }
  }

  if (isIdentity) {
    // No permutation needed - direct assign
    z->assign(x);
  } else {
    z->assign(x->permute(permutationVector, false, false));
  }

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(permute) {
  getOpDescriptor()->setAllowedInputTypes(0, ANY)->setAllowedInputTypes(1, {ALL_INTS})->setSameMode(true);
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(permute) {
  auto x = INPUT_VARIABLE(0);

  // Handle empty input - return same shape
  if (x->isEmpty()) {
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(x->dataType(), x->ordering(), *x->getShapeAsVector()));
  }

  // Handle scalar input - permute is a no-op for scalars, return scalar shape
  if (x->rankOf() == 0) {
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(x->dataType()));
  }

  if (block.width() == 1 && block.getIArguments()->size() == 0) {
    auto ret = ShapeUtils::evalTransposeShapeInfo(*x, block.workspace(), true);
    return SHAPELIST(ret);
  }
  std::vector<LongType> permutationVector = block.width() > 1 ? INPUT_VARIABLE(1)->asVectorT<LongType>() : *block.getIArguments();

  // Handle empty permutation vector - return input shape unchanged
  if (permutationVector.empty()) {
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(x->dataType(), x->ordering(), *x->getShapeAsVector()));
  }

  // Handle dynamic shape mismatch: if permutation vector size doesn't match input rank,
  // try to adapt by truncating to match input rank (keeping first N dimensions)
  if (permutationVector.size() != static_cast<size_t>(x->rankOf())) {
    sd_printf("PERMUTE shape function: permutation vector size (%lld) != input rank (%d), adapting permutation\n",
              (long long)permutationVector.size(), x->rankOf());

    // If permutation vector is larger than input rank, we need to remap
    // Find the valid indices and create a new permutation for the actual rank
    std::vector<LongType> validIndices;
    for (size_t i = 0; i < permutationVector.size(); ++i) {
      if (permutationVector[i] < x->rankOf()) {
        validIndices.push_back(permutationVector[i]);
      }
    }

    // If we have exactly the right number of valid indices, use them
    if (validIndices.size() == static_cast<size_t>(x->rankOf())) {
      permutationVector = validIndices;
    } else {
      // Fall back to identity permutation (no change)
      permutationVector.clear();
      for (int i = 0; i < x->rankOf(); ++i) {
        permutationVector.push_back(i);
      }
    }
  }

  auto outputShapeInfo =
      ShapeUtils::evalPermShapeInfo(permutationVector.data(), x->rankOf(), x, block.workspace(), true);
  return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
