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

  // View path: if Java initializeOutputs already set up z as a view of x's buffer
  // (shared DataBuffer with permuted strides), the view is correct — nothing to do.
  if (x->dataBuffer() == z->dataBuffer()) {
    return Status::OK;
  }

  std::vector<LongType> permutationVector;
  if (block.width() == 1 && block.getIArguments()->size() == 0) {
    NDArray *t = x->transpose();
    z->assign(t);
    if (t != nullptr && !t->isView()) {
      delete t;
    }
    return Status::OK;
  }

  if (block.width() > 1) {
    // Read permutation indices directly from synced host buffer to avoid e<T>() bugs on CUDA.
    auto* permArr = INPUT_VARIABLE(1);
    permArr->syncToHost();
    auto* db = permArr->dataBuffer();
    auto numEl = permArr->lengthOf();
    auto dtype = permArr->dataType();
    auto arrOffset = permArr->offset();
    if (dtype == INT64) {
      auto* hostLongs = reinterpret_cast<LongType*>(db->primary()) + arrOffset;
      for (sd::LongType i = 0; i < numEl; i++)
        permutationVector.push_back(hostLongs[i]);
    } else if (dtype == INT32) {
      auto* hostInts = reinterpret_cast<int*>(db->primary()) + arrOffset;
      for (sd::LongType i = 0; i < numEl; i++)
        permutationVector.push_back(static_cast<LongType>(hostInts[i]));
    } else {
      permutationVector = permArr->asVectorT<LongType>();
    }
  } else {
    permutationVector = *block.getIArguments();
  }

  // Check for identity permutation [0,1,2,...]
  if (permutationVector.size() == static_cast<size_t>(x->rankOf())) {
    bool isIdentity = true;
    for (size_t i = 0; i < permutationVector.size(); i++) {
      if (permutationVector[i] != static_cast<LongType>(i)) {
        isIdentity = false;
        break;
      }
    }
    if (isIdentity) {
      z->assign(x);
      return Status::OK;
    }
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
    auto permuted = x->permute(permutationVector, false, false);
    z->assign(permuted);
    delete permuted;
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

  // Handle empty input - permute dimensions but preserve ARRAY_EMPTY flag.
  // We must get the permutation vector first, then apply it to the shape.
  if (x->isEmpty()) {
    std::vector<LongType> permVec;
    if (block.width() > 1) {
      auto* permArr = INPUT_VARIABLE(1);
      permArr->syncToHost();
      permVec = permArr->asVectorT<LongType>();
    } else if (block.getIArguments()->size() > 0) {
      permVec = *block.getIArguments();
    }

    auto inputShape = *x->getShapeAsVector();
    std::vector<LongType> permutedShape;
    if (!permVec.empty() && permVec.size() == inputShape.size()) {
      permutedShape.resize(inputShape.size());
      for (size_t i = 0; i < permVec.size(); i++) {
        permutedShape[i] = inputShape[permVec[i]];
      }
    } else {
      permutedShape = inputShape;
    }

    auto emptyShape = ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(x->dataType(), permutedShape);
    return SHAPELIST(CONSTANT(emptyShape));
  }

  // Handle scalar input - permute is a no-op for scalars, return scalar shape
  if (x->rankOf() == 0) {
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(x->dataType()));
  }

  if (block.width() == 1 && block.getIArguments()->size() == 0) {
    auto temp = ShapeUtils::evalTransposeShapeInfo(*x, nullptr, true);
    auto ret = ConstantShapeHelper::getInstance().createFromExisting(temp);
    RELEASE(temp, nullptr);
    return SHAPELIST(ret);
  }
  std::vector<LongType> permutationVector;
  if (block.width() > 1) {
    auto* permArr = INPUT_VARIABLE(1);
    permArr->syncToHost();
    auto* db = permArr->dataBuffer();
    auto numEl = permArr->lengthOf();
    auto dtype = permArr->dataType();
    auto arrOffset = permArr->offset();
    if (dtype == INT64) {
      auto* hostLongs = reinterpret_cast<LongType*>(db->primary()) + arrOffset;
      for (sd::LongType i = 0; i < numEl; i++)
        permutationVector.push_back(hostLongs[i]);
    } else if (dtype == INT32) {
      auto* hostInts = reinterpret_cast<int*>(db->primary()) + arrOffset;
      for (sd::LongType i = 0; i < numEl; i++)
        permutationVector.push_back(static_cast<LongType>(hostInts[i]));
    } else {
      permutationVector = permArr->asVectorT<LongType>();
    }
  } else {
    permutationVector = *block.getIArguments();
  }

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

  auto temp =
      ShapeUtils::evalPermShapeInfo(permutationVector.data(), x->rankOf(), x, nullptr, true);
  auto outputShapeInfo = ConstantShapeHelper::getInstance().createFromExisting(temp);
  RELEASE(temp, nullptr);
  return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
