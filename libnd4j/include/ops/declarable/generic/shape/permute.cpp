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
    // Read permutation indices using bulk host sync — avoids per-element GPU->CPU copies.
    permutationVector = ShapeUtils::readIntParams(INPUT_VARIABLE(1));
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
  // adapt the permutation to work with the actual input rank.
  // Common case: expand_dims adds size-1 leading dimensions (e.g., rank-4 [1,C,H,W] becomes
  // rank-5 [1,1,C,H,W]) but the permutation vector is still for the original rank-4.
  // Strategy: find leading size-1 dimensions that account for the rank difference,
  // keep them in place (identity), and shift the original permutation indices.
  if (permutationVector.size() != static_cast<size_t>(x->rankOf())) {
    sd_printf("PERMUTE OP: permutation vector size (%lld) != input rank (%d), adapting permutation\n",
              (long long)permutationVector.size(), x->rankOf());

    int permSize = static_cast<int>(permutationVector.size());
    int inputRank = x->rankOf();
    int extraDims = inputRank - permSize;

    if (extraDims > 0) {
      // Input has more dimensions than the permutation expects.
      // Count leading size-1 dimensions that can be treated as identity-mapped.
      int leadingOnes = 0;
      for (int i = 0; i < inputRank && leadingOnes < extraDims; ++i) {
        if (x->sizeAt(i) == 1) {
          leadingOnes++;
        } else {
          break;
        }
      }

      if (leadingOnes >= extraDims) {
        // We have enough leading size-1 dims to account for the rank difference.
        // Build new permutation: identity for the extra leading dims, then shift original.
        std::vector<LongType> adapted;
        for (int i = 0; i < extraDims; ++i) {
          adapted.push_back(i);  // identity for extra leading size-1 dims
        }
        for (int i = 0; i < permSize; ++i) {
          adapted.push_back(permutationVector[i] + extraDims);  // shift original indices
        }
        sd_printf("PERMUTE OP: adapted permutation by prepending %d identity dims\n", extraDims);
        permutationVector = adapted;
      } else {
        // Can't find enough leading size-1 dims; fall back to identity
        permutationVector.clear();
        for (int i = 0; i < inputRank; ++i) {
          permutationVector.push_back(i);
        }
      }
    } else {
      // Permutation vector is larger than input rank (extraDims < 0).
      // Filter to valid indices only.
      std::vector<LongType> validIndices;
      for (size_t i = 0; i < permutationVector.size(); ++i) {
        if (permutationVector[i] < inputRank) {
          validIndices.push_back(permutationVector[i]);
        }
      }
      if (validIndices.size() == static_cast<size_t>(inputRank)) {
        permutationVector = validIndices;
      } else {
        permutationVector.clear();
        for (int i = 0; i < inputRank; ++i) {
          permutationVector.push_back(i);
        }
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
      permVec = ShapeUtils::readIntParams(INPUT_VARIABLE(1));
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
    // Read permutation indices using bulk host sync — avoids per-element GPU->CPU copies.
    permutationVector = ShapeUtils::readIntParams(INPUT_VARIABLE(1));
  } else {
    permutationVector = *block.getIArguments();
  }

  // Handle empty permutation vector - return input shape unchanged
  if (permutationVector.empty()) {
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(x->dataType(), x->ordering(), *x->getShapeAsVector()));
  }

  // Handle dynamic shape mismatch: adapt permutation to actual input rank.
  // Common case: expand_dims adds size-1 leading dimensions (e.g., rank-4 becomes rank-5)
  // but the permutation vector is still for the original smaller rank.
  if (permutationVector.size() != static_cast<size_t>(x->rankOf())) {
    sd_printf("PERMUTE shape function: permutation vector size (%lld) != input rank (%d), adapting permutation\n",
              (long long)permutationVector.size(), x->rankOf());

    int permSize = static_cast<int>(permutationVector.size());
    int inputRank = x->rankOf();
    int extraDims = inputRank - permSize;

    if (extraDims > 0) {
      // Input has more dimensions than the permutation expects.
      // Count leading size-1 dimensions that can be identity-mapped.
      int leadingOnes = 0;
      for (int i = 0; i < inputRank && leadingOnes < extraDims; ++i) {
        if (x->sizeAt(i) == 1) {
          leadingOnes++;
        } else {
          break;
        }
      }

      if (leadingOnes >= extraDims) {
        std::vector<LongType> adapted;
        for (int i = 0; i < extraDims; ++i) {
          adapted.push_back(i);  // identity for extra leading size-1 dims
        }
        for (int i = 0; i < permSize; ++i) {
          adapted.push_back(permutationVector[i] + extraDims);  // shift original indices
        }
        sd_printf("PERMUTE shape function: adapted permutation by prepending %d identity dims\n", extraDims);
        permutationVector = adapted;
      } else {
        permutationVector.clear();
        for (int i = 0; i < inputRank; ++i) {
          permutationVector.push_back(i);
        }
      }
    } else {
      // Permutation vector is larger than input rank — filter to valid indices
      std::vector<LongType> validIndices;
      for (size_t i = 0; i < permutationVector.size(); ++i) {
        if (permutationVector[i] < inputRank) {
          validIndices.push_back(permutationVector[i]);
        }
      }
      if (validIndices.size() == static_cast<size_t>(inputRank)) {
        permutationVector = validIndices;
      } else {
        permutationVector.clear();
        for (int i = 0; i < inputRank; ++i) {
          permutationVector.push_back(i);
        }
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
