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
namespace {

std::vector<LongType> normalizePermutationForInput(
    NDArray* input, std::vector<LongType> permutation) {
  const int inputRank = input->rankOf();
  if (permutation.empty()) {
    permutation.resize(inputRank);
    for (int dimension = 0; dimension < inputRank; ++dimension) {
      permutation[dimension] = dimension;
    }
    return permutation;
  }

  const int permutationRank = static_cast<int>(permutation.size());
  for (auto& dimension : permutation) {
    if (dimension < 0) dimension += permutationRank;
    if (dimension < 0 || dimension >= permutationRank) {
      THROW_EXCEPTION(
          "PERMUTE OP: permutation contains an axis outside its source rank");
    }
  }

  if (permutationRank < inputRank) {
    const int leadingDimensions = inputRank - permutationRank;
    for (int dimension = 0; dimension < leadingDimensions; ++dimension) {
      if (input->sizeAt(dimension) != 1) {
        THROW_EXCEPTION(
            "PERMUTE OP: a shorter permutation requires matching leading "
            "size-one input dimensions");
      }
    }

    std::vector<LongType> adapted;
    adapted.reserve(inputRank);
    for (int dimension = 0; dimension < leadingDimensions; ++dimension) {
      adapted.push_back(dimension);
    }
    for (const auto dimension : permutation) {
      adapted.push_back(dimension + leadingDimensions);
    }
    permutation = adapted;
  } else if (permutationRank > inputRank) {
    THROW_EXCEPTION(
        "PERMUTE OP: permutation rank exceeds the input rank");
  }

  std::vector<bool> seen(inputRank, false);
  for (const auto dimension : permutation) {
    if (dimension < 0 || dimension >= inputRank || seen[dimension]) {
      THROW_EXCEPTION(
          "PERMUTE OP: permutation must contain every input axis exactly once");
    }
    seen[dimension] = true;
  }

  return permutation;
}

}  // namespace

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
  permutationVector = normalizePermutationForInput(x, permutationVector);

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
  getOpDescriptor()->addTraits(OP_TRAIT_VIEW_PRODUCING);
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(permute) {
  auto x = INPUT_VARIABLE(0);

  // Handle empty input: apply the same validated permutation semantics while
  // preserving ARRAY_EMPTY on the result descriptor.
  if (x->isEmpty()) {
    std::vector<LongType> permutation;
    if (block.width() > 1) {
      permutation = ShapeUtils::readIntParams(INPUT_VARIABLE(1));
    } else if (!block.getIArguments()->empty()) {
      permutation = *block.getIArguments();
    } else {
      for (int dimension = x->rankOf() - 1; dimension >= 0; --dimension) {
        permutation.push_back(dimension);
      }
    }
    permutation = normalizePermutationForInput(x, permutation);

    const auto inputShape = *x->getShapeAsVector();
    std::vector<LongType> permutedShape(inputShape.size());
    for (size_t dimension = 0; dimension < permutation.size(); ++dimension) {
      permutedShape[dimension] = inputShape[permutation[dimension]];
    }

    auto emptyShape = ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(
        x->dataType(), permutedShape);
    return SHAPELIST(CONSTANT(emptyShape));
  }

  // Handle scalar input - permute is a no-op for scalars, return scalar shape
  if (x->rankOf() == 0) {
    return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(x->dataType()));
  }

  if (block.width() == 1 && block.getIArguments()->size() == 0) {
    auto temp = ShapeUtils::evalTransposeShapeInfo(*x, nullptr, false);
    ArrayOptions::setPropertyBit(temp, ARRAY_COPY_OFFSET_INPUT_0);
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

  permutationVector = normalizePermutationForInput(x, permutationVector);

  auto temp =
      ShapeUtils::evalPermShapeInfo(permutationVector.data(), x->rankOf(), x, nullptr, false);
  ArrayOptions::setPropertyBit(temp, ARRAY_COPY_OFFSET_INPUT_0);
  auto outputShapeInfo = ConstantShapeHelper::getInstance().createFromExisting(temp);
  RELEASE(temp, nullptr);
  return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
