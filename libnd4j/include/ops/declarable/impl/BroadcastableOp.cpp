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
// Created by raver on 6/6/2018.
//
#include <helpers/ShapeUtils.h>
#include <ops/declarable/BroadcastableOp.h>
#include <system/op_boilerplate.h>
#include <system/env_functions.h>

namespace sd {
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
BroadcastableOp::BroadcastableOp(const char *name, int numTArgs, int numIArgs)
    : DeclarableCustomOp::DeclarableCustomOp(2, 1, name, false, numTArgs, numIArgs) {
  _descriptor->addTraits(OP_TRAIT_BINARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}

ShapeList *BroadcastableOp::calculateOutputShape(ShapeList *inputShape, sd::graph::Context &block) {
  // Registration normally initializes this metadata eagerly. Keep the call here
  // idempotent for directly constructed ops that bypass OpRegistrator.
  initializeDescriptor();

  auto shapeList = SHAPELIST();
  auto x = inputShape->at(0);
  auto y = inputShape->at(1);
  auto outputs = _descriptor->getOutputTypesForOutput(0);
  sd::DataType dtype = block.dataType(0);
  if (block.dataType(0) != sd::DataType::BOOL && !(outputs.size() == 1 && outputs[0] == sd::DataType::BOOL)) {
    // Always use pickPairwiseResultType to resolve mixed floating-point types.
    // HALF+FLOAT must produce FLOAT output regardless of operand order.
    if (shape::length(y) > shape::length(x)) {
      dtype = DataTypeUtils::pickPairwiseResultType(y, x);
    } else {
      dtype = DataTypeUtils::pickPairwiseResultType(x, y);
    }
  } else
    dtype = sd::DataType::BOOL;

  // Helper lambda: create a contiguous shape info from an input shape info.
  // This prevents non-contiguous strides (from views like permute) from leaking
  // into the output shape. Output shape info must always have contiguous strides
  // because downstream code (e.g., Java-side raw buffer copy) assumes contiguous layout.
  auto contiguousShapeFrom = [&](const LongType* si) -> LongType* {
    int rank = shape::rank(si);
    std::vector<LongType> shapeVec(rank);
    for (int d = 0; d < rank; d++) {
      shapeVec[d] = shape::shapeOf(si)[d];
    }
    return ConstantShapeHelper::getInstance().createShapeInfo(dtype, shape::order(si), shapeVec);
  };

  // Also check using the NDArray objects when available (fastpath).
  // The native shape info may not have the ARRAY_EMPTY bit set for Java-created empty arrays
  // (the Java singleton Nd4j.empty() stores ARRAY_EMPTY in javaShapeInformation but the
  // native shape info only has the dtype bit). NDArray::isEmpty() has a fallback that checks
  // rank==0 && _buffer==nullptr, which correctly identifies these arrays as empty.
  bool xEmptyFromArray = false, yEmptyFromArray = false;
  if (block.isFastPath()) {
    const auto& fp = block.fastpath_in();
    if (fp.size() > 0 && fp[0] != nullptr) xEmptyFromArray = fp[0]->isEmpty();
    if (fp.size() > 1 && fp[1] != nullptr) yEmptyFromArray = fp[1]->isEmpty();
  }

  if (shape::isEmptyConst(x) || shape::isEmptyConst(y) || xEmptyFromArray || yEmptyFromArray) {
    // When either operand is rank-0 empty, return a rank-0 empty with the correct output dtype.
    // When both operands have shape dimensions (rank > 0), compute the broadcast shape and
    // mark it empty so that structural shape info is preserved (e.g. [0,2]).
    bool xIsRank0Empty = (shape::isEmptyConst(x) && shape::rank(x) == 0) || (xEmptyFromArray && shape::rank(x) == 0);
    bool yIsRank0Empty = (shape::isEmptyConst(y) && shape::rank(y) == 0) || (yEmptyFromArray && shape::rank(y) == 0);
    if (xIsRank0Empty || yIsRank0Empty) {
      // Rank-0 empty input: return rank-0 empty with correct dtype
      auto desc = ShapeBuilders::emptyShapeInfo(dtype);
      shapeList->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(desc)->primary());
      delete[] desc;
      return shapeList;
    }
    // Both inputs have rank > 0 with at least one empty: broadcast shapes and mark empty
    sd::LongType *newshape = nullptr;
    ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());
    // evalBroadcastShapeInfo sets EMPTY bit and uses DataTypeUtils::pickPairwiseResultType
    // for the dtype. Override dtype if needed (e.g. BOOL output for logical ops).
    if (dtype != ArrayOptions::dataType(newshape)) {
      int rank = shape::rank(newshape);
      std::vector<LongType> shapeVec(rank);
      for (int d = 0; d < rank; d++) shapeVec[d] = shape::shapeOf(newshape)[d];
      shapeList->push_back(ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(dtype, shapeVec));
    } else {
      shapeList->push_back(newshape);
    }
    return shapeList;
  }

  // Defensive fallback: if either input is empty (e.g. rank-0 empty scalar) and the
  // early-exit block above was bypassed for any reason, emit a rank-0 empty output.
  // This guards against equalsSoft returning true for any two rank-0 arrays (empty or not)
  // and against isScalar returning non-zero for rank-0 non-empty scalars sharing a
  // rank with a rank-0 empty input.
  if (shape::isEmptyConst(x) || shape::isEmptyConst(y) || xEmptyFromArray || yEmptyFromArray) {
    auto desc = ShapeBuilders::emptyShapeInfo(dtype);
    shapeList->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(desc)->primary());
    delete[] desc;
    return shapeList;
  }

  if (shape::isScalar(x) && shape::isScalar(y)) {
    if (shape::rank(x) >= shape::rank(y)) {
      shapeList->push_back(contiguousShapeFrom(x));
    } else {
      shapeList->push_back(contiguousShapeFrom(y));
    }
  } else if (shape::equalsSoft(x, y)) {
    shapeList->push_back(contiguousShapeFrom(x));
  } else if (shape::isScalar(x) && !shape::isScalar(y)) {
    shapeList->push_back(contiguousShapeFrom(y));
  } else if (!shape::isScalar(x) && shape::isScalar(y)) {
    shapeList->push_back(contiguousShapeFrom(x));
  } else if (ShapeUtils::areShapesBroadcastable(x, y)) {
    sd::LongType *newshape = nullptr;
    ShapeUtils::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());
    // NOTE: newshape is already a cached pointer from ConstantShapeHelper, don't delete it
    shapeList->push_back(newshape);
  } else {
    shapeList->push_back(contiguousShapeFrom(x));
  }

  return shapeList;
}
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd
