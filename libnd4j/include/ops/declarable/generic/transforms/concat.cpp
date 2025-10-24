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
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>

#include <array>
#if NOT_EXCLUDED(OP_concat)

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(concat, -1, 1, false, 0, 0) {
  REQUIRE_TRUE(block.width() > 0, 0, "CONCAT op: No input arrays were provided");

  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

  const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

  // first of all take into account possible presence of empty arrays
  // also if scalar is present -> copy its value to vector with length=1
  std::vector<NDArray*> nonEmptyArrs;
  std::vector<NDArray*> arrsToDelete;  // Track allocated arrays for cleanup
  LongType index = 0;
  bool allOfSameType = true;
  auto rankOfFirstArr = block.width() > 0 ? INPUT_VARIABLE(0)->rankOf() : 0;
  auto typeOfFirstArr = block.width() > 0 ? INPUT_VARIABLE(0)->dataType() : block.dataType();

  for (LongType i = 0; i < numOfInArrs; ++i) {
    auto input = INPUT_VARIABLE(i);
    if (!input->isEmpty()) {
      allOfSameType &= (typeOfFirstArr == input->dataType());

      if (input->rankOf() == 0) {
        std::vector<sd::LongType> shape = {1};
        NDArray* vec = nullptr;
        try {
          vec = new NDArray('c', shape, input->dataType(), block.launchContext());
          vec->assign(input);
          nonEmptyArrs.push_back(vec);
          arrsToDelete.push_back(vec);  // Mark for cleanup
        } catch (...) {
          // If allocation fails, clean up what we've created so far
          if (vec) delete vec;
          for (auto arr : arrsToDelete) {
            delete arr;
          }
          throw;
        }
      } else {
        nonEmptyArrs.push_back(input);
      }
      ++index;
    }
  }

  const LongType numOfNonEmptyArrs = nonEmptyArrs.size();

  if (numOfNonEmptyArrs == 0) {
    // Clean up allocated temporary arrays before returning
    for (auto arr : arrsToDelete) {
      if(arr != nullptr) {
        delete arr;

      }
    }
    // All inputs are empty arrays -> return empty, mainly for TF import compatibility (no op)
    REQUIRE_TRUE(OUTPUT_VARIABLE(0)->isEmpty(), 0, "CONCAT op: If all input variables are empty, output must be empty");
    return Status::OK;
  }

  const LongType rank = nonEmptyArrs[0]->rankOf();  //  look up to first non-empty array
  LongType axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<LongType>(0) : INT_ARG(0);
  if (axis < 0) {
    axis += rank;
  }

  // ******** input validation ******** //
  if (!allOfSameType) {
    for (auto arr : arrsToDelete) delete arr;
    REQUIRE_TRUE(false, 0, "CONCAT op: all of input arrays must have same type !");
  }
  
  if (nonEmptyArrs[0]->dataType() != OUTPUT_VARIABLE(0)->dataType()) {
    for (auto arr : arrsToDelete) delete arr;
    REQUIRE_TRUE(false, 0, "CONCAT op: output array should have the same type as inputs arrays !");
  }
  
  if (!(0 <= axis && (axis < rank || (axis == 0 && rank == 0)))) {
    for (auto arr : arrsToDelete) delete arr;
    REQUIRE_TRUE(false, 0, "CONCAT op: input axis must be in range [0, %i], but got %i instead!", rank - 1, axis);
  }

  for (LongType i = 1; i < numOfNonEmptyArrs; ++i) {
    if (nonEmptyArrs[i]->rankOf() != rank) {
      std::string error;
      error += "CONCAT op: array at index ";
      error += std::to_string(i);
      error += " did not have same rank. Expected rank: ";
      error += std::to_string(rank);
      error += " but was: ";
      error += std::to_string(nonEmptyArrs[i]->rankOf());
      
      // Cleanup before throwing
      for (auto arr : arrsToDelete) delete arr;
      REQUIRE_TRUE(false, 0, error.c_str());
    }

    for (LongType dim = 0; dim < rank; ++dim) {
      if (dim != axis) {
        if (nonEmptyArrs[i]->sizeAt(dim) != nonEmptyArrs[0]->sizeAt(dim)) {
          std::string error;
          error += "CONCAT op: array at index ";
          error += std::to_string(i);
          error += " did not have same dimension at position ";
          error += std::to_string(dim);
          error += ". Expected dimension: ";
          error += std::to_string(nonEmptyArrs[0]->sizeAt(dim));
          error += " but was: ";
          error += std::to_string(nonEmptyArrs[i]->sizeAt(dim));
          
          // Cleanup before throwing
          for (auto arr : arrsToDelete) delete arr;
          REQUIRE_TRUE(false, 0, error.c_str());
        }
      }
    }
  }

  // ******** end of input validation ******** //

  auto output = OUTPUT_VARIABLE(0);

  helpers::concat(block.launchContext(), nonEmptyArrs, *output, axis);

  // Clean up allocated temporary arrays
  for (auto arr : arrsToDelete) {
    delete arr;
  }

  return Status::OK;
}

DECLARE_SYN(ParallelConcat, concat);
DECLARE_SYN(concat_v2, concat);
DECLARE_SYN(concatv2, concat);

DECLARE_TYPES(concat) {
  getOpDescriptor()->setAllowedInputTypes(ANY);
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(concat) {
  REQUIRE_TRUE(block.width() > 0, 0, "CONCAT op: No input arrays were provided");

  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

  //used for copying shape later if we have a mix of empty and non empty
  //all arrays but empty should fit same pattern
  int firstNonEmptyShapeIdx = -1;
  const LongType numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();
  // first of all take into account possible presence of empty arrays
  // also if scalar is present -> use the shape of vector with length=1 instead
  ShapeList arrShapes;
  std::vector<LongType> shapesToDelete;
  LongType numOfNonEmptyArrs = 0;
  const LongType rank = shape::rank(INPUT_VARIABLE(0)->shapeInfo());
  LongType newDim = 0;
  LongType axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<LongType>(0) : INT_ARG(0);
  if (axis < 0) {
    axis += rank;
  }

  for (LongType i = 0; i < numOfInArrs; i++) {
    if (shape::rank(inputShape->at(i)) <= 1) {
      if (shape::isEmptyConst(inputShape->at(i))) {
        int isScalar = shape::isScalar(inputShape->at(i));
        int len = isScalar ? 1 : shape::length(inputShape->at(i));
        newDim += len;
        arrShapes.push_back(inputShape->at(i));
      } else {
        int isScalar = shape::isScalar(inputShape->at(i));
        int len = isScalar ? 1 : shape::length(inputShape->at(i));
        newDim += len;
        arrShapes.push_back(ConstantShapeHelper::getInstance().vectorShapeInfo(len, INPUT_VARIABLE(0)->dataType()));

        if (firstNonEmptyShapeIdx < 0)
          firstNonEmptyShapeIdx = i;
        numOfNonEmptyArrs++;
      }
    } else {
      if (!shape::isEmptyConst(inputShape->at(i))) {
        numOfNonEmptyArrs++;
        if (firstNonEmptyShapeIdx < 0)
          firstNonEmptyShapeIdx = i;
        auto currShape = shape::shapeOf(inputShape->at(i));
        newDim += currShape[axis];
      } else {
        //empty arrays can still have a shape and should be accounted for
        auto currShape = shape::shapeOf(inputShape->at(i));
        newDim += currShape[axis];
      }

      arrShapes.push_back(inputShape->at(i));
    }
  }

  if (numOfNonEmptyArrs < 1) {
    //this case is all empty arrays
    //in this case we need to set the shape to be
    //whatever the number of empty arrays is
    //plus the shape of whatever the rest of the array is
    //for example if empty shape is 1,2,1,0 and we have 3
    // arrays a concat at axis 0 would be 3,2,1,0
    LongType* outShapeInfo(nullptr);
    COPY_SHAPE(arrShapes.at(0), outShapeInfo);
    auto currShape = shape::shapeOf(outShapeInfo);
    currShape[axis] = newDim;
    std::vector<LongType> shapeVec;
    for (int i = 0; i < rank; i++) {
      shapeVec.push_back(currShape[i]);
    }

    // All inputs are empty arrays -> return empty, mainly for TF import compatibility (no op)
    auto newShape = ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(INPUT_VARIABLE(0)->dataType(), shapeVec);
    delete[] outShapeInfo;

    // Clean up allocated vectors
    for (auto idx : shapesToDelete) {
      delete[] const_cast<LongType*>(arrShapes.at(idx));
    }

    return SHAPELIST(newShape);
  }

  // ******** input validation ******** //
  //axis needs to be flexible between 0 and 1
  if (axis > 1)
    REQUIRE_TRUE(0 <= axis && axis < rank, 0, "CONCAT op: input axis must be in range [0, %i], but got %i instead!",
                 rank - 1, axis);

  // ******** end of input validation ******** //

  if (shape::isScalar(arrShapes.at(firstNonEmptyShapeIdx))) {
    //concat of scalar should be  a 1d vector
    auto newShape = ConstantShapeHelper::getInstance().vectorShapeInfo(newDim, INPUT_VARIABLE(0)->dataType());
    return SHAPELIST(CONSTANT(newShape));
  } else {
    LongType* outShapeInfo(nullptr);
    COPY_SHAPE(arrShapes.at(firstNonEmptyShapeIdx), outShapeInfo);
    //reset flags: if an array is empty we can have unintended side effects from the flags
    //in our case by this point we handled empty and should only need the data type.
    ArrayOptions::resetFlags(outShapeInfo);
    // case when we have only one input array
    if (numOfNonEmptyArrs == 1) {
      ShapeUtils::updateStridesAndType(outShapeInfo, arrShapes.at(firstNonEmptyShapeIdx), shape::order(arrShapes.at(firstNonEmptyShapeIdx)));
      auto result = CONSTANT(outShapeInfo);
      delete[] outShapeInfo;
      return SHAPELIST(result);
    }

    auto currShape = shape::shapeOf(outShapeInfo);
    currShape[axis] = newDim;
    ShapeUtils::updateStridesAndType(outShapeInfo, arrShapes.at(firstNonEmptyShapeIdx), shape::order(arrShapes.at(firstNonEmptyShapeIdx)));

    //note: always ensure that the constant shape helper is used, otherwise we could end up with
    //some modification of pre existing cache values.
    auto result = ConstantShapeHelper::getInstance().createFromExisting(outShapeInfo);
    delete[] outShapeInfo;
    return SHAPELIST(result);
  }
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(concat_bp, -1, -1, false, 0, 0) {
  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

  const LongType numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

  auto epsilonNext = INPUT_VARIABLE(numOfInArrs - 1);

  auto first = INPUT_VARIABLE(0);

  const LongType axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<int>(0)
                                        : (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + INPUT_VARIABLE(0)->rankOf());

  LongType startPos = 0;

  for (LongType e = 0; e < numOfInArrs - 1; e++) {
    auto originalChunk = INPUT_VARIABLE(e);
    auto epsilonChunk = OUTPUT_VARIABLE(e);
    std::vector<LongType> indices(2 * epsilonNext->rankOf());

    int width = originalChunk->sizeAt(axis);

    for (LongType e2 = 0; e2 < epsilonNext->rankOf(); e2++) {
      if (e2 == axis)
        indices[2 * e2 + 1] = (indices[2 * e2] = startPos) + width;
      else
        indices[2 * e2 + 1] = indices[2 * e2] = 0;
    }

    auto subarray = (*epsilonNext)(indices, true);
    epsilonChunk->assign(subarray);

    delete subarray;
    startPos += width;
  }

  return Status::OK;
}

DECLARE_TYPES(concat_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(concat_bp) {
  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

  const LongType numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

  auto shapeList = SHAPELIST();

  for (int e = 0; e < numOfInArrs - 1; e++) {
    auto inShape = inputShape->at(e);
    shapeList->push_back(ConstantShapeHelper::getInstance().bufferForShapeInfo(ArrayOptions::dataType(inShape),
                                                                               shape::order(inShape),
                                                                               shape::rank(inShape),
                                                                               shape::shapeOf(inShape))->primary());
  }

  return shapeList;
}

}  // namespace ops
}  // namespace sd

#endif
