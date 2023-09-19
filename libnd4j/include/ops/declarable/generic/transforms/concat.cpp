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
  std::vector<const NDArray*> nonEmptyArrs;
  std::vector<sd::LongType> arrsToDelete;
  sd::LongType index = 0;
  bool allOfSameType = true;
  auto rankOfFirstArr = block.width() > 0 ? INPUT_VARIABLE(0)->rankOf() : 0;
  auto typeOfFirstArr = block.width() > 0 ? INPUT_VARIABLE(0)->dataType() : block.dataType();

  for (sd::LongType i = 0; i < numOfInArrs; ++i) {
    auto input = INPUT_VARIABLE(i);
    if (!input->isEmpty()) {
      allOfSameType &= (typeOfFirstArr == input->dataType());

      if (input->rankOf() == 0) {
        auto vec = new NDArray('c', {1}, input->dataType(), block.launchContext());
        vec->assign(input);
        nonEmptyArrs.push_back(vec);
        arrsToDelete.push_back(index);
      } else {
        nonEmptyArrs.push_back(input);
      }
      ++index;
    }
  }

  const sd::LongType numOfNonEmptyArrs = nonEmptyArrs.size();

  if (numOfNonEmptyArrs == 0) {
    // All inputs are empty arrays -> return empty, mainly for TF import compatibility (no op)
    REQUIRE_TRUE(OUTPUT_VARIABLE(0)->isEmpty(), 0, "CONCAT op: If all input variables are empty, output must be empty");
    return sd::Status::OK;
  }

  const sd::LongType rank = nonEmptyArrs[0]->rankOf();  //  look up to first non-empty array
  sd::LongType axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<sd::LongType>(0) : INT_ARG(0);
  if (axis < 0) {
    axis += rank;
  }

  // ******** input validation ******** //
  REQUIRE_TRUE(allOfSameType, 0, "CONCAT op: all of input arrays must have same type !");
  REQUIRE_TRUE(nonEmptyArrs[0]->dataType() == OUTPUT_VARIABLE(0)->dataType(), 0,
               "CONCAT op: output array should have the same type as inputs arrays !");
  REQUIRE_TRUE(0 <= axis && (axis < rank || (axis == 0 && rank == 0)), 0,
               "CONCAT op: input axis must be in range [0, %i], but got %i instead!", rank - 1, axis);

  for (sd::LongType i = 1; i < numOfNonEmptyArrs; ++i) {
    if(nonEmptyArrs[i]->rankOf() != rank) {
      std::string error;
      error += std::string("CONCAT op: array at index: ");
      error += std::to_string(i);
      error += std::string(" ");
      error += std::string(" did not have same rank. Expected rank: " + rank);
      error += std::string(" but was: " + nonEmptyArrs[i]->rankOf());
      REQUIRE_TRUE(nonEmptyArrs[i]->rankOf() == rank, 0,error.c_str());
    }

    for (sd::LongType dim = 0; dim < rank; ++dim) {
      if (dim != axis) {
        if(nonEmptyArrs[i]->sizeAt(dim) != nonEmptyArrs[0]->sizeAt(dim)) {
          std::string error;
          error += std::string("CONCAT op: array at index: ");
          error += std::to_string(i);
          error += std::string(" ");
          error += std::string(" did not have same dimension. Expected dimension : " + nonEmptyArrs[0]->sizeAt(dim));
          error += std::string(" but was: " + nonEmptyArrs[i]->sizeAt(dim));
          REQUIRE_TRUE(nonEmptyArrs[i]->rankOf() == rank, 0,error.c_str());
        }
      }
    }

  }

  // ******** end of input validation ******** //

  auto output = OUTPUT_VARIABLE(0);


  helpers::concat(block.launchContext(), nonEmptyArrs, *output, axis);

  for(int i = 0; i < arrsToDelete.size(); i++) {
    delete nonEmptyArrs[arrsToDelete[i]];
  }

  return sd::Status::OK;
}

DECLARE_SYN(ParallelConcat, concat);
DECLARE_SYN(concat_v2, concat);
DECLARE_SYN(concatv2, concat);

DECLARE_TYPES(concat) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY);
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(concat) {
  REQUIRE_TRUE(block.width() > 0, 0, "CONCAT op: No input arrays were provided");

  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

  const sd::LongType numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();
  // first of all take into account possible presence of empty arrays
  // also if scalar is present -> use the shape of vector with length=1 instead
  ShapeList arrShapes;
  std::vector<sd::LongType> shapesToDelete;
  sd::LongType index = 0;
  for (sd::LongType i = 0; i < numOfInArrs; i++) {
    if (shape::rank(inputShape->at(i)) <= 1) {
      if(shape::isEmpty(inputShape->at(i))) {
        auto newShape = ConstantShapeHelper::getInstance().emptyShapeInfo(INPUT_VARIABLE(0)->dataType());
        arrShapes.push_back(newShape);
      } else {
        int isScalar = shape::isScalar(inputShape->at(i));
        int len = isScalar ? 1 : shape::length(inputShape->at(i));
        arrShapes.push_back(ConstantShapeHelper::getInstance().vectorShapeInfo(len, INPUT_VARIABLE(0)->dataType()));
      }

    } else {
      arrShapes.push_back(inputShape->at(i));
    }
    index++;
  }



  const sd::LongType numOfNonEmptyArrs = arrShapes.size();

  if(numOfNonEmptyArrs < 1) {
    // All inputs are empty arrays -> return empty, mainly for TF import compatibility (no op)
    auto newShape = ConstantShapeHelper::getInstance().emptyShapeInfo(INPUT_VARIABLE(0)->dataType());
    return SHAPELIST(newShape);
  }

  const sd::LongType rank = shape::rank(arrShapes.at(0));

  sd::LongType axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<sd::LongType>(0) : INT_ARG(0);
  if (axis < 0) {
    axis += rank;
  }

  // ******** input validation ******** //
  //axis needs to be flexible between 0 and 1
  if(axis > 1)
  REQUIRE_TRUE(0 <= axis && axis < rank, 0, "CONCAT op: input axis must be in range [0, %i], but got %i instead!",
               rank - 1, axis);



  // ******** end of input validation ******** //

  sd::LongType* outShapeInfo(nullptr);
  COPY_SHAPE(arrShapes.at(0), outShapeInfo);
  //reset flags: if an array is empty we can have unintended side effects from the flags
  //in our case by this point we handled empty and should only need the data type.
  ArrayOptions::resetFlags(outShapeInfo);
  ArrayOptions::setDataType(outShapeInfo, INPUT_VARIABLE(0)->dataType());
  printf("Out shape info concat copy\n");
  shape::printShapeInfo(outShapeInfo);
  // case when we have only one input array
  if (numOfNonEmptyArrs == 1) {
    ShapeUtils::updateStridesAndType(outShapeInfo, arrShapes.at(0), shape::order(arrShapes.at(0)));
    return SHAPELIST(CONSTANT(outShapeInfo));
  }


  int newDim = 0;
  for (sd::LongType i = 0; i < numOfNonEmptyArrs; i++) {
    auto newShape = shape::shapeOf(arrShapes.at(i));
    //print the shape based on the shape info rank for this current iteration
    printf("shape of arrShapes at %d\n",i);
    shape::printShapeInfo(arrShapes.at(i));

    if(!shape::isEmpty(arrShapes.at(i))) {
      auto newDim2 = newShape[axis];
      if(newDim2 < 1) {
        printf("new dim 2 is %d\n",newDim2);
        newDim += 1;
      }
      else
        newDim += newDim2;
    }

    printf("new dim is %d axis %d\n",newDim,axis);
  }

  if(newDim < 1)
    newDim = 1;

  //concat can't output scalars
  if(rank < 1) {
    outShapeInfo[0] = 1;
  }


  auto outShape = shape::shapeOf(outShapeInfo);
  outShape[axis] = newDim;


  ShapeUtils::updateStridesAndType(outShapeInfo, arrShapes.at(0), shape::order(arrShapes.at(0)));

  auto desc = new ShapeDescriptor(outShapeInfo);
  printf("number of in arrays %d new dim is %d desc is empty %d\n",numOfInArrs,newDim,desc->isEmpty());
  auto result = ConstantShapeHelper::getInstance().createShapeInfo(desc);
  delete desc;
  return SHAPELIST(result);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(concat_bp, -1, -1, false, 0, 0) {
  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

  const sd::LongType numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

  auto epsilonNext = INPUT_VARIABLE(numOfInArrs - 1);

  auto first = INPUT_VARIABLE(0);

  const sd::LongType axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<int>(0)
                                            : (INT_ARG(0) >= 0 ? INT_ARG(0) : INT_ARG(0) + INPUT_VARIABLE(0)->rankOf());

  sd::LongType startPos = 0;

  for (sd::LongType e = 0; e < numOfInArrs - 1; e++) {
    auto originalChunk = INPUT_VARIABLE(e);
    auto epsilonChunk = OUTPUT_VARIABLE(e);
    std::vector<sd::LongType> indices(2 * epsilonNext->rankOf());

    int width = originalChunk->sizeAt(axis);

    for (sd::LongType e = 0; e < epsilonNext->rankOf(); e++) {
      if (e == axis)
        indices[2 * e + 1] = (indices[2 * e] = startPos) + width;
      else
        indices[2 * e + 1] = indices[2 * e] = 0;
    }

    auto subarray = (*epsilonNext)(indices, true);
    epsilonChunk->assign(subarray);

    startPos += width;
  }

  return sd::Status::OK;
}

DECLARE_TYPES(concat_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(concat_bp) {
  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);

  const sd::LongType numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();

  auto shapeList = SHAPELIST();

  for (int e = 0; e < numOfInArrs - 1; e++) {
    auto inShape = inputShape->at(e);
    auto desc = new ShapeDescriptor(
        ArrayOptions::dataType(inShape), shape::order(inShape), shape::shapeOf(inShape), shape::rank(inShape));
    shapeList->push_back(ConstantShapeHelper::getInstance().createShapeInfo(desc));
    delete desc;
  }

  return shapeList;
}

}  // namespace ops
}  // namespace sd

#endif
