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
    auto currentRank = input->rankOf();
    auto *shapeInfoCast = input->shapeInfo();

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

  // delete dynamically allocated vectors with length=1
  //for (sd::LongType index : arrsToDelete) delete nonEmptyArrs[index];

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
  sd_printf("Number of in arrays: %lld\n",numOfInArrs);
  // first of all take into account possible presence of empty arrays
  // also if scalar is present -> use the shape of vector with length=1 instead
  ShapeList arrShapes;
  std::vector<sd::LongType> shapesToDelete;
  sd::LongType index = 0;
  for (sd::LongType i = 0; i < numOfInArrs; ++i) {
    if (inputShape->at(i)[0] == 0) {
      if (shape::isEmpty(inputShape->at(i))) {
        sd_printf("Empty array at index: %lld\n",i);
        arrShapes.push_back(ConstantShapeHelper::getInstance().vectorShapeInfo(0, INPUT_VARIABLE(0)->dataType()));
      } else {
        sd_printf("Non Empty array at index: %lld\n",i);
        shape::printShapeInfo(ConstantShapeHelper::getInstance().vectorShapeInfo(1, INPUT_VARIABLE(0)->dataType()));
        arrShapes.push_back(ConstantShapeHelper::getInstance().vectorShapeInfo(1, INPUT_VARIABLE(0)->dataType()));
      }
    } else {
      sd_printf("Concat shape at  at index: %lld\n",i);
      shape::printShapeInfo(inputShape->at(i));
      arrShapes.push_back(inputShape->at(i));
    }
    ++index;
  }

  const sd::LongType numOfNonEmptyArrs = arrShapes.size();

  const sd::LongType rank = shape::rank(arrShapes.at(0));

  sd::LongType axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<sd::LongType>(0) : INT_ARG(0);
  if (axis < 0) {
    axis += rank;
  }

  // ******** input validation ******** //
  REQUIRE_TRUE(0 <= axis && axis < rank, 0, "CONCAT op: input axis must be in range [0, %i], but got %i instead!",
               rank - 1, axis);


  for (sd::LongType i = 1; i < numOfNonEmptyArrs; ++i) {
    if (shape::rank(arrShapes.at(i)) != rank) {
      std::string error;
      error += std::string("CONCAT op: array at index: ");
      error += std::string("" + i);
      error += std::string(" ");
      error += std::string(" did not have same rank. Expected rank: " + rank);
      error += std::string(" but was: " + shape::rank(arrShapes.at(i)));
      THROW_EXCEPTION(error.c_str());
    }

    for (sd::LongType dim = 0; dim < rank; ++dim) {
      if (dim != axis) {
        if (arrShapes.at(i)[dim + 1] != arrShapes.at(0)[dim + 1]) {
          std::string error;
          error += std::string("CONCAT op: array at index: ");
          error += std::string("" + i);
          error += std::string(" ");
          error += std::string(" did not have same dimension. Expected dimension : " + arrShapes.at(0)[dim + 1]);
          error += std::string(" but was: " + arrShapes.at(0)[dim + 1]);
          THROW_EXCEPTION(error.c_str());
        }
      }
    }
  }

  // ******** end of input validation ******** //

  sd::LongType* outShapeInfo(nullptr);
  COPY_SHAPE(arrShapes.at(0), outShapeInfo);
  sd_printf("Copied shape info\n",0);
  shape::printShapeInfo(outShapeInfo);
  // case when we have only one input array
  if (numOfNonEmptyArrs == 1) {
    sd_printf("concat Updating strides: \n",0);
    ShapeUtils::updateStridesAndType(outShapeInfo, arrShapes.at(0), shape::order(arrShapes.at(0)));
    return SHAPELIST(CONSTANT(outShapeInfo));
  }


  for (sd::LongType i = 1; i < numOfNonEmptyArrs; ++i) {
    outShapeInfo[axis + 1] += arrShapes.at(i)[axis + 1];
  }


  sd_printf("Final concat code path\n",0);
  for (sd::LongType i = 1; i < numOfNonEmptyArrs; ++i) {
    outShapeInfo[axis + 1] += arrShapes.at(i)[axis + 1];
  }

  sd_printf("Shape info before update strides: \n",0);
  ShapeUtils::updateStridesAndType(outShapeInfo, arrShapes.at(0), shape::order(arrShapes.at(0)));
  shape::printShapeInfo(outShapeInfo);
  sd_printf("First input Shape info after update strides: \n",0);
  shape::printShapeInfo(arrShapes.at(0));

  auto desc = new ShapeDescriptor(outShapeInfo);
  auto result = ConstantShapeHelper::getInstance().createShapeInfo(desc);
  sd_printf("Final concat shape info: \n",0);
  shape::printShapeInfo(result);
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
