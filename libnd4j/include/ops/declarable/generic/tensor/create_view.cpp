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
// @author Adam Gibson
//

#include <system/op_boilerplate.h>
#include <indexing/NDIndexUtils.h>
#include <helpers/ConstantShapeHelper.h>
#include <array/ShapeDescriptor.h>
#if NOT_EXCLUDED(OP_create_view)

#include <ops/declarable/headers/parity_ops.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(create_view, -2, -1, true, 0, -2) {
  auto inputBase = INPUT_VARIABLE(0);
  auto numNewAxis = 0;
  auto numPoint = 0;
  auto indicesPerIndex = std::vector<std::vector<LongType>>();
  auto indexTypes = std::vector<LongType>();
  auto numIndicesPerIndex = std::vector<LongType>();
  auto inclusive = std::vector<LongType>();


  auto baseOffset = inputBase->offset();
  auto outIdx = 0;
  auto inIdx = 0;
  std::vector<std::vector<LongType>> indexVectors;
  //note we iterate from i + 1 for each input so we only go to block input size - 1
  for (size_t i = 0; i < block.width() - 1; i++) {
    //first element is the input we are creating the view from
    auto inputIndex = INPUT_VARIABLE(i + 1);
    auto indexVector = inputIndex->asVectorT<LongType>();
    indexVectors.push_back(indexVector);
    auto indexType = indexVector[0];

    if(indexType == POINT_TYPE) {
      numPoint++;
      inclusive.push_back(1);
    } else if(indexType == INTERVAL_TYPE) {
      //the end indicates inclusive or not
      inclusive.push_back(indexVector[indexVector.size() - 1]);
    } else if(indexType == ALL_TYPE) {
      inclusive.push_back(1);
    } else if(indexType == NEW_AXIS) {
      numNewAxis++;
      inclusive.push_back(1);
    }
  }

  auto outRank = inputBase->rankOf() + numNewAxis - numPoint;
  auto outputShape = std::vector<LongType>(outRank);
  auto outputStrides = std::vector<LongType>(outRank);



  auto numIndices = block.width() - 1;

  auto all = NDIndexUtils::createAll();
  // Padding remaining dimensions with all() index if too few indices provided
  if (numIndices - numNewAxis < static_cast<size_t>(inputBase->rankOf())) {
    for (int e = numIndices; e < inputBase->rankOf() + numNewAxis; e++) {
      indexTypes.push_back(ALL_TYPE);
      indexVectors.push_back(all->asVectorT<LongType>());
    }
  }

  for (size_t i = 0; i < indexVectors.size(); i++) {
    auto indexVector = indexVectors[i];
    auto indexType = indexVector[0];
    auto currDimension = i;

    indexTypes.push_back(indexType);
    auto stride = indexVector[2];
    //point should start at 3 for indices, interval is 4 (start,end)
    auto indexIndices = std::vector<LongType>();
    int indexOffset = 3;


    //accumulate the target indices
    //prevent out of bounds
    for (size_t j = 0; j < indexVector.size() - indexOffset; j++) {
      indexIndices.push_back(indexVector[j + indexOffset]);
    }


    indicesPerIndex.push_back(indexVector);

    if(indexType ==  POINT_TYPE) { //point index
      //Point indexes don't appear in output
      auto pointOffset = indexIndices[i];
      baseOffset += pointOffset * ( inputBase->strideAt(inIdx));
      inIdx++;

    } else if(indexType ==  ALL_TYPE) { // all index
      //All index: doesn't change offset. Axis is in both in and output arrays
      outputShape[outIdx] = inputBase->sizeAt(inIdx);
      outputStrides[outIdx] = inputBase->strideAt(inIdx);
      inIdx++;
      outIdx++;
    } else if(indexType == INTERVAL_TYPE) { //interval index
      //Interval index: Axis is in both in and output arrays, but output might be smaller
      auto start = indexIndices[0];
      auto end = indexIndices[1];
      // For INTERVAL_TYPE, actual stride is at indexIndices[2] (position 5 in vector)
      // indexVector[2] is a hardcoded header value, not the actual stride
      auto intervalStride = indexIndices.size() > 2 ? indexIndices[2] : stride;
      auto dimSize = inputBase->sizeAt(inIdx);

      // Handle negative indices: any negative end means "to end of dimension"
      if (start < 0) start += dimSize;
      if (start < 0) start = 0;
      if (start > dimSize) start = dimSize;
      if (end < 0) end = dimSize;
      if (end > dimSize) end = dimSize;

      auto endInc = end - (inclusive[currDimension] > 0 ? 0 : 1);
      if (endInc > dimSize) {
        std::string errorMessage;
        errorMessage += "CREATE_VIEW: Indices are out of range: Cannot get interval index ";
        errorMessage += std::to_string(endInc);
        errorMessage += " on dimension ";
        errorMessage += std::to_string(dimSize);
        THROW_EXCEPTION(errorMessage.c_str());
      }

      auto length = (endInc - start) / intervalStride + 1;

      baseOffset += start * inputBase->strideAt(inIdx);
      outputShape[outIdx] = length;
      outputStrides[outIdx] = intervalStride *  inputBase->strideAt(inIdx);

      inIdx++;
      outIdx++;
    } else if(indexType == NEW_AXIS) {
      //New axis: appends a 1 in shape. Axis not present in input, but is present in output
      outputShape[outIdx] = 1;
      if (outIdx > 0) { //Stride doesn't matter for 1 size axis anyway...
        outputStrides[outIdx] = outputStrides[outIdx - 1];
      } else {
        outputStrides[outIdx] = 1;
      }
      outIdx++;
    }
  }

  delete all;


  auto outputLength = shape::prodLong(outputShape.data(),outRank);

  // Create a ShapeDescriptor with the computed strides (not standard C-order strides)
  // so the view correctly maps back into the original array's memory layout.
  auto descriptor = new ShapeDescriptor(inputBase->dataType(), 'c', outputShape, outputStrides);
  auto viewArray = NDArray(inputBase->dataBuffer(), descriptor, inputBase->getContext(), baseOffset);

  // When a pre-allocated output exists, copy the viewed data into it.
  // Creating a view and trying to overwrite the output doesn't work reliably
  // in all execution paths (SameDiff pre-allocates output buffers).
  if(block.isFastPath() && block.fastpath_out().size() > 0) {
    auto output = OUTPUT_VARIABLE(0);
    output->assign(&viewArray);
  } else {
    auto newResult = new NDArray(inputBase->dataBuffer(), new ShapeDescriptor(inputBase->dataType(), 'c', outputShape, outputStrides), inputBase->getContext(), baseOffset);
    STORE_RESULT(newResult);
  }
  delete descriptor;
  return Status::OK;
}

DECLARE_SHAPE_FN(create_view) {
  auto inputBase = INPUT_VARIABLE(0);
  auto numNewAxis = 0;
  auto numPoint = 0;
  std::vector<std::vector<LongType>> indexVectors;
  std::vector<LongType> inclusive;

  for (size_t i = 0; i < block.width() - 1; i++) {
    auto inputIndex = INPUT_VARIABLE(i + 1);
    auto indexVector = inputIndex->asVectorT<LongType>();
    indexVectors.push_back(indexVector);
    auto indexType = indexVector[0];

    if (indexType == POINT_TYPE) {
      numPoint++;
      inclusive.push_back(1);
    } else if (indexType == INTERVAL_TYPE) {
      inclusive.push_back(indexVector[indexVector.size() - 1]);
    } else if (indexType == ALL_TYPE) {
      inclusive.push_back(1);
    } else if (indexType == NEW_AXIS) {
      numNewAxis++;
      inclusive.push_back(1);
    }
  }

  auto outRank = inputBase->rankOf() + numNewAxis - numPoint;
  std::vector<LongType> outputShape(outRank);

  auto numIndices = block.width() - 1;
  auto all = NDIndexUtils::createAll();
  if (numIndices - numNewAxis < static_cast<size_t>(inputBase->rankOf())) {
    for (int e = numIndices; e < inputBase->rankOf() + numNewAxis; e++) {
      indexVectors.push_back(all->asVectorT<LongType>());
    }
  }

  int outIdx = 0;
  int inIdx = 0;
  for (size_t i = 0; i < indexVectors.size(); i++) {
    auto indexVector = indexVectors[i];
    auto indexType = indexVector[0];
    auto stride = indexVector[2];
    int indexOffset = 3;
    std::vector<LongType> indexIndices;
    for (size_t j = 0; j < indexVector.size() - indexOffset; j++) {
      indexIndices.push_back(indexVector[j + indexOffset]);
    }

    if (indexType == POINT_TYPE) {
      inIdx++;
    } else if (indexType == ALL_TYPE) {
      outputShape[outIdx] = inputBase->sizeAt(inIdx);
      inIdx++;
      outIdx++;
    } else if (indexType == INTERVAL_TYPE) {
      auto start = indexIndices[0];
      auto end = indexIndices[1];
      // For INTERVAL_TYPE, actual stride is at indexIndices[2] (position 5 in vector)
      auto intervalStride = indexIndices.size() > 2 ? indexIndices[2] : stride;
      auto dimSize = inputBase->sizeAt(inIdx);

      // Handle negative indices: any negative end means "to end of dimension"
      if (start < 0) start += dimSize;
      if (start < 0) start = 0;
      if (start > dimSize) start = dimSize;
      if (end < 0) end = dimSize;
      if (end > dimSize) end = dimSize;

      auto endInc = end - (inclusive[i] > 0 ? 0 : 1);
      auto length = (endInc - start) / intervalStride + 1;
      outputShape[outIdx] = length;
      inIdx++;
      outIdx++;
    } else if (indexType == NEW_AXIS) {
      outputShape[outIdx] = 1;
      outIdx++;
    }
  }
  delete all;

  auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(
      inputBase->dataType(), inputBase->ordering(), outRank, outputShape.data(), 0);
  return SHAPELIST(newShape);
}

DECLARE_TYPES(create_view) { getOpDescriptor()->setAllowedInputTypes({ANY})->setAllowedOutputTypes(ANY); }
}  // namespace ops
}  // namespace sd

#endif
