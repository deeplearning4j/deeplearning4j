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
#if NOT_EXCLUDED(OP_create_view)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(create_view, -2, -1, true, 0, -2) {
  auto inputBase = INPUT_VARIABLE(0);
  auto numInterval = 0;
  auto numAll = 0;
  auto numNewAxis = 0;
  auto numPoint = 0;
  auto indicesPerIndex = std::vector<std::vector<sd::LongType>>();
  auto indexTypes = std::vector<sd::LongType>();
  auto numIndicesPerIndex = std::vector<sd::LongType>();
  auto inclusive = std::vector<sd::LongType>();
  auto outRank = inputBase->rankOf() + numNewAxis - numPoint;
  auto outputShape = std::vector<sd::LongType>();
  auto outputStrides = std::vector<sd::LongType>();
  auto baseOffset = inputBase->bufferOffset();
  auto outIdx = 0;
  auto inIdx = 0;
  std::vector<std::vector<sd::LongType>> indexVectors;
  //note we iterate from i + 1 for each input so we only go to block input size - 1
  for(sd::LongType i = 0; i < block.fastpath_in().size() - 1; i++) {
    //first element is the input we are creating the view from
    auto inputIndex = INPUT_VARIABLE(i + 1);
    auto indexVector = inputIndex->asVectorT<sd::LongType>();
    indexVectors.push_back(indexVector);
    auto indexType = indexVector[0];

    if(indexType == POINT_TYPE) {
      numPoint++;
      inclusive.push_back(1);
    } else if(indexType == 1) {
      numInterval++;
      //the end indicates inclusive or not
      inclusive.push_back(indexVector[indexVector.size() - 1]);
    } else if(indexType == ALL_TYPE) {
      numAll++;
      inclusive.push_back(1);
    } else if(indexType == NEW_AXIS) {
      numNewAxis++;
      inclusive.push_back(1);
    }
  }

  auto numIndices = block.fastpath_in().size() - 1;

  // Padding remaining dimensions with all() index if too few indices provided
  if (numIndices - numNewAxis < inputBase->rankOf()) {
    for (int e = numIndices; e < inputBase->rankOf() + numNewAxis; e++) {
      numAll++;
      indexTypes.push_back(ALL_TYPE);
      indexVectors.push_back(NDIndexUtils::createAll().asVectorT<sd::LongType>());
    }
  }

  for(sd::LongType i = 0; i < indexVectors.size(); i++) {
    auto indexVector = indexVectors[i];
    auto indexType = indexVector[0];
    auto currDimension = i;

    indexTypes.push_back(indexType);
    auto stride = indexVector[2];
    auto indexIndices = std::vector<sd::LongType>();
    //accumulate the target indices
    for(sd::LongType j = 0; j < numIndices; j++) {
      indexIndices.push_back(indexVector[j + 3]);
    }

    indicesPerIndex.push_back(indexVector);

    if(indexType ==  POINT_TYPE) { //point index
      //Point indexes don't appear in output
      auto pointOffset = indexIndices[0];
      baseOffset += pointOffset * ( inputBase->strideAt(inIdx));
      inIdx++;
    } else if(indexType ==  ALL_TYPE) { // all index
      //All index: doesn't change offset. Axis is in both in and output arrays
      outputShape.push_back(inputBase->sizeAt(inIdx));
      outputStrides.push_back(inputBase->strideAt(inIdx));
      inIdx++;
      outIdx++;
    } else if(indexType == INTERVAL_TYPE) { //interval index
      //Interval index: Axis is in both in and output arrays, but output might be smaller
      auto start = indexIndices[0];
      auto end = indexIndices[1];
      auto endInc = end - (inclusive[currDimension] > 0 ? 0 : 1);
      if (endInc >= inputBase->sizeAt(inIdx)) {
        throw std::runtime_error("CREATE_VIEW: Indices are out of range: Cannot get interval index ");
      }

      auto length = (endInc - start) / stride + 1;

      baseOffset += start * inputBase->strideAt(inIdx);
      outputShape.push_back(length);
      outputStrides.push_back(stride *  inputBase->strideAt(inIdx));
      inIdx++;
      outIdx++;
    } else if(indexType == NEW_AXIS) {
      //New axis: appends a 1 in shape. Axis not present in input, but is present in output
      outputShape.push_back(1);
      if (outIdx > 0) { //Stride doesn't matter for 1 size axis anyway...
        outputStrides.push_back(outputStrides[outIdx - 1]);
      } else {
        outputStrides.push_back(1);
      }
      outIdx++;
    }
  }

  auto newResult = new NDArray(inputBase->dataBuffer(),'c',outputShape,inputBase->dataType(),inputBase->getContext(),false,true,baseOffset);
  //note we pass in delete false here so we don't cause a double free
  //overwrite first calls push ndarray which has an option to delete the array if it's not relevant
  //we also call delete later when it's removable.
  if(block.isFastPath() && block.fastpath_out().size() > 0) {
    OVERWRITE_RESULT_NO_DELETE(newResult);
  } else if(block.isFastPath() && block.fastpath_out().size() < 1) {
    STORE_RESULT(newResult);
  }
  return sd::Status::OK;
}

DECLARE_SHAPE_FN(create_view) {
  auto shapeInput = INPUT_VARIABLE(0);
  return SHAPELIST(shapeInput->shapeInfo());
}

DECLARE_TYPES(create_view) { getOpDescriptor()->setAllowedInputTypes({sd::DataType::ANY})->setAllowedOutputTypes(sd::DataType::ANY); }
}  // namespace ops
}  // namespace sd

#endif
