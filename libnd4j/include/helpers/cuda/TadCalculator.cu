/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <helpers/ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ShapeBufferCreatorHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/cuda/CudaShapeBufferCreator.h>

#include "array/CudaPointerDeallocator.h"
#include "array/TadCalculator.h"

namespace sd {

TadCalculator::TadCalculator(LongType* originalShape)
    : _originalShape(originalShape), _numTads(0) {}

TadCalculator::~TadCalculator() {
  if (_tadOffsets) {
    delete _tadOffsets;
    _tadOffsets = nullptr;
  }
  if (_tadShape) {
    delete _tadShape;
    _tadShape = nullptr;
  }
}


void TadCalculator::createTadPack(const std::vector<LongType>& dimensions) {
  if (!_originalShape) {
    THROW_EXCEPTION("Original shape is null");
  }

  auto shapeInfo = ConstantShapeHelper::getInstance().createFromExisting(_originalShape);
  const LongType rank = shape::rank(shapeInfo);

  const std::vector<LongType>* dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions.size(), dimensions.data());
  if (!dimsToExclude) {
    THROW_EXCEPTION("Failed to evaluate dimensions to exclude");
  }

  bool hasSize1Dimension = false;
  for (auto dim : dimensions) {
    if (shape::shapeOf(shapeInfo)[dim] == 1) {
      hasSize1Dimension = true;
      break;
    }
  }

  if (dimsToExclude->size() == 0 || dimsToExclude->size() == rank || hasSize1Dimension) {
    const LongType totalElements = shape::length(shapeInfo);
    
    auto scalarShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(shapeInfo));
    auto scalarShapeBuffer = CudaShapeBufferCreator::getInstance().create(scalarShapeInfo, 0);
    
    LongType* offsetsBuf = new LongType[totalElements];
    
    for (LongType i = 0; i < totalElements; ++i) {
      offsetsBuf[i] = i;
    }
    
    auto oPtr = std::make_shared<PointerWrapper>(offsetsBuf);
    auto offDPtr = std::make_shared<PointerWrapper>(
        ConstantHelper::getInstance().replicatePointer(oPtr->pointer(), totalElements * sizeof(LongType)),
        std::make_shared<CudaPointerDeallocator>());
    
    _tadShape = scalarShapeBuffer;
    _tadOffsets = new ConstantOffsetsBuffer(oPtr, offDPtr);
    _numTads = totalElements;
    
    delete dimsToExclude;
    return;
  }

  const LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(shapeInfo, *dimsToExclude);

  if (numOfSubArrs > 0) {
    const LongType subArrRank = (static_cast<size_t>(rank) == dimsToExclude->size() || false) ? rank : rank - dimsToExclude->size();

    LongType* shapeInfoBuf = new LongType[shape::shapeInfoLength(subArrRank)];
    LongType* offsetsBuf = new LongType[numOfSubArrs];

    shape::calcSubArrsShapeInfoAndOffsets(
        shapeInfo,
        numOfSubArrs,
        dimsToExclude->size(),
        dimsToExclude->data(),
        shapeInfoBuf,
        offsetsBuf,
        false);

    ConstantShapeBuffer* shapesBuffer = CudaShapeBufferCreator::getInstance().create(shapeInfoBuf, subArrRank);

    auto oPtr = std::make_shared<PointerWrapper>(offsetsBuf);
    auto offDPtr = std::make_shared<PointerWrapper>(
        ConstantHelper::getInstance().replicatePointer(oPtr->pointer(), numOfSubArrs * sizeof(LongType)),
        std::make_shared<CudaPointerDeallocator>());
    
    _tadOffsets = new ConstantOffsetsBuffer(oPtr, offDPtr);
    _tadShape = shapesBuffer;
    _numTads = numOfSubArrs;
    

  } else {
    const LongType subArrRank = rank;

    LongType* shapeInfoBuf = new LongType[shape::shapeInfoLength(subArrRank)];

    auto nonConstant = const_cast<LongType*>(shapeInfo);
    shape::copyTo<LongType>(shape::shapeInfoLength(subArrRank), nonConstant, shapeInfoBuf);

    ConstantShapeBuffer* shapesBuffer = CudaShapeBufferCreator::getInstance().create(shapeInfoBuf, subArrRank);

    LongType* baseOffset = new LongType[1];
    baseOffset[0] = 0;
    auto oPtr = std::make_shared<PointerWrapper>(baseOffset);
    auto offDPtr = std::make_shared<PointerWrapper>(
        ConstantHelper::getInstance().replicatePointer(oPtr->pointer(), 1 * sizeof(LongType)),
        std::make_shared<CudaPointerDeallocator>());
    
    _tadOffsets = new ConstantOffsetsBuffer(oPtr, offDPtr);
    _tadShape = shapesBuffer;
    _numTads = 1;

  }

  delete dimsToExclude;
}

} // namespace sd