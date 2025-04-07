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

void TadCalculator::createTadPack(const std::vector<LongType>& dimensions) {
  // Validate input and create shape info from original shape
  if (!_originalShape) {
    THROW_EXCEPTION("Original shape is null");
  }

  auto shapeInfo = ConstantShapeHelper::getInstance().createFromExisting(_originalShape);
  const LongType rank = shape::rank(shapeInfo);

  // Calculate dimensions to exclude
  const std::vector<LongType>* dimsToExclude = ShapeUtils::evalDimsToExclude(rank, dimensions.size(), dimensions.data());
  if (!dimsToExclude) {
    THROW_EXCEPTION("Failed to evaluate dimensions to exclude");
  }

  // Calculate number of sub-arrays
  const LongType numOfSubArrs = ShapeUtils::getNumOfSubArrs(shapeInfo, *dimsToExclude);

  if (numOfSubArrs > 0) {
    // Calculate sub-array rank
    const LongType subArrRank = (static_cast<size_t>(rank) == dimsToExclude->size() || false) ? rank : rank - dimsToExclude->size();

    // Allocate memory for shapes and offsets
    LongType* shapeInfoBuf = new LongType[shape::shapeInfoLength(subArrRank)];
    LongType* offsetsBuf = new LongType[numOfSubArrs];

    // Calculate shapes and offsets
    shape::calcSubArrsShapeInfoAndOffsets(
        shapeInfo,
        numOfSubArrs,
        dimsToExclude->size(),
        dimsToExclude->data(),
        shapeInfoBuf,
        offsetsBuf,
        false);  // areUnitiesInShape

    // Use the CUDA ShapeBufferCreator for shape buffer creation
    ConstantShapeBuffer* shapesBuffer = CudaShapeBufferCreator::getInstance().create(shapeInfoBuf, subArrRank);

    // Create offsets buffer with CUDA device copy
    auto oPtr = std::make_shared<PointerWrapper>(offsetsBuf);
    auto offDPtr = std::make_shared<PointerWrapper>(
        ConstantHelper::getInstance().replicatePointer(oPtr->pointer(), numOfSubArrs * sizeof(LongType)),
        std::make_shared<CudaPointerDeallocator>());
    
    _tadOffsets = new ConstantOffsetsBuffer(oPtr, offDPtr);
    _tadShape = shapesBuffer;
    _numTads = numOfSubArrs;
    

  } else {
    // Base case: number of sub arrays is zero, use original shape
    const LongType subArrRank = rank;

    // Allocate and copy shape info
    LongType* shapeInfoBuf = new LongType[shape::shapeInfoLength(subArrRank)];

    // Copy shape info
    auto nonConstant = const_cast<LongType*>(shapeInfo);
    shape::copyTo<LongType>(shape::shapeInfoLength(subArrRank), nonConstant, shapeInfoBuf);

    // Use the CUDA ShapeBufferCreator for shape buffer creation
    ConstantShapeBuffer* shapesBuffer = CudaShapeBufferCreator::getInstance().create(shapeInfoBuf, subArrRank);

    // Create base offset
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
