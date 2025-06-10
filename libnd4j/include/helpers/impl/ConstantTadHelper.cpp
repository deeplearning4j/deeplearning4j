/* ******************************************************************************
 *
 * Copyright (c) 2024 Konduit K.K.
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <array/TadDescriptor.h>
#include <array/TadPack.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>

namespace sd {

ConstantTadHelper& ConstantTadHelper::getInstance() {
  static ConstantTadHelper instance;
  return instance;
}

TadPack* ConstantTadHelper::tadForDimensions(LongType* originalShape, LongType dimension) {
  return tadForDimensions(originalShape, &dimension, 1);
}

TadPack* ConstantTadHelper::tadForDimensions(LongType* originalShape, std::vector<LongType>* dimensions) {
  if (dimensions == nullptr) {
    THROW_EXCEPTION("Dimensions vector is null");
  }
  return tadForDimensions(originalShape, const_cast<LongType*>(dimensions->data()), dimensions->size());
}

TadPack* ConstantTadHelper::tadForDimensions(TadDescriptor* descriptor) {
  if (descriptor == nullptr) {
    THROW_EXCEPTION("TadDescriptor is null");
  }
  return tadForDimensions(descriptor->originalShape(), descriptor->axis().data(),
                          descriptor->axis().size());
}

TadPack* ConstantTadHelper::tadForDimensions(LongType* originalShape, LongType* dimensions, LongType dimLength) {
    if (originalShape == nullptr) {
        THROW_EXCEPTION("Original shape is null");
    }
    if (dimensions == nullptr) {
        THROW_EXCEPTION("Dimensions array is null");
    }
    if (dimLength <= 0) {
        THROW_EXCEPTION("Invalid dimension length");
    }

    sd::LongType rank = shape::rank(originalShape);
    if (rank < 0) {
        THROW_EXCEPTION("Invalid shape rank");
    }
    
    // Additional validation: check if dimensions are within valid range
    for (LongType i = 0; i < dimLength; i++) {
        LongType dim = dimensions[i];
        if (dim < 0) dim += rank;  // Handle negative dimensions
        if (dim < 0 || dim >= rank) {
            THROW_EXCEPTION("Dimension index is out of bounds");
        }
    }

    std::vector<LongType> dims(dimensions, dimensions + dimLength);

    // Single attempt pattern - no double locking
    try {
        return _trie.getOrCreate(dims, originalShape);
    } catch (const std::exception& e) {
        THROW_EXCEPTION("Failed to create or retrieve TAD pack");
    }
}

} // namespace sd