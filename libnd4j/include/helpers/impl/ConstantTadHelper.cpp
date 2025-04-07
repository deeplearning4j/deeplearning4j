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
  return tadForDimensions(originalShape, const_cast<LongType*>(dimensions->data()), dimensions->size());
}

TadPack* ConstantTadHelper::tadForDimensions(TadDescriptor* descriptor) {
  return tadForDimensions(descriptor->originalShape(), descriptor->axis().data(),
                          descriptor->axis().size());
}

TadPack* ConstantTadHelper::tadForDimensions(LongType* originalShape, LongType* dimensions, LongType dimLength) {
    if (!originalShape) THROW_EXCEPTION("Original shape is null");
    if (!dimensions) THROW_EXCEPTION("Dimensions array is null");
    if (dimLength <= 0) THROW_EXCEPTION("Invalid dimension length");

    sd::LongType rank = shape::rank(originalShape);
    if (rank < 0) THROW_EXCEPTION("Invalid shape rank");

    std::vector<LongType> dims(dimensions, dimensions + dimLength);

    // Single attempt pattern - no double locking
    return _trie.getOrCreate(dims, originalShape);
}

} // namespace sd