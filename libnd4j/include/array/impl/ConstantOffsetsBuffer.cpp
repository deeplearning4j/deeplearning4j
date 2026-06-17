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

//
// @author raver119@gmail.com
//
#include <array/ConstantOffsetsBuffer.h>
#include <system/PointerValidation.h>

namespace sd {
ConstantOffsetsBuffer::ConstantOffsetsBuffer(const std::shared_ptr<PointerWrapper> &primary)
    : ConstantOffsetsBuffer(primary, std::shared_ptr<PointerWrapper>(nullptr)) {
  //
}

ConstantOffsetsBuffer::ConstantOffsetsBuffer(const std::shared_ptr<PointerWrapper> &primary,
                                             const std::shared_ptr<PointerWrapper> &special) {
  _magic = MAGIC_VALID;  // Mark as valid/constructed
  _primaryOffsets = primary;
  _specialOffsets = special;
}

ConstantOffsetsBuffer::ConstantOffsetsBuffer() {
  _magic = MAGIC_VALID;  // Mark default-constructed objects as valid
  // shared_ptr members are automatically initialized to nullptr
}

ConstantOffsetsBuffer::~ConstantOffsetsBuffer() {
  _magic = MAGIC_DESTROYED;  // Mark as destroyed - helps detect use-after-free
}

LongType *ConstantOffsetsBuffer::primary() {
  SD_VALIDATE_MAGIC(_magic, MAGIC_VALID, "ConstantOffsetsBuffer::primary()");
  if (_primaryOffsets == nullptr || !_primaryOffsets) {
    THROW_EXCEPTION("ConstantOffsetsBuffer::primary: _primaryOffsets is nullptr");
  }
  return reinterpret_cast<LongType *>(_primaryOffsets->pointer());
}

LongType *ConstantOffsetsBuffer::special() {
  return _specialOffsets ? reinterpret_cast<LongType *>(_specialOffsets->pointer()) : nullptr;
}

LongType *ConstantOffsetsBuffer::platform() {
#ifdef SD_CUDA
  return special();
#else
  return primary();
#endif  // CUDABLAS
}

}  // namespace sd
