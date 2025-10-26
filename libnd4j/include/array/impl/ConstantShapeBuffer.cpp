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
#include <array/ConstantShapeBuffer.h>

namespace sd {
ConstantShapeBuffer::ConstantShapeBuffer( PointerWrapper* primary)
    : ConstantShapeBuffer(primary, nullptr) {
#if defined(SD_GCC_FUNCTRACE)
  st = backward::StackTrace();
  st.load_here(32);
#endif

}
ConstantShapeBuffer::ConstantShapeBuffer() {
  _primaryShapeInfo = nullptr;
  _specialShapeInfo = nullptr;
}
ConstantShapeBuffer::~ConstantShapeBuffer() {
  if(_primaryShapeInfo != nullptr)
    delete _primaryShapeInfo;
  _primaryShapeInfo = nullptr;

  if(_specialShapeInfo != nullptr)
    delete _specialShapeInfo;
  _specialShapeInfo = nullptr;
}

ConstantShapeBuffer::ConstantShapeBuffer( PointerWrapper* primary,
                                          PointerWrapper* special) {
  _primaryShapeInfo = primary;
  _specialShapeInfo = special;
#if defined(SD_GCC_FUNCTRACE)
  st = backward::StackTrace();
  st.load_here(32);
#endif
}

LongType *ConstantShapeBuffer::primary()  {
  if(_primaryShapeInfo != nullptr) {
    return reinterpret_cast<LongType *>(_primaryShapeInfo->pointer());
  }

  return nullptr;
}

LongType *ConstantShapeBuffer::special()  {
  if(_specialShapeInfo != nullptr) {
    return reinterpret_cast<LongType *>(_specialShapeInfo->pointer());
  }

  return nullptr;
}

LongType *ConstantShapeBuffer::platform()  {
#ifdef SD_CUDA
  return special();
#else
  return primary();
#endif  // CUDABLAS
}

}  // namespace sd
