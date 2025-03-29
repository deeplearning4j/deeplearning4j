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
ConstantShapeBuffer::ConstantShapeBuffer(const std::shared_ptr<PointerWrapper> &primary)
    : ConstantShapeBuffer(primary, std::shared_ptr<PointerWrapper>(nullptr)) {
#if defined(SD_GCC_FUNCTRACE)
  st = backward::StackTrace();
  st.load_here(32);
#endif

}

ConstantShapeBuffer::ConstantShapeBuffer(const std::shared_ptr<PointerWrapper> &primary,
                                         const std::shared_ptr<PointerWrapper> &special) {
  _primaryShapeInfo = primary;
  _specialShapeInfo = special;
#if defined(SD_GCC_FUNCTRACE)
  st = backward::StackTrace();
  st.load_here(32);
#endif
}

LongType *ConstantShapeBuffer::primary()  {
  return reinterpret_cast<LongType *>(_primaryShapeInfo->pointer());
}

 LongType *ConstantShapeBuffer::special()  {
  return reinterpret_cast<LongType *>(_specialShapeInfo->pointer());
}

 LongType *ConstantShapeBuffer::platform()  {
#ifdef __CUDABLAS__
  return special();
#else
  return primary();
#endif  // CUDABLAS
}

}  // namespace sd
