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
ConstantShapeBuffer::ConstantShapeBuffer(const std::shared_ptr<PointerWrapper> &primary) :
                     ConstantShapeBuffer(primary, std::shared_ptr<PointerWrapper>(nullptr)) {
  //
}

ConstantShapeBuffer::ConstantShapeBuffer(const std::shared_ptr<PointerWrapper> &primary,
                                         const std::shared_ptr<PointerWrapper> &special) {
  _primaryShapeInfo = primary;
  _specialShapeInfo = special;
}

const Nd4jLong *ConstantShapeBuffer::primary() const {
  return reinterpret_cast<Nd4jLong*>(_primaryShapeInfo->pointer());
}

const Nd4jLong *ConstantShapeBuffer::special() const {
  return _specialShapeInfo ? reinterpret_cast<Nd4jLong*>(_specialShapeInfo->pointer()) : nullptr;
}

const Nd4jLong *ConstantShapeBuffer::platform() const {
#ifdef __CUDABLAS__
  return special();
#else
  return primary();
#endif // CUDABLAS
}

} // namespace sd
