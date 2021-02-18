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

#ifndef SD_ARRAY_CONSTANTSHAPEBUFFER_H_
#define SD_ARRAY_CONSTANTSHAPEBUFFER_H_

#include <system/dll.h>
#include <system/pointercast.h>
#include <array/PointerWrapper.h>
#include <memory>

namespace sd {

class ND4J_EXPORT ConstantShapeBuffer {
 private:
  std::shared_ptr<PointerWrapper> _primaryShapeInfo;
  std::shared_ptr<PointerWrapper> _specialShapeInfo;

 public:
  ConstantShapeBuffer(const std::shared_ptr<PointerWrapper> &primary);
  ConstantShapeBuffer(const std::shared_ptr<PointerWrapper> &primary, const std::shared_ptr<PointerWrapper> &special);
  ConstantShapeBuffer() = default;
  ~ConstantShapeBuffer() = default;

  const Nd4jLong* primary() const;
  const Nd4jLong* special() const;
  const Nd4jLong* platform() const;
};

} // namespace sd

#endif //SD_ARRAY_CONSTANTSHAPEBUFFER_H_
