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
// @author raver119@gmail.com

#ifndef LIBND4J_CONSTANTDATABUFFER_H
#define LIBND4J_CONSTANTDATABUFFER_H

#include <array/DataType.h>
#include <array/PointerWrapper.h>
#include <system/common.h>

#include <memory>

namespace sd {
class SD_LIB_EXPORT ConstantDataBuffer {
 private:
  std::shared_ptr<PointerWrapper> _primaryBuffer;
  std::shared_ptr<PointerWrapper> _specialBuffer = nullptr;
  uint64_t _length = 0;
  uint8_t _sizeOf = 0;

 public:
  ConstantDataBuffer(const std::shared_ptr<PointerWrapper> &primary, uint64_t numEelements, DataType dype);
  ConstantDataBuffer(const std::shared_ptr<PointerWrapper> &primary, const std::shared_ptr<PointerWrapper> &special,
                     uint64_t numEelements, DataType dype);
  ConstantDataBuffer(const ConstantDataBuffer &other);
  ConstantDataBuffer() = default;
  ~ConstantDataBuffer() = default;

  uint8_t sizeOf() const;
  uint64_t length() const;

  void *primary() const;
  void *special() const;

  ConstantDataBuffer &operator=(const ConstantDataBuffer &other) = default;
  ConstantDataBuffer &operator=(ConstantDataBuffer &&other) noexcept = default;

  template <typename T>
  T *primaryAsT() const;

  template <typename T>
  T *specialAsT() const;
};
}  // namespace sd

#endif  // DEV_TESTS_CONSTANTDATABUFFER_H
