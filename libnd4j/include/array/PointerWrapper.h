/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
 *
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

//
//  @author raver119@gmail.com
//

#ifndef SD_ARRAY_POINTER_H_
#define SD_ARRAY_POINTER_H_

#include <system/dll.h>
#include <system/pointercast.h>
#include <array/PointerDeallocator.h>
#include <memory>

namespace sd {
class ND4J_EXPORT PointerWrapper {
 private:
  void* _pointer = nullptr;
  std::shared_ptr<PointerDeallocator> _deallocator;

 public:
  PointerWrapper(void* ptr, const std::shared_ptr<PointerDeallocator> &deallocator = {});
  PointerWrapper() = default;
  ~PointerWrapper();

  void* pointer() const;

  template <typename T>
  T* pointerAsT() const {
    return reinterpret_cast<T*>(pointer());
  }
};
} // namespace sd

#endif //SD_ARRAY_POINTER_H_
