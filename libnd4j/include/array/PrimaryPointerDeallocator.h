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

#ifndef SD_PRIMARYPOINTERDEALLOCATOR_H_
#define SD_PRIMARYPOINTERDEALLOCATOR_H_

#include <system/dll.h>
#include <system/pointercast.h>
#include <array/PointerDeallocator.h>

namespace sd {
class ND4J_EXPORT PrimaryPointerDeallocator : public PointerDeallocator {
 public:
  PrimaryPointerDeallocator() = default;
  ~PrimaryPointerDeallocator() = default;

  void release(void* ptr) override;
};
}

#endif //SD_PRIMARYPOINTERDEALLOCATOR_H_
