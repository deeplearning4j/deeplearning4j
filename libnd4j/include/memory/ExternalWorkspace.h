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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_EXTERNALWORKSPACE_H
#define LIBND4J_EXTERNALWORKSPACE_H
#include <system/common.h>

namespace sd {
namespace memory {
class SD_LIB_EXPORT ExternalWorkspace {
 private:
  void *_ptrH = nullptr;
  void *_ptrD = nullptr;

  LongType _sizeH = 0L;
  LongType _sizeD = 0L;

 public:
  ExternalWorkspace() = default;
  ~ExternalWorkspace() = default;

  ExternalWorkspace(Pointer ptrH, LongType sizeH, Pointer ptrD, LongType sizeD);

  void *pointerHost();
  void *pointerDevice();

  LongType sizeHost();
  LongType sizeDevice();
};
}  // namespace memory
}  // namespace sd

#endif
