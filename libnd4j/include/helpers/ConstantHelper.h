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

#ifndef DEV_TESTS_CONSTANTHELPER_H
#define DEV_TESTS_CONSTANTHELPER_H
#include <array/ConstantDataBuffer.h>
#include <array/ConstantDescriptor.h>
#include <array/ConstantHolder.h>
#include <memory/Workspace.h>
#include <system/BackendNamespace.h>
#include <system/op_boilerplate.h>

#include <map>
#include <mutex>
#include <vector>

SD_BACKEND_ABI_NAMESPACE_BEGIN
class SD_LIB_EXPORT ConstantHelper {
 private:
  ConstantHelper();

  std::vector<SD_MAP_IMPL<ConstantDescriptor, ConstantHolder*>> _cache;

  // Per-device constant-space ownership for device backends.
  std::vector<Pointer> _devicePointers;
  std::vector<LongType> _deviceOffsets;
  std::mutex _mutex;
  std::mutex _mutexHolder;

  std::vector<LongType> _counters;

 public:
  ~ConstantHelper();
  void *getConstantSpace();
  static ConstantHelper& getInstance();
  static int getCurrentDevice();
  static int getNumberOfDevices();
  void* replicatePointer(void* src, size_t numBytes, memory::Workspace* workspace = nullptr);

  ConstantDataBuffer* constantBuffer(const ConstantDescriptor& descriptor, DataType dataType);

  LongType getCachedAmount(int deviceId);
};
SD_BACKEND_ABI_NAMESPACE_END
SD_BACKEND_ABI_ALIAS(ConstantHelper)

#endif  // DEV_TESTS_CONSTANTHELPER_H
