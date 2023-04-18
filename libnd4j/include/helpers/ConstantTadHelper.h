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

#ifndef DEV_TESTS_CONSTANTTADHELPER_H
#define DEV_TESTS_CONSTANTTADHELPER_H

#include <array/ConstantDataBuffer.h>
#include <array/ConstantDescriptor.h>
#include <array/ShapeDescriptor.h>
#include <array/TadDescriptor.h>
#include <array/TadPack.h>
#include <system/op_boilerplate.h>

#include <map>
#include <mutex>
#include <vector>
namespace sd {
class SD_LIB_EXPORT ConstantTadHelper {
 private:
  std::mutex _mutex;
  std::vector<SD_MAP_IMPL<TadDescriptor, TadPack>> _cache;

  ConstantTadHelper();

 public:
  ~ConstantTadHelper() = default;

  static ConstantTadHelper &getInstance();

  /**
   * These methods calculate Tensor-Along-Dimension(s) shape and offsets
   *
   * @param originalShape
   * @param dimensions
   * @param keepUnitiesInShape
   * @return
   */
  TadPack tadForDimensions(const sd::LongType *originalShape, const std::vector<LongType> &dimensions,
                           const bool keepUnitiesInShape = false);
  TadPack tadForDimensions(const sd::LongType *originalShape, LongType *dimensions, LongType dimLength,
                           const bool keepUnitiesInShape = false);
  TadPack tadForDimensions(const sd::LongType *originalShape, LongType dimension, const bool keepUnitiesInShape = false);
  TadPack tadForDimensions(ShapeDescriptor &descriptor, std::vector<LongType> &dimensions,
                           const bool keepUnitiesInShape = false);
  TadPack tadForDimensions(TadDescriptor &descriptor);

  /**
   * This method returns number of cached TAD shapes/offsets on specific device
   * @return
   */
  SD_INLINE int cachedEntriesForDevice(int deviceId) {
    if (deviceId > _cache.size()) throw std::runtime_error("deviceId > number of actual devices");

    return _cache[deviceId].size();
  }

  /**
   * This method returns total number of cached TAD shapes/offsets on all devices
   * @return
   */
  SD_INLINE int totalCachedEntries() {
    int total = 0;

    for (int e = 0; e < _cache.size(); e++) total += _cache[e].size();

    return total;
  }
};
}  // namespace sd

#endif  // DEV_TESTS_CONSTANTTADHELPER_H
