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

#ifndef DEV_TESTS_CONSTANTSHAPEHELPER_H
#define DEV_TESTS_CONSTANTSHAPEHELPER_H

#include <array/ConstantShapeBuffer.h>
#include <array/ShapeDescriptor.h>
#include <memory/Workspace.h>
#include <system/op_boilerplate.h>

#include <map>
#include <mutex>
#include <vector>

namespace sd {

class SD_LIB_EXPORT ConstantShapeHelper {
 private:
  std::mutex _mutex;
  std::vector<SD_MAP_IMPL<ShapeDescriptor , ConstantShapeBuffer *>> _cache;
#if defined(__NEC__)
  bool _cache_existing_pointers = true;
#endif
  ConstantShapeHelper();

 public:

#if defined(__NEC__)
  //Warning: Use it with caution. please, restore it to the previous state to avoid interfering internals
  void disableExistingPointerCaching(){
    _cache_existing_pointers = false;
  }

  void enableExistingPointerCaching(){
    _cache_existing_pointers = true;
  }
#endif

  static ConstantShapeHelper& getInstance();

  ~ConstantShapeHelper() {}
  ConstantShapeBuffer* bufferForShapeInfo(sd::DataType dataType, char order, const std::vector<sd::LongType>& shape);
  ConstantShapeBuffer* bufferForShapeInfo(ShapeDescriptor *descriptor);
  ConstantShapeBuffer* bufferForShapeInfo(const sd::LongType* shapeInfo);
  ConstantShapeBuffer* bufferForShapeInfo(sd::DataType dataType, char order, int rank, const sd::LongType* shape);
  ConstantShapeBuffer* createShapeInfoWithUnitiesForBroadcast(const sd::LongType* maxShapeInfo,
                                                              const sd::LongType* minShapeInfo,
                                                              sd::memory::Workspace* workspace = nullptr,
                                                              const std::vector<LongType>& dimensions = {});
  ConstantShapeBuffer* createShapeInfoWithNoUnitiesForReduce(const sd::LongType* maxShapeInfo,
                                                             const std::vector<LongType>* dimsWithUnities,
                                                             sd::memory::Workspace* workspace = nullptr);
  ConstantShapeBuffer* createSubArrShapeInfo(const sd::LongType* inShapeInfo, const LongType* dims,
                                             const LongType dimsSize,
                                             sd::memory::Workspace* workspace = nullptr);

  const sd::LongType* emptyShapeInfo(sd::DataType dataType);
  const sd::LongType* scalarShapeInfo(sd::DataType dataType);
  const sd::LongType* vectorShapeInfo(sd::LongType length, sd::DataType dataType);
  const sd::LongType* createShapeInfo(ShapeDescriptor *descriptor);
  const sd::LongType* createShapeInfo(sd::DataType dataType, char order, const std::vector<sd::LongType>& shape);
  const sd::LongType* createShapeInfo(sd::DataType dataType, char order, int rank, const sd::LongType* shape);
  const sd::LongType* createShapeInfo(sd::DataType dataType, const sd::LongType* shapeInfo);
  const sd::LongType* createFromExisting(const sd::LongType* shapeInfo, sd::memory::Workspace* workspace);
  const sd::LongType* createFromExisting(const sd::LongType* shapeInfo, bool destroyOriginal = true);

  const sd::LongType* createFromExisting(sd::LongType* shapeInfo, sd::memory::Workspace* workspace);
  const sd::LongType* createFromExisting(sd::LongType* shapeInfo, bool destroyOriginal = true);

  bool checkBufferExistenceForShapeInfo(ShapeDescriptor *descriptor);

  /**
   * This method returns number of cached TAD shapes/offsets on specific device
   * @return
   */
  SD_INLINE int cachedEntriesForDevice(int deviceId) {
    if (deviceId > _cache.size()) THROW_EXCEPTION("deviceId > number of actual devices");

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

#endif  // DEV_TESTS_CONSTANTSHAPEHELPER_H
