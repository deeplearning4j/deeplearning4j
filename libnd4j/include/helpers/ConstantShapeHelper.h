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

  ConstantShapeHelper();

 public:



  static ConstantShapeHelper& getInstance();

  ~ConstantShapeHelper();
  ConstantShapeBuffer* bufferForShapeInfo(DataType dataType, char order, const std::vector<LongType>& shape);
  ConstantShapeBuffer* bufferForShapeInfo(ShapeDescriptor *descriptor);
  ConstantShapeBuffer* bufferForShapeInfo(const LongType* shapeInfo);
  ConstantShapeBuffer* bufferForShapeInfo(DataType dataType, char order, int rank, const LongType* shape);
  ConstantShapeBuffer* createShapeInfoWithUnitiesForBroadcast(const LongType* maxShapeInfo,
                                                              const LongType* minShapeInfo,
                                                              memory::Workspace* workspace = nullptr,
                                                              const std::vector<LongType>& dimensions = {});
  ConstantShapeBuffer* createShapeInfoWithNoUnitiesForReduce(const LongType* maxShapeInfo,
                                                             const std::vector<LongType>* dimsWithUnities,
                                                             memory::Workspace* workspace = nullptr);
  ConstantShapeBuffer* createSubArrShapeInfo(const LongType* inShapeInfo, const LongType* dims,
                                             const LongType dimsSize,
                                             memory::Workspace* workspace = nullptr);

  const LongType* emptyShapeInfo(DataType dataType);
  const LongType* scalarShapeInfo(DataType dataType);
  const LongType* vectorShapeInfo(LongType length, DataType dataType);
  const LongType* createShapeInfo(ShapeDescriptor *descriptor);
  const LongType* createShapeInfo(DataType dataType, char order, const std::vector<LongType>& shape);
  const LongType* createShapeInfo(const DataType dataType, const char order, const int rank,
                                      const LongType* shape, LongType extraProperties);
  const LongType* createShapeInfo(DataType dataType, const LongType* shapeInfo);
  const LongType* createFromExisting(const LongType* shapeInfo, sd::memory::Workspace* workspace);
  const LongType* createFromExisting(const LongType* shapeInfo, bool destroyOriginal = true);

  const LongType* createFromExisting( sd::LongType* shapeInfo, sd::memory::Workspace* workspace);
  const LongType* createFromExisting( sd::LongType* shapeInfo, bool destroyOriginal = true);

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
  ConstantShapeBuffer* storeAndWrapBuffer(ShapeDescriptor* descriptor);
  ShapeDescriptor* findBufferForShapeInfo(ShapeDescriptor *descriptor);
  const LongType* emptyShapeInfoWithShape(const DataType dataType, std::vector<LongType>& shape);
};
}  // namespace sd

#endif  // DEV_TESTS_CONSTANTSHAPEHELPER_H
