/* ******************************************************************************
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
#include "DirectShapeTrie.h"
#include <mutex>

namespace sd {

class SD_LIB_EXPORT ConstantShapeHelper {
 private:
  std::mutex _mutex;
  DirectShapeTrie _shapeTrie;

  ConstantShapeHelper();

 public:
  static ConstantShapeHelper& getInstance();

  ~ConstantShapeHelper();
  ConstantShapeBuffer* bufferForShapeInfo(DataType dataType, char order, const std::vector<LongType>& shape);
  ConstantShapeBuffer* bufferForShapeInfo(LongType* shapeInfo);
  ConstantShapeBuffer* bufferForShapeInfo(DataType dataType, char order, int rank,  LongType* shape);
  ConstantShapeBuffer* createShapeInfoWithUnitiesForBroadcast( LongType* maxShapeInfo,
                                                               LongType* minShapeInfo,
                                                               memory::Workspace* workspace = nullptr,
                                                               const std::vector<LongType>& dimensions = {});
  ConstantShapeBuffer* createShapeInfoWithNoUnitiesForReduce( LongType* maxShapeInfo,
                                                              const std::vector<LongType>* dimsWithUnities,
                                                              memory::Workspace* workspace = nullptr);

  LongType* emptyShapeInfo(DataType dataType);
  LongType * scalarShapeInfo(DataType dataType);
  LongType* vectorShapeInfo(LongType length, DataType dataType);
  LongType* createShapeInfo(ShapeDescriptor *descriptor);
  LongType* createShapeInfo(DataType dataType, char order, const std::vector<LongType>& shape);
  LongType* createShapeInfo(DataType dataType, const char order, const int rank,  LongType* shape, LongType extraProperties);
  LongType* createShapeInfo(DataType dataType,  LongType* shapeInfo);
  LongType* createFromExisting(LongType* shapeInfo);



  LongType* castToDataType( LongType* shapeInfo,  DataType newType);
  LongType* emptyShapeInfoWithShape(const DataType dataType, std::vector<LongType>& shape);
  ConstantShapeBuffer* createConstBuffFromExisting(sd::LongType* shapeInfo);
  ConstantShapeBuffer* createSubArrShapeInfo(LongType* shapeInfo, LongType* dims, LongType rank);
};
}  // namespace sd

#endif  // DEV_TESTS_CONSTANTSHAPEHELPER_H