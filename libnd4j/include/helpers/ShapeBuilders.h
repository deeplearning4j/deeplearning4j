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
//

#ifndef DEV_TESTS_SHAPEBUILDERS_H
#define DEV_TESTS_SHAPEBUILDERS_H
#include <array/ArrayOptions.h>
#include <array/DataType.h>
#include <helpers/shape.h>
#include <memory/Workspace.h>

#include <vector>

#include "array/ShapeDescriptor.h"

namespace sd {
class SD_LIB_EXPORT ShapeBuilders {
 public:

  static sd::LongType* createShapeInfoFrom(ShapeDescriptor* descriptor);

  static sd::LongType* createScalarShapeInfo(sd::DataType dataType, sd::memory::Workspace* workspace = nullptr);

  static sd::LongType* createVectorShapeInfo(const sd::DataType dataType, const sd::LongType length,
                                             sd::memory::Workspace* workspace = nullptr);

  /**
   *   create shapeInfo for given order basing on shape stored in shapeOnly vector
   *   memory allocation for shapeInfo is on given workspace
   */
  static sd::LongType* createShapeInfo(const sd::DataType dataType, const char order, int rank,
                                       const sd::LongType* shapeOnly, memory::Workspace* workspace = nullptr);
  static sd::LongType* createShapeInfo(const sd::DataType dataType, const char order,
                                       const std::vector<sd::LongType>& shapeOnly,
                                       memory::Workspace* workspace = nullptr);
  static sd::LongType* createShapeInfo(const sd::DataType dataType, const char order,
                                       const std::initializer_list<sd::LongType>& shapeOnly,
                                       memory::Workspace* workspace = nullptr);

  /**
   *   allocates memory for new shapeInfo and copy all information from inShapeInfo to new shapeInfo
   *   if copyStrides is false then strides for new shapeInfo are recalculated
   */
  static sd::LongType* copyShapeInfo(const sd::LongType* inShapeInfo, const bool copyStrides,
                                     memory::Workspace* workspace = nullptr);
  static sd::LongType* copyShapeInfoAndType(const sd::LongType* inShapeInfo, const DataType dtype,
                                            const bool copyStrides, memory::Workspace* workspace = nullptr);
  static sd::LongType* copyShapeInfoAndType(const sd::LongType* inShapeInfo, const sd::LongType* shapeInfoToGetTypeFrom,
                                            const bool copyStrides, memory::Workspace* workspace = nullptr);

  /**
   * allocates memory for sub-array shapeInfo and copy shape and strides at axes(positions) stored in dims
   */
  static sd::LongType* createSubArrShapeInfo(const sd::LongType* inShapeInfo, const LongType* dims, const int dimsSize,
                                             memory::Workspace* workspace = nullptr);

  static sd::LongType* emptyShapeInfo(const sd::DataType dataType, memory::Workspace* workspace = nullptr);

  static sd::LongType* emptyShapeInfo(const sd::DataType dataType, const char order,
                                      const std::vector<sd::LongType>& shape, memory::Workspace* workspace = nullptr);

  static sd::LongType* emptyShapeInfo(const sd::DataType dataType, const char order, int rank,
                                       const sd::LongType* shapeOnly, memory::Workspace* workspace = nullptr);

};
}  // namespace sd

#endif  // DEV_TESTS_SHAPEBUILDERS_H
