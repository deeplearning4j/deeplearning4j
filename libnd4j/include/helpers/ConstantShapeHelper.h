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
#include <unordered_set>

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
  ConstantShapeBuffer* createShapeInfoWithNoUnitiesForReduce( const LongType* maxShapeInfo,
                                                              const std::vector<LongType>* dimsWithUnities,
                                                              memory::Workspace* workspace = nullptr);

  /**
 * Create a cached shape buffer with the view flag set
 * @param shapeInfo Source shape info to fork
 * @return Cached shape buffer with ARRAY_IS_VIEW flag set
   */
  ConstantShapeBuffer* bufferForShapeInfoWithView(LongType* shapeInfo);

  /**
 * Create a cached shape buffer with the view flag unset
 * @param shapeInfo Source shape info to fork
 * @return Cached shape buffer with ARRAY_IS_VIEW flag cleared
   */
  ConstantShapeBuffer* bufferForShapeInfoWithoutView(LongType* shapeInfo);

  /**
 * Create a cached shape buffer with the needs copy flag set
 * @param shapeInfo Source shape info to fork
 * @return Cached shape buffer with ARRAY_NEEDS_COPY flag set
   */
  ConstantShapeBuffer* bufferForShapeInfoWithNeedsCopy(LongType* shapeInfo);

  /**
 * Create a cached shape buffer with the needs copy flag unset
 * @param shapeInfo Source shape info to fork
 * @return Cached shape buffer with ARRAY_NEEDS_COPY flag cleared
   */
  ConstantShapeBuffer* bufferForShapeInfoWithoutNeedsCopy(LongType* shapeInfo);

  /**
 * Create a cached shape buffer with a copy offset flag set for a specific input
 * @param shapeInfo Source shape info to fork
 * @param inputIndex Input index (0-10) to set the copy offset flag for
 * @return Cached shape buffer with the specified ARRAY_COPY_OFFSET_INPUT_N flag set
   */
  ConstantShapeBuffer* bufferForShapeInfoWithCopyOffset(LongType* shapeInfo, int inputIndex);

  /**
 * Create a cached shape buffer with a copy offset flag cleared for a specific input
 * @param shapeInfo Source shape info to fork
 * @param inputIndex Input index (0-10) to clear the copy offset flag for
 * @return Cached shape buffer with the specified ARRAY_COPY_OFFSET_INPUT_N flag cleared
   */
  ConstantShapeBuffer* bufferForShapeInfoWithoutCopyOffset(LongType* shapeInfo, int inputIndex);

  /**
 * Create a cached shape buffer with all copy offset flags cleared
 * @param shapeInfo Source shape info to fork
 * @return Cached shape buffer with all ARRAY_COPY_OFFSET_INPUT_* flags cleared
   */
  ConstantShapeBuffer* bufferForShapeInfoWithoutAllCopyOffsets(LongType* shapeInfo);

  /**
 * Create a cached shape buffer with custom flags applied
 * @param shapeInfo Source shape info to fork
 * @param flagsToSet Flags to set (OR operation)
 * @param flagsToUnset Flags to unset (AND NOT operation)
 * @return Cached shape buffer with modified flags
   */
  ConstantShapeBuffer* bufferForShapeInfoWithFlags(LongType* shapeInfo, LongType flagsToSet, LongType flagsToUnset);

  /**
 * Create a cached shape buffer configured as a view with copy offset
 * @param shapeInfo Source shape info to fork
 * @param inputIndex Input index (0-10) to set the copy offset flag for
 * @return Cached shape buffer with ARRAY_IS_VIEW and ARRAY_COPY_OFFSET_INPUT_N flags set
   */
  ConstantShapeBuffer* bufferForShapeInfoAsViewWithOffset(LongType* shapeInfo, int inputIndex);

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

  /**
   * Clears all cached shape buffers to prevent memory leaks.
   * This is called during application shutdown to free accumulated cache memory.
   */
  void clearCache();

  /**
   * Get the total number of cached shape entries.
   *
   * @return Total number of cached shape buffers across all stripes
   */
  LongType getCachedEntries() const;

  /**
   * Get the total memory used by cached shape buffers in bytes.
   * This includes the shape_info buffer sizes across all cached entries.
   *
   * @return Total memory used in bytes
   */
  LongType getCachedBytes() const;

  /**
   * Get the peak number of entries that were cached simultaneously.
   *
   * @return Peak number of cached entries
   */
  LongType getPeakCachedEntries() const;

  /**
   * Get the peak memory usage by cached shape buffers in bytes.
   *
   * @return Peak memory usage in bytes
   */
  LongType getPeakCachedBytes() const;

  /**
   * Generate a human-readable string representation of the shape cache.
   * Shows the trie structure with cached shape buffers for debugging.
   *
   * @param maxDepth Maximum depth to traverse (default: 10, -1 for unlimited)
   * @param maxEntries Maximum number of entries to show (default: 100, -1 for unlimited)
   * @return String representation of the cache
   */
  std::string toString(int maxDepth = 10, int maxEntries = 100) const;

  /**
   * Get all ConstantShapeBuffer pointers currently in the cache.
   * This is used by lifecycle tracking to distinguish cached entries from real leaks.
   *
   * @param out_pointers Set to fill with pointers to all cached ConstantShapeBuffer objects
   */
  void getCachedPointers(std::unordered_set<void*>& out_pointers) const;
};
}  // namespace sd

#endif  // DEV_TESTS_CONSTANTSHAPEHELPER_H