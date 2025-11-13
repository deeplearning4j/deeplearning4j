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

#ifndef DEV_TESTS_CONSTANTTADHELPER_H
#define DEV_TESTS_CONSTANTTADHELPER_H

#include <array/ConstantDataBuffer.h>
#include <array/ConstantDescriptor.h>
#include <array/ShapeDescriptor.h>
#include <array/TadDescriptor.h>
#include <array/TadPack.h>
#include <helpers/DirectTadTrie.h>
#include <system/op_boilerplate.h>

#include <map>
#include <mutex>
#include <vector>

namespace sd {
class SD_LIB_EXPORT ConstantTadHelper {
 private:
  DirectTadTrie _trie;  // Single trie for device 0

  ConstantTadHelper() = default;

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

  TadPack *tadForDimensions(LongType *originalShape, LongType *dimensions, LongType dimLength);
  TadPack *tadForDimensions(ShapeDescriptor &descriptor, std::vector<LongType> &dimensions,
                           const bool keepUnitiesInShape = false);
  TadPack *tadForDimensions(TadDescriptor *descriptor);

  TadPack *tadForDimensions(LongType *originalShape, LongType dimension);
  TadPack *tadForDimensions(LongType *originalShape, std::vector<LongType> *dimensions);

  /**
   * Clear all cached TAD packs to prevent memory leaks during testing
   */
  void clearCache();

  /**
   * Get the total number of cached TAD pack entries.
   *
   * @return Total number of cached TAD packs across all stripes
   */
  LongType getCachedEntries() const;

  /**
   * Get the total memory used by cached TAD packs in bytes.
   * This includes both shape_info and offset buffer sizes.
   *
   * @return Total memory used in bytes
   */
  LongType getCachedBytes() const;

  /**
   * Get the peak number of TAD pack entries that were cached simultaneously.
   *
   * @return Peak number of cached TAD packs
   */
  LongType getPeakCachedEntries() const;

  /**
   * Get the peak memory usage by cached TAD packs in bytes.
   *
   * @return Peak memory usage in bytes
   */
  LongType getPeakCachedBytes() const;

  /**
   * Generate a human-readable string representation of the TAD cache.
   * Shows the trie structure with cached TAD packs for debugging.
   *
   * @param maxDepth Maximum depth to traverse (default: 10, -1 for unlimited)
   * @param maxEntries Maximum number of entries to show (default: 100, -1 for unlimited)
   * @return String representation of the cache
   */
  std::string toString(int maxDepth = 10, int maxEntries = 100) const;
};
}  // namespace sd

#endif  // DEV_TESTS_CONSTANTTADHELPER_H