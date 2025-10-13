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

#ifndef DEV_TESTS_TADPACK_H
#define DEV_TESTS_TADPACK_H

#include <array/ConstantOffsetsBuffer.h>
#include <array/ConstantShapeBuffer.h>
#include <system/common.h>

#include <array/NDArray.h>
#ifndef __JAVACPP_HACK__
namespace sd {
class SD_LIB_EXPORT TadPack {
 private:
  ConstantShapeBuffer *_tadShape;
  ConstantOffsetsBuffer *_tadOffsets;
  LongType _numTads = 0;
  LongType _shapeInfoLength = 0;
  LongType* _dimensions = nullptr;
  LongType _dimensionsLength = 0;
  size_t _packHash = 0;  // Cache the hash for quick comparison

 public:
  explicit TadPack( ConstantShapeBuffer *shapes,
                    ConstantOffsetsBuffer *offets, LongType numTads,
                   LongType* dimensions = nullptr, LongType dimLength = 0);
  TadPack() = default;
  ~TadPack();

  LongType* primaryShapeInfo();
  LongType* primaryOffsets();

  LongType* specialShapeInfo();
  LongType* specialOffsets();

  LongType numberOfTads() const;
  LongType shapeInfoLength();
  /**
   * Extracts an NDArray view for the given TAD index.
   * @param input The input NDArray.
   * @param tadIndex The index of the TAD to extract.
   * @return A new NDArray view representing the TAD.
   */
  NDArray *extractTadView(NDArray* input, sd::LongType tadIndex) {
    auto shapeInfo = primaryShapeInfo();
    auto offsets = primaryOffsets();

    auto tadOffset = offsets[tadIndex];
    auto x = input->buffer();
    NDArray *ret = new NDArray(x,shapeInfo,LaunchContext::defaultContext(),false,tadOffset);
    return ret;
  }

  void computeHash() {
    size_t hash = 17;

    // Add dimensions to hash
    for (LongType i = 0; i < _dimensionsLength; i++) {
      hash = hash * 31 + static_cast<size_t>(_dimensions[i]);
    }

    // Add shape info to hash if available
    LongType* primaryShape = primaryShapeInfo();
    if (primaryShape) {
      int rank = shape::rank(primaryShape);

      // Add rank
      hash = hash * 13 + static_cast<size_t>(rank);

      // Add shape dimensions
      LongType* shapeValues = shape::shapeOf(primaryShape);
      for (int i = 0; i < rank; i++) {
        hash = hash * 17 + static_cast<size_t>(shapeValues[i]);
      }

      // Add strides
      LongType* strides = shape::stride(primaryShape);
      for (int i = 0; i < rank; i++) {
        hash = hash * 23 + static_cast<size_t>(strides[i]);
      }

      // Add order and data type
      hash = hash * 29 + static_cast<size_t>(shape::order(primaryShape));
      hash = hash * 37 + static_cast<size_t>(ArrayOptions::dataType(primaryShape));
    }

    // Add number of TADs
    hash = hash * 41 + static_cast<size_t>(_numTads);

    _packHash = hash;
  }


  /**
   * These methods return either primary or special pointers depending on platform binaries were compiled for
   * @return
   */
  LongType* platformShapeInfo();
  LongType* platformOffsets();

  void print(const char* msg);
  bool operator==( TadPack& other);
};
}  // namespace sd

#endif  // DEV_TESTS_TADPACK_H
#endif
