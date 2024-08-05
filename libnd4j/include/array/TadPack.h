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

namespace sd {
class SD_LIB_EXPORT TadPack {
 private:
  ConstantShapeBuffer _tadShape;
  ConstantOffsetsBuffer _tadOffsets;
  LongType _numTads = 0;
  LongType _shapeInfoLength = 0;
  LongType* _dimensions = nullptr;
  LongType _dimensionsLength = 0;
 public:
  explicit TadPack(const ConstantShapeBuffer& shapes,
                   const ConstantOffsetsBuffer& offets, LongType numTads,
                   LongType* dimensions = nullptr, LongType dimLength = 0);
  TadPack() = default;
  ~TadPack() {};

  const LongType* primaryShapeInfo() const;
  const LongType* primaryOffsets() const;

  const LongType* specialShapeInfo() const;
  const LongType* specialOffsets() const;

  LongType numberOfTads() const;
  LongType shapeInfoLength() const;


  /**
   * These methods return either primary or special pointers depending on platform binaries were compiled for
   * @return
   */
  const LongType* platformShapeInfo() const;
  const LongType* platformOffsets() const;

  void print(const char* msg) const;
};
}  // namespace sd

#endif  // DEV_TESTS_TADPACK_H
