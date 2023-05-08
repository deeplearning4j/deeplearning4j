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

namespace sd {
class SD_LIB_EXPORT TadPack {
 private:
  ConstantShapeBuffer _tadShape;
  ConstantOffsetsBuffer _tadOffsets;
  sd::LongType _numTads = 0;
  int _shapeInfoLength = 0;

 public:
  explicit TadPack(const ConstantShapeBuffer& shapes, const ConstantOffsetsBuffer& offets, sd::LongType numTads);
  TadPack() = default;
  ~TadPack() {};

  const sd::LongType* primaryShapeInfo() const;
  const sd::LongType* primaryOffsets() const;

  const sd::LongType* specialShapeInfo() const;
  const sd::LongType* specialOffsets() const;

  sd::LongType numberOfTads() const;
  int shapeInfoLength() const;

  /**
   * These methods return either primary or special pointers depending on platform binaries were compiled for
   * @return
   */
  const sd::LongType* platformShapeInfo() const;
  const sd::LongType* platformOffsets() const;
};
}  // namespace sd

#endif  // DEV_TESTS_TADPACK_H
