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
#include "../TadPack.h"

#include <helpers/shape.h>
#include <system/Environment.h>

namespace sd {
TadPack::TadPack(const ConstantShapeBuffer& shapes,
                 const ConstantOffsetsBuffer& offets, LongType numTads,
                 LongType* dimensions, LongType dimLength)
    : _tadShape(shapes),
      _tadOffsets(offets) {
  _numTads = numTads;
  _dimensionsLength = dimLength;
  if(dimensions != nullptr) {
    _dimensions = new LongType[dimLength];
    for(int i = 0; i < dimLength; i++) {
      _dimensions[i] = dimensions[i];
    }
  }

}

LongType* TadPack::primaryShapeInfo() {
  if(_tadShape.primary() == nullptr)
    THROW_EXCEPTION("TadPack::primaryShapeInfo: primary shape info is nullptr!");
  return _tadShape.primary();
}

LongType* TadPack::primaryOffsets() {
  return _tadOffsets.primary();
}

LongType* TadPack::specialShapeInfo() { return _tadShape.special(); }

LongType* TadPack::specialOffsets() { return _tadOffsets.special(); }

LongType TadPack::numberOfTads() const { return _numTads; }

LongType* TadPack::platformShapeInfo() {
  return Environment::getInstance().isCPU() ? primaryShapeInfo() : specialShapeInfo();
}

LongType* TadPack::platformOffsets() {
  return Environment::getInstance().isCPU() ? primaryOffsets() : specialOffsets();
}


void TadPack::print(const char* msg) {
  printf("---------------------------\n");
  printf("%s: ", msg);
  printf("Offsets:\n");
  for (int e = 0; e < _numTads; e++) {
    printf("%lld, ", _tadOffsets.primary()[e]);
  }
  printf("\n");

  printf("Dimensions:\n");
  if (_dimensions == nullptr || _dimensionsLength == 0) {
    printf("none\n");
  } else {
    for (int i = 0; i < _dimensionsLength; i++) {
      printf("%lld, ", _dimensions[i]);
    }
    printf("\n");
  }

  printf("tad pack shape info:");
  shape::printShapeInfo(_tadShape.primary());
  printf("\n");
  printf("number of tads: %lld\n", _numTads);
  printf("shape info length: %lld\n", _shapeInfoLength);
  printf("---------------------------\n");
}

LongType TadPack::shapeInfoLength() { return shape::shapeInfoLength(primaryShapeInfo()); }
}  // namespace sd
