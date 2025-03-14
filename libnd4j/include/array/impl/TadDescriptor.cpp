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

#include "../TadDescriptor.h"

#include <algorithm>
#include <helpers/ModularHasher.h>
namespace sd {
TadDescriptor::TadDescriptor(const TadDescriptor &other) {
  _originalShape = other._originalShape;
  _axis = other._axis;
  _unitiesInShape = other._unitiesInShape;
}

TadDescriptor::TadDescriptor(const LongType *originalShape, const LongType *dimensions, const LongType length,
                             const bool keepUnitiesInShape) {

  _axis.resize(length);
  for (LongType e = 0; e < length; e++) {
    _axis[e] = dimensions[e];
  }

  if (length > 1) std::sort(_axis.begin(), _axis.end());

  _originalShape = const_cast<sd::LongType *>(originalShape);
  _unitiesInShape = keepUnitiesInShape;
}

bool TadDescriptor::operator==(const TadDescriptor &other) const {
  return std::tie(_originalShape, _axis, _unitiesInShape) ==
         std::tie(other._originalShape, other._axis, other._unitiesInShape);
}

bool TadDescriptor::operator<(const TadDescriptor &other) const {
  return std::tie(_originalShape, _axis, _unitiesInShape) <
         std::tie(other._originalShape, other._axis, other._unitiesInShape);
}

std::vector<LongType> &TadDescriptor::axis() { return _axis; }

LongType *TadDescriptor::originalShape() { return _originalShape; }

bool TadDescriptor::areUnitiesinShape() const { return _unitiesInShape; }
}  // namespace sd

namespace std {
size_t hash<sd::TadDescriptor>::operator()(const sd::TadDescriptor &k) const {
  using namespace sd::helpers::detail;

  // Start with initial hash from unities flag
  uint64_t hash = ModularHasher::hash_scalar(k.areUnitiesinShape());
  // Hash the axis vector
  auto& axes = const_cast<sd::TadDescriptor&>(k).axis();
  if (!axes.empty()) {
    hash = ModularHasher::hash_vector(axes, hash);
  }

  return hash;
}
}  // namespace std
