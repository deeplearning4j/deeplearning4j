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
// Created by Yurii Shyrma on 02.01.2018
//
#include <helpers/hhSequence.h>
#include <helpers/householder.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
HHsequence::HHsequence(NDArray* vectors, NDArray* coeffs, const char type)
    : _vectors(vectors), _coeffs(coeffs) {
  _diagSize = math::sd_min(_vectors->sizeAt(0), _vectors->sizeAt(1));
  _shift = 0;
  _type = type;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHsequence::mulLeft_(NDArray* matrix) {
  const int rows = _vectors->sizeAt(0);
  const int cols = _vectors->sizeAt(1);
  const int inRows = matrix->sizeAt(0);
  NDArray matrixRef = *matrix;
  NDArray vectorsRef =  *_vectors;
  for (int i = _diagSize - 1; i >= 0; --i) {
    if (_type == 'u') {
      NDArray *blockPtr = matrixRef({inRows - rows + _shift + i, inRows, 0, 0}, true);
      NDArray block = *blockPtr;
      
      NDArray *vectorPtr = vectorsRef({i + 1 + _shift, rows, i, i + 1}, true);
      NDArray vector = *vectorPtr;
      
      Householder<T>::mulLeft(block, vector, _coeffs->t<T>(i));
      
      delete blockPtr;
      delete vectorPtr;
    } else {
      NDArray *blockPtr = matrixRef({inRows - cols + _shift + i, inRows, 0, 0}, true);
      NDArray block = *blockPtr;
      
      NDArray *vectorPtr = vectorsRef({i, i + 1, i + 1 + _shift, cols}, true);
      NDArray vector = *vectorPtr;
      
      Householder<T>::mulLeft(block, vector, _coeffs->t<T>(i));
      
      delete blockPtr;
      delete vectorPtr;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
NDArray HHsequence::getTail(const int idx) const {
  int first = idx + 1 + _shift;
  NDArray vectorsRef =  *_vectors;

  if (_type == 'u') {
    NDArray *tailPtr = vectorsRef({first, -1, idx, idx + 1}, true);
    NDArray tail = *tailPtr;
    delete tailPtr;
    return tail;
  } else {
    NDArray *tailPtr = vectorsRef({idx, idx + 1, first, -1}, true);
    NDArray tail = *tailPtr;
    delete tailPtr;
    return tail;
  }
}
//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHsequence::applyTo_(NDArray* dest) {
  int size = _type == 'u' ? _vectors->sizeAt(0) : _vectors->sizeAt(1);
 NDArray *originalDest = dest;
 NDArray destRef = *dest;
  std::vector<LongType> sizeShape = {size,size};
  if (dest->rankOf() != 2 || (dest->sizeAt(0) != size && dest->sizeAt(1) != size)) {
    dest = new NDArray(dest->ordering(), sizeShape, dest->dataType(), dest->getContext());
    destRef = *dest;
  }
  dest->setIdentity();

  for (int k = _diagSize - 1; k >= 0; --k) {
    int curNum = size - k - _shift;
    if (curNum < 1 || (k + 1 + _shift) >= size) continue;
    
    NDArray *blockPtr = destRef({dest->sizeAt(0) - curNum, dest->sizeAt(0), dest->sizeAt(1) - curNum, dest->sizeAt(1)}, true);
    NDArray block = *blockPtr;

    NDArray tailK = getTail(k);
    Householder<T>::mulLeft(block, tailK, _coeffs->t<T>(k));
    
    delete blockPtr;
  }

  if(originalDest != dest) {
    delete dest;
  }
}

//////////////////////////////////////////////////////////////////////////
void HHsequence::applyTo(NDArray* dest) {
  auto xType = _coeffs->dataType();
  BUILD_SINGLE_SELECTOR(xType, applyTo_, (dest), SD_FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
void HHsequence::mulLeft(NDArray* matrix) {
  auto xType = _coeffs->dataType();
  BUILD_SINGLE_SELECTOR(xType, mulLeft_, (matrix), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE( void HHsequence::applyTo_, (sd::NDArray * dest), SD_FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE( void HHsequence::mulLeft_, (NDArray * matrix), SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
