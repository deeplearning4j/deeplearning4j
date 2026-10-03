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
// The essential part of the idx-th Householder vector as a view of the vectors matrix (the column below the
// diagonal for type 'u', the row right of it for type 'v'). The view is handed back through a pointer: a named
// NDArray returned by value goes through NDArray's move constructor, which does not carry the view's offset over,
// so the tail came back pointing at the first element of the vectors' buffer instead of at its own column or row.
static NDArray *tailView(NDArray &vectors, const char type, const int shift, const int idx) {
  const int first = idx + 1 + shift;

  if (type == 'u') return vectors({first, -1, idx, idx + 1}, true);

  return vectors({idx, idx + 1, first, -1}, true);
}

//////////////////////////////////////////////////////////////////////////
NDArray HHsequence::getTail(const int idx) const {
  NDArray vectorsRef = *_vectors;
  NDArray *tailPtr = tailView(vectorsRef, _type, _shift, idx);

  // Frees the view once the array handed back has been built from it.
  struct ViewHolder {
    NDArray *view;
    explicit ViewHolder(NDArray *v) : view(v) {}
    ~ViewHolder() { delete view; }
  } holder(tailPtr);

  // A temporary built by the copy constructor, which keeps the view's offset. Returning a named local would
  // use the move constructor, which resets the offset to 0 unless the compiler elides the move.
  return NDArray(*tailPtr);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHsequence::applyTo_(NDArray* dest) {
  const int size = _type == 'u' ? _vectors->sizeAt(0) : _vectors->sizeAt(1);

  // A destination of another size is replaced by a size x size array, as Eigen's evalTo resizes its destination.
  // The replacement goes into *dest: it used to be built as a separate array that was freed before the caller
  // could see it.
  if (dest->rankOf() != 2 || (dest->sizeAt(0) != size && dest->sizeAt(1) != size)) {
    std::vector<LongType> sizeShape = {size, size};
    *dest = NDArray(dest->ordering(), sizeShape, dest->dataType(), dest->getContext());
  }
  dest->setIdentity();

  NDArray destRef = *dest;
  NDArray vectorsRef = *_vectors;

  for (int k = _diagSize - 1; k >= 0; --k) {
    int curNum = size - k - _shift;
    if (curNum < 1 || (k + 1 + _shift) >= size) continue;

    NDArray *blockPtr = destRef({dest->sizeAt(0) - curNum, dest->sizeAt(0), dest->sizeAt(1) - curNum, dest->sizeAt(1)}, true);
    NDArray block = *blockPtr;
    delete blockPtr;

    NDArray *tailPtr = tailView(vectorsRef, _type, _shift, k);
    Householder<T>::mulLeft(block, *tailPtr, _coeffs->t<T>(k));
    delete tailPtr;
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
