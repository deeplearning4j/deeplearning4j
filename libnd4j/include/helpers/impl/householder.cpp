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
// Created by Yurii Shyrma on 18.12.2017
//
#include <helpers/householder.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixData(NDArray& x, NDArray& tail, T& coeff, T& normX) {
  // input validation
  if (x.rankOf() != 1 && !x.isScalar())
    THROW_EXCEPTION(
        "ops::helpers::Householder::evalHHmatrixData method: input array must have rank = 1 or to be scalar!");

  if (!x.isScalar() && x.lengthOf() != tail.lengthOf() + 1)
    THROW_EXCEPTION(
        "ops::helpers::Householder::evalHHmatrixData method: input tail vector must have length less than unity "
        "compared to input x vector!");

  const auto xLen = x.lengthOf();

  NDArray *xTailPtr = xLen > 1 ? x({1, -1}) : nullptr;
  NDArray xTail = xTailPtr != nullptr ? *xTailPtr : NDArray();
  if (xTailPtr != nullptr) delete xTailPtr;

  T tailXnorm;
  if (xLen > 1) {
    auto* tailNormPtr = xTail.reduceNumber(reduce::SquaredNorm);
    tailXnorm = tailNormPtr->t<T>(0);
    delete tailNormPtr;
  } else {
    tailXnorm = (T)0;
  }

  const auto xFirstElem = x.t<T>(0);

  if (tailXnorm <= DataTypeUtils::min_positive<T>()) {
    normX = xFirstElem;
    coeff = (T)0.f;
    tail = (T)0.f;
  } else {
    normX = math::sd_sqrt<T, T>(xFirstElem * xFirstElem + tailXnorm);

    if (xFirstElem >= (T)0.f) normX = -normX;  // choose opposite sign to lessen roundoff error

    coeff = (normX - xFirstElem) / normX;
    T divisor = xFirstElem - normX;
    NDArray *tailAssign = xTail / divisor;
    tail.assign(tailAssign);
    delete tailAssign;
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::evalHHmatrixDataI(NDArray& x, T& coeff, T& normX) {
  // input validation
  if (x.rankOf() != 1 && !x.isScalar())
    THROW_EXCEPTION(
        "ops::helpers::Householder::evalHHmatrixDataI method: input array must have rank = 1 or to be scalar!");

  int rows = (int)x.lengthOf() - 1;
  int num = 1;

  if (rows == 0) {
    rows = 1;
    num = 0;
  }

  NDArray *tailPtr = x({num, -1});
  NDArray tail = *tailPtr;
  delete tailPtr;

  evalHHmatrixData(x, tail, coeff, normX);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulLeft(NDArray& matrix, NDArray& tail, const T coeff) {
  if (matrix.sizeAt(0) == 1 && coeff != (T)0) {
    NDArray *scaledResult = matrix * ((T)1.f - coeff);
    matrix.assign(scaledResult);
    delete scaledResult;
  } else if (coeff != (T)0.f) {
    NDArray *bottomPartPtr = matrix({1, matrix.sizeAt(0), 0, 0}, true);
    NDArray bottomPart = *bottomPartPtr;
    delete bottomPartPtr;
    
    NDArray *fistRowPtr = matrix({0, 1, 0, 0}, true);
    NDArray fistRow = *fistRowPtr;
    delete fistRowPtr;
    
    NDArray *tailTranspose = tail.transpose();
    if (tail.isColumnVector()) {
      NDArray *resultingRow = mmul(*tailTranspose, bottomPart);
      NDArray *rowPlusFirst = (*resultingRow) + fistRow;
      delete resultingRow;
      resultingRow = rowPlusFirst;
      
      NDArray *scaledRow = (*resultingRow) * coeff;
      delete resultingRow;
      resultingRow = scaledRow;
      
      NDArray *firstMinusRow = fistRow - (*resultingRow);
      fistRow.assign(firstMinusRow);
      delete firstMinusRow;
      
      NDArray *tailMulRow = mmul(tail, *resultingRow);
      NDArray *bottomMinusTailMul = bottomPart - (*tailMulRow);
      bottomPart.assign(bottomMinusTailMul);
      delete tailMulRow;
      delete bottomMinusTailMul;
      delete resultingRow;
    } else {
      NDArray *resultingRow = mmul(tail, bottomPart);
      NDArray *rowPlusFirst = (*resultingRow) + fistRow;
      delete resultingRow;
      resultingRow = rowPlusFirst;
      
      NDArray *scaledRow = (*resultingRow) * coeff;
      delete resultingRow;
      resultingRow = scaledRow;
      
      NDArray *firstMinusRow = fistRow - (*resultingRow);
      fistRow.assign(firstMinusRow);
      delete firstMinusRow;
      
      NDArray *transTailMulRow = mmul(*tailTranspose, *resultingRow);
      NDArray *bottomMinusTrans = bottomPart - (*transTailMulRow);
      bottomPart.assign(bottomMinusTrans);
      delete transTailMulRow;
      delete bottomMinusTrans;
      delete resultingRow;
    }
    delete tailTranspose;
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Householder<T>::mulRight(NDArray& matrix, NDArray& tail, const T coeff) {
  if (matrix.sizeAt(1) == 1 && coeff != (T)0) {
    NDArray *scaledResult = matrix * ((T)1.f - coeff);
    matrix.assign(scaledResult);
    delete scaledResult;
  } else if (coeff != (T)0.f) {
    NDArray *rightPartPtr = matrix({0, 0, 1, matrix.sizeAt(1)}, true);
    NDArray rightPart = *rightPartPtr;
    delete rightPartPtr;
    
    NDArray *fistColPtr = matrix({0, 0, 0, 1}, true);
    NDArray fistCol = *fistColPtr;
    delete fistColPtr;

    NDArray *transposedTail = tail.transpose();
    if (tail.isColumnVector()) {
      NDArray *resultingCol = mmul(rightPart, tail);
      NDArray *colPlusFirst = (*resultingCol) + fistCol;
      delete resultingCol;
      resultingCol = colPlusFirst;
      
      NDArray *scaledCol = (*resultingCol) * coeff;
      delete resultingCol;
      resultingCol = scaledCol;
      
      NDArray *firstMinusCol = fistCol - (*resultingCol);
      fistCol.assign(firstMinusCol);
      delete firstMinusCol;
      
      NDArray *colMulTransTail = mmul(*resultingCol, *transposedTail);
      NDArray *rightMinusColMul = rightPart - (*colMulTransTail);
      rightPart.assign(rightMinusColMul);
      delete colMulTransTail;
      delete rightMinusColMul;
      delete resultingCol;
    } else {
      NDArray *resultingCol = mmul(rightPart, *transposedTail);
      NDArray *colPlusFirst = (*resultingCol) + fistCol;
      delete resultingCol;
      resultingCol = colPlusFirst;
      
      NDArray *scaledCol = (*resultingCol) * coeff;
      delete resultingCol;
      resultingCol = scaledCol;
      
      NDArray *firstMinusCol = fistCol - (*resultingCol);
      fistCol.assign(firstMinusCol);
      delete firstMinusCol;
      
      NDArray *colMulTail = mmul(*resultingCol, tail);
      NDArray *rightMinusColMul = rightPart - (*colMulTail);
      rightPart.assign(rightMinusColMul);
      delete colMulTail;
      delete rightMinusColMul;
      delete resultingCol;
    }

    delete transposedTail;
  }
}

BUILD_SINGLE_TEMPLATE( class Householder, , SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
