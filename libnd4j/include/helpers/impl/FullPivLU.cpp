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
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/FullPivLU.h>
#include <ops/declarable/helpers/triangular_solve.h>

#include <numeric>

#if NOT_EXCLUDED(OP_triangular_solve)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
// A{M,K} * x{K,N} = b{M,N}
template <typename T>
void FullPivLU<T>::solve(NDArray &A, NDArray &b, NDArray& x) {
  if (A.rankOf() != 2) THROW_EXCEPTION("FullPivLU::solve: input matrix A must be 2D !");

  if (A.sizeAt(0) != b.sizeAt(0))
    THROW_EXCEPTION("FullPivLU::solve: A and b must have the same number of rows !");

  if (A.sizeAt(1) != x.sizeAt(0))
    THROW_EXCEPTION("FullPivLU::solve: number of A columns must be equal to number of x rows !");

  NDArray *LU = A.dup(A.ordering());
  NDArray luRef = *LU;
  const int rows = LU->sizeAt(0);
  const int cols = LU->sizeAt(1);
  const int diagLen = math::sd_min<int>(rows, cols);

  std::vector<int> rowsInds(rows), colsInds(cols);

  int nonZeroPivots1 = diagLen;

  T maxPivot = T(0);

  for (int k = 0; k < diagLen; ++k) {
    NDArray *bottomRightCornerPtr = luRef({k, rows, k, cols}, true);
    NDArray bottomRightCorner = *bottomRightCornerPtr;
    delete bottomRightCornerPtr;
    
    NDArray *indexNum = bottomRightCorner.indexReduceNumber(indexreduce::IndexAbsoluteMax);
    const int indPivot = static_cast<int>(indexNum->t<LongType>(0));

    int colPivot = indPivot % (cols - k);
    int rowPivot = indPivot / (cols - k);

    T currentMax = math::sd_abs<T,T>(bottomRightCorner.t<T>(rowPivot, colPivot));

    // take into account that this was calculated in corner, not in whole LU
    rowPivot += k;
    colPivot += k;

    if (currentMax == T(0)) {
      nonZeroPivots1 = k;

      for (int i = k; i < diagLen; ++i) rowsInds[i] = colsInds[i] = i;

      delete indexNum;
      break;
    }

    if (currentMax > maxPivot) maxPivot = currentMax;

    rowsInds[k] = rowPivot;
    colsInds[k] = colPivot;

    if (k != rowPivot) {
      NDArray *row1Ptr = luRef({k, k + 1, 0, 0}, true);
      NDArray *row2Ptr = luRef({rowPivot, rowPivot + 1, 0, 0}, true);
      row1Ptr->swapUnsafe(*row2Ptr);
      delete row1Ptr;
      delete row2Ptr;
    }
    if (k != colPivot) {
      NDArray *col1Ptr = luRef({0, 0, k, k + 1}, true);
      NDArray *col2Ptr = luRef({0, 0, colPivot, colPivot + 1}, true);
      col1Ptr->swapUnsafe(*col2Ptr);
      delete col1Ptr;
      delete col2Ptr;
    }

    if (k < rows - 1) {
      NDArray *divViewPtr = luRef({k + 1, rows, k, k + 1}, true);
      *divViewPtr /= luRef.t<T>(k, k);
      delete divViewPtr;
    }

    if (k < diagLen - 1) {
      NDArray *leftPtr = luRef({k + 1, rows, k, k + 1}, true);
      NDArray *rightPtr = luRef({k, k + 1, k + 1, cols}, true);
      NDArray *targetPtr = luRef({k + 1, rows, k + 1, cols}, true);
      NDArray left = *leftPtr;
      NDArray right = *rightPtr;
      NDArray *mulResult = mmul(left, right);
      *targetPtr -= *mulResult;
      delete mulResult;
      delete leftPtr;
      delete rightPtr;
      delete targetPtr;
    }

    delete indexNum;
  }
  //***************************************************//

  const T threshold = maxPivot * DataTypeUtils::eps<T>() * (T)diagLen;

  int nonZeroPivots2 = 0;
  for (int i = 0; i < nonZeroPivots1; ++i)
    nonZeroPivots2 += static_cast<int>(math::sd_abs<T,T>(luRef.t<T>(i, i)) > threshold);

  if (nonZeroPivots2 == 0) {
    x.nullify();
    delete LU;
    return;
  }

  //***************************************************//

  std::vector<int> rowsPermut1(rows), rowsPermut2(rows), colsPermut(cols);
  std::iota(rowsPermut1.begin(), rowsPermut1.end(), 0);
  std::iota(colsPermut.begin(), colsPermut.end(), 0);

  for (int k = diagLen - 1; k >= 0; --k) math::sd_swap<int>(rowsPermut1[k], rowsPermut1[rowsInds[k]]);

  for (int k = 0; k < diagLen; ++k) math::sd_swap<int>(colsPermut[k], colsPermut[colsInds[k]]);

  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < rows; ++j)
      if (i == rowsPermut1[j]) {
        rowsPermut2[i] = j;
        break;
      }

  //***************************************************//

  NDArray *bUlike = b.ulike();
  NDArray c = *bUlike;

  for (int i = 0; i < rows; ++i) {
    NDArray *cAssignPtr = b({rowsPermut2[i], rowsPermut2[i] + 1, 0, 0}, true);
    NDArray cAssign = *cAssignPtr;
    delete cAssignPtr;
    
    NDArray *cTargetPtr = c({i, i + 1, 0, 0}, true);
    cTargetPtr->assign(&cAssign);
    delete cTargetPtr;
  }
  
  NDArray *cTopRows1Ptr = c({0, diagLen, 0, 0}, true);
  NDArray cTopRows1 = *cTopRows1Ptr;
  delete cTopRows1Ptr;
  
  NDArray *luDiagPtr = luRef({0, diagLen, 0, diagLen}, true);
  // TriangularSolver<T>::solve(LU({0,diagLen, 0,diagLen}, true), cTopRows1, true, true, cTopRows1);
  helpers::triangularSolve2D<T>(nullptr, *luDiagPtr, cTopRows1, true, true, cTopRows1);
  delete luDiagPtr;

  if (rows > cols) {
    NDArray *leftPtr = luRef({cols, -1, 0, 0}, true);
    NDArray *rightPtr = c({0, cols, 0, 0}, true);
    NDArray *targetPtr = c({cols, -1, 0, 0}, true);
    NDArray left = *leftPtr;
    NDArray right = *rightPtr;
    NDArray *mulResult = mmul(left, right);
    *targetPtr -= *mulResult;
    delete mulResult;
    delete leftPtr;
    delete rightPtr;
    delete targetPtr;
  }
  
  NDArray *cTopRows2Ptr = c({0, nonZeroPivots2, 0, 0}, true);
  NDArray cTopRows2 = *cTopRows2Ptr;
  delete cTopRows2Ptr;
  
  NDArray *luNonZeroPtr = luRef({0, nonZeroPivots2, 0, nonZeroPivots2}, true);
  helpers::triangularSolve2D<T>(nullptr, *luNonZeroPtr, cTopRows2, false, false, cTopRows2);
  delete luNonZeroPtr;

  for (int i = 0; i < nonZeroPivots2; ++i) {
    NDArray *cAssignPtr = c({i, i + 1, 0, 0}, true);
    NDArray cAssign = *cAssignPtr;
    delete cAssignPtr;
    
    NDArray *xTargetPtr = x({colsPermut[i], colsPermut[i] + 1, 0, 0}, true);
    xTargetPtr->assign(&cAssign);
    delete xTargetPtr;
  }
  
  for (int i = nonZeroPivots2; i < cols; ++i) {
    NDArray *xNullifyPtr = x({colsPermut[i], colsPermut[i] + 1, 0, 0}, true);
    xNullifyPtr->nullify();
    delete xNullifyPtr;
  }

  delete LU;
  delete bUlike;
}

BUILD_SINGLE_TEMPLATE( class FullPivLU, , SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
