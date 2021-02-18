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


namespace sd      {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// A{M,K} * x{K,N} = b{M,N}
template <typename T>
void FullPivLU<T>::solve(const NDArray& A, const NDArray& b, NDArray& x) {

    if(A.rankOf() != 2)
        throw std::runtime_error("FullPivLU::solve: input matrix A must be 2D !");

    if(A.sizeAt(0) !=  b.sizeAt(0))
        throw std::runtime_error("FullPivLU::solve: A and b must have the same number of rows !");

    if(A.sizeAt(1) !=  x.sizeAt(0))
        throw std::runtime_error("FullPivLU::solve: number of A columns must be equal to number of x rows !");

    NDArray LU = A.dup();

    const int rows    = LU.sizeAt(0);
    const int cols    = LU.sizeAt(1);
    const int diagLen = math::nd4j_min<int>(rows, cols);

    std::vector<int> rowsInds(rows), colsInds(cols);

    int numOfTranspos = 0;
    int nonZeroPivots1 = diagLen;

    T maxPivot = T(0);

    for(int k = 0; k < diagLen; ++k) {

        NDArray bottomRightCorner = LU({k,rows, k,cols}, true);
        const int indPivot = static_cast<int>(bottomRightCorner.indexReduceNumber(indexreduce::IndexAbsoluteMax).t<Nd4jLong>(0));

        int colPivot = indPivot % (cols-k);
        int rowPivot = indPivot / (cols-k);

        T currentMax = math::nd4j_abs<T>(bottomRightCorner.t<T>(rowPivot, colPivot));

        // take into account that this was calculated in corner, not in whole LU
        rowPivot += k;
        colPivot += k;

        if(currentMax == T(0)) {

            nonZeroPivots1 = k;

            for(int i = k; i < diagLen; ++i)
                rowsInds[i] = colsInds[i] = i;

            break;
        }

        if(currentMax > maxPivot)
            maxPivot = currentMax;

        rowsInds[k] = rowPivot;
        colsInds[k] = colPivot;

        if(k != rowPivot) {
            NDArray row1 = LU({k,k+1, 0,0}, true);
            NDArray row2 = LU({rowPivot,rowPivot+1, 0,0}, true);
            row1.swapUnsafe(row2);
            ++numOfTranspos;
        }
        if(k != colPivot) {
            NDArray col1 = LU({0,0, k,k+1}, true);
            NDArray col2 = LU({0,0, colPivot,colPivot+1}, true);
            col1.swapUnsafe(col2);
            ++numOfTranspos;
        }

        if(k < rows-1)
            LU({k+1,rows, k,k+1}, true) /=  LU.t<T>(k, k);

        if(k < diagLen-1)
            LU({k+1,rows, k+1,cols},true) -= mmul(LU({k+1,rows, k,k+1},true), LU({k,k+1, k+1,cols},true));
    }

    //***************************************************//

    const T threshold = maxPivot * DataTypeUtils::eps<T>() * (T)diagLen;

    int  nonZeroPivots2 = 0;
    for(int i = 0; i < nonZeroPivots1; ++i)
        nonZeroPivots2 += static_cast<int>(math::nd4j_abs<T>(LU.t<T>(i,i)) > threshold);

    if(nonZeroPivots2 == 0) {
        x.nullify();
        return;
    }

    //***************************************************//

    std::vector<int> rowsPermut1(rows), rowsPermut2(rows), colsPermut(cols);
    std::iota(rowsPermut1.begin(), rowsPermut1.end(), 0);
    std::iota(colsPermut.begin(), colsPermut.end(), 0);

    for(int k = diagLen-1; k >= 0; --k)
        math::nd4j_swap<int>(rowsPermut1[k], rowsPermut1[rowsInds[k]]);

    for(int k = 0; k < diagLen; ++k)
        math::nd4j_swap<int>(colsPermut[k], colsPermut[colsInds[k]]);

    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < rows; ++j)
            if(i == rowsPermut1[j]) { rowsPermut2[i] = j; break; }

    //***************************************************//

    NDArray c = b.ulike();

    for (int i = 0; i < rows; ++i)
        c({i,i+1, 0,0}, true).assign(b({rowsPermut2[i],rowsPermut2[i]+1, 0,0}, true));


    NDArray cTopRows1 = c({0,diagLen, 0,0}, true);
    // TriangularSolver<T>::solve(LU({0,diagLen, 0,diagLen}, true), cTopRows1, true, true, cTopRows1);
    ops::helpers::triangularSolve2D<T>(nullptr, LU({0,diagLen, 0,diagLen}, true), cTopRows1,true,true, cTopRows1);

    if(rows > cols)
        c({cols,-1, 0,0}, true) -= mmul(LU({cols,-1, 0,0},true),  c({0,cols, 0,0}, true));

    NDArray cTopRows2 = c({0,nonZeroPivots2, 0,0}, true);
    // TriangularSolver<T>::solve(LU({0,nonZeroPivots2, 0,nonZeroPivots2}, true), cTopRows2, false, false, cTopRows2);
    ops::helpers::triangularSolve2D<T>(nullptr, LU({0,nonZeroPivots2, 0,nonZeroPivots2}, true),cTopRows2,false,false, cTopRows2);

    for(int i = 0; i < nonZeroPivots2; ++i)
        x({colsPermut[i],colsPermut[i]+1, 0,0}, true).assign(c({i,i+1, 0,0}, true));

    for(int i = nonZeroPivots2; i < cols; ++i)
        x({colsPermut[i],colsPermut[i]+1, 0,0}, true).nullify();
}

template class ND4J_EXPORT FullPivLU<float>;
template class ND4J_EXPORT FullPivLU<float16>;
template class ND4J_EXPORT FullPivLU<bfloat16>;
template class ND4J_EXPORT FullPivLU<double>;

}
}
}
