/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <helpers/Sqrtm.h>
#include <ops/declarable/helpers/lup.h>
#include <helpers/EigenValsAndVecs.h>
#include <helpers/HessenbergAndSchur.h>
#include <helpers/FullPivLU.h>
#include <helpers/MmulHelper.h>


namespace sd      {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void sqrtmQuasiTrianDiag(const NDArray& matrixT, NDArray& sqrtT ) {

    const int rows = matrixT.sizeAt(0);

    for(int i = 0; i < rows; i++) {

        if (i == rows - 1 || matrixT.t<T>(i+1, i) == (T)0) {
            const auto elemT = matrixT.t<T>(i, i);
            if(elemT < (T)0)
                throw std::runtime_error("ops::helpers::Sqrtm::sqrtmQuasiTrianDiag: can't take sqrt of negative diagonal element of T matrix !");
            sqrtT.r<T>(i,i) = math::nd4j_sqrt<T,T>(elemT);
        }
        else {

            EigenValsAndVecs<T> es(matrixT({i,i+2, i,i+2}, true));  // es._Vecs {2,2,2}, es._Vals{2,2}

            const NDArray& vecs = es._Vecs;
            const NDArray& vals = es._Vals;

            const T& vecsReal00 = vecs.t<T>(0,0,0);
            const T& vecsImag00 = vecs.t<T>(0,0,1);
            const T& vecsReal01 = vecs.t<T>(0,1,0);
            const T& vecsImag01 = vecs.t<T>(0,1,1);
            const T& vecsReal10 = vecs.t<T>(1,0,0);
            const T& vecsImag10 = vecs.t<T>(1,0,1);
            const T& vecsReal11 = vecs.t<T>(1,1,0);
            const T& vecsImag11 = vecs.t<T>(1,1,1);

            // es.eigenvalues().cwiseSqrt().asDiagonal()
            T eigenValsSqrt[2][2];
            eigenValsSqrt[0][0] = vals.t<T>(0,0);
            eigenValsSqrt[0][1] = vals.t<T>(0,1);
            eigenValsSqrt[1][0] = vals.t<T>(1,0);
            eigenValsSqrt[1][1] = vals.t<T>(1,1);
            EigenValsAndVecs<T>::sqrtComplexNum(eigenValsSqrt[0][0], eigenValsSqrt[0][1]);
            EigenValsAndVecs<T>::sqrtComplexNum(eigenValsSqrt[1][0], eigenValsSqrt[1][1]);

            // es.eigenvectors() * es.eigenvalues().cwiseSqrt().asDiagonal()
            T vecsElem[2][2][2];
            EigenValsAndVecs<T>::multiplyComplexNums(vecsReal00,vecsImag00,  eigenValsSqrt[0][0],eigenValsSqrt[0][1],  vecsElem[0][0][0],vecsElem[0][0][1]);
            EigenValsAndVecs<T>::multiplyComplexNums(vecsReal01,vecsImag01,  eigenValsSqrt[1][0],eigenValsSqrt[1][1],  vecsElem[0][1][0],vecsElem[0][1][1]);
            EigenValsAndVecs<T>::multiplyComplexNums(vecsReal10,vecsImag10,  eigenValsSqrt[0][0],eigenValsSqrt[0][1],  vecsElem[1][0][0],vecsElem[1][0][1]);
            EigenValsAndVecs<T>::multiplyComplexNums(vecsReal11,vecsImag11,  eigenValsSqrt[1][0],eigenValsSqrt[1][1],  vecsElem[1][1][0],vecsElem[1][1][1]);

            // es.eigenvectors().inverse()
            T vecsElemInv[2][2][2];

            T tempReal, tempImag, divisorReal, divisorImag;
            EigenValsAndVecs<T>::multiplyComplexNums(vecsReal00,vecsImag00,  vecsReal11,vecsImag11,  divisorReal,divisorImag);
            EigenValsAndVecs<T>::multiplyComplexNums(vecsReal01,vecsImag01,  vecsReal10,vecsImag10,  tempReal,tempImag);
            divisorReal -= tempReal;
            divisorImag -= tempImag;

            EigenValsAndVecs<T>::divideComplexNums(vecsReal11,vecsImag11,    divisorReal,divisorImag,  vecsElemInv[0][0][0],vecsElemInv[0][0][1]);
            EigenValsAndVecs<T>::divideComplexNums(-vecsReal01,-vecsImag01,  divisorReal,divisorImag,  vecsElemInv[0][1][0],vecsElemInv[0][1][1]);
            EigenValsAndVecs<T>::divideComplexNums(-vecsReal10,-vecsImag10,  divisorReal,divisorImag,  vecsElemInv[1][0][0],vecsElemInv[1][0][1]);
            EigenValsAndVecs<T>::divideComplexNums(vecsReal00,vecsImag00,    divisorReal,divisorImag,  vecsElemInv[1][1][0],vecsElemInv[1][1][1]);

            // result
            T result[2][2][2];

            EigenValsAndVecs<T>::multiplyComplexNums(vecsElem[0][0][0],vecsElem[0][0][1],  vecsElemInv[0][0][0],vecsElemInv[0][0][1],  tempReal,tempImag);
            EigenValsAndVecs<T>::multiplyComplexNums(vecsElem[0][1][0],vecsElem[0][1][1],  vecsElemInv[1][0][0],vecsElemInv[1][0][1],  result[0][0][0],result[0][0][1]);
            result[0][0][0] += tempReal;

            EigenValsAndVecs<T>::multiplyComplexNums(vecsElem[0][0][0],vecsElem[0][0][1],  vecsElemInv[0][1][0],vecsElemInv[0][1][1],  tempReal,tempImag);
            EigenValsAndVecs<T>::multiplyComplexNums(vecsElem[0][1][0],vecsElem[0][1][1],  vecsElemInv[1][1][0],vecsElemInv[1][1][1],  result[0][1][0],result[0][1][1]);
            result[0][1][0] += tempReal;

            EigenValsAndVecs<T>::multiplyComplexNums(vecsElem[1][0][0],vecsElem[1][0][1],  vecsElemInv[0][0][0],vecsElemInv[0][0][1],  tempReal,tempImag);
            EigenValsAndVecs<T>::multiplyComplexNums(vecsElem[1][1][0],vecsElem[1][1][1],  vecsElemInv[1][0][0],vecsElemInv[1][0][1],  result[1][0][0],result[1][0][1]);
            result[1][0][0] += tempReal;

            EigenValsAndVecs<T>::multiplyComplexNums(vecsElem[1][0][0],vecsElem[1][0][1],  vecsElemInv[0][1][0],vecsElemInv[0][1][1],  tempReal,tempImag);
            EigenValsAndVecs<T>::multiplyComplexNums(vecsElem[1][1][0],vecsElem[1][1][1],  vecsElemInv[1][1][0],vecsElemInv[1][1][1],  result[1][1][0],result[1][1][1]);
            result[1][1][0] += tempReal;

            sqrtT.r<T>(i,i)     = result[0][0][0];
            sqrtT.r<T>(i,i+1)   = result[0][1][0];
            sqrtT.r<T>(i+1,i)   = result[1][0][0];
            sqrtT.r<T>(i+1,i+1) = result[1][1][0];

            ++i;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// all matrices are {2,2} here
template <typename T>
static void sqrtmQuasiTrianAuxEq(const NDArray& A, const NDArray& B, const NDArray& C, NDArray& X) {

    NDArray tempMatrix(A.ordering(), {4,4}, A.dataType(), A.getContext());

    tempMatrix.r<T>(0,0) = A.t<T>(0,0) + B.t<T>(0,0);
    tempMatrix.r<T>(1,1) = A.t<T>(0,0) + B.t<T>(1,1);
    tempMatrix.r<T>(2,2) = A.t<T>(1,1) + B.t<T>(0,0);
    tempMatrix.r<T>(3,3) = A.t<T>(1,1) + B.t<T>(1,1);
    tempMatrix.r<T>(0,1) = B.t<T>(1,0);
    tempMatrix.r<T>(0,2) = A.t<T>(0,1);
    tempMatrix.r<T>(1,0) = B.t<T>(0,1);
    tempMatrix.r<T>(1,3) = A.t<T>(0,1);
    tempMatrix.r<T>(2,0) = A.t<T>(1,0);
    tempMatrix.r<T>(2,3) = B.t<T>(1,0);
    tempMatrix.r<T>(3,1) = A.t<T>(1,0);
    tempMatrix.r<T>(3,2) = B.t<T>(0,1);
    tempMatrix.r<T>(0,3) = (T)0;
    tempMatrix.r<T>(1,2) = (T)0;
    tempMatrix.r<T>(2,1) = (T)0;
    tempMatrix.r<T>(3,0) = (T)0;

    NDArray result(A.ordering(), {4,1}, A.dataType(), A.getContext());
    result.r<T>(0,0) = C.t<T>(0,0);
    result.r<T>(1,0) = C.t<T>(0,1);
    result.r<T>(2,0) = C.t<T>(1,0);
    result.r<T>(3,0) = C.t<T>(1,1);

    FullPivLU<T>::solve(tempMatrix, result, result);

    X.r<T>(0,0) = result.t<T>(0);
    X.r<T>(0,1) = result.t<T>(1);
    X.r<T>(1,0) = result.t<T>(2);
    X.r<T>(1,1) = result.t<T>(3);
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void sqrtmQuasiTrianOffDiag(const NDArray& matrixT, NDArray& sqrtT ) {

    const int rows = matrixT.sizeAt(0);

    for (int j = 1; j < rows; j++) {

        if (matrixT.t<T>(j, j-1) != (T)0)
            continue;

        for (int i = j - 1; i >= 0; i--) {

            if (i > 0 && matrixT.t<T>(i, i-1) != (T)0)
                continue;

            const bool iBlockIs2x2 = (i < rows - 1) && (matrixT.t<T>(i+1, i) != (T)0);
            const bool jBlockIs2x2 = (j < rows - 1) && (matrixT.t<T>(j+1, j) != (T)0);

            if (iBlockIs2x2 && jBlockIs2x2) {

                NDArray A = sqrtT({i,i+2, i,i+2}, true);
                NDArray B = sqrtT({j,j+2, j,j+2}, true);
                NDArray X = matrixT({i,i+2, j,j+2}, true);//.dup();

                if (j - i > 2)
                    X -= mmul(sqrtT({i,i+2, i+2,j}, true), sqrtT({i+2,j, j,j+2}, true));

                sqrtmQuasiTrianAuxEq<T>(A, B, X, X);

                sqrtT.syncToDevice();
                sqrtT({i,i+2, j,j+2}, true).assign(X);
            }
            else if (iBlockIs2x2 && !jBlockIs2x2) {

                NDArray rhs = matrixT({i,i+2, j,j+1}, true);//.dup();

                if (j - i > 2)
                    rhs -= mmul(sqrtT({i,i+2, i+2,j}, true), sqrtT({i+2,j, j,j+1}, true));

                NDArray A(matrixT.ordering(), {2,2}, matrixT.dataType(), matrixT.getContext());
                A.r<T>(0,0) = A.r<T>(1,1) = sqrtT.t<T>(j,j);
                A.r<T>(0,1) = A.r<T>(1,0) = T(0);
                A += sqrtT({i,i+2, i,i+2}, true);

                FullPivLU<T>::solve(A,rhs,rhs);

                // sqrtT.syncToDevice();
                sqrtT({i,i+2, j,j+1}, true).assign(rhs);
            }
            else if (!iBlockIs2x2 && jBlockIs2x2) {

                NDArray rhs = matrixT({i,i+1, j,j+2}, true);//.dup();

                if (j - i > 1)
                    rhs -= mmul(sqrtT({i,i+1, i+1,j}, true), sqrtT({i+1,j, j,j+2}, true));

                NDArray A(matrixT.ordering(), {2,2}, matrixT.dataType(), matrixT.getContext());
                A.r<T>(0,0) = A.r<T>(1,1) = sqrtT.t<T>(i,i);
                A.r<T>(0,1) = A.r<T>(1,0) = T(0);
                A += sqrtT({j,j+2, j,j+2}, true).transpose();

                NDArray rhsT = rhs.transpose();
                FullPivLU<T>::solve(A,rhsT,rhsT);

                // sqrtT.syncToDevice();
                sqrtT({i,i+1, j,j+2}, true).assign(rhs);
            }
            else if (!iBlockIs2x2 && !jBlockIs2x2) {

                T temp = mmul(sqrtT({i,i+1, i+1,j}), sqrtT({i+1,j, j,j+1})).t<T>(0);        // dot
                sqrtT.r<T>(i,j) = (matrixT.t<T>(i,j) - temp ) / (sqrtT.t<T>(i,i) + sqrtT.t<T>(j,j));
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Sqrtm<T>::calc(const NDArray& in, NDArray& out) {

    if(in.rankOf() != 2 || in.sizeAt(0) != in.sizeAt(1))
        throw std::runtime_error("ops::helpers::Sqrtm::calc: input matrix must have rank 2 and be square !");
    if(!out.isSameShape(in))
        throw std::runtime_error("ops::helpers::Sqrtm::calc: output matrix must have the same shape as input one!");

    if(in.lengthOf() == 1) {
        out.r<T>(0) = math::nd4j_sqrt<T,T>(in.t<T>(0));
        return;
    }

    ops::helpers::Schur<T> schur(in);

    const NDArray& t1 = schur._T;
    const NDArray& t2 = schur._U;

    NDArray sqrtT = in.ulike();
    sqrtT.nullify();

    sqrtmQuasiTrianDiag<T>(schur._T, sqrtT);
    sqrtmQuasiTrianOffDiag<T>(schur._T, sqrtT);

    // out = U * sqrtT * U^T;
    NDArray temp = mmul(sqrtT, schur._U.transpose());
    MmulHelper::mmul(&schur._U, &temp, &out);
}

template class ND4J_EXPORT Sqrtm<float>;
template class ND4J_EXPORT Sqrtm<float16>;
template class ND4J_EXPORT Sqrtm<bfloat16>;
template class ND4J_EXPORT Sqrtm<double>;


}
}
}