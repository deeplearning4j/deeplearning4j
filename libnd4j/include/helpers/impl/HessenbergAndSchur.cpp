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

#include <helpers/HessenbergAndSchur.h>
#include <helpers/householder.h>
#include <helpers/hhSequence.h>
#include <helpers/jacobiSVD.h>


namespace sd      {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
Hessenberg<T>::Hessenberg(const NDArray& matrix) {

    if(matrix.rankOf() != 2)
        throw std::runtime_error("ops::helpers::Hessenberg constructor: input matrix must be 2D !");

    if(matrix.sizeAt(0) == 1) {
        _Q = NDArray(matrix.ordering(), {1,1}, matrix.dataType(), matrix.getContext());
        _Q = 1;
        _H = matrix.dup();
        return;
    }

    if(matrix.sizeAt(0) != matrix.sizeAt(1))
        throw std::runtime_error("ops::helpers::Hessenberg constructor: input array must be 2D square matrix !");

    _H = matrix.dup();
    _Q = matrix.ulike();

    evalData();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Hessenberg<T>::evalData() {

    const int rows = _H.sizeAt(0);

    NDArray hhCoeffs(_H.ordering(), {rows - 1}, _H.dataType(), _H.getContext());

    // calculate _H
    for(uint i = 0; i < rows - 1; ++i) {

        T coeff, norm;

        NDArray tail1 = _H({i+1,-1, i,i+1});
        NDArray tail2 = _H({i+2,-1, i,i+1}, true);

        Householder<T>::evalHHmatrixDataI(tail1, coeff, norm);

        _H({0,0, i,i+1}). template r<T>(i+1) = norm;
        hhCoeffs. template r<T>(i) = coeff;

        NDArray bottomRightCorner = _H({i+1,-1, i+1,-1}, true);
        Householder<T>::mulLeft(bottomRightCorner, tail2, coeff);

        NDArray rightCols = _H({0,0, i+1,-1}, true);
        Householder<T>::mulRight(rightCols, tail2.transpose(), coeff);
    }

    // calculate _Q
    HHsequence hhSeq(_H, hhCoeffs, 'u');
    hhSeq._diagSize = rows - 1;
    hhSeq._shift = 1;
    hhSeq.applyTo_<T>(_Q);

    // fill down with zeros starting at first subdiagonal
    _H.fillAsTriangular<T>(0, -1, 0, _H, 'l');
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
Schur<T>::Schur(const NDArray& matrix) {

    if(matrix.rankOf() != 2)
        throw std::runtime_error("ops::helpers::Schur constructor: input matrix must be 2D !");

    if(matrix.sizeAt(0) != matrix.sizeAt(1))
        throw std::runtime_error("ops::helpers::Schur constructor: input array must be 2D square matrix !");

    evalData(matrix);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void Schur<T>::evalData(const NDArray& matrix) {

    const T scale = matrix.reduceNumber(reduce::AMax).template t<T>(0);

    const T almostZero = DataTypeUtils::min<T>();

    if(scale < DataTypeUtils::min<T>()) {

        _T = matrix.ulike();
        _U = matrix.ulike();

        _T.nullify();
        _U.setIdentity();

        return;
    }

    // perform Hessenberg decomposition
    Hessenberg<T> hess(matrix / scale);

    _T = std::move(hess._H);
    _U = std::move(hess._Q);

    calcFromHessenberg();

    _T *= scale;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void Schur<T>::splitTwoRows(const int ind, const T shift) {

    const int numCols = _T.sizeAt(1);

    T p = (T)0.5 * (_T.t<T>(ind-1, ind-1) - _T.t<T>(ind, ind));

    T q = p*p + _T.t<T>(ind, ind-1) * _T.t<T>(ind-1, ind);

    _T.r<T>(ind, ind) += shift;
    _T.r<T>(ind-1, ind-1) += shift;

    if (q >= (T)0) {

        T z = math::nd4j_sqrt<T,T>(math::nd4j_abs<T>(q));

        NDArray rotation(_T.ordering(), {2, 2}, _T.dataType(), _T.getContext());

        if (p >= (T)0)
            JacobiSVD<T>::createJacobiRotationGivens(p+z, _T.t<T>(ind, ind-1), rotation);
        else
            JacobiSVD<T>::createJacobiRotationGivens(p-z, _T.t<T>(ind, ind-1), rotation);

        NDArray rightCols = _T({0,0, ind-1,-1});
        JacobiSVD<T>::mulRotationOnLeft(ind-1, ind, rightCols, rotation.transpose());

        NDArray topRows = _T({0,ind+1, 0,0});
        JacobiSVD<T>::mulRotationOnRight(ind-1, ind, topRows, rotation);

        JacobiSVD<T>::mulRotationOnRight(ind-1, ind, _U, rotation);

        _T.r<T>(ind, ind-1) = (T)0;
    }

    if (ind > 1)
        _T.r<T>(ind-1, ind-2) = (T)0;
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void Schur<T>::calcShift(const int ind, const int iter, T& shift, NDArray& shiftVec) {

    // shiftVec has length = 3

    shiftVec.r<T>(0) = _T.t<T>(ind, ind);
    shiftVec.r<T>(1) = _T.t<T>(ind-1, ind-1);
    shiftVec.r<T>(2) = _T.t<T>(ind, ind-1) * _T.t<T>(ind-1, ind);

    if (iter == 10) {
        shift += shiftVec.t<T>(0);

        for (int i = 0; i <= ind; ++i)
            _T.r<T>(i,i) -= shiftVec.t<T>(0);

        T s = math::nd4j_abs<T>(_T.t<T>(ind, ind-1)) + math::nd4j_abs<T>(_T.t<T>(ind-1, ind-2));

        shiftVec.r<T>(0) = T(0.75) * s;
        shiftVec.r<T>(1) = T(0.75) * s;
        shiftVec.r<T>(2) = T(-0.4375) * s*s;
    }

    if (iter == 30) {

        T s = (shiftVec.t<T>(1) - shiftVec.t<T>(0)) / T(2.0);
        s = s*s + shiftVec.t<T>(2);

        if (s > T(0)) {

            s = math::nd4j_sqrt<T,T>(s);

            if (shiftVec.t<T>(1) < shiftVec.t<T>(0))
                s = -s;

            s = s + (shiftVec.t<T>(1) - shiftVec.t<T>(0)) / T(2.0);
            s = shiftVec.t<T>(0) - shiftVec.t<T>(2) / s;
            shift += s;

            for (int i = 0; i <= ind; ++i)
                _T.r<T>(i,i) -= s;

            shiftVec = T(0.964);
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void Schur<T>::initFrancisQR(const int ind1,  const int ind2, const NDArray& shiftVec, int& ind3, NDArray& householderVec) {

  // shiftVec has length = 3

  for (ind3 = ind2-2; ind3 >= ind1; --ind3) {

        const T mm = _T.t<T>(ind3, ind3);
        const T r = shiftVec.t<T>(0) - mm;
        const T s = shiftVec.t<T>(1) - mm;

        householderVec.r<T>(0) = (r * s - shiftVec.t<T>(2)) / _T.t<T>(ind3+1, ind3) + _T.t<T>(ind3, ind3+1);
        householderVec.r<T>(1) = _T.t<T>(ind3+1, ind3+1) - mm - r - s;
        householderVec.r<T>(2) = _T.t<T>(ind3+2, ind3+1);

        if (ind3 == ind1)
          break;

        const T lhs = _T.t<T>(ind3,ind3-1) * (math::nd4j_abs<T>(householderVec.t<T>(1)) + math::nd4j_abs<T>(householderVec.t<T>(2)));
        const T rhs = householderVec.t<T>(0) * (math::nd4j_abs<T>(_T.t<T>(ind3-1, ind3-1)) + math::nd4j_abs<T>(mm) + math::nd4j_abs<T>(_T.t<T>(ind3+1, ind3+1)));

        if(math::nd4j_abs<T>(lhs) < DataTypeUtils::eps<T>() * rhs)
            break;
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void Schur<T>::doFrancisQR(const int ind1, const int ind2, const int ind3, const NDArray& householderVec) {

    if(!(ind2 >= ind1))
        throw std::runtime_error("ops::helpers::Schur:doFrancisQR: wrong input indexes, condition ind2 >= ind1 must be true !");
    if(!(ind2 <= ind3-2))
        throw std::runtime_error("ops::helpers::Schur:doFrancisQR: wrong input indexes, condition iind2 <= ind3-2 must be true !");

    const int numCols = _T.sizeAt(1);

    for (int k = ind2; k <= ind3-2; ++k) {

        const bool firstIter = (k == ind2);

        T coeff, normX;
        NDArray tail(_T.ordering(), {2, 1}, _T.dataType(), _T.getContext());
        Householder<T>::evalHHmatrixData(firstIter ? householderVec : _T({k,k+3, k-1,k}), tail, coeff, normX);

        if (normX != T(0)) {

            if (firstIter && k > ind1)
                _T.r<T>(k, k-1) = -_T.t<T>(k, k-1);
            else if (!firstIter)
                _T.r<T>(k, k-1) = normX;

            NDArray block1 = _T({k,k+3, k,numCols}, true);
            Householder<T>::mulLeft(block1, tail, coeff);

            NDArray block2 = _T({0,math::nd4j_min<int>(ind3,k+3)+1, k,k+3}, true);
            Householder<T>::mulRight(block2, tail, coeff);

            NDArray block3 = _U({0,numCols, k,k+3}, true);
            Householder<T>::mulRight(block3, tail, coeff);
        }
    }

    T coeff, normX;
    NDArray tail(_T.ordering(), {1, 1}, _T.dataType(), _T.getContext());
    Householder<T>::evalHHmatrixData(_T({ind3-1,ind3+1, ind3-2,ind3-1}), tail, coeff, normX);

    if (normX != T(0)) {

        _T.r<T>(ind3-1, ind3-2) = normX;

        NDArray block1 = _T({ind3-1,ind3+1, ind3-1,numCols}, true);
        Householder<T>::mulLeft(block1, tail, coeff);

        NDArray block2 = _T({0,ind3+1, ind3-1,ind3+1}, true);
        Householder<T>::mulRight(block2, tail, coeff);

        NDArray block3 = _U({0,numCols, ind3-1,ind3+1}, true);
        Householder<T>::mulRight(block3, tail, coeff);
    }

    for (int i = ind2+2; i <= ind3; ++i) {
        _T.r<T>(i, i-2) = T(0);
        if (i > ind2+2)
            _T.r<T>(i, i-3) = T(0);
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void Schur<T>::calcFromHessenberg() {

    const int maxIters = _maxItersPerRow * _T.sizeAt(0);

    const int numCols = _T.sizeAt(1);
    int iu = numCols - 1;
    int iter = 0;
    int totalIter = 0;

    T shift = T(0);

    T norm = 0;
    for (int j = 0; j < numCols; ++j)
        norm += _T({0,math::nd4j_min<int>(numCols,j+2), j,j+1}).reduceNumber(reduce::ASum).template t<T>(0);

    if(norm != T(0)) {

        while (iu >= 0) {

            const int il = getSmallSubdiagEntry(iu);

            if (il == iu) {

                _T.r<T>(iu,iu) = _T.t<T>(iu,iu) + shift;
                if (iu > 0)
                    _T.r<T>(iu, iu-1) = T(0);
                iu--;
                iter = 0;

            }
            else if (il == iu-1) {

                splitTwoRows(iu, shift);
                iu -= 2;
                iter = 0;
            }
            else  {

                NDArray householderVec(_T.ordering(), {3}, _T.dataType(), _T.getContext());
                NDArray shiftVec      (_T.ordering(), {3}, _T.dataType(), _T.getContext());

                calcShift(iu, iter, shift, shiftVec);

                ++iter;
                ++totalIter;

                if (totalIter > maxIters)
                    break;

                int im;
                initFrancisQR(il, iu, shiftVec, im, householderVec);
                doFrancisQR(il, im, iu, householderVec);
            }
        }
    }
}

template class ND4J_EXPORT Hessenberg<float>;
template class ND4J_EXPORT Hessenberg<float16>;
template class ND4J_EXPORT Hessenberg<bfloat16>;
template class ND4J_EXPORT Hessenberg<double>;

template class ND4J_EXPORT Schur<float>;
template class ND4J_EXPORT Schur<float16>;
template class ND4J_EXPORT Schur<bfloat16>;
template class ND4J_EXPORT Schur<double>;

}
}
}