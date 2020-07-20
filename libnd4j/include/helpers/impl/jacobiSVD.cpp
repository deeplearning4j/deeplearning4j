/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
// Created by Yurii Shyrma on 11.01.2018
//

#include <helpers/jacobiSVD.h>
#include <helpers/hhColPivQR.h>
#include <helpers/MmulHelper.h>

namespace sd {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
JacobiSVD<T>::JacobiSVD(const NDArray& matrix, const bool calcU, const bool calcV, const bool fullUV) {

    if(matrix.rankOf() != 2 || matrix.isScalar())
        throw std::runtime_error("ops::helpers::JacobiSVD constructor: input array must be 2D matrix !");

    _rows = static_cast<int>(matrix.sizeAt(0));
    _cols = static_cast<int>(matrix.sizeAt(1));
    _diagSize = math::nd4j_min<int>(_rows, _cols);

    _calcU = calcU;
    _calcV = calcV;
    _fullUV = fullUV;

    _s = NDArray(matrix.ordering(), {_diagSize, 1}, matrix.dataType(), matrix.getContext());

    if(_calcU) {
        if(_fullUV)
            _u = NDArray(matrix.ordering(), {_rows, _rows}, matrix.dataType(), matrix.getContext());
        else
            _u = NDArray(matrix.ordering(), {_rows, _diagSize}, matrix.dataType(), matrix.getContext());
    }
    else
        _u = NDArray(matrix.ordering(), {_rows, 1}, matrix.dataType(), matrix.getContext());

    if(_calcV) {
        if(_fullUV)
            _v = NDArray(matrix.ordering(), {_cols, _cols}, matrix.dataType(), matrix.getContext());
        else
            _v = NDArray(matrix.ordering(), {_cols, _diagSize}, matrix.dataType(), matrix.getContext());
    }
    else
        _v = NDArray(matrix.ordering(), {_cols, 1}, matrix.dataType(), matrix.getContext());

    _m = NDArray(matrix.ordering(), {_diagSize, _diagSize}, matrix.dataType(), matrix.getContext());

    evalData(matrix);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::mulRotationOnLeft(const int i, const int j, NDArray& block, const NDArray& rotation) {

    if(i < j) {

        if(j+1 > block.sizeAt(0))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnLeft: second arguments is out of array row range !");

        auto temp = block({i,j+1,j-i,  0,0,0}, true, true);
        temp.assign(mmul(rotation, temp));

        // auto pTemp = block({i,j+1,j-i,  0,0,0}, true, true);
        // auto temp = pTemp.dup();
        // pTemp.assign(mmul(rotation, temp));
    }
    else {

        if(j+1 > block.sizeAt(0) || i+1 > block.sizeAt(0))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnLeft: some or both integer arguments are out of array row range !");

        NDArray temp(block.ordering(), {2, block.sizeAt(1)}, block.dataType(), block.getContext());
        auto row1     = block({i,i+1, 0,0}, true);
        auto row2     = block({j,j+1, 0,0}, true);
        auto rowTemp1 = temp({0,1, 0,0}, true);
        auto rowTemp2 = temp({1,2, 0,0}, true);
        rowTemp1.assign(row1);
        rowTemp2.assign(row2);
        temp.assign(mmul(rotation, temp));
        row1.assign(rowTemp1);
        row2.assign(rowTemp2);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::mulRotationOnRight(const int i, const int j, NDArray& block, const NDArray& rotation) {

    if(i < j) {

        if(j+1 > block.sizeAt(1))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnRight: second argument is out of array column range !");

        auto temp = block({0,0,0,  i,j+1,j-i}, true, true);
        temp.assign(mmul(temp, rotation));

        // auto pTemp = block({0,0,0,  i,j+1,j-i}, true, true);
        // auto temp = pTemp.dup();
        // pTemp.assign(mmul(temp, rotation));
    }
    else {

        if(j+1 > block.sizeAt(1) || i+1 > block.sizeAt(1))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnRight: some or both integer arguments are out of array column range !");

        NDArray temp(block.ordering(), {block.sizeAt(0), 2}, block.dataType(), block.getContext());
        auto col1     = block({0,0, i,i+1}, true);
        auto col2     = block({0,0, j,j+1}, true);
        auto colTemp1 = temp({0,0, 0,1}, true);
        auto colTemp2 = temp({0,0, 1,2}, true);
        colTemp1.assign(col1);
        colTemp2.assign(col2);
        temp.assign(mmul(temp, rotation));
        col1.assign(colTemp1);
        col2.assign(colTemp2);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool JacobiSVD<T>::isBlock2x2NotDiag(NDArray& block, int p, int q, T& maxElem) {

    NDArray rotation(_m.ordering(), {2, 2}, _m.dataType(), _m.getContext());

    T n = math::nd4j_sqrt<T,T>(block.t<T>(p, p) * block.t<T>(p, p)  + block.t<T>(q, p)*block.t<T>(q, p));

    const T almostZero = DataTypeUtils::min<T>();
    const T precision = DataTypeUtils::eps<T>();

    if(n == (T)0.f) {
        block.r<T>(p, p) = (T)0;
        block.r<T>(q, p) = (T)0;
    } else {
        T v = block.t<T>(p, p) / n;

        rotation.r<T>(0,0) = rotation.r<T>(1,1) = v;

        v = block.t<T>(q, p) / n;
        rotation.r<T>(0,1) = v;

        rotation.r<T>(1,0) = -rotation.template t<T>(0,1);
        mulRotationOnLeft(p, q, block, rotation);

        if(_calcU)
            mulRotationOnRight(p, q, _u, rotation.transpose());
    }

    maxElem = math::nd4j_max<T>(maxElem, math::nd4j_max<T>(math::nd4j_abs<T>(block.t<T>(p, p)), math::nd4j_abs<T>(block.t<T>(q, q))));
    T threshold = math::nd4j_max<T>(almostZero, precision * maxElem);

    return math::nd4j_abs<T>(block.t<T>(p, q)) > threshold || math::nd4j_abs<T>(block.t<T>(q, p)) > threshold;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool JacobiSVD<T>::createJacobiRotation(const T& x, const T& y, const T& z, NDArray& rotation) {

    T denom = (T)(2.f)* math::nd4j_abs<T>(y);

    if(denom < DataTypeUtils::min<T>()) {

        rotation.r<T>(0,0) = rotation.r<T>(1,1) = (T)1.f;
        rotation.r<T>(0,1) = rotation.r<T>(1,0) = (T)0.f;

        return false;
    }
    else {

        T tau = (x-z)/denom;
        T w = math::nd4j_sqrt<T,T>(tau*tau + (T)1.f);
        T t;

        if(tau > (T)0.)
            t = (T)1.f / (tau + w);
        else
            t = (T)1.f / (tau - w);

        T sign = t > (T)0. ? (T)1.f : (T)-1.f;

        T cos = (T)1.f / math::nd4j_sqrt<T,T>(t*t + (T)1.f);
        T sin = -sign * (y / math::nd4j_abs<T>(y)) * math::nd4j_abs<T>(t) * cos;

        rotation.r<T>(0,1) = sin;
        rotation.r<T>(1,0) = -sin;
        rotation.r<T>(0,0) = rotation.r<T>(1,1) = cos;

        return true;
    }
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void JacobiSVD<T>::createJacobiRotationGivens(const T& p, const T& q, NDArray& rotation) {

    T cos, sin;

    if(q == (T)0) {

        cos = p < (T)0 ? (T)-1 : (T)1;
        sin = (T)0;
    }
    else if(p == (T)0) {

        cos = (T)0;
        sin = q < (T)0 ? (T)1 : (T)-1;
    }
    else if(math::nd4j_abs<T>(p) > math::nd4j_abs<T>(q)) {

        T t = q / p;
        T u = math::nd4j_sqrt<T,T>((T)1 + t*t);
        if(p < (T)0)
            u = -u;
        cos = (T)1 / u;
        sin = -t * cos;
    }
    else {
        T t = p / q;
        T u = math::nd4j_sqrt<T,T>((T)1 + t*t);
        if(q < (T)0)
            u = -u;
        sin = -(T)1 / u;
        cos = -t * sin;
    }

    rotation.r<T>(0,1) = sin;
    rotation.r<T>(1,0) = -sin;
    rotation.r<T>(0,0) = rotation.r<T>(1,1) = cos;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::svd2x2(const NDArray& block, int p, int q, NDArray& left, NDArray& right) {

    NDArray m(block.ordering(), {2, 2}, block.dataType(), block.getContext());
    m.r<T>(0,0) = block.t<T>(p,p);
    m.r<T>(0,1) = block.t<T>(p,q);
    m.r<T>(1,0) = block.t<T>(q,p);
    m.r<T>(1,1) = block.t<T>(q,q);

    NDArray rotation(block.ordering(), {2, 2}, block.dataType(), block.getContext());
    T t = m.t<T>(0,0) + m.t<T>(1,1);
    T d = m.t<T>(1,0) - m.t<T>(0,1);

    if(math::nd4j_abs<T>(d) < DataTypeUtils::min<T>()) {

        rotation.r<T>(0,0) = rotation.r<T>(1,1) = (T)1;
        rotation.r<T>(0,1) = rotation.r<T>(1,0) = (T)0;
    }
    else {

        T u = t / d;
        T tmp = math::nd4j_sqrt<T,T>((T)1.f + u*u);
        rotation.r<T>(0,0) = rotation.r<T>(1,1) = u / tmp;
        rotation.r<T>(0,1) =  (T)1.f / tmp;
        rotation.r<T>(1,0) =  -rotation.t<T>(0,1);
    }

    m.assign(mmul(rotation, m));

    createJacobiRotation(m.t<T>(0,0), m.t<T>(0,1), m.t<T>(1,1), right);

    left.assign(mmul(rotation, right.transpose()));
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::evalData(const NDArray& matrix) {

    const T precision  = (T)2.f * DataTypeUtils::eps<T>();
    const T almostZero = DataTypeUtils::min<T>();

    T scale = matrix.reduceNumber(reduce::AMax).template t<T>(0);
    if(scale== (T)0.f)
        scale = (T)1.f;

    if(_rows > _cols) {

        HHcolPivQR qr(matrix / scale);
        _m.assign(qr._qr({0,_cols, 0,_cols}));
        _m.fillAsTriangular<T>(0., 0, 0, _m, 'l');

        HHsequence hhSeg(qr._qr, qr._coeffs, 'u');

        if(_fullUV)
            hhSeg.applyTo(_u);
        else if(_calcU) {
            _u.setIdentity();
            hhSeg.mulLeft(_u);
        }

        if(_calcV)
            _v.assign(qr._permut);
    }
    else if(_rows < _cols) {

        HHcolPivQR qr(matrix.transpose() / scale);
        _m.assign(qr._qr({0,_rows, 0,_rows}));
        _m.fillAsTriangular<T>(0., 0, 0, _m, 'l');
        _m.transposei();

        HHsequence hhSeg(qr._qr, qr._coeffs, 'u');          // type = 'u' is not mistake here !

        if(_fullUV)
            hhSeg.applyTo(_v);
        else if(_calcV) {
            _v.setIdentity();
            hhSeg.mulLeft(_v);
        }

        if(_calcU)
            _u.assign(qr._permut);
    }
    else {

        _m.assign(matrix({0,_diagSize, 0,_diagSize}) / scale);

        if(_calcU)
            _u.setIdentity();

        if(_calcV)
            _v.setIdentity();
    }

    T maxDiagElem = 0.;
    for(int i = 0; i < _diagSize; ++i) {
        T current = math::nd4j_abs<T>(_m.t<T>(i,i));
        if(maxDiagElem < current )
            maxDiagElem = current;
    }

    bool stop = false;

    while(!stop) {

        stop = true;

        for(int p = 1; p < _diagSize; ++p) {

            for(int q = 0; q < p; ++q) {

                T threshold = math::nd4j_max<T>(almostZero, precision * maxDiagElem);

                if(math::nd4j_abs<T>(_m.t<T>(p,q)) > threshold || math::nd4j_abs<T>(_m.t<T>(q,p)) > threshold){

                    stop = false;

                    // if(isBlock2x2NotDiag(_m, p, q, maxDiagElem))
                    {
                        NDArray rotLeft(_m.ordering(), {2, 2}, _m.dataType(), _m.getContext());
                        NDArray rotRight(_m.ordering(), {2, 2}, _m.dataType(), _m.getContext());
                        svd2x2(_m, p, q, rotLeft, rotRight);

                        mulRotationOnLeft(p, q, _m, rotLeft);

                        if(_calcU)
                            mulRotationOnRight(p, q, _u, rotLeft.transpose());

                        mulRotationOnRight(p, q, _m, rotRight);

                        if(_calcV)
                            mulRotationOnRight(p, q, _v, rotRight);

                        maxDiagElem = math::nd4j_max<T>(maxDiagElem, math::nd4j_max<T>(math::nd4j_abs<T>(_m.t<T>(p,p)), math::nd4j_abs<T>(_m.t<T>(q,q))));
                    }
                }
            }
        }
    }

    for(int i = 0; i < _diagSize; ++i) {

        _s.r<T>(i) = math::nd4j_abs<T>(_m.t<T>(i,i));

        if(_calcU && _m.t<T>(i,i) < (T)0.) {
            auto temp = _u({0,0, i,i+1}, true);
            temp.applyTransform(transform::Neg, temp, nullptr);
        }
    }

    _s *= scale;

    for(int i = 0; i < _diagSize; i++) {

        int pos = (_s({i,-1, 0,0}).indexReduceNumber(indexreduce::IndexMax, nullptr)).template e<int>(0);
        T maxSingVal = _s({i,-1, 0,0}).reduceNumber(reduce::Max).template t<T>(0);

        if(maxSingVal == (T)0.)
            break;

        if(pos) {

            pos += i;

            math::nd4j_swap<T>(_s.r<T>(i), _s.r<T>(pos));

            if(_calcU) {
                auto temp1 = _u({0,0, pos,pos+1}, true);
                auto temp2 = _u({0,0, i,i+1}, true);
                temp1.swapUnsafe(temp2);
            }

            if(_calcV) {
                auto temp1 = _v({0,0, pos, pos+1}, true);
                auto temp2 = _v({0,0, i, i+1}, true);
                temp1.swapUnsafe(temp2);
            }
        }
    }
}


template class ND4J_EXPORT JacobiSVD<float>;
template class ND4J_EXPORT JacobiSVD<float16>;
template class ND4J_EXPORT JacobiSVD<bfloat16>;
template class ND4J_EXPORT JacobiSVD<double>;







}
}
}

