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
#include <helpers/EigenValsAndVecs.h>


namespace sd      {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
EigenValsAndVecs<T>::EigenValsAndVecs(const NDArray& matrix) {

    if(matrix.rankOf() != 2)
        throw std::runtime_error("ops::helpers::EigenValsAndVecs constructor: input matrix must be 2D !");

    if(matrix.sizeAt(0) != matrix.sizeAt(1))
        throw std::runtime_error("ops::helpers::EigenValsAndVecs constructor: input array must be 2D square matrix !");

    Schur<T> schur(matrix);

    NDArray& schurMatrixU = schur._U;
    NDArray& schurMatrixT = schur._T;

    _Vecs = NDArray(matrix.ordering(), {schurMatrixU.sizeAt(1), schurMatrixU.sizeAt(1), 2}, matrix.dataType(), matrix.getContext());
    _Vals = NDArray(matrix.ordering(), {matrix.sizeAt(1), 2}, matrix.dataType(), matrix.getContext());

    // sequence of methods calls matters
    calcEigenVals(schurMatrixT);
    calcPseudoEigenVecs(schurMatrixT, schurMatrixU);    // pseudo-eigenvectors are real and will be stored in schurMatrixU
    calcEigenVecs(schurMatrixU);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void EigenValsAndVecs<T>::calcEigenVals(const NDArray& schurMatrixT) {

    const int numOfCols = schurMatrixT.sizeAt(1);

    // calculate eigenvalues _Vals
    int i = 0;
    while (i < numOfCols) {

        if (i == numOfCols - 1 || schurMatrixT.t<T>(i+1, i) == T(0.f)) {

            _Vals.r<T>(i, 0) = schurMatrixT.t<T>(i, i); // real part
            _Vals.r<T>(i, 1) = T(0);                    // imaginary part

            if(!math::nd4j_isfin<T>(_Vals.t<T>(i, 0))) {
                throw std::runtime_error("ops::helpers::igenValsAndVec::calcEigenVals: got infinite eigen value !");
                return;
            }

            ++i;
        }
        else {

            T p = T(0.5) * (schurMatrixT.t<T>(i, i) - schurMatrixT.t<T>(i+1, i+1));
            T z;
            {
                T t0 = schurMatrixT.t<T>(i+1, i);
                T t1 = schurMatrixT.t<T>(i, i+1);
                T maxval = math::nd4j_max<T>(math::nd4j_abs<T>(p), math::nd4j_max<T>(math::nd4j_abs<T>(t0), math::nd4j_abs<T>(t1)));
                t0 /= maxval;
                t1 /= maxval;
                T p0 = p / maxval;
                z = maxval * math::nd4j_sqrt<T,T>(math::nd4j_abs<T>(p0 * p0 + t0 * t1));
            }

            _Vals.r<T>(i, 0)  = _Vals.r<T>(i+1, 0) = schurMatrixT.t<T>(i+1, i+1) + p;
            _Vals.r<T>(i, 1)  = z;
            _Vals.r<T>(i+1,1) = -z;

            if(!(math::nd4j_isfin<T>(_Vals.t<T>(i,0)) && math::nd4j_isfin<T>(_Vals.t<T>(i+1,0)) && math::nd4j_isfin<T>(_Vals.t<T>(i,1))) && math::nd4j_isfin<T>(_Vals.t<T>(i+1,1))) {
                throw std::runtime_error("ops::helpers::igenValsAndVec::calcEigenVals: got infinite eigen value !");
                return;
            }

            i += 2;
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void EigenValsAndVecs<T>::calcPseudoEigenVecs(NDArray& schurMatrixT, NDArray& schurMatrixU) {

    const int numOfCols = schurMatrixU.sizeAt(1);

    T norm = 0;
    for (int j = 0; j < numOfCols; ++j)
        norm += schurMatrixT({j,j+1, math::nd4j_max<Nd4jLong>(j-1, 0),numOfCols}).reduceNumber(reduce::ASum).template t<T>(0);

    if (norm == T(0))
        return;

    for (int n = numOfCols-1; n >= 0; n--) {

        T p = _Vals.t<T>(n, 0);     // real part
        T q = _Vals.t<T>(n, 1);     // imaginary part

        if(q == (T)0) {    // not complex

            T lastr((T)0), lastw((T)0);
            int l = n;

            schurMatrixT.r<T>(n, n) = T(1);

            for (int i = n-1; i >= 0; i--) {

                T w = schurMatrixT.t<T>(i,i) - p;
                T r = mmul(schurMatrixT({i,i+1, l,n+1}, true), schurMatrixT({l,n+1, n,n+1}, true)).template t<T>(0); // dot

                if (_Vals.t<T>(i, 1) < T(0)) {
                    lastw = w;
                    lastr = r;
                }
                else {

                    l = i;
                    if (_Vals.t<T>(i, 1) == T(0)) {

                        if (w != T(0))
                            schurMatrixT.r<T>(i, n) = -r / w;
                        else
                            schurMatrixT.r<T>(i, n) = -r / (DataTypeUtils::eps<T>() * norm);
                    }
                    else {

                        T x = schurMatrixT.t<T>(i, i+1);
                        T y = schurMatrixT.t<T>(i+1, i);
                        T denom = (_Vals.t<T>(i, 0) - p) * (_Vals.t<T>(i, 0) - p) + _Vals.t<T>(i, 1) * _Vals.t<T>(i, 1);
                        T t = (x * lastr - lastw * r) / denom;
                        schurMatrixT.r<T>(i, n) = t;

                        if (math::nd4j_abs<T>(x) > math::nd4j_abs<T>(lastw))
                          schurMatrixT.r<T>(i+1, n) = (-r - w * t) / x;
                        else
                          schurMatrixT.r<T>(i+1, n) = (-lastr - y * t) / lastw;
                      }


                    T t = math::nd4j_abs<T>(schurMatrixT.t<T>(i, n));
                    if((DataTypeUtils::eps<T>() * t) * t > T(1))
                        schurMatrixT({schurMatrixT.sizeAt(0)-numOfCols+i,-1, n,n+1}) /= t;
                }
            }
        }
        else if(q < T(0) && n > 0) {           // complex

            T lastra(0), lastsa(0), lastw(0);
            int l = n - 1;

            if(math::nd4j_abs<T>(schurMatrixT.t<T>(n, n-1)) > math::nd4j_abs<T>(schurMatrixT.t<T>(n-1, n))) {

                schurMatrixT.r<T>(n-1, n-1) = q / schurMatrixT.t<T>(n, n-1);
                schurMatrixT.r<T>(n-1, n)   = -(schurMatrixT.t<T>(n, n) - p) / schurMatrixT.t<T>(n, n-1);
            }
            else {
                divideComplexNums(T(0),-schurMatrixT.t<T>(n-1,n),  schurMatrixT.t<T>(n-1,n-1)-p,q,  schurMatrixT.r<T>(n-1,n-1),schurMatrixT.r<T>(n-1,n));
            }

            schurMatrixT.r<T>(n,n-1) = T(0);
            schurMatrixT.r<T>(n,n)   = T(1);

            for (int i = n-2; i >= 0; i--) {

                T ra = mmul(schurMatrixT({i,i+1, l,n+1}, true), schurMatrixT({l,n+1, n-1,n}, true)).template t<T>(0);            // dot
                T sa = mmul(schurMatrixT({i,i+1, l,n+1}, true), schurMatrixT({l,n+1, n,n+1}, true)).template t<T>(0);            // dot

                T w = schurMatrixT.t<T>(i,i) - p;

                if (_Vals.t<T>(i, 1) < T(0)) {
                    lastw = w;
                    lastra = ra;
                    lastsa = sa;
                }
                else {

                    l = i;

                    if (_Vals.t<T>(i, 1) == T(0)) {
                        divideComplexNums(-ra,-sa, w,q, schurMatrixT.r<T>(i,n-1),schurMatrixT.r<T>(i,n));
                    }
                    else {

                        T x = schurMatrixT.t<T>(i,i+1);
                        T y = schurMatrixT.t<T>(i+1,i);
                        T vr = (_Vals.t<T>(i, 0) - p) * (_Vals.t<T>(i, 0) - p) + _Vals.t<T>(i, 1) * _Vals.t<T>(i, 1) - q * q;
                        T vi = (_Vals.t<T>(i, 0) - p) * T(2) * q;

                        if ((vr == T(0)) && (vi == T(0)))
                            vr = DataTypeUtils::eps<T>() * norm * (math::nd4j_abs<T>(w) + math::nd4j_abs<T>(q) + math::nd4j_abs<T>(x) + math::nd4j_abs<T>(y) + math::nd4j_abs<T>(lastw));

                        divideComplexNums(x*lastra-lastw*ra+q*sa,x*lastsa-lastw*sa-q*ra, vr,vi, schurMatrixT.r<T>(i,n-1),schurMatrixT.r<T>(i,n));

                        if(math::nd4j_abs<T>(x) > (math::nd4j_abs<T>(lastw) + math::nd4j_abs<T>(q))) {

                            schurMatrixT.r<T>(i+1,n-1) = (-ra - w * schurMatrixT.t<T>(i,n-1) + q * schurMatrixT.t<T>(i,n))   / x;
                            schurMatrixT.r<T>(i+1,n)   = (-sa - w * schurMatrixT.t<T>(i,n)   - q * schurMatrixT.t<T>(i,n-1)) / x;
                        }
                        else
                            divideComplexNums(-lastra-y*schurMatrixT.t<T>(i,n-1),-lastsa-y*schurMatrixT.t<T>(i,n), lastw,q, schurMatrixT.r<T>(i+1,n-1),schurMatrixT.r<T>(i+1,n));
                    }

                    T t = math::nd4j_max<T>(math::nd4j_abs<T>(schurMatrixT.t<T>(i, n-1)), math::nd4j_abs<T>(schurMatrixT.t<T>(i,n)));
                    if ((DataTypeUtils::eps<T>() * t) * t > T(1))
                        schurMatrixT({i,numOfCols, n-1,n+1}) /= t;
                }
            }
            n--;
        }
        else
            throw std::runtime_error("ops::helpers::EigenValsAndVecs::calcEigenVecs: internal bug !");
    }

    for (int j = numOfCols-1; j >= 0; j--)
        schurMatrixU({0,0, j,j+1}, true).assign( mmul(schurMatrixU({0,0, 0,j+1}, true), schurMatrixT({0,j+1, j,j+1}, true)) );
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void EigenValsAndVecs<T>::calcEigenVecs(const NDArray& schurMatrixU) {

    const T precision = T(2) * DataTypeUtils::eps<T>();

    const int numOfCols = schurMatrixU.sizeAt(1);

    for (int j = 0; j < numOfCols; ++j) {

        if(math::nd4j_abs<T>(_Vals.t<T>(j, 1)) <= math::nd4j_abs<T>(_Vals.t<T>(j, 0)) * precision || j+1 == numOfCols) {    // real

            _Vecs.syncToDevice();
            _Vecs({0,0, j,j+1, 0,1}).assign(schurMatrixU({0,0, j,j+1}));
            _Vecs({0,0, j,j+1, 1,2}) = (T)0;

            // normalize
            const T norm2 = _Vecs({0,0, j,j+1, 0,1}).reduceNumber(reduce::SquaredNorm).template t<T>(0);
            if(norm2 > (T)0)
                _Vecs({0,0, j,j+1, 0,1}) /= math::nd4j_sqrt<T,T>(norm2);
        }
        else { // complex

            for (int i = 0; i < numOfCols; ++i) {
                _Vecs.r<T>(i, j, 0)   = _Vecs.r<T>(i, j+1, 0) = schurMatrixU.t<T>(i, j);
                _Vecs.r<T>(i, j, 1)   = schurMatrixU.t<T>(i, j+1);
                _Vecs.r<T>(i, j+1, 1) = -schurMatrixU.t<T>(i, j+1);
            }

            // normalize
            T norm2 = _Vecs({0,0, j,j+1, 0,0}).reduceNumber(reduce::SquaredNorm).template t<T>(0);
            if(norm2 > (T)0)
                _Vecs({0,0, j,j+1, 0,0}) /= math::nd4j_sqrt<T,T>(norm2);

            // normalize
            norm2 = _Vecs({0,0, j+1,j+2, 0,0}).reduceNumber(reduce::SquaredNorm).template t<T>(0);
            if(norm2 > (T)0)
                _Vecs({0,0, j+1,j+2, 0,0}) /= math::nd4j_sqrt<T,T>(norm2);

            ++j;
        }
    }
}


template class ND4J_EXPORT EigenValsAndVecs<float>;
template class ND4J_EXPORT EigenValsAndVecs<float16>;
template class ND4J_EXPORT EigenValsAndVecs<bfloat16>;
template class ND4J_EXPORT EigenValsAndVecs<double>;

}
}
}