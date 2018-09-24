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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 03.01.2018
//

#include <ops/declarable/helpers/svd.h>
#include <ops/declarable/helpers/jacobiSVD.h>
#include <ops/declarable/helpers/biDiagonalUp.h>
#include <array/ResultSet.h>
#include <NDArrayFactory.h>


namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
SVD<T>::SVD(const NDArray& matrix, const int switchSize, const bool calcU, const bool calcV, const bool fullUV ) {

    if(matrix.rankOf() != 2 || matrix.isScalar())
        throw std::runtime_error("ops::helpers::SVD constructor: input array must be 2D matrix !");

    const int rows = matrix.sizeAt(0);
    const int cols = matrix.sizeAt(1);
    
    if(cols > rows) {

        _transp = true;
        _diagSize = rows;
    }
    else {

        _transp = false;
        _diagSize = cols;
    }

    _switchSize = switchSize;
    _calcU = calcU;
    _calcV = calcV;
    _fullUV = fullUV;    

    if (_transp)
        math::nd4j_swap<bool>(_calcU, _calcV);

    _s = NDArrayFactory::_create<T>(matrix.ordering(), {_diagSize, 1}, matrix.getWorkspace());
    _m = NDArrayFactory::_create<T>(matrix.ordering(), {_diagSize + 1, _diagSize}, matrix.getWorkspace());
    _m.assign(0.);

    if (_calcU)
        _u = NDArrayFactory::_create<T>(matrix.ordering(), {_diagSize + 1, _diagSize + 1}, matrix.getWorkspace());
    else         
        _u = NDArrayFactory::_create<T>(matrix.ordering(), {2, _diagSize + 1}, matrix.getWorkspace());
    _u.assign(0.);

    if (_calcV) {
        _v = NDArrayFactory::_create<T>(matrix.ordering(), {_diagSize, _diagSize}, matrix.getWorkspace());
        _v.assign(0.);
    }

    evalData(matrix);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SVD<T>::SVD(const NDArray& matrix, const int switchSize, const bool calcU, const bool calcV, const bool fullUV, const char t) {

    if(matrix.rankOf() != 2 || matrix.isScalar())
        throw std::runtime_error("ops::helpers::SVD constructor: input array must be 2D matrix !");

    const int rows = matrix.sizeAt(0);
    const int cols = matrix.sizeAt(1);
    
    if(cols > rows) {

        _transp = true;
        _diagSize = rows;
    }
    else {

        _transp = false;
        _diagSize = cols;
    }

    _switchSize = switchSize;
    _calcU = calcU;
    _calcV = calcV;
    _fullUV = fullUV;    

    if (_transp)
        math::nd4j_swap<bool>(_calcU, _calcV);

    _s = NDArrayFactory::_create<T>(matrix.ordering(), {_diagSize, 1}, matrix.getWorkspace());
    _m = NDArrayFactory::_create<T>(matrix.ordering(), {_diagSize + 1, _diagSize}, matrix.getWorkspace());
    _m.assign(0.f);

    if (_calcU)
        _u = NDArrayFactory::_create<T>(matrix.ordering(), {_diagSize + 1, _diagSize + 1}, matrix.getWorkspace());
    else         
        _u = NDArrayFactory::_create<T>(matrix.ordering(), {2, _diagSize + 1}, matrix.getWorkspace());
    _u.assign(0.);

    if (_calcV) {
        _v = NDArrayFactory::_create<T>(matrix.ordering(), {_diagSize, _diagSize}, matrix.getWorkspace());
        _v.assign(0.);
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::deflation1(int col1, int shift, int ind, int size) {

    if(ind <= 0)
        throw std::runtime_error("ops::helpers::SVD::deflation1 method: input int must satisfy condition ind > 0 !");

    int first = col1 + shift;    
    T cos = _m.e<T>(first, first);
    T sin = _m.e<T>(first+ind, first);
    T denom = math::nd4j_sqrt<T, T>(cos*cos + sin*sin);

    if (denom == (T)0.) {
        
        _m.p(first+ind, first+ind, 0.f);
        return;
    }

    cos /= denom;
    sin /= denom;

    _m.p(first,first, denom);
    _m.p(first+ind, first, 0.f);
    _m.p(first+ind, first+ind, 0.f);
        
    auto rotation = NDArrayFactory::_create<T>(_m.ordering(), {2, 2},  _m.getWorkspace());
    rotation.p(0, 0, cos);
    rotation.p(0, 1, -sin);
    rotation.p(1, 0, sin);
    rotation.p(1, 1, cos);

    if (_calcU) {        
        auto temp = _u.subarray({{col1, col1 + size + 1}, {}});
        JacobiSVD<T>::mulRotationOnRight(col1, col1+ind, *temp, rotation);
        delete temp;
    }
    else
        JacobiSVD<T>::mulRotationOnRight(col1, col1+ind, _u, rotation);        
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::deflation2(int col1U , int col1M, int row1W, int col1W, int ind1, int ind2, int size) {
  
    if(ind1 >= ind2)
        throw std::runtime_error("ops::helpers::SVD::deflation2 method: input intes must satisfy condition ind1 < ind2 !");

    if(size <= 0)
        throw std::runtime_error("ops::helpers::SVD::deflation2 method: input size must satisfy condition size > 0 !");

    T cos = _m.e<T>(col1M+ind1, col1M);
    T sin = _m.e<T>(col1M+ind2, col1M);
    T denom = math::nd4j_sqrt<T,T>(cos*cos + sin*sin);
    
    if (denom == (T)0.)  {
      
      _m.p(col1M + ind1, col1M + ind1, _m.e<T>(col1M + ind2, col1M + ind2));
      return;
    }

    cos /= denom;
    sin /= denom;
    _m.p(col1M + ind1, col1M, denom);
    _m.p(col1M + ind2, col1M + ind2, _m.e<T>(col1M + ind1, col1M + ind1));
    _m.p(col1M + ind2, col1M, 0.f);
    
    auto rotation = NDArrayFactory::_create<T>(_m.ordering(), {2, 2}, _m.getWorkspace());
    rotation.p(0,0, cos);
    rotation.p(1,1, cos);

    rotation.p(0,1, -sin);
    rotation.p(1,0, sin);
    
    if (_calcU) {
        auto temp = _u.subarray({{col1U, col1U + size + 1},{}});
        JacobiSVD<T>::mulRotationOnRight(col1U+ind1, col1U+ind2, *temp, rotation);
        delete temp;
    }
    else
        JacobiSVD<T>::mulRotationOnRight(col1U+ind1, col1U+ind2, _u, rotation);    
    
    if (_calcV)  {
        auto temp = _v.subarray({{row1W, row1W + size},{}});
        JacobiSVD<T>::mulRotationOnRight(col1W+ind1, col1W+ind2, *temp, rotation);        
        delete temp;
    }
}

//////////////////////////////////////////////////////////////////////////
// has effect on block from (col1+shift, col1+shift) to (col2+shift, col2+shift) inclusively 
template <typename T>
void SVD<T>::deflation(int col1, int col2, int ind, int row1W, int col1W, int shift)
{
    
    const int len = col2 + 1 - col1;

    auto colVec0 = _m.subarray({{col1+shift, col1+shift+len },{col1+shift, col1+shift+1}});
        
    auto diagInterval = _m({col1+shift, col1+shift+len, col1+shift,col1+shift+len}, true).diagonal('c');
  
    const T almostZero = DataTypeUtils::min<T>();
    T maxElem;
    if(len == 1)
        maxElem = math::nd4j_abs<T>(diagInterval->template e<T>(0));
    else
        maxElem = (*diagInterval)({1,-1, 0,0}, true).reduceNumber(reduce::AMax).template e<T>(0);
    T maxElem0 = colVec0->reduceNumber(reduce::AMax).template e<T>(0);

    T eps = math::nd4j_max<T>(almostZero, DataTypeUtils::eps<T>() * maxElem);
    T epsBig = (T)8. * DataTypeUtils::eps<T>() * math::nd4j_max<T>(maxElem0, maxElem);        

    if(diagInterval->template e<T>(0) < epsBig)
        diagInterval->p(Nd4jLong(0), epsBig);
  
    for(int i=1; i < len; ++i)
        if(math::nd4j_abs<T>(colVec0->template e<T>(i)) < eps)
            colVec0->p(i, 0.f);

    for(int i=1; i < len; i++)
        if(diagInterval->template e<T>(i) < epsBig) {
            deflation1(col1, shift, i, len);    
            for(int i = 0; i < len; ++i)
                diagInterval->p(i, _m.e<T>(col1+shift+i,col1+shift+i));
        }
    
    {
        
        bool totDefl = true;    
        for(int i=1; i < len; i++)
            if(colVec0->template e<T>(i) >= almostZero) {
                totDefl = false;
                break;
            }
        
        int* permut = nullptr;    
        ALLOCATE(permut, _m.getWorkspace(), 3*_diagSize, int);
        {
            permut[0] = 0;
            int p = 1;          
            
            for(int i=1; i<len; ++i)
                if(math::nd4j_abs<T>(diagInterval->template e<T>(i)) < almostZero)
                    permut[p++] = i;            
            
            int k = 1, m = ind+1;
            
            for( ; p < len; ++p) {
                if(k > ind)             
                    permut[p] = m++;                
                else if(m >= len)
                    permut[p] = k++;
                else if(diagInterval->template e<T>(k) < diagInterval->template e<T>(m))
                    permut[p] = m++;
                else                        
                    permut[p] = k++;
            }
        }
    
        if(totDefl) {
            for(int i=1; i<len; ++i) {
                int ki = permut[i];
                if(math::nd4j_abs<T>(diagInterval->template e<T>(ki)) < almostZero || diagInterval->template e<T>(0) < diagInterval->template e<T>(ki))
                    permut[i-1] = permut[i];
                else {
                    permut[i-1] = 0;
                    break;
                }
            }
        }
        
        int *tInd = permut + len;
        int *tCol = permut + 2*len;
    
        for(int m = 0; m < len; m++) {
            tCol[m] = m;
            tInd[m] = m;
        }
    
        for(int i = totDefl ? 0 : 1; i < len; i++) {
        
            const int ki = permut[len - (totDefl ? i+1 : i)];
            const int jac = tCol[ki];

            T _e0 = diagInterval->template e<T>(jac);
            //math::nd4j_swap<T>(diagInterval)(i), (*diagInterval)(jac));
            diagInterval->p(jac, diagInterval->template e<T>(i));
            diagInterval->p(i, _e0);

            if(i!=0 && jac!=0) {
                _e0 = colVec0->template e<T>(jac);
                //math::nd4j_swap<T>((*colVec0)(i), (*colVec0)(jac));
                colVec0->p(jac, colVec0->template e<T>(i));
                colVec0->p(i, _e0);
            }
      
            NDArray* temp1 = nullptr, *temp2 = nullptr;
            if (_calcU) {
                temp1 = _u.subarray({{col1, col1+len+1},{col1+i,   col1+i+1}});
                temp2 = _u.subarray({{col1, col1+len+1},{col1+jac, col1+jac+1}});                     
                auto temp3 = *temp1;
                temp1->assign(temp2);
                temp2->assign(temp3);                
            }        
            else {
                temp1 = _u.subarray({{0, 2},{col1+i,   col1+i+1}});
                temp2 = _u.subarray({{0, 2},{col1+jac, col1+jac+1}});                
                auto temp3 = *temp1;
                temp1->assign(temp2);
                temp2->assign(temp3);                
            }            
            delete temp1;
            delete temp2;

            if(_calcV) {
                temp1 = _v.subarray({{row1W, row1W+len},{col1W+i,   col1W+i+1}});
                temp2 = _v.subarray({{row1W, row1W+len},{col1W+jac, col1W+jac+1}});               
                auto temp3 = *temp1;
                temp1->assign(temp2);
                temp2->assign(temp3);
                delete temp1;
                delete temp2;                
            }
      
            const int tI = tInd[i];
            tCol[tI] = jac;
            tCol[ki] = i;
            tInd[jac] = tI;
            tInd[i] = ki;
        }

        RELEASE(permut, _m.getWorkspace());
    }
    
    {
        int i = len-1;
        
        while(i > 0 && (math::nd4j_abs<T>(diagInterval->template e<T>(i)) < almostZero || math::nd4j_abs<T>(colVec0->template e<T>(i)) < almostZero))
            --i;
        
        for(; i > 1; --i) {
            if( (diagInterval->template e<T>(i) - diagInterval->template e<T>(i-1)) < DataTypeUtils::eps<T>()*maxElem ) {
                if (math::nd4j_abs<T>(diagInterval->template e<T>(i) - diagInterval->template e<T>(i-1)) >= epsBig)
                    throw std::runtime_error("ops::helpers::SVD::deflation: diagonal elements are not properly sorted !");
                deflation2(col1, col1 + shift, row1W, col1W, i-1, i, len);
            }
        }
    }  

    delete colVec0;
    delete diagInterval;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
T SVD<T>::secularEq(const T diff, const NDArray& col0, const NDArray& diag, const NDArray& permut, const NDArray& diagShifted, const T shift) {

    auto len = permut.lengthOf();
    T res = 1.;
    T item;
    for(int i=0; i<len; ++i) {
        auto j = permut.e<int>(i);
        item = col0.e<T>(j) / ((diagShifted.e<T>(j) - diff) * (diag.e<T>(j) + shift + diff));
        res += item * col0.e<T>(j);
    }
  
    return res;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcSingVals(const NDArray& col0, const NDArray& diag, const NDArray& permut, NDArray& singVals, NDArray& shifts, NDArray& mus) {
  
    auto len = col0.lengthOf();
    auto curLen = len;
    
    while(curLen > 1 && col0.e<T>(curLen-1) == (T)0.f)
        --curLen;

    for (int k = 0; k < len; ++k)  {
    
        if (col0.e<T>(k) == (T)0.f || curLen==1) {
    
            singVals.p(k, k==0 ? col0.e<T>(0) : diag.e<T>(k));
            mus.p(k, 0.f);
            shifts.p(k, k==0 ? col0.e<T>(0) : diag.e<T>(k));
            continue;
        } 
    
        T left = diag.e<T>(k);
        T right;
    
        if(k==curLen-1)
            right = diag.e<T>(curLen-1) + col0.reduceNumber(reduce::Norm2).e<T>(0);
        else {
      
            int l = k+1;
            while(col0.e<T>(l) == (T)0.f) {
                ++l; 
                if(l >= curLen)
                    throw std::runtime_error("ops::helpers::SVD::calcSingVals method: l >= curLen !");
            }
        
            right = diag.e<T>(l);
        }
    
        T mid = left + (right - left) / (T)2.;
        T fMid = secularEq(mid, col0, diag, permut, diag, 0.);
        T shift = (k == curLen-1 || fMid > (T)0.) ? left : right;

        auto diagShifted = diag - shift;
        
        T muPrev, muCur;
        if (shift == left) {
            muPrev = (right - left) * 0.1;
            if (k == curLen-1) 
                muCur = right - left;
            else               
                muCur = (right - left) * 0.5;
        }
        else {
            muPrev = -(right - left) * 0.1;
            muCur  = -(right - left) * 0.5;
        }

        T fPrev = secularEq(muPrev, col0, diag, permut, diagShifted, shift);
        T fCur = secularEq(muCur, col0, diag, permut, diagShifted, shift);
        
        if (math::nd4j_abs<T>(fPrev) < math::nd4j_abs<T>(fCur)) {      
            math::nd4j_swap<T>(fPrev, fCur);
            math::nd4j_swap<T>(muPrev, muCur);
        }
    
        bool useBisection = fPrev * fCur > (T)0.;
        while (fCur != (T).0 && 
               math::nd4j_abs<T>(muCur - muPrev) > (T)8. * DataTypeUtils::eps<T>() * math::nd4j_max<T>(math::nd4j_abs<T>(muCur), math::nd4j_abs<T>(muPrev)) 
               && math::nd4j_abs<T>(fCur - fPrev) > DataTypeUtils::eps<T>() && !useBisection) {
                        
            T a = (fCur - fPrev) / ((T)1./muCur - (T)1./muPrev);
            T jac = fCur - a / muCur;      
            T muZero = -a/jac;
            T fZero = secularEq(muZero, col0, diag, permut, diagShifted, shift);            
      
            muPrev = muCur;
            fPrev = fCur;
            muCur = muZero;
            fCur = fZero;            
            
            if (shift == left  && (muCur < (T)0. || muCur > right - left)) 
                useBisection = true;
            if (shift == right && (muCur < -(right - left) || muCur > (T)0.)) 
                useBisection = true;
            if (math::nd4j_abs<T>(fCur) > math::nd4j_abs<T>(fPrev)) 
                useBisection = true;
        }
        

        if (useBisection) {

            T leftShifted, rightShifted;
            if (shift == left) {
                leftShifted = DataTypeUtils::min<T>();
                rightShifted = (k==curLen-1) ? right : ((right - left) * (T)0.6); 
            }
            else {
           
                leftShifted = -(right - left) * (T)0.6;
                rightShifted = -DataTypeUtils::min<T>();
            }
      
            T fLeft  = secularEq(leftShifted,  col0, diag, permut, diagShifted, shift);
            T fRight = secularEq(rightShifted, col0, diag, permut, diagShifted, shift);            
            // if(fLeft * fRight >= (T)0.)
                // throw "ops::helpers::SVD::calcSingVals method: fLeft * fRight >= (T)0. !";        
      
            while (rightShifted - leftShifted > (T)2.f * DataTypeUtils::eps<T>() * math::nd4j_max<T>(math::nd4j_abs<T>(leftShifted), math::nd4j_abs<T>(rightShifted))) {
            
                T midShifted = (leftShifted + rightShifted) / (T)2.;
                fMid = secularEq(midShifted, col0, diag, permut, diagShifted, shift);
                if (fLeft * fMid < (T)0.)        
                    rightShifted = midShifted;        
                else {
                    leftShifted = midShifted;
                    fLeft = fMid;
                }
            }
            muCur = (leftShifted + rightShifted) / (T)2.;
        }        
        singVals.p(k, shift + muCur);
        shifts.p(k, shift);
        mus.p(k, muCur);
    }

}


//////////////////////////////////////////////////////////////////////////
template <typename T> 
void SVD<T>::perturb(const NDArray& col0, const NDArray& diag, const NDArray& permut, const NDArray& singVals,  const NDArray& shifts, const NDArray& mus, NDArray& zhat) {
    
    int n = col0.lengthOf();
    int m = permut.lengthOf();
    if(m==0) {
        zhat.assign(0.);
        return;
    }
    
    int last = permut.e<int>(m-1);
  
    for (int k = 0; k < n; ++k) {
        
        if (col0.e<T>(k) == (T)0.f)
            zhat.p(k, (T)0.f);
        else {            
            T dk   = diag.e<T>(k);
            T prod = (singVals.e<T>(last) + dk) * (mus.e<T>(last) + (shifts.e<T>(last) - dk));

            for(int l = 0; l<m; ++l) {
                int i = permut.e<int>(l);
                if(i!=k) {
                    int j = i<k ? i : permut.e<int>(l-1);
                    prod *= ((singVals.e<T>(j)+dk) / ((diag.e<T>(i)+dk))) * ((mus.e<T>(j)+(shifts.e<T>(j)-dk)) / ((diag.e<T>(i)-dk)));
                }
            }
        T tmp = math::nd4j_sqrt<T,T>(prod);
        zhat.p(k, col0.e<T>(k) > (T)0.f ? tmp : -tmp);
        }  
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcSingVecs(const NDArray& zhat, const NDArray& diag, const NDArray& perm, const NDArray& singVals,
                             const NDArray& shifts, const NDArray& mus, NDArray& U, NDArray& V) {
  
    int n = zhat.lengthOf();
    int m = perm.lengthOf();
  
    for (int k = 0; k < n; ++k) {
        
        auto colU = U.subarray({{},{k, k+1}});
        *colU = 0.;
        NDArray* colV = nullptr;
        
        if (_calcV) {
            colV = V.subarray({{},{k, k+1}});
            *colV = 0.;
        }

        if (zhat.e<T>(k) == (T)0.f) {
            colU->p(k, 1.f);
            
            if (_calcV)            
                colV->p(k, 1.f);
        }
        else {
      
            for(int l = 0; l < m; ++l) {
                int i = perm.e<int>(l);
                U.p(i,k, zhat.e<T>(i)/(((diag.e<T>(i) - shifts.e<T>(k)) - mus.e<T>(k)) )/( (diag.e<T>(i) + singVals.e<T>(k))));
            }
            U.p(n,k, 0.f);
            *colU /= colU->reduceNumber(reduce::Norm2);
    
            if (_calcV) {
        
                for(int l = 1; l < m; ++l){
                    int i = perm.e<T>(l);
                    V.p(i,k, diag.e<T>(i) * zhat.e<T>(i) / (((diag.e<T>(i) - shifts.e<T>(k)) - mus.e<T>(k)) )/( (diag.e<T>(i) + singVals.e<T>(k))));
                }
                V.p(0,k, -1.f);
                *colV /= colV->reduceNumber(reduce::Norm2);
            }
        }
        delete colU;  
        if (_calcV)    
            delete colV;
    }
    
    auto colU = U.subarray({{},{n, n+1}});
    *colU = 0.;
    colU->p(n, 1.);
    delete colU;    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::calcBlockSVD(int col1, int size, NDArray& U, NDArray& singVals, NDArray& V) {
    
    const T almostZero = DataTypeUtils::min<T>();
    auto col0 = _m({col1, col1+size, col1, col1+1}, true);
    auto diagP = _m({col1, col1+size, col1, col1+size}, true).diagonal('c');
    auto diag = *diagP;
    delete diagP;

    diag.p(Nd4jLong(0), T(0));
    singVals = NDArrayFactory::_create<T>(_m.ordering(), {size, 1}, _m.getWorkspace());
    U = NDArrayFactory::_create<T>(_u.ordering(), {size+1, size+1}, _u.getWorkspace());
    if (_calcV) 
        V = NDArrayFactory::_create<T>(_v.ordering(), {size, size}, _v.getWorkspace());
    
    int curSize = size;
    while(curSize > 1 && diag.template e<T>(curSize-1) == (T)0.f)
        --curSize;
    
    int m = 0; 
    std::vector<T> indices;
    for(int k = 0; k < curSize; ++k)
        if(math::nd4j_abs<T>(col0.template e<T>(k)) > almostZero)
            indices.push_back((T)k);            
  
    auto permut = NDArrayFactory::_create<T>(_m.ordering(), {1, (int)indices.size()}, indices, _m.getWorkspace());
    auto shifts = NDArrayFactory::_create<T>(_m.ordering(), {size, 1}, _m.getWorkspace());
    auto mus    = NDArrayFactory::_create<T>(_m.ordering(), {size, 1}, _m.getWorkspace());
    auto zhat   = NDArrayFactory::_create<T>(_m.ordering(), {size, 1}, _m.getWorkspace());
          
    calcSingVals(col0, diag, permut, singVals, shifts, mus);
    perturb(col0, diag, permut, singVals, shifts, mus, zhat);
    calcSingVecs(zhat, diag, permut, singVals, shifts, mus, U, V);      
        
    for(int i=0; i<curSize-1; ++i) {        
        
        if(singVals.e<T>(i) > singVals.e<T>(i+1)) {
            T _e0 = singVals.e<T>(i);
            T _e1 = singVals.e<T>(i+1);
            //math::nd4j_swap<T>(singVals(i),singVals(i+1));
            singVals.p(i, _e1);
            singVals.p(i+1, _e0);

            auto temp1 = U.subarray({{},{i,i+1}});
            auto temp2 = U.subarray({{},{i+1,i+2}});
            auto temp3 = *temp1;
            temp1->assign(temp2);
            temp2->assign(temp3);            
            delete temp1;
            delete temp2;
            
            if(_calcV) {
                auto temp1 = V.subarray({{},{i,i+1}});
                auto temp2 = V.subarray({{},{i+1,i+2}});
                auto temp3 = *temp1;
                temp1->assign(temp2);
                temp2->assign(temp3);            
                delete temp1;
                delete temp2;                    
            }
        }
    }
    
    auto temp1 = singVals.subarray({{0, curSize},{}});
    for (int e = 0; e < curSize / 2; ++e) {
        T tmp = temp1->e<T>(e);
        temp1->p(e, temp1->e<T>(curSize-1-e));
        temp1->p(curSize-1-e, tmp);
    }
    delete temp1;
    
    auto temp2 = U.subarray({{},{0, curSize}});
    for(int i = 0; i < curSize/2; ++i) {
        auto temp3 = temp2->subarray({{},{i,i+1}});
        auto temp4 = temp2->subarray({{},{curSize-1-i,curSize-i}});
        auto  temp5 = *temp3;
        temp3->assign(temp4);
        temp4->assign(temp5);        
        delete temp3;
        delete temp4;
    }
    delete temp2;
    
    if (_calcV) {
        auto temp2 = V.subarray({{},{0, curSize}});
        for(int i = 0; i < curSize/2; ++i) {
            auto temp3 = temp2->subarray({{}, {i,i+1}});
            auto temp4 = temp2->subarray({{}, {curSize-1-i,curSize-i}});
            auto  temp5 = *temp3;
            temp3->assign(temp4);
            temp4->assign(temp5);
            delete temp3;
            delete temp4;
        }
        delete temp2;
    }     
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void SVD<T>::DivideAndConquer(int col1, int col2, int row1W, int col1W, int shift) {
    
    // requires rows = cols + 1;
    const int n = col2 - col1 + 1;
    const int k = n/2;
    const T almostZero = DataTypeUtils::min<T>();
    T alphaK;
    T betaK; 
    T r0; 
    T lambda, phi, c0, s0;    
    auto l = NDArrayFactory::_create<T>(_u.ordering(), {1, k}, _u.getWorkspace());
    auto f = NDArrayFactory::_create<T>(_u.ordering(), {1, n-k-1}, _u.getWorkspace());
    
    if(n < _switchSize) { 
                            
        JacobiSVD<T> jac(_m({col1,col1+n+1, col1,col1+n}, true), _calcU, _calcV, _fullUV);
        
        if (_calcU) {
            auto temp = _u.subarray({{col1, col1+n+1},{col1, col1+n+1}});
            temp->assign(jac._u);            
            delete temp;
        }
        else {
            auto temp1 = _u.subarray({{0, 1},{col1, col1+n+1}});
            temp1->assign(jac._u({0,1, 0,0}, true));
            delete temp1;
            auto temp2 = _u.subarray({{1, 2},{col1, col1+n+1}});
            temp2->assign(jac._u({n,n+1, 0,0}, true));    
            delete temp2;
        }
    
        if (_calcV) {
            auto temp = _v.subarray({{row1W, row1W+n},{col1W, col1W+n}});
            temp->assign(jac._v);
            delete temp;
        }
            
        auto temp = _m.subarray({{col1+shift, col1+shift+n+1}, {col1+shift, col1+shift+n}});
        temp->assign(0.);
        delete temp;
        auto diag = _m.diagonal('c');
        (*diag)({col1+shift, col1+shift+n, 0,0}, true).assign(jac._s({0,n, 0,0}, true));
        delete diag;        
    
        return;
    }
      
    alphaK = _m.e<T>(col1 + k, col1 + k);
    betaK  = _m.e<T>(col1 + k + 1, col1 + k);
  
    DivideAndConquer(k + 1 + col1, col2, k + 1 + row1W, k + 1 + col1W, shift);
    DivideAndConquer(col1, k - 1 + col1, row1W, col1W + 1, shift + 1);

    if (_calcU) {
        lambda = _u.e<T>(col1 + k, col1 + k);
        phi    = _u.e<T>(col1 + k + 1, col2 + 1);
    } 
    else {
        lambda = _u.e<T>(1, col1 + k);
        phi    = _u.e<T>(0, col2 + 1);
    }
    
    r0 = math::nd4j_sqrt<T, T>((math::nd4j_abs<T>(alphaK * lambda) * math::nd4j_abs<T>(alphaK * lambda)) + math::nd4j_abs<T>(betaK * phi) * math::nd4j_abs<T>(betaK * phi));
    
    if(_calcU) {
        l.assign(_u({col1+k,  col1+k+1,  col1,col1+k}, true));
        f.assign(_u({col1+k+1,col1+k+2,  col1+k+1,col1+n}, true));
    }
    else {
        l.assign(_u({1,2, col1, col1+k}, true));
        f.assign(_u({0,1, col1+k+1, col1+n}, true));
    }

    // UofSVD.printIndexedBuffer();
    // VofSVD.printIndexedBuffer();
    // singVals.printIndexedBuffer();
    // printf("!! \n");
    
    if (_calcV) 
        _v.p(row1W+k, col1W, 1.f);
  
    if (r0 < almostZero){
        c0 = 1.;
        s0 = 0.;
    }
    else {
        c0 = alphaK * lambda / r0;
        s0 = betaK * phi / r0;
    }
  
    if (_calcU) {
    
        auto temp = _u.subarray({{col1, col1+k+1},{col1+k, col1+k+1}});
        NDArray q1(*temp);
        delete temp;

        for (int i = col1 + k - 1; i >= col1; --i) {
            auto temp = _u.subarray({{col1, col1+k+1}, {i+1, i+2}});
            temp->assign(_u({col1, col1+k+1, i, i+1}, true));
            delete temp;
        }

        auto temp1 = _u.subarray({{col1, col1+k+1}, {col1, col1+1}});
        temp1->assign(q1 * c0);
        delete temp1;
        auto temp2 = _u.subarray({{col1, col1+k+1}, {col2+1, col2+2}});
        temp2->assign(q1 * (-s0));
        delete temp2;
        auto temp3 = _u.subarray({{col1+k+1, col1+n+1}, {col1, col1+1}});
        temp3->assign(_u({col1+k+1, col1+n+1, col2+1, col2+2}, true) * s0);        
        delete temp3;
        auto temp4 =_u.subarray({{col1+k+1, col1+n+1}, {col2+1, col2+2}});
        *temp4 *= c0;
        delete temp4;
    } 
    else  {
    
        T q1 = _u.e<T>(0, col1 + k);
    
        for (int i = col1 + k - 1; i >= col1; --i) 
            _u.p(0, i+1, _u.e<T>(0, i));

        _u.p(0, col1, q1 * c0);
        _u.p(0, col2+1, -q1*s0);
        _u.p(1, col1, _u.e<T>(1, col2+1) * s0);
        _u.p(1, col2 + 1,  _u.e<T>(1, col2 + 1) * c0);
        _u({1,2,  col1+1, col1+k+1}, true) = 0.f;
        _u({0,1,  col1+k+1, col1+n}, true) = 0.f;
    }
    
    _m.p(col1 + shift, col1 + shift, r0);
    auto temp1 = _m.subarray({{col1+shift+1, col1+shift+k+1}, {col1+shift, col1+shift+1}});
    temp1->assign(l*alphaK);
    delete temp1;
    auto temp2 = _m.subarray({{col1+shift+k+1, col1+shift+n}, {col1+shift, col1+shift+1}});
    temp2->assign(f*betaK);    
    delete temp2;

    deflation(col1, col2, k, row1W, col1W, shift);
      
    NDArray UofSVD, VofSVD, singVals;
    calcBlockSVD(col1 + shift, n, UofSVD, singVals, VofSVD);    
    
    if(_calcU) {
        auto pTemp = _u.subarray({{col1, col1+n+1},{col1, col1+n+1}});
        auto temp = *pTemp;
        pTemp->assign(mmul(temp, UofSVD));
        delete pTemp;
    }
    else {
        auto pTemp = _u.subarray({{}, {col1, col1+n+1}});
        auto temp = *pTemp;
        pTemp->assign(mmul(temp, UofSVD));
        delete pTemp;
    }
  
    if (_calcV) {
        auto pTemp = _v.subarray({{row1W, row1W+n},{row1W, row1W+n}});
        auto temp = *pTemp;
        pTemp->assign(mmul(temp, VofSVD));
        delete pTemp;
    }

    auto blockM = _m.subarray({{col1+shift, col1+shift+n},{col1+shift, col1+shift+n}});
    *blockM = 0.f;
    auto diag = blockM->diagonal('c');
    diag->assign(singVals);
    delete diag;
    delete blockM;
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void SVD<T>::exchangeUV(const HHsequence& hhU, const HHsequence& hhV, const NDArray& U, const NDArray& V) {
    
    if (_calcU) {
        
        int colsU = _fullUV ? hhU.rows() : _diagSize;        
        auto temp1 = NDArrayFactory::_create<T>(_u.ordering(), {hhU.rows(), colsU}, _u.getWorkspace());
        temp1.setIdentity();
        _u = temp1;        

        auto temp2 = _u.subarray({{0, _diagSize},{0, _diagSize}});
        temp2->assign(V({0,_diagSize, 0,_diagSize}, true));
        const_cast<HHsequence&>(hhU).mulLeft(_u);
        delete temp2;
    }
    
    if (_calcV) {
        
        int colsV = _fullUV ? hhV.rows() : _diagSize;        
        auto temp1 = NDArrayFactory::_create<T>(_v.ordering(), {hhV.rows(), colsV}, _v.getWorkspace());
        temp1.setIdentity();
        _v = temp1;

        auto temp2 = _v.subarray({{0, _diagSize},{0, _diagSize}});
        temp2->assign(U({0,_diagSize, 0,_diagSize}, true));
        const_cast<HHsequence&>(hhV).mulLeft(_v);
        delete temp2;        
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void SVD<T>::evalData(const NDArray& matrix) {

    const T almostZero = DataTypeUtils::min<T>();
    
    if(matrix.sizeAt(1) < _switchSize) {
    
        JacobiSVD<T> jac(matrix, _calcU, _calcV, _fullUV);

        if(_calcU) 
            _u = jac._u;
        if(_calcV) 
            _v = jac._v;

        _s.assign(jac._s);

        return;
    }
          
    T scale = matrix.reduceNumber(reduce::AMax).e<T>(0);
    
    if(scale == (T)0.) 
        scale = 1.;
    
    NDArray copy;
    if(_transp) {        
        copy = NDArrayFactory::_create<T>(matrix.ordering(), {matrix.sizeAt(1), matrix.sizeAt(0)}, matrix.getWorkspace());
        for(int i = 0; i < copy.sizeAt(0); ++i)
            for(int j = 0; j < copy.sizeAt(1); ++j)
                copy.p<T>(i, j, matrix.e<T>(j,i) / scale);
    }
    else
        copy = matrix / scale;
  
    BiDiagonalUp biDiag(copy);

    _u = 0.;
    _v = 0.;
  
    auto temp1 = biDiag._HHbidiag.transpose();
    auto temp2 = _m.subarray({{0, _diagSize},{}});
    temp2->assign(temp1);
    delete temp1;
    delete temp2;

    auto temp3 = _m.subarray({{_m.sizeAt(0)-1, _m.sizeAt(0)},{}});
    temp3->assign(0.);
    delete temp3;

    DivideAndConquer(0, _diagSize - 1, 0, 0, 0);      
    
    for (int i = 0; i < _diagSize; ++i) {
        T a = math::nd4j_abs<T>(_m.e<T>(i, i));
        _s.p(i, a * scale);
        if (a < almostZero) {            
            auto temp = _s.subarray({{i+1, _diagSize}, {}});
            temp->assign(0.);
            delete temp;
            break;
        }
        else if (i == _diagSize-1)         
            break;
    }
    
    if(_transp) 
        exchangeUV(biDiag.makeHHsequence('v'), biDiag.makeHHsequence('u'), _v, _u);
    else
        exchangeUV(biDiag.makeHHsequence('u'), biDiag.makeHHsequence('v'), _u, _v);
}


BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT SVD,,FLOAT_TYPES);


//////////////////////////////////////////////////////////////////////////
// svd operation, this function is not method of SVD class, it is standalone function
template <typename T>
static void svd_(const NDArray* x, const std::vector<NDArray*>& outArrs, const bool fullUV, const bool calcUV, const int switchNum) {

    auto s = outArrs[0];
    auto u = outArrs[1];
    auto v = outArrs[2];

    const int rank =  x->rankOf();    
    const int sRank = rank - 1; 

    auto listX = x->allTensorsAlongDimension({rank-2, rank-1});
    auto listS = s->allTensorsAlongDimension({sRank-1});
    ResultSet* listU(nullptr), *listV(nullptr);
    
    if(calcUV) {                
        listU = u->allTensorsAlongDimension({rank-2, rank-1});
        listV = v->allTensorsAlongDimension({rank-2, rank-1});
    }

    for(int i = 0; i < listX->size(); ++i) {
        
        // NDArray<T> matrix(x->ordering(), {listX->at(i)->sizeAt(0), listX->at(i)->sizeAt(1)}, block.getWorkspace());
        // matrix.assign(listX->at(i));
        helpers::SVD<T> svdObj(*(listX->at(i)), switchNum, calcUV, calcUV, fullUV);
        listS->at(i)->assign(svdObj._s);

        if(calcUV) {
            listU->at(i)->assign(svdObj._u);
            listV->at(i)->assign(svdObj._v);
        }        
    }

    delete listX;
    delete listS;
    
    if(calcUV) {
        delete listU;
        delete listV;
    }
}

    void svd(const NDArray* x, const std::vector<NDArray*>& outArrs, const bool fullUV, const bool calcUV, const int switchNum) {
        BUILD_SINGLE_SELECTOR(x->dataType(), svd_, (x, outArrs, fullUV, calcUV, switchNum), FLOAT_TYPES);
    }


}
}
}

