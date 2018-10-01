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

#include <ops/declarable/helpers/jacobiSVD.h>
#include <ops/declarable/helpers/hhColPivQR.h>
#include <NDArrayFactory.h>


namespace nd4j {
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

    _s = NDArrayFactory::create(matrix.ordering(), {_diagSize, 1}, matrix.dataType(), matrix.getWorkspace());

    if(_calcU) {
        if(_fullUV)
            _u = NDArrayFactory::create(matrix.ordering(), {_rows, _rows}, matrix.dataType(), matrix.getWorkspace());
        else
            _u = NDArrayFactory::create(matrix.ordering(), {_rows, _diagSize}, matrix.dataType(), matrix.getWorkspace());
    }
    else 
        _u = NDArrayFactory::create(matrix.ordering(), {_rows, 1}, matrix.dataType(), matrix.getWorkspace());

    if(_calcV) {
        if(_fullUV)
            _v = NDArrayFactory::create(matrix.ordering(), {_cols, _cols}, matrix.dataType(), matrix.getWorkspace());
        else
            _v = NDArrayFactory::create(matrix.ordering(), {_cols, _diagSize}, matrix.dataType(), matrix.getWorkspace());
    }
    else 
        _v = NDArrayFactory::create(matrix.ordering(), {_cols, 1}, matrix.dataType(), matrix.getWorkspace());
    
    _m = NDArrayFactory::create(matrix.ordering(), {_diagSize, _diagSize}, matrix.dataType(), matrix.getWorkspace());
    
    evalData(matrix);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::mulRotationOnLeft(const int i, const int j, NDArray& block, const NDArray& rotation) {

    if(i < j) {

        if(j+1 > block.sizeAt(0))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnLeft: second arguments is out of array row range !");
        
        IndicesList indices({NDIndex::interval(i, j+1, j-i), NDIndex::all()});
        auto pTemp = block.subarray(indices);
        auto temp = *pTemp;
        pTemp->assign(mmul(rotation, temp));
        delete pTemp;
    }
    else {

        if(j+1 > block.sizeAt(0) || i+1 > block.sizeAt(0))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnLeft: some or both integer arguments are out of array row range !");
        
        auto temp = NDArrayFactory::create(block.ordering(), {2, block.sizeAt(1)}, block.dataType(), block.getWorkspace());
        auto row1 = block.subarray({{i, i+1}, {}});
        auto row2 = block.subarray({{j, j+1}, {}});
        auto rowTemp1 = temp.subarray({{0, 1}, {}});
        auto rowTemp2 = temp.subarray({{1, 2}, {}});
        rowTemp1->assign(row1);
        rowTemp2->assign(row2);
        temp.assign(mmul(rotation, temp));
        row1->assign(rowTemp1);
        row2->assign(rowTemp2);
        
        delete row1;
        delete row2;
        delete rowTemp1;
        delete rowTemp2;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::mulRotationOnRight(const int i, const int j, NDArray& block, const NDArray& rotation) {

    if(i < j) {

        if(j+1 > block.sizeAt(1))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnRight: second argument is out of array column range !");
        
        IndicesList indices({NDIndex::all(), NDIndex::interval(i, j+1, j-i)});
        auto pTemp = block.subarray(indices);
        auto temp = *pTemp;
        pTemp->assign(mmul(temp, rotation));
        delete pTemp;
    }
    else {

        if(j+1 > block.sizeAt(1) || i+1 > block.sizeAt(1))
            throw std::runtime_error("ops::helpers::JacobiSVD mulRotationOnRight: some or both integer arguments are out of array column range !");
        
        auto temp = NDArrayFactory::create(block.ordering(), {block.sizeAt(0), 2}, block.dataType(), block.getWorkspace());
        auto col1 = block.subarray({{}, {i, i+1}});
        auto col2 = block.subarray({{}, {j, j+1}});
        auto colTemp1 = temp.subarray({{}, {0, 1}});
        auto colTemp2 = temp.subarray({{}, {1, 2}});
        colTemp1->assign(col1);
        colTemp2->assign(col2);
        temp.assign(mmul(temp, rotation));
        col1->assign(colTemp1);
        col2->assign(colTemp2);
        
        delete col1;
        delete col2;
        delete colTemp1;
        delete colTemp2;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool JacobiSVD<T>::isBlock2x2NotDiag(NDArray& block, int p, int q, T& maxElem) {
        
    auto rotation = NDArrayFactory::create(_m.ordering(), {2, 2}, _m.dataType(), _m.getWorkspace());
    T n = math::nd4j_sqrt<T,T>(block.e<T>(p,p) * block.e<T>(p,p) + block.e<T>(q,p) * block.e<T>(q,p));

    const T almostZero = DataTypeUtils::min<T>();
    const T precision = DataTypeUtils::eps<T>();

    if(n == (T)0.f) {
        block.p(p, p, 0.f);
        block.p(q, p, 0.f);
    } else {
        T v = block.e<T>(p, p) / n;

        rotation.p(0, 0, v);
        rotation.p(1,1, v);

        v = block.e<T>(q,p) / n;
        rotation.p(0, 1, v);

        rotation.p(1,0, -rotation.template e<T>(0, 1));
        mulRotationOnLeft(p, q, block, rotation);        

        if(_calcU) {
            auto temp2 = rotation.transpose();
            mulRotationOnRight(p, q, _u, *temp2);
            delete temp2;
        }
    }
    
    maxElem = math::nd4j_max<T>(maxElem, math::nd4j_max<T>(math::nd4j_abs<T>(block.e<T>(p,p)), math::nd4j_abs<T>(block.e<T>(q,q))));
    T threshold = math::nd4j_max<T>(almostZero, precision * maxElem);
    const bool condition1 = math::nd4j_abs<T>(block.e<T>(p,q)) > threshold;
    const bool condition2 = math::nd4j_abs<T>(block.e<T>(q,p)) > threshold;

    return condition1 || condition2;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
bool JacobiSVD<T>::createJacobiRotation(const T& x, const T& y, const T& z, NDArray& rotation) {
  
    T denom = 2.* math::nd4j_abs<T>(y);

    if(denom < DataTypeUtils::min<T>()) {
        
        rotation.p(0,0, 1.f);
        rotation.p(1,1, 1.f);
        rotation.p(0,1, 0.f);
        rotation.p(1,0, 0.f);
        return false;
    } 
    else {
        
        T tau = (x-z)/denom;
        T w = math::nd4j_sqrt<T,T>(tau*tau + 1.);
        T t;
  
        if(tau > (T)0.)
            t = 1. / (tau + w);
        else
            t = 1. / (tau - w);
  
        T sign = t > (T)0. ? 1. : -1.;
        T n = 1. / math::nd4j_sqrt<T,T>(t*t + 1.f);
        rotation.p(0,0, n);
        rotation.p(1,1, n);

        rotation.p(0,1,  -sign * (y / math::nd4j_abs<T>(y)) * math::nd4j_abs<T>(t) * n);
        rotation.p(1,0, -rotation.e<T>(0,1));

        return true;
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::svd2x2(const NDArray& block, int p, int q, NDArray& left, NDArray& right) {
        
    auto m = NDArrayFactory::create(block.ordering(), {2, 2}, block.dataType(), block.getWorkspace());
    m.p<T>(0,0, block.e<T>(p,p));
    m.p<T>(0,1, block.e<T>(p,q));
    m.p<T>(1,0, block.e<T>(q,p));
    m.p<T>(1,1, block.e<T>(q,q));
  
    auto rotation = NDArrayFactory::create(block.ordering(), {2, 2}, block.dataType(), block.getWorkspace());
    T t = m.e<T>(0,0) + m.e<T>(1,1);
    T d = m.e<T>(1,0) - m.e<T>(0,1);

    if(math::nd4j_abs<T>(d) < DataTypeUtils::min<T>()) {
    
        rotation.p(0,0, 1.f);
        rotation.p(1,1, 1.f);
        rotation.p(0,1, 0.f);
        rotation.p(1,0, 0.f);
    }
    else {    
    
        T u = t / d;
        T tmp = math::nd4j_sqrt<T,T>(1. + u*u);
        rotation.p(0,0, u / tmp);
        rotation.p(1,1, u / tmp);
        rotation.p(0,1, 1.f / tmp);
        rotation.p(1,0, -rotation.e<T>(0,1));
    }
              
    m.assign(mmul(rotation, m));

    auto _x = m.e<T>(0,0);
    auto _y = m.e<T>(0,1);
    auto _z = m.e<T>(1,1);

    createJacobiRotation(_x, _y, _z, right);

    m.p<T>(0, 0, _x);
    m.p<T>(0, 1, _y);
    m.p<T>(1, 1, _z);

    auto temp = right.transpose();
    left.assign(mmul(rotation, *temp));
    delete temp;
    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void JacobiSVD<T>::evalData(const NDArray& matrix) {

    const T precision  = (T)2.f * DataTypeUtils::eps<T>();
    const T almostZero = DataTypeUtils::min<T>();

    T scale = matrix.reduceNumber(reduce::AMax).e<T>(0);
    if(scale== (T)0.f)
        scale = (T)1.f;

    if(_rows > _cols) {

        HHcolPivQR qr(matrix / scale);
        _m.assign(qr._qr({0,_cols, 0,_cols}));
        _m.setValueInDiagMatrix(0., -1, 'l');
            
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

        auto matrixT = matrix.transpose();
        HHcolPivQR qr(*matrixT / scale);
        _m.assign(qr._qr({0,_rows, 0,_rows}));
        _m.setValueInDiagMatrix(0., -1, 'l');
        _m.transposei();
    
        HHsequence  hhSeg(qr._qr, qr._coeffs, 'u');          // type = 'u' is not mistake here !

        if(_fullUV)
            hhSeg.applyTo(_v);             
        else if(_calcV) {            
            _v.setIdentity();
            hhSeg.mulLeft(_v);        
        }
                        
        if(_calcU)
            _u.assign(qr._permut);
        
        delete matrixT;      
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
        T current = math::nd4j_abs<T>(_m.e<T>(i,i));
        if(maxDiagElem < current )
            maxDiagElem = current;
    }    

    bool stop = false;

    while(!stop) {        

        stop = true;            

        for(int p = 1; p < _diagSize; ++p) {
            
            for(int q = 0; q < p; ++q) {
        
                T threshold = math::nd4j_max<T>(almostZero, precision * maxDiagElem);                
                
                if(math::nd4j_abs<T>(_m.e<T>(p,q)) > threshold || math::nd4j_abs<T>(_m.e<T>(q,p)) > threshold){
                    
                    stop = false;
                    
                    // if(isBlock2x2NotDiag(_m, p, q, maxDiagElem)) 
                    {                                                                       
                        auto rotLeft = NDArrayFactory::create(_m.ordering(), {2, 2}, _m.dataType(), _m.getWorkspace());
                        auto rotRight = NDArrayFactory::create(_m.ordering(), {2, 2}, _m.dataType(), _m.getWorkspace());
                        svd2x2(_m, p, q, rotLeft, rotRight);

                        mulRotationOnLeft(p, q, _m, rotLeft);
                                                    
                        if(_calcU) {                            
                            auto temp = rotLeft.transpose();
                            mulRotationOnRight(p, q, _u, *temp);
                            delete temp;
                        }                        
                        
                        mulRotationOnRight(p, q, _m, rotRight);                        

                        if(_calcV)
                            mulRotationOnRight(p, q, _v, rotRight);
            
                        maxDiagElem = math::nd4j_max<T>(maxDiagElem, math::nd4j_max<T>(math::nd4j_abs<T>(_m.e<T>(p,p)), math::nd4j_abs<T>(_m.e<T>(q,q))));
                    }
                }
            }
        }
    }
    
    for(int i = 0; i < _diagSize; ++i) {                
        _s.p(i, math::nd4j_abs<T>(_m.e<T>(i,i)));
        if(_calcU && _m.e<T>(i,i) < (T)0.) {
            auto temp = _u.subarray({{},{i, i+1}});
            temp->applyTransform(transform::Neg, temp, nullptr);
            delete temp;
        }
    }
  
    _s *= scale;

    for(int i = 0; i < _diagSize; i++) {
                
        int pos = (_s({i,-1, 0,0}).indexReduceNumber(indexreduce::IndexMax, nullptr)).template e<int>(0);
        T maxSingVal =  _s({i,-1, 0,0}).reduceNumber(reduce::Max).template e<T>(0);

        if(maxSingVal == (T)0.)   
            break;

        if(pos) {
            
            pos += i;

            T _e0 = _s.e<T>(i);
            T _e1 = _s.e<T>(pos);
            _s.p(pos, _e0);
            _s.p(i, _e1);
            //math::nd4j_swap<T>(_s(i), _s(pos));
            
            if(_calcU) {
                auto temp1 = _u.subarray({{}, {pos, pos+1}});
                auto temp2 = _u.subarray({{}, {i, i+1}});
                auto  temp3 = *temp1;
                temp1->assign(temp2);
                temp2->assign(temp3);
                delete temp1;
                delete temp2;                
            }
            
            if(_calcV) { 
                auto temp1 = _v.subarray({{}, {pos, pos+1}});
                auto temp2 = _v.subarray({{}, {i, i+1}});
                auto temp3 = *temp1;
                temp1->assign(temp2);
                temp2->assign(temp3);
                delete temp1;
                delete temp2;                                
            }
        }
    }  
}




template class ND4J_EXPORT JacobiSVD<float>;
template class ND4J_EXPORT JacobiSVD<float16>;
template class ND4J_EXPORT JacobiSVD<double>;







}
}
}

