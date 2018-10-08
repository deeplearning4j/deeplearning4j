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

#include <ops/declarable/helpers/hhColPivQR.h>
#include <ops/declarable/helpers/householder.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
HHcolPivQR<T>::HHcolPivQR(const NDArray<T>& matrix) {

    _qr = matrix;
    _diagSize = math::nd4j_min<int>(matrix.sizeAt(0), matrix.sizeAt(1));    
    _coeffs = NDArray<T>(matrix.ordering(), {1, _diagSize}, matrix.getWorkspace());   
    
    _permut = NDArray<T>(matrix.ordering(), {matrix.sizeAt(1), matrix.sizeAt(1)}, matrix.getWorkspace());   

    evalData();    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHcolPivQR<T>::evalData() {

    int rows = _qr.sizeAt(0);
    int cols = _qr.sizeAt(1);    
        
    NDArray<T> transp  (_qr.ordering(), {1, cols}, _qr.getWorkspace());
    NDArray<T> normsUpd(_qr.ordering(), {1, cols}, _qr.getWorkspace());
    NDArray<T> normsDir(_qr.ordering(), {1, cols}, _qr.getWorkspace());
          
    int transpNum = 0;

    for (int k = 0; k < cols; ++k) {
        
        T norm = _qr({0,0, k,k+1}).template reduceNumber<simdOps::Norm2<T>>();
        normsDir(k) = norm;
        normsUpd(k) = norm;
    }

    T normScaled = (normsUpd.template reduceNumber<simdOps::Max<T>>()) * DataTypeUtils::eps<T>();
    T threshold1 = normScaled * normScaled / (T)rows;     
    T threshold2 = math::nd4j_sqrt<T>(DataTypeUtils::eps<T>());

    T nonZeroPivots = _diagSize; 
    T maxPivot = 0.;

    for(int k = 0; k < _diagSize; ++k) {
    
        int biggestColIndex = (int)(normsUpd({0,0, k,-1}).template indexReduceNumber<simdOps::IndexMax<T>>());
        T biggestColNorm = normsUpd({0,0, k,-1}).template reduceNumber<simdOps::Max<T>>();
        T biggestColSqNorm = biggestColNorm * biggestColNorm;
        biggestColIndex += k;
    
        if(nonZeroPivots == (T)_diagSize && biggestColSqNorm < threshold1 * (T)(rows-k))
            nonZeroPivots = k;
        
        transp(k) = (T)biggestColIndex;

        if(k != biggestColIndex) {
        
            NDArray<T>* temp1 = _qr.subarray({{}, {k,k+1}});
            NDArray<T>* temp2 = _qr.subarray({{}, {biggestColIndex, biggestColIndex+1}});
            NDArray<T>  temp3 = *temp1;
            temp1->assign(temp2);
            temp2->assign(temp3);
            delete temp1;
            delete temp2;

            math::nd4j_swap<T>(normsUpd(k), normsUpd(biggestColIndex));
            math::nd4j_swap<T>(normsDir(k), normsDir(biggestColIndex));
                        
            ++transpNum;
        }
        
        T normX;
        NDArray<T>* qrBlock = nullptr;
        qrBlock = _qr.subarray({{k, rows}, {k,k+1}});
        Householder<T>::evalHHmatrixDataI(*qrBlock, _coeffs(k), normX);            
        delete qrBlock;        

        _qr(k,k) = normX;
        
        if(math::nd4j_abs<T>(normX) > maxPivot) 
            maxPivot = math::nd4j_abs<T>(normX);
        
        if(k < rows && (k+1) < cols) {
            qrBlock = _qr.subarray({{k, rows},{k+1, cols}});
            NDArray<T>* tail = _qr.subarray({{k+1, rows}, {k, k+1}});
            Householder<T>::mulLeft(*qrBlock, *tail, _coeffs(k));
            delete qrBlock;
            delete tail;
        }

        for (int j = k + 1; j < cols; ++j) {            
            
            if (normsUpd(j) != (T)0.) {            
                T temp = math::nd4j_abs<T>(_qr(k, j)) / normsUpd(j);
                temp = (1. + temp) * (1. - temp);
                temp = temp < (T)0. ? (T)0. : temp;
                T temp2 = temp * normsUpd(j) * normsUpd(j) / (normsDir(j)*normsDir(j));
                
                if (temp2 <= threshold2) {          
                    if(k+1 < rows && j < cols)
                        normsDir(j) = _qr({k+1,rows, j,j+1}).template reduceNumber<simdOps::Norm2<T>>();                    
                    normsUpd(j) = normsDir(j);
                } 
                else 
                    normsUpd(j) *= math::nd4j_sqrt<T>(temp);                
            }
        }
    }

    _permut.setIdentity();
    
    for(int k = 0; k < _diagSize; ++k) {

        int idx = (int)transp(k);
        NDArray<T>* temp1 = _permut.subarray({{}, {k, k+1}});
        NDArray<T>* temp2 = _permut.subarray({{}, {idx, idx+1}});
        NDArray<T>  temp3 = *temp1;
        temp1->assign(temp2);
        temp2->assign(temp3);
        delete temp1;
        delete temp2;
    }    
}




template class ND4J_EXPORT HHcolPivQR<float>;
template class ND4J_EXPORT HHcolPivQR<float16>;
template class ND4J_EXPORT HHcolPivQR<double>;







}
}
}

