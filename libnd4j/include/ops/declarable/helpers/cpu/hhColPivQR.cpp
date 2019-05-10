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
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
HHcolPivQR::HHcolPivQR(const NDArray& matrix) {

    _qr = matrix;
    _diagSize = math::nd4j_min<int>(matrix.sizeAt(0), matrix.sizeAt(1));    
    _coeffs = NDArrayFactory::create(matrix.ordering(), {1, _diagSize}, matrix.dataType(), matrix.getWorkspace());
    
    _permut = NDArrayFactory::create(matrix.ordering(), {matrix.sizeAt(1), matrix.sizeAt(1)}, matrix.dataType(), matrix.getWorkspace());

    evalData();    
}

    void HHcolPivQR::evalData() {
        BUILD_SINGLE_SELECTOR(_qr.dataType(), _evalData, (), FLOAT_TYPES);
    }

//////////////////////////////////////////////////////////////////////////
template <typename T>
void HHcolPivQR::_evalData() {

    int rows = _qr.sizeAt(0);
    int cols = _qr.sizeAt(1);    
        
    auto transp = NDArrayFactory::create(_qr.ordering(), {1, cols}, _qr.dataType(), _qr.getWorkspace());
    auto normsUpd = NDArrayFactory::create(_qr.ordering(), {1, cols}, _qr.dataType(), _qr.getWorkspace());
    auto normsDir = NDArrayFactory::create(_qr.ordering(), {1, cols}, _qr.dataType(), _qr.getWorkspace());
          
    int transpNum = 0;

    for (int k = 0; k < cols; ++k) {
        
        T norm = _qr({0,0, k,k+1}).reduceNumber(reduce::Norm2).e<T>(0);
        normsDir.p<T>(k, norm);
        normsUpd.p<T>(k, norm);
    }

    T normScaled = (normsUpd.reduceNumber(reduce::Max)).e<T>(0) * DataTypeUtils::eps<T>();
    T threshold1 = normScaled * normScaled / (T)rows;     
    T threshold2 = math::nd4j_sqrt<T,T>(DataTypeUtils::eps<T>());

    T nonZeroPivots = _diagSize; 
    T maxPivot = 0.;

    for(int k = 0; k < _diagSize; ++k) {

        int biggestColIndex = normsUpd({0,0, k,-1}).indexReduceNumber(indexreduce::IndexMax).e<int>(0);
        T biggestColNorm = normsUpd({0,0, k,-1}).reduceNumber(reduce::Max).e<T>(0);
        T biggestColSqNorm = biggestColNorm * biggestColNorm;
        biggestColIndex += k;
    
        if(nonZeroPivots == (T)_diagSize && biggestColSqNorm < threshold1 * (T)(rows-k))
            nonZeroPivots = k;
        
        transp.p<T>(k, (T)biggestColIndex);

        if(k != biggestColIndex) {
        
            auto temp1 = new NDArray(_qr({0,0, k,k+1}, true));
            auto temp2 = new NDArray(_qr({0,0, biggestColIndex,biggestColIndex+1}, true));
            auto temp3 = *temp1;
            temp1->assign(temp2);
            temp2->assign(temp3);
            delete temp1;
            delete temp2;

            T e0 = normsUpd.e<T>(k);
            T e1 = normsUpd.e<T>(biggestColIndex);
            normsUpd.p(k, e1);
            normsUpd.p(biggestColIndex, e0);
            //math::nd4j_swap<T>(normsUpd(k), normsUpd(biggestColIndex));

            e0 = normsDir.e<T>(k);
            e1 = normsDir.e<T>(biggestColIndex);
            normsDir.p(k, e1);
            normsDir.p(biggestColIndex, e0);
            //math::nd4j_swap<T>(normsDir(k), normsDir(biggestColIndex));
                        
            ++transpNum;
        }
        
        T normX;
        NDArray* qrBlock = new NDArray(_qr({k,rows, k,k+1}, true));
        T c;
        Householder<T>::evalHHmatrixDataI(*qrBlock, c, normX);
        _coeffs.p<T>(k, c);
        delete qrBlock;        

        _qr.p<T>(k,k, normX);
        
        T max = math::nd4j_abs<T>(normX);
        if(max > maxPivot) 
            maxPivot = max;
        
        if(k < rows && (k+1) < cols) {
            qrBlock   = new NDArray(_qr({k,  rows,  k+1,cols}, true));
            auto tail = new NDArray(_qr({k+1,rows,  k, k+1},   true));
            Householder<T>::mulLeft(*qrBlock, *tail, _coeffs.e<T>(k));
            delete qrBlock;
            delete tail;
        }

        for (int j = k + 1; j < cols; ++j) {            
            
            if (normsUpd.e<T>(j) != (T)0.f) {
                T temp = math::nd4j_abs<T>(_qr.e<T>(k, j)) / normsUpd.e<T>(j);
                temp = (1. + temp) * (1. - temp);
                temp = temp < (T)0. ? (T)0. : temp;
                T temp2 = temp * normsUpd.e<T>(j) * normsUpd.e<T>(j) / (normsDir.e<T>(j)*normsDir.e<T>(j));
                
                if (temp2 <= threshold2) {          
                    if(k+1 < rows && j < cols)
                        normsDir.p<T>(j, _qr({k+1,rows, j,j+1}).reduceNumber(reduce::Norm2).e<T>(0));

                    normsUpd.p<T>(j, normsDir.e<T>(j));
                } 
                else 
                    normsUpd.p<T>(j, normsUpd.e<T>(j) * math::nd4j_sqrt<T, T>(temp));
            }
        }
    }

    _permut.setIdentity();
    
    for(int k = 0; k < _diagSize; ++k) {

        int idx = transp.e<int>(k);
        auto temp1 = new NDArray(_permut({0,0, k, k+1},    true));
        auto temp2 = new NDArray(_permut({0,0, idx,idx+1}, true));
        auto  temp3 = *temp1;
        temp1->assign(temp2);
        temp2->assign(temp3);
        delete temp1;
        delete temp2;
    }    
}

    BUILD_SINGLE_TEMPLATE(template void HHcolPivQR::_evalData, (), FLOAT_TYPES);

}
}
}

