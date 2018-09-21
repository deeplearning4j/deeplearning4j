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
    _coeffs = NDArrayFactory::_create(matrix.ordering(), {1, _diagSize}, matrix.dataType(), matrix.getWorkspace());
    
    _permut = NDArrayFactory::_create(matrix.ordering(), {matrix.sizeAt(1), matrix.sizeAt(1)}, matrix.dataType(), matrix.getWorkspace());

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
        
    auto transp = NDArrayFactory::_create(_qr.ordering(), {1, cols}, _qr.dataType(), _qr.getWorkspace());
    auto normsUpd = NDArrayFactory::_create(_qr.ordering(), {1, cols}, _qr.dataType(), _qr.getWorkspace());
    auto normsDir = NDArrayFactory::_create(_qr.ordering(), {1, cols}, _qr.dataType(), _qr.getWorkspace());
          
    int transpNum = 0;

    for (int k = 0; k < cols; ++k) {
        
        T norm = _qr({0,0, k,k+1}).reduceNumber(reduce::Norm2).getScalar<T>(0);
        normsDir.putScalar<T>(k, norm);
        normsUpd.putScalar<T>(k, norm);
    }

    T normScaled = (normsUpd.reduceNumber(reduce::Max)).getScalar<T>(0) * DataTypeUtils::eps<T>();
    T threshold1 = normScaled * normScaled / (T)rows;     
    T threshold2 = math::nd4j_sqrt<T,T>(DataTypeUtils::eps<T>());

    T nonZeroPivots = _diagSize; 
    T maxPivot = 0.;

    for(int k = 0; k < _diagSize; ++k) {
    
        int biggestColIndex = (int)(normsUpd({0,0, k,-1}).indexReduceNumber(indexreduce::IndexMax));
        T biggestColNorm = normsUpd({0,0, k,-1}).reduceNumber(reduce::Max).getScalar<T>(0);
        T biggestColSqNorm = biggestColNorm * biggestColNorm;
        biggestColIndex += k;
    
        if(nonZeroPivots == (T)_diagSize && biggestColSqNorm < threshold1 * (T)(rows-k))
            nonZeroPivots = k;
        
        transp.putScalar<T>(k, (T)biggestColIndex);

        if(k != biggestColIndex) {
        
            auto temp1 = _qr.subarray({{}, {k,k+1}});
            auto temp2 = _qr.subarray({{}, {biggestColIndex, biggestColIndex+1}});
            auto temp3 = *temp1;
            temp1->assign(temp2);
            temp2->assign(temp3);
            delete temp1;
            delete temp2;

            T _e0 = normsUpd.getScalar<T>(k);
            T _e1 = normsUpd.getScalar<T>(biggestColIndex);
            normsUpd.putScalar(k, _e1);
            normsUpd.putScalar(biggestColIndex, _e0);
            //math::nd4j_swap<T>(normsUpd(k), normsUpd(biggestColIndex));

            _e0 = normsDir.getScalar<T>(k);
            _e1 = normsDir.getScalar<T>(biggestColIndex);
            normsDir.putScalar(k, _e1);
            normsDir.putScalar(biggestColIndex, _e0);
            //math::nd4j_swap<T>(normsDir(k), normsDir(biggestColIndex));
                        
            ++transpNum;
        }
        
        T normX;
        NDArray* qrBlock = nullptr;
        qrBlock = _qr.subarray({{k, rows}, {k,k+1}});

        T _x = _coeffs.getScalar<T>(k);

        Householder<T>::evalHHmatrixDataI(*qrBlock, _x, normX);

        _coeffs.putScalar<T>(k, _x);

        delete qrBlock;        

        _qr.putScalar<T>(k,k, normX);
        
        if(math::nd4j_abs<T>(normX) > maxPivot) 
            maxPivot = math::nd4j_abs<T>(normX);
        
        if(k < rows && (k+1) < cols) {
            qrBlock = _qr.subarray({{k, rows},{k+1, cols}});
            auto tail = _qr.subarray({{k+1, rows}, {k, k+1}});
            Householder<T>::mulLeft(*qrBlock, *tail, _coeffs.getScalar<T>(k));
            delete qrBlock;
            delete tail;
        }

        for (int j = k + 1; j < cols; ++j) {            
            
            if (normsUpd.getScalar<T>(j) != (T)0.f) {
                T temp = math::nd4j_abs<T>(_qr.getScalar<T>(k, j)) / normsUpd.getScalar<T>(j);
                temp = (1. + temp) * (1. - temp);
                temp = temp < (T)0. ? (T)0. : temp;
                T temp2 = temp * normsUpd.getScalar<T>(j) * normsUpd.getScalar<T>(j) / (normsDir.getScalar<T>(j)*normsDir.getScalar<T>(j));
                
                if (temp2 <= threshold2) {          
                    if(k+1 < rows && j < cols)
                        normsDir.putScalar(j, _qr({k+1,rows, j,j+1}).reduceNumber(reduce::Norm2));

                    normsUpd.putScalar<T>(j, normsDir.getScalar<T>(j));
                } 
                else 
                    normsUpd.putScalar<T>(j, normsUpd.getScalar<T>(j) * math::nd4j_sqrt<T, T>(temp));
            }
        }
    }

    _permut.setIdentity();
    
    for(int k = 0; k < _diagSize; ++k) {

        int idx = transp.getScalar<int>(k);
        auto temp1 = _permut.subarray({{}, {k, k+1}});
        auto temp2 = _permut.subarray({{}, {idx, idx+1}});
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

