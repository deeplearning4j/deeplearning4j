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

#include <svd.h>

namespace nd4j {
namespace ops {
namespace helpers {

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

