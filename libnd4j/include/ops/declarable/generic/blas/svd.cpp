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
//  @author Yurii Shyrma (iuriish@yahoo.com), created on 20.01.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_svd)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/svd.h>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(svd, 1, 1, false, 0, 3) {
    auto x = INPUT_VARIABLE(0);
    
    const int rank =  x->rankOf();
    REQUIRE_TRUE(rank >= 2 , 0, "SVD OP: the rank of input array must be >=2, but got %i instead!", rank);

    const bool fullUV = (bool)INT_ARG(0);
    const bool calcUV = (bool)INT_ARG(1);
    const int switchNum = INT_ARG(2);    
    
    helpers::svd(x, {OUTPUT_VARIABLE(0), calcUV ? OUTPUT_VARIABLE(1) : nullptr, calcUV ? OUTPUT_VARIABLE(2) : nullptr}, fullUV, calcUV, switchNum);
         
    return Status::OK();;
}


DECLARE_SHAPE_FN(svd) {

    auto inShapeInfo = inputShape->at(0);
    bool fullUV = (bool)INT_ARG(0);
    bool calcUV = (bool)INT_ARG(1);
    
    const int rank = inShapeInfo[0];
    REQUIRE_TRUE(rank >= 2 , 0, "SVD OP: the rank of input array must be >=2, but got %i instead!", rank);

    const int diagSize = inShapeInfo[rank] < inShapeInfo[rank-1] ? inShapeInfo[rank] : inShapeInfo[rank-1];        
    
    Nd4jLong* sShapeInfo(nullptr);
    if(rank == 2) {
        ALLOCATE(sShapeInfo, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong); 
        sShapeInfo[0] = 1;
        sShapeInfo[1] = diagSize;
    }
    else {
        ALLOCATE(sShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank-1), Nd4jLong); 
        sShapeInfo[0] = rank - 1;
        for(int i=1; i <= rank-2; ++i)
            sShapeInfo[i] = inShapeInfo[i];
        sShapeInfo[rank-1] = diagSize;
    }
    
    shape::updateStrides(sShapeInfo, shape::order(inShapeInfo));
    
    if(calcUV){

        Nd4jLong *uShapeInfo(nullptr), *vShapeInfo(nullptr);
        COPY_SHAPE(inShapeInfo, uShapeInfo);
        COPY_SHAPE(inShapeInfo, vShapeInfo);

        if(fullUV) {
            uShapeInfo[rank]   = uShapeInfo[rank-1];
            vShapeInfo[rank-1] = vShapeInfo[rank];
        }
        else {
            uShapeInfo[rank] = diagSize;
            vShapeInfo[rank-1] = vShapeInfo[rank];
            vShapeInfo[rank] = diagSize;
        }
    
        shape::updateStrides(uShapeInfo, shape::order(inShapeInfo));
        shape::updateStrides(vShapeInfo, shape::order(inShapeInfo));
    
        return SHAPELIST(sShapeInfo, uShapeInfo, vShapeInfo);        
    }         
    return SHAPELIST(sShapeInfo);
}



}
}

#endif