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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_range)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/range.h>

namespace nd4j {
namespace ops {

CUSTOM_OP_IMPL(range, -2, 1, false, -2, -2) {

    NDArray<T>* output = OUTPUT_VARIABLE(0);

    const int numInArrs = block.width();
    const int numTArgs  = block.getTArguments()->size();
    const int numIArgs  = block.getIArguments()->size();    

    T start(T(0)), limit, delta(T(1));
    
    if (numIArgs > 0) {
    
        if(numIArgs == 1) 
            limit = INT_ARG(0);
        else if(numIArgs == 2) {
            start = INT_ARG(0);
            limit = INT_ARG(1);
        }
        else {
            start = INT_ARG(0);
            limit = INT_ARG(1);
            delta = INT_ARG(2);
        }
    }         
    else if (numTArgs > 0) {

        if(numTArgs == 1) 
            limit = T_ARG(0);
        else if(numTArgs == 2) {
            start = T_ARG(0);
            limit = T_ARG(1);
        }
        else {
            start = T_ARG(0);
            limit = T_ARG(1);
            delta = T_ARG(2);
        }
    } 
    else if (numInArrs > 0) {

        if(numInArrs == 1) 
            limit = (*INPUT_VARIABLE(0))(0.);
        else if(numInArrs == 2) {
            start = (*INPUT_VARIABLE(0))(0.);
            limit = (*INPUT_VARIABLE(1))(0.);
        }
        else {
            start = (*INPUT_VARIABLE(0))(0.);
            limit = (*INPUT_VARIABLE(1))(0.);
            delta = (*INPUT_VARIABLE(2))(0.);
        }
    } 
    else
        REQUIRE_TRUE(false, 0, "CUSTOM RANGE OP: op should have inputs defined in any possible way: T_args, INT_args, or INPUT variables!");

    REQUIRE_TRUE(limit != start, 0, "CUSTOM RANGE OP: limit and start values should be different, but got both equal to %f !", limit);
    REQUIRE_TRUE(delta != T(0), 0, "CUSTOM RANGE OP: delta should not be equal to zero !");

        
    return Status::OK();
}

DECLARE_SHAPE_FN(range) {
    
    const int numInArrs = block.width();
    const int numTArgs  = block.getTArguments()->size();
    const int numIArgs  = block.getIArguments()->size();    

    T start(T(0)), limit, delta(T(1));
    
    if (numIArgs > 0) {
    
        if(numIArgs == 1) 
            limit = INT_ARG(0);
        else if(numIArgs == 2) {
            start = INT_ARG(0);
            limit = INT_ARG(1);
        }
        else {
            start = INT_ARG(0);
            limit = INT_ARG(1);
            delta = INT_ARG(2);
        }
    }         
    else if (numTArgs > 0) {

        if(numTArgs == 1) 
            limit = T_ARG(0);
        else if(numTArgs == 2) {
            start = T_ARG(0);
            limit = T_ARG(1);
        }
        else {
            start = T_ARG(0);
            limit = T_ARG(1);
            delta = T_ARG(2);
        }
    } 
    else if (numInArrs > 0) {

        if(numInArrs == 1) 
            limit = (*INPUT_VARIABLE(0))(0.);
        else if(numInArrs == 2) {
            start = (*INPUT_VARIABLE(0))(0.);
            limit = (*INPUT_VARIABLE(1))(0.);
        }
        else {
            start = (*INPUT_VARIABLE(0))(0.);
            limit = (*INPUT_VARIABLE(1))(0.);
            delta = (*INPUT_VARIABLE(2))(0.);
        }
    } 
    else
        REQUIRE_TRUE(false, 0, "CUSTOM RANGE OP: op should have inputs defined in any possible way: T_args, INT_args, or INPUT variables!");

    REQUIRE_TRUE(limit != start, 0, "CUSTOM RANGE OP: limit and start values should be different, but got both equal to %f !", limit);
    REQUIRE_TRUE(delta != T(0), 0, "CUSTOM RANGE OP: delta should not be equal to zero !");
    
    Nd4jLong* vecShapeInfo(nullptr);        
    ALLOCATE(vecShapeInfo, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);    
    shape::shapeVector((limit - start) / delta, vecShapeInfo);
    
    return SHAPELIST(vecShapeInfo);
}


}
}

#endif