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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.02.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_meshgrid)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/meshgrid.h>
#include <numeric>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(meshgrid, -1, -1, false, 0, 0) {    
    
    int rank = block.width();

    if(rank == 1) {
        OUTPUT_VARIABLE(0)->assign(INPUT_VARIABLE(0));
        return Status::OK();
    }

    bool swapFirst2Dims = block.getIArguments()->size() > 0 ? (bool)INT_ARG(0) : true;

    std::vector<NDArray<T>*> inArrs(rank);
    std::vector<NDArray<T>*> outArrs(rank);

    for(int i = 0; i < rank; ++i) {        
        inArrs[i]  = INPUT_VARIABLE(i);
        outArrs[i] = OUTPUT_VARIABLE(i);
    }

    helpers::meshgrid<T>(inArrs, outArrs, swapFirst2Dims);

    return Status::OK();
}



DECLARE_SHAPE_FN(meshgrid) {

    bool swapFirst2Dims = block.getIArguments()->size() > 0 ? (bool)INT_ARG(0) : true;
    
    int rank = block.width();
    Nd4jLong* outShapeInfo = nullptr;
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);
    outShapeInfo[0] = rank;    
    for(int i = 1; i <= rank; ++i)
        outShapeInfo[i] = (int)shape::length(inputShape->at(i - 1));
    
    if(swapFirst2Dims && rank > 1)
        math::nd4j_swap<Nd4jLong>(outShapeInfo[1], outShapeInfo[2]);
    
    shape::updateStrides(outShapeInfo, shape::order(inputShape->at(0)));

    auto shapes = SHAPELIST();
    shapes->push_back(outShapeInfo);
    
    Nd4jLong* tempShapeInfo = nullptr;
    for(int i = 2; i <= rank; ++i) {
        COPY_SHAPE(outShapeInfo, tempShapeInfo);
        shapes->push_back(tempShapeInfo);
    }
    
    return shapes;
}

}
}

#endif