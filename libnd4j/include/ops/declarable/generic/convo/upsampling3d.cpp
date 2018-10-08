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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 04.05.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_upsampling3d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(upsampling3d, 1, 1, false, 0, 3) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);             // [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC) 
    NDArray<T>* output = OUTPUT_VARIABLE(0);            // [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)
            
    const int factorD = INT_ARG(0);
    const int factorH = INT_ARG(1);
    const int factorW = INT_ARG(2);
    const int isNCDHW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 0;       // INT_ARG(3): 0-NCDHW,  1-NDHWC

    REQUIRE_TRUE(input->rankOf() == 5, 0, "UPSAMPLING3D op: input should be 5D, but got %i instead!", input->rankOf());
    REQUIRE_TRUE(output->rankOf() == 5, 0, "UPSAMPLING3D op: output should be 5D, but got %i instead!", output->rankOf());

    ConvolutionUtils<T>::upsampling3d(*input, *output, factorD, factorH, factorW, (bool)isNCDHW);

    return Status::OK();
}

        
DECLARE_SHAPE_FN(upsampling3d) {
    
    auto inputShapeInfo = inputShape->at(0);
    
    REQUIRE_TRUE(inputShapeInfo[0] == 5, 0, "UPSAMPLING2D op: input should be 5D, but got %i instead!", inputShapeInfo[0]);

    const int factorD = INT_ARG(0);
    const int factorH = INT_ARG(1);
    const int factorW = INT_ARG(2);
    const int isNCDHW = block.getIArguments()->size() > 3 ? INT_ARG(3) : 0;       // INT_ARG(3): 0-NCHW,  1-NHWC

    Nd4jLong *outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo[0]), Nd4jLong);

    outputShapeInfo[0] = inputShapeInfo[0];
    outputShapeInfo[1] = inputShapeInfo[1];
    
    if(isNCDHW) {
        outputShapeInfo[2] = inputShapeInfo[2];
        outputShapeInfo[3] = inputShapeInfo[3] * factorD;
        outputShapeInfo[4] = inputShapeInfo[4] * factorH;
        outputShapeInfo[5] = inputShapeInfo[5] * factorW;
    }
    else {        
        outputShapeInfo[2] = inputShapeInfo[2] * factorD;
        outputShapeInfo[3] = inputShapeInfo[3] * factorH;
        outputShapeInfo[4] = inputShapeInfo[4] * factorW;
        outputShapeInfo[5] = inputShapeInfo[5];
    }

    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}

//////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(upsampling3d_bp, 2, 1, false, 0, 0) {
    
    // NDArray<T>* input = INPUT_VARIABLE(0);             // [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC) 
    NDArray<T>* gradO = INPUT_VARIABLE(1);             // [bS, iC, factorD*iD, factorH*iH, factorW*iW ] (NCDHW) or [bS, factorD*iD, factorH*iH, factorW*iW, iC] (NDHWC)
    NDArray<T>* gradI = OUTPUT_VARIABLE(0);            // [bS, iC, iD, iH, iW] (NCDHW) or [bS, iD, iH, iW, iC] (NDHWC) 
                
    const int isNCDHW  = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;       // INT_ARG(0): 0-NCHW,  1-NHWC

    // REQUIRE_TRUE(input->rankOf() == 5, 0, "UPSAMPLING3D_BP op: input array must be 4D, but got %i instead!", input->rankOf());
    REQUIRE_TRUE(gradO->rankOf() == 5, 0, "UPSAMPLING3D_BP op: output's gradient array must be 4D, but got %i instead!", gradO->rankOf());
    REQUIRE_TRUE(gradI->rankOf() == 5, 0, "UPSAMPLING3D_BP op: input's gradient array must be 4D, but got %i instead!", gradI->rankOf());

    ConvolutionUtils<T>::upsampling3dBP(*gradO, *gradI, (bool)isNCDHW);

    return Status::OK();
}

        
DECLARE_SHAPE_FN(upsampling3d_bp) {
    
    REQUIRE_TRUE(inputShape->at(0)[0] == 5, 0, "UPSAMPLING3D_BP op: input array must be 4D, but got %i instead!", inputShape->at(0)[0]);
    REQUIRE_TRUE(inputShape->at(1)[0] == 5, 0, "UPSAMPLING3D_BP op: output's gradient array must be 4D, but got %i instead!", inputShape->at(1)[0]);
    
    Nd4jLong *gradIShapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIShapeInfo);
    
    return SHAPELIST(gradIShapeInfo);
}

}
}

#endif