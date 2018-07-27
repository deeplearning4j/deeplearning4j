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
// @author raver119, created on 29/10/17.
// @author Yurii Shyrma (iuriish@yahoo.com), changed on 03.05.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_upsampling2d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(upsampling2d, 1, 1, false, 0, 2) {

    NDArray<T>* input  = INPUT_VARIABLE(0);             // [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)
    NDArray<T>* output = OUTPUT_VARIABLE(0);            // [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)

    const int factorH = INT_ARG(0);
    const int factorW = INT_ARG(1);
    const int isNCHW  = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;       // 1-NCHW,  0-NHWC

    REQUIRE_TRUE(input->rankOf() == 4, 0, "UPSAMPLING2D op: input should be 4D, but got %i instead!", input->rankOf());
    REQUIRE_TRUE(output->rankOf() == 4, 0, "UPSAMPLING2D op: output should be 4D, but got %i instead!", output->rankOf());

    ConvolutionUtils<T>::upsampling2d(*input, *output, factorH, factorW, (bool)isNCHW);

    return Status::OK();
}
DECLARE_SYN(upsampling, upsampling2d);



DECLARE_SHAPE_FN(upsampling2d) {

    auto inputShapeInfo = inputShape->at(0);

    REQUIRE_TRUE(inputShapeInfo[0] == 4, 0, "UPSAMPLING2D op: input should be 4D, but got %i instead!", inputShapeInfo[0]);

    const int factorH = INT_ARG(0);
    const int factorW = INT_ARG(1);
    const int isNCHW  = block.getIArguments()->size() > 2 ? INT_ARG(2) : 0;       // 1-NCHW,  0-NHWC

    Nd4jLong *outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo[0]), Nd4jLong);

    outputShapeInfo[0] = inputShapeInfo[0];
    outputShapeInfo[1] = inputShapeInfo[1];

    if(isNCHW) {
        outputShapeInfo[2] = inputShapeInfo[2];
        outputShapeInfo[3] = inputShapeInfo[3] * factorH;
        outputShapeInfo[4] = inputShapeInfo[4] * factorW;
    }
    else {
        outputShapeInfo[2] = inputShapeInfo[2] * factorH;
        outputShapeInfo[3] = inputShapeInfo[3] * factorW;
        outputShapeInfo[4] = inputShapeInfo[4];
    }

    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}


//////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(upsampling2d_bp, 2, 1, false, 0, 0) {

    // NDArray<T>* input = INPUT_VARIABLE(0);             // [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)
    NDArray<T>* gradO = INPUT_VARIABLE(1);             // [bS, iC, factorH*iH, factorW*iW ] (NCHW) or [bS, factorH*iH, factorW*iW, iC] (NHWC)
    NDArray<T>* gradI = OUTPUT_VARIABLE(0);            // [bS, iC, iH, iW] (NCHW) or [bS, iH, iW, iC] (NHWC)

    const int isNCHW  = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;       // 1-NCHW,  0-NHWC

    // REQUIRE_TRUE(input->rankOf() == 4, 0, "UPSAMPLING2D_BP op: input array must be 4D, but got %i instead!", input->rankOf());
    REQUIRE_TRUE(gradO->rankOf() == 4, 0, "UPSAMPLING2D_BP op: output's gradient array must be 4D, but got %i instead!", gradO->rankOf());
    REQUIRE_TRUE(gradI->rankOf() == 4, 0, "UPSAMPLING2D_BP op: input's gradient array must be 4D, but got %i instead!", gradI->rankOf());

    ConvolutionUtils<T>::upsampling2dBP(*gradO, *gradI, (bool)isNCHW);

    return Status::OK();
}
DECLARE_SYN(upsampling_bp, upsampling2d_bp);


DECLARE_SHAPE_FN(upsampling2d_bp) {

    REQUIRE_TRUE(inputShape->at(0)[0] == 4, 0, "UPSAMPLING2D_BP op: input array must be 4D, but got %i instead!", inputShape->at(0)[0]);
    REQUIRE_TRUE(inputShape->at(1)[0] == 4, 0, "UPSAMPLING2D_BP op: output's gradient array must be 4D, but got %i instead!", inputShape->at(1)[0]);

    Nd4jLong* gradIShapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIShapeInfo);

    return SHAPELIST(gradIShapeInfo);
}

}
}

#endif