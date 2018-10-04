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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 24.07.2018
//


#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_prelu)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/activations.h>
#include <numeric>

namespace nd4j {
namespace ops  {


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(prelu, 2, 1, true, 0, 0) {
    auto input  = INPUT_VARIABLE(0);
    auto alpha  = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    std::vector<int> sharedAxes = *block.getIArguments();
    
    const int inputRank     = input->rankOf();
    const int alphaRank     = alpha->rankOf();
    const int numSharedAxes = sharedAxes.size();            // can be zero as well
    const Nd4jLong inputLen = input->lengthOf();
    const Nd4jLong alphaLen = alpha->lengthOf();
    const std::vector<Nd4jLong> inputShape = input->getShapeAsVector();
    const std::vector<Nd4jLong> alphaShape = alpha->getShapeAsVector();

    //***** input validation *****//
    std::vector<Nd4jLong> expectedAlphaShape(&inputShape[1], &inputShape[inputRank]);

    REQUIRE_TRUE(inputRank > 1, 0, "PRELU OP: wrong rank of input array, expected rank should be > 1, but got %i instead !", inputRank);   
    
    for(int i = 0; i < numSharedAxes; ++i) {
        if(sharedAxes[i] <= 0)
            sharedAxes[i] += inputRank - 1;
        REQUIRE_TRUE(1 <= sharedAxes[i] && sharedAxes[i] <= inputRank - 1, 0, "PRELU OP: wrong axis value %i in sharedAxes at position %i, axis value must be within range [1, input_rank-1] !", sharedAxes[i], i);            
        expectedAlphaShape[sharedAxes[i] - 1] = 1;
    }

    //const Nd4jLong expectedAlphaLen =  0; //std::accumulate(expectedAlphaShape.begin(), expectedAlphaShape.end(), 1, std::multiplies<T>());

    REQUIRE_TRUE(helpers::checkAlphaShapeLen(expectedAlphaShape, alphaLen), 0, "PRELU OP: wrong shape of alpha array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedAlphaShape).c_str(), ShapeUtils::shapeAsString(alphaShape).c_str());
    // ***** end of validation ***** //

    if(alphaShape != expectedAlphaShape)
        alpha = alpha->reshape(alpha->ordering(), expectedAlphaShape);

    helpers::prelu(*input, *alpha, *output);

    if(alphaShape != expectedAlphaShape)
        delete alpha;
    
    return Status::OK();
}


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(prelu_bp, 3, 2, true, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto alpha = INPUT_VARIABLE(1);
    auto dLdO  = INPUT_VARIABLE(2);
    
    auto dLdI = OUTPUT_VARIABLE(0);
    auto dLdA = OUTPUT_VARIABLE(1);

    std::vector<int> sharedAxes = *block.getIArguments();
    
    const int inputRank     = input->rankOf();
    const int alphaRank     = alpha->rankOf();
    const int numSharedAxes = sharedAxes.size();            // can be zero as well
    const Nd4jLong inputLen = input->lengthOf();
    const Nd4jLong alphaLen = alpha->lengthOf();
    const std::vector<Nd4jLong> inputShape = input->getShapeAsVector();
    const std::vector<Nd4jLong> alphaShape = alpha->getShapeAsVector();

    //***** input validation *****//
    std::vector<Nd4jLong> expectedAlphaShape(&inputShape[1], &inputShape[inputRank]);

    REQUIRE_TRUE(inputRank > 1, 0, "PRELU_BP OP: wrong rank of input array, expected rank should be > 1, but got %i instead !", inputRank);   
    
    for(int i = 0; i < numSharedAxes; ++i) {
        if(sharedAxes[i] <= 0)
            sharedAxes[i] += inputRank - 1;
        REQUIRE_TRUE(1 <= sharedAxes[i] && sharedAxes[i] <= inputRank - 1, 0, "PRELU_BP OP: wrong axis value %i in sharedAxes at position %i, axis value must be within range [1, input_rank-1] !", sharedAxes[i], i);            
        expectedAlphaShape[sharedAxes[i] - 1] = 1;
    }

    REQUIRE_TRUE(helpers::checkAlphaShapeLen(expectedAlphaShape, alphaLen), 0, "PRELU_BP OP: wrong shape of alpha array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedAlphaShape).c_str(), ShapeUtils::shapeAsString(alphaShape).c_str());
    // ***** end of validation ***** //
    
    if(alphaShape != expectedAlphaShape) {
        alpha = alpha->reshape(alpha->ordering(), expectedAlphaShape);
        dLdA  = dLdA->reshape(dLdA->ordering(), expectedAlphaShape);
    }

    helpers::preluBP(*input, *alpha, *dLdO, *dLdI, *dLdA);

    if(alphaShape != expectedAlphaShape) {        
        delete alpha;
        delete dLdA;
    }
    
    return Status::OK();
}



}
}

#endif