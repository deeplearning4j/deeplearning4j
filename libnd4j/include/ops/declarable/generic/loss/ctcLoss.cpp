/*******************************************************************************
 * Copyright (c) 2021 Konduit K.K.
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
// @author AbdelRauf
//


#include <system/op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/ctcLoss.h>

namespace sd {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(ctc_loss, 4, 1, false, 0, 1) {

    auto targetLabels = INPUT_VARIABLE(0);
    auto logitInput = INPUT_VARIABLE(1);
    auto targetLabelLengths = INPUT_VARIABLE(2);
    auto logitInputLengths = INPUT_VARIABLE(3); 
    auto outputLosses = OUTPUT_VARIABLE(0);

    int blankIndex = INT_ARG(0);

    REQUIRE_TRUE(targetLabels->rankOf()==2, 0, "CtcLoss: target labels fails to meet rank requirement (batch_size, max_label_sequence_length): %i == 2 ", targetLabels->rankOf());
    REQUIRE_TRUE(logitInput->rankOf()==3, 0, "CtcLoss: logit Input fails to meet rank requirement (batch_size, frames, classes): %i == 3 ", logitInput->rankOf());
    REQUIRE_TRUE(targetLabelLengths->rankOf()==1, 0, "CtcLoss: target label length fails to meet rank requirement (batch_size): %i == 1 ", targetLabelLengths->rankOf());
    REQUIRE_TRUE(logitInputLengths->rankOf()==1, 0, "CtcLoss: logit Input lengths fails to meet rank requirement (batch_size): %i == 1 ", logitInputLengths->rankOf());

    int batchSize0 = targetLabels->sizeAt(0);
    int batchSize1 = logitInput->sizeAt(0);
    int batchSize2 = targetLabelLengths->sizeAt(0);
    int batchSize3 = logitInputLengths->sizeAt(0);
    int batchSize4 = outputLosses->sizeAt(0);

    bool check_batches = (batchSize0 == batchSize1) && (batchSize2 == batchSize3);
    check_batches = check_batches && (batchSize0 == batchSize4) && (batchSize0 == batchSize2);

    REQUIRE_TRUE(check_batches, 0, "CtcLoss: All batch sizes should be equal %i", batchSize0);
    REQUIRE_TRUE(outputLosses->isSameShape(targetLabelLengths), 0, "CtcLoss: wrong shape of output array, expected is %s but got %s instead !", ShapeUtils::shapeAsString(targetLabelLengths).c_str(), ShapeUtils::shapeAsString(outputLosses).c_str());

    auto emptyGradients = NDArrayFactory::empty<float>();
    sd::ops::helpers::ctcLoss(block, *logitInput, *targetLabels, *logitInputLengths, *targetLabelLengths, *outputLosses, emptyGradients, blankIndex);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(ctc_loss) {
        getOpDescriptor()->setAllowedInputTypes({ALL_INDICES})
                     ->setAllowedInputTypes(1,{ALL_FLOATS})
                     ->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(ctc_loss) {
    auto yShapeInfo =  inputShape->at(1);
    auto zShapeInfo = inputShape->at(2); 

    auto dtype = ArrayOptions::dataType(yShapeInfo);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(zShapeInfo, dtype)));
}



//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(ctc_loss_grad, 4, 1, false, 1, 1) {
 
    auto targetLabels = INPUT_VARIABLE(0);
    auto logitInput = INPUT_VARIABLE(1);
    auto targetLabelLengths = INPUT_VARIABLE(2);
    auto logitInputLengths = INPUT_VARIABLE(3); 
    auto outputGradients = OUTPUT_VARIABLE(0);

    int blankIndex = INT_ARG(0);

    REQUIRE_TRUE(targetLabels->rankOf()==2, 0, "CtcLoss: target labels fails to meet rank requirement (batch_size, max_label_sequence_length): %i == 2 ", targetLabels->rankOf());
    REQUIRE_TRUE(logitInput->rankOf()==3, 0, "CtcLoss: logit Input fails to meet rank requirement (batch_size, frames, classes): %i == 3 ", logitInput->rankOf());
    REQUIRE_TRUE(targetLabelLengths->rankOf()==1, 0, "CtcLoss: target label length fails to meet rank requirement (batch_size): %i == 1 ", targetLabelLengths->rankOf());
    REQUIRE_TRUE(logitInputLengths->rankOf()==1, 0, "CtcLoss: logit Input lengths fails to meet rank requirement (batch_size): %i == 1 ", logitInputLengths->rankOf());

    int batchSize0 = targetLabels->sizeAt(0);
    int batchSize1 = logitInput->sizeAt(0);
    int batchSize2 = targetLabelLengths->sizeAt(0);
    int batchSize3 = logitInputLengths->sizeAt(0);
    int batchSize4 = outputGradients->sizeAt(0);

    bool check_batches = (batchSize0 == batchSize1) && (batchSize2 == batchSize3);
    check_batches = check_batches && (batchSize0 == batchSize4) && (batchSize0 == batchSize2);

    REQUIRE_TRUE(check_batches, 0, "CtcLoss Gradient: All batch sizes should be equal %i", batchSize0);
    REQUIRE_TRUE(outputGradients->isSameShape(logitInput), 0, "CtcLoss Gradient: wrong shape of output array, expected is %s but got %s instead !", ShapeUtils::shapeAsString(logitInput).c_str(), ShapeUtils::shapeAsString(outputGradients).c_str());

    auto emptyLoss = NDArrayFactory::empty<float>();
    sd::ops::helpers::ctcLoss(block, *logitInput, *targetLabels, *logitInputLengths, *targetLabelLengths, emptyLoss, *outputGradients, blankIndex);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(ctc_loss_grad) {
        getOpDescriptor()->setAllowedInputTypes({ALL_INDICES})
                     ->setAllowedInputTypes(1,{ALL_FLOATS})
                     ->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(ctc_loss_grad) {
    auto yShapeInfo =  inputShape->at(1);
    auto dtype = ArrayOptions::dataType(yShapeInfo);
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(yShapeInfo, dtype)));
  
}



}
}

