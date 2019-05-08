/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 29.08.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_sparse_softmax_cross_entropy_loss_with_logits)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/ScatterHelper.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sparse_softmax_cross_entropy_loss_with_logits, 2, 1, false, 0, 0) {
  	auto labels  = INPUT_VARIABLE(0);
    auto logits  = INPUT_VARIABLE(1);

    auto output  = OUTPUT_VARIABLE(0);

    const int labelsRank = labels->rankOf();
    const int logitsRank = logits->rankOf();
   
    // input validation    		       
    REQUIRE_TRUE(labelsRank == logitsRank - 1, 0, "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: input arrays should satisfy relation (labels_rank = logits_rank - 1), but got labels_rank = %i and logits_rank = %i instead !", labelsRank, logitsRank);

    std::vector<Nd4jLong> labelsShape = labels->getShapeAsVector(); // this is correct
    std::vector<Nd4jLong> logitsShape = logits->getShapeAsVector();
    logitsShape.pop_back();
    bool equalSoft = logitsShape == labelsShape;
    
    REQUIRE_TRUE(equalSoft, 0, "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: wrong shape of labels array, its shape should be the same as logits shape with last dimension excluded, however got labels_shape = %s and logits_shape = %s instead !", ShapeUtils::shapeAsString(labelsShape).c_str(), ShapeUtils::shapeAsString(logitsShape).c_str());

    std::vector<int> dimension = {-1};

    auto maxAlongDim = logits->reduceAlongDims(reduce::Max, dimension, true);    
    auto logitsExp = (*logits - maxAlongDim).transform(transform::Exp, nullptr);
    auto logSoftMax = ( logitsExp / logitsExp.reduceAlongDims(reduce::Sum, dimension, true) ).transform(transform::Log);

    ScatterHelper::scatterForLoss(*labels, -logSoftMax, *output, false);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(sparse_softmax_cross_entropy_loss_with_logits) {
    
    getOpDescriptor()->setAllowedInputTypes(0, {ALL_INTS})->setAllowedInputTypes(1, {ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS});            
}


//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(sparse_softmax_cross_entropy_loss_with_logits) {

    auto labelsShapeInfo = inputShape->at(0);
    auto logitsShapeInfo = inputShape->at(1);

    REQUIRE_TRUE(labelsShapeInfo[0] == logitsShapeInfo[0] - 1, 0, "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: input arrays should satisfy relation (labels_rank = logits_rank - 1), but got labels_rank = %i and logits_rank = %i instead !", labelsShapeInfo[0], logitsShapeInfo[0]);

    bool equalSoft = true;
    for (int i = 1; i < labelsShapeInfo[0]; ++i)
        if (labelsShapeInfo[i] != logitsShapeInfo[i]) {
            equalSoft = false;
            break;
        }    
    
    REQUIRE_TRUE(equalSoft, 0, "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: wrong shape of labels array, its shape should be the same as logits shape with last dimension excluded, however got labels_shape = %s and logits_shape = %s instead !", ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());

    Nd4jLong *outShapeInfo =  ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, logitsShapeInfo, false, block.getWorkspace());
    
    return SHAPELIST(outShapeInfo);
}







//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sparse_softmax_cross_entropy_loss_with_logits_grad, 2, 1, false, 0, 0) {
    
    auto labels  = INPUT_VARIABLE(0);
    auto logits  = INPUT_VARIABLE(1);

    auto dLdp  = OUTPUT_VARIABLE(0);    // dL/dlogits

    const int labelsRank = labels->rankOf();
    const int logitsRank = logits->rankOf();
   
    // input validation                
    REQUIRE_TRUE(labelsRank == logitsRank - 1, 0, "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: input arrays should satisfy relation (labels_rank = logits_rank - 1), but got labels_rank = %i and logits_rank = %i instead !", labelsRank, logitsRank);

    std::vector<Nd4jLong> labelsShape = labels->getShapeAsVector(); // this is correct
    std::vector<Nd4jLong> logitsShape = logits->getShapeAsVector();
    logitsShape.pop_back();
    bool equalSoft = logitsShape == labelsShape;
    
    REQUIRE_TRUE(equalSoft, 0, "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: wrong shape of labels array, its shape should be the same as logits shape with last dimension excluded, however got labels_shape = %s and logits_shape = %s instead !", ShapeUtils::shapeAsString(labelsShape).c_str(), ShapeUtils::shapeAsString(logitsShape).c_str());

    std::vector<int> dimension = {-1};

    NDArray softmax = (*logits - logits->reduceAlongDims(reduce::Max, dimension, true)).transform(transform::Exp);
    softmax /= softmax.reduceAlongDims(reduce::Sum, dimension, true);

    // dEdp = softmax - 1 (or 0)
    dLdp->assign(softmax);

    // subtract unities at appropriate indexes of dLdp array    
    ScatterHelper::scatterForLoss(*labels, *dLdp, *labels /*actually third array is unnecessary for gradient calculation*/, true);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(sparse_softmax_cross_entropy_loss_with_logits_grad) {
    
    getOpDescriptor()->setAllowedInputTypes(0, {ALL_INTS})->setAllowedInputTypes(1, {ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS});            
}


//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(sparse_softmax_cross_entropy_loss_with_logits_grad) {

    auto labelsShapeInfo = inputShape->at(0);
    auto logitsShapeInfo = inputShape->at(1);

    REQUIRE_TRUE(labelsShapeInfo[0] == logitsShapeInfo[0] - 1, 0, "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: input arrays should satisfy relation (labels_rank = logits_rank - 1), but got labels_rank = %i and logits_rank = %i instead !", labelsShapeInfo[0], logitsShapeInfo[0]);

    bool equalSoft = true;
    for (int i = 1; i < labelsShapeInfo[0]; ++i)
        if (labelsShapeInfo[i] != logitsShapeInfo[i]) {
            equalSoft = false;
            break;
        }    
    
    REQUIRE_TRUE(equalSoft, 0, "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS_GRAD OP: wrong shape of labels array, its shape should be the same as logits shape with last dimension excluded, however got labels_shape = %s and logits_shape = %s instead !", ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());

    DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));    

    Nd4jLong *dLdpShapeInfo = ShapeBuilders::copyShapeInfoAndType(logitsShapeInfo, outType, false, block.getWorkspace());    
    
    return SHAPELIST(dLdpShapeInfo);
}



}
}

#endif