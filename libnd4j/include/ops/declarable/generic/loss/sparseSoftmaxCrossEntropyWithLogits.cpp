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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 29.08.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_sparse_softmax_cross_entropy_loss_with_logits)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/losses.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
OP_IMPL(sparse_softmax_cross_entropy_loss_with_logits, 2, 1, false) {
  	auto labels  = INPUT_VARIABLE(0);
    auto logits  = INPUT_VARIABLE(1);

    auto output  = OUTPUT_VARIABLE(0);

    const int labelsRank = labels->rankOf();
    const int logitsRank = logits->rankOf();
   
    // input validation    		       
    REQUIRE_TRUE(labelsRank == logitsRank - 1, 0, "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: input arrays should satisfy relation (labels_rank = logits_rank - 1), but got labels_rank = %i and logits_rank = %i instead !", labelsRank, logitsRank);

    std::vector<Nd4jLong> labelsShape = labels->getShapeAsVector();
    std::vector<Nd4jLong> logitsShape = logits->getShapeAsVector();
    REQUIRE_TRUE(shape::equalsSoft(labels->shapeInfo(), logits->shapeInfo()), 0, "SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_WITH_LOGITS OP: wrong shape of labels array, its shape should be the same as logits shape with last dimension excluded, however got labels_shape = %s and logits_shape = %s instead !", ShapeUtils::shapeAsString(labelsShape).c_str(), ShapeUtils::shapeAsString(logitsShape).c_str());

    helpers::sparseSoftmaxCrossEntropyLossWithLogits(*labels, *logits, *output);

    return Status::OK();
}



}
}

#endif