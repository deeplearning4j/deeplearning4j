/*******************************************************************************
 * Copyright (c) 2021 Deeplearning4j Contributors
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
 *******************************************************************************/

//
// @author AbdelRauf
//

#include <system/op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/ctc.h>

namespace sd {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(ctc_beam, 2, 3, false, 0, -2) {

    auto logit = INPUT_VARIABLE(0);
    auto sequence_length = INPUT_VARIABLE(1);
    auto result_sequences = OUTPUT_VARIABLE(0);
    auto result_probs = OUTPUT_VARIABLE(1);
    auto result_sequences_length = OUTPUT_VARIABLE(2);
    auto arg_size =  block.getIArguments()->size();
    auto normalize_logits = block.numB() > 0 ? B_ARG(0) : false;
    
    int blank_index = arg_size>0 ? INT_ARG(0) : -1;
    int beam_width = arg_size>1 ? INT_ARG(1) : 25;
    int nbest_len = arg_size>2? INT_ARG(2): 1;

    REQUIRE_TRUE(logit->rankOf()==3, 0, "Ctc Beam Search: logit Input fails to meet rank requirement {BATCH_LEN, MAX_FRAME_LEN, CLASS_LEN }: %i == 3 ", logit->rankOf());
    REQUIRE_TRUE(sequence_length->rankOf()==1, 0, "Ctc Beam Search: sequence frame length (sequence_length) Input fails to meet rank requirement {BATCH_LEN}: %i == 1 ", sequence_length->rankOf());

    REQUIRE_TRUE(result_sequences->rankOf()==3, 0, "Ctc Beam Search: result_sequences Output fails to meet rank requirement {BATCH_LEN, NBEST_LEN, MAX_FRAME_LEN }: %i == 3 ", result_sequences->rankOf());
    REQUIRE_TRUE(result_probs->rankOf()==2, 0, "Ctc Beam Search: result_probs Output fails to meet rank requirement {BATCH_LEN, NBEST_LEN}: %i == 2 ", result_probs->rankOf());
    REQUIRE_TRUE(result_sequences_length->rankOf()==2, 0, "Ctc Beam Search: result_sequences_length Output fails to meet rank requirement {BATCH_LEN, NBEST_LEN}: %i == 2 ", result_sequences_length->rankOf());

    auto batchSize0 = logit->sizeAt(0);
    auto batchSize1 = sequence_length->sizeAt(0);
    auto batchSize2 = result_sequences->sizeAt(0);
    auto batchSize3 = result_probs->sizeAt(0);
    auto batchSize4 = result_sequences_length->sizeAt(0);

    bool check_batches = (batchSize0 == batchSize1) && (batchSize2 == batchSize3);
    check_batches = check_batches && (batchSize0 == batchSize4) && (batchSize0 == batchSize2);

    REQUIRE_TRUE(nbest_len>0 && nbest_len <=beam_width, 0, "Ctc Beam Search: nbest_len %i should be > 0 and <= %i", nbest_len, beam_width);
    REQUIRE_TRUE(check_batches, 0, "Ctc Beam Search: All batch sizes should be %i", batchSize0);
    auto max_t = logit->sizeAt(1);
    REQUIRE_TRUE(result_sequences->sizeAt(1) == nbest_len && result_sequences->sizeAt(2) == max_t  , 0, "Ctc Beam Search: shape of the result_sequences should be {%i, %i, %i} but got { %i, %i, %i}", 
    batchSize0, nbest_len, max_t, batchSize1, result_sequences->sizeAt(1), result_sequences->sizeAt(2));
    REQUIRE_TRUE(result_probs->sizeAt(1) == nbest_len  , 0, "Ctc Beam Search: shape of the result_probs should be {%i, %i} but got { %i, %i}", 
    batchSize0, nbest_len, batchSize3, result_sequences->sizeAt(1));
    REQUIRE_TRUE(result_sequences_length->sizeAt(1) == nbest_len  , 0, "Ctc Beam Search: shape of the result_sequences_length should be {%i, %i} but got { %i, %i}", 
    batchSize0, nbest_len, batchSize4, result_sequences_length->sizeAt(1));
    REQUIRE_TRUE(result_sequences->ews()==1 && result_sequences->ordering()=='c', 0, "Ctc Beam Search: result_sequences output should be ews()==1 and c order: %d == ews(1) %c == order(c) ", result_sequences->ews(), result_sequences->ordering()); 
    REQUIRE_TRUE(result_probs->ews()==1 && result_probs->ordering()=='c', 0, "Ctc Beam Search: result_probs output should be ews()==1 and c order: %d == ews(1) %c == order(c) ",  result_probs->ews(), result_probs->ordering()); 
    REQUIRE_TRUE(result_sequences_length->ews()==1 && result_sequences_length->ordering()=='c', 0, "Ctc Beam Search: result_sequences_length output should be ews()==1 and c order: %d == ews(1) %c == order(c) ",  result_sequences_length->ews(), result_sequences_length->ordering()); 

    sd::ops::helpers::beamSearch(*logit, *sequence_length, *result_sequences, *result_probs, *result_sequences_length, blank_index, beam_width, nbest_len, normalize_logits);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(ctc_beam) {
        getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})
                     ->setAllowedInputTypes(1,{ALL_INDICES})
                     ->setAllowedOutputTypes(0, {ALL_INDICES})
                     ->setAllowedOutputTypes(1, {ALL_FLOATS})
                     ->setAllowedOutputTypes(2, {ALL_INDICES});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(ctc_beam) {
    auto logitShapeInfo =  inputShape->at(0);
    auto sequenceShapeInfo =  inputShape->at(1);
    auto arg_size =  block.getIArguments()->size();

    auto nbest_len = arg_size>2? INT_ARG(2): 1;

    REQUIRE_TRUE(logitShapeInfo[0] ==3 , 0, "Ctc Beam Search: logit Input fails to meet rank requirement {BATCH_LEN, MAX_FRAME_LEN, CLASS_LEN }: %i == 3",
                 logitShapeInfo[0]);

    auto batch_size = shape::shapeOf(logitShapeInfo)[0] ;
    auto max_t = shape::shapeOf(logitShapeInfo)[1] ;

    auto dtype_float = ArrayOptions::dataType(logitShapeInfo);
    auto dtype_index = ArrayOptions::dataType(sequenceShapeInfo);

    auto output0 = ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(dtype_index, 'c', {batch_size, nbest_len, max_t}));
    auto output1 = ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(dtype_float, 'c', {batch_size, nbest_len})); 
    auto output2 = ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(dtype_index, 'c', {batch_size, nbest_len})); 
    return SHAPELIST(output0, output1, output2);
}

}
}
