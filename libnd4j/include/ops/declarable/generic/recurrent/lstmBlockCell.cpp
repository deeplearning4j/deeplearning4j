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
// @author Alex Black
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_lstmBlockCell)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstm.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmBlockCell, 8, 7, false, 2, 1) {
	//Notation: mostly following https://arxiv.org/pdf/1503.04069.pdf
    auto xt    = INPUT_VARIABLE(0);                   // input [bS, inSize] at time t
    auto cLast = INPUT_VARIABLE(1);                   // previous cell state  [bS, numUnits], time t-1
	auto yLast = INPUT_VARIABLE(2);                   // previous output [bS, numUnits], time t-1

    auto W    = INPUT_VARIABLE(3);                    // Weights - concatenated (input-to-hidden, hidden-to-hidden weights)  weights, [(inSize+numUnits), 4*numUnits]
	auto Wci  = INPUT_VARIABLE(4);                    // weights - cell peephole (t-1) connections to input modulation gate, [numUnits]
	auto Wcf  = INPUT_VARIABLE(5);                    // weights - cell peephole (t-1) connections to forget gate, [numUnits]
	auto Wco  = INPUT_VARIABLE(6);                    // weights - cell peephole (t) connections to output gate, [numUnits]
    auto b    = INPUT_VARIABLE(7);                    // biases, [4*numUnits]

	auto z   =  OUTPUT_VARIABLE(0);                   // Output - input gate activations [bs, numUnits]
	auto i   =  OUTPUT_VARIABLE(1);                   // Output - input modulation gate activations [bS, numUnits]
    auto f   =  OUTPUT_VARIABLE(2);                   // Output - forget gate activations [bs, numUnits]
	auto o   =  OUTPUT_VARIABLE(3);                   // Output - output gate activations [bs, numUnits]
	auto h   =  OUTPUT_VARIABLE(4);                   // Activations, pre input gate [bs, numUnits]
	auto c   =  OUTPUT_VARIABLE(5);		               // Activations, cell state [bs, numUnits]
	auto y   =  OUTPUT_VARIABLE(6);                   // current cell output [bS, numProj], time t


    const int peephole   = INT_ARG(0);                // if 1, provide peephole connections

	const double forgetBias   = T_ARG(0);
    const double clippingCellValue  = T_ARG(0);       // clipping value for ct, if it is not equal to zero, then cell state is clipped

    const int rank     = xt->rankOf();
    const int bS       = xt->sizeAt(0);
    const int inSize   = xt->sizeAt(1);
    const int numProj  = ht_1->sizeAt(1);
    const int numUnits = ct_1->sizeAt(1);

    // input shapes validation
	REQUIRE_TRUE(xt->rankOf()==2 && cLast->rankOf()==2 && yLast->rankOf()==2, 0, "Input ranks must be 2 for inputs 0/1/2 (x, cLast, outLast) - got %i, %i, %i", xt->rankOf(), cLast->rankOf(), yLast->rankOf());

	//TODO: other checks


    // calculations
    helpers::lstmCell(xt, cLast, yLast, W, Wci, Wcf, Wco, b, z, i, f, o, h, c, y, {(double)peephole, forgetBias, clippingCellValue});

    return Status::OK();
}

        DECLARE_TYPES(lstmCell) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }


DECLARE_SHAPE_FN(lstmBlockCell) {
    auto xt    = inputShape->at(0);                   // input [bS, inSize] at time t
    auto cLast = inputShape->at(1);                   // previous cell state  [bS, numUnits], time t-1
	auto yLast = inputShape->at(2);                   // previous output [bS, numUnits], time t-1

    auto W    = inputShape->at(3);                    // Weights - concatenated (input-to-hidden, hidden-to-hidden weights)  weights, [(inSize+numUnits), 4*numUnits]
	auto Wci  = inputShape->at(4);                    // weights - cell peephole (t-1) connections to input modulation gate, [numUnits]
	auto Wcf  = inputShape->at(5);                    // weights - cell peephole (t-1) connections to forget gate, [numUnits]
	auto Wco  = inputShape->at(6);                    // weights - cell peephole (t) connections to output gate, [numUnits]
    auto b    = inputShape->at(7);                    // biases, [4*numUnits]

    auto xtShapeInfo   = inputShape->at(0);                   // input [bS x inSize]
    auto ht_1ShapeInfo = inputShape->at(1);                   // previous cell output [bS x numProj],  that is at previous time step t-1, in case of projection=false -> numProj=numUnits!!!
    auto ct_1ShapeInfo = inputShape->at(2);                   // previous cell state  [bS x numUnits], that is at previous time step t-1

    auto WxShapeInfo   = inputShape->at(3);                   // input-to-hidden  weights, [inSize  x 4*numUnits]
    auto WhShapeInfo   = inputShape->at(4);                   // hidden-to-hidden weights, [numProj x 4*numUnits]
    auto WcShapeInfo   = inputShape->at(5);                   // diagonal weights for peephole connections [3*numUnits]
    auto WpShapeInfo   = inputShape->at(6);                   // projection weights [numUnits x numProj]
    auto bShapeInfo    = inputShape->at(7);                   // biases, [4*numUnits]

    //TODO: shape validation

    // evaluate output shapeInfos
    Nd4jLong *s(nullptr);
    ALLOCATE(s, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [bS, numUnits]

    s[0] = rank;
    s[1] = bS;
    s[2] = numUnits;

    ShapeUtils::updateStridesAndType(s, xt, 'c');

    //7 outputs, all same shape: z, i, f, o, h, c, y
    return SHAPELIST(s, s, s, s, s, s, s);
}

}
}

#endif