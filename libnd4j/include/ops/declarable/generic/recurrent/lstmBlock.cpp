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
// @author Alex Black
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_lstmBlock)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstmBlock.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmBlock, 9, 7, false, 2, 2) {
    auto maxTSLength = INPUT_VARIABLE(0);
    auto xt    = INPUT_VARIABLE(1);                   // input [seqLen, bS, inSize] at time t
    auto cLast = INPUT_VARIABLE(2);                   // previous cell state  [bS, numUnits], time t-1
	auto yLast = INPUT_VARIABLE(3);                   // previous output [bS, numUnits], time t-1

    auto W    = INPUT_VARIABLE(4);                    // Weights - concatenated (input-to-hidden, hidden-to-hidden weights)  weights, [(inSize+numUnits), 4*numUnits]
	auto Wci  = INPUT_VARIABLE(5);                    // weights - cell peephole (t-1) connections to input modulation gate, [numUnits]
	auto Wcf  = INPUT_VARIABLE(6);                    // weights - cell peephole (t-1) connections to forget gate, [numUnits]
	auto Wco  = INPUT_VARIABLE(7);                    // weights - cell peephole (t) connections to output gate, [numUnits]
    auto b    = INPUT_VARIABLE(8);                    // biases, [4*numUnits]

    auto i   =  OUTPUT_VARIABLE(0);                   // Output - input modulation gate activations [seqLen, bS, numUnits]
	auto c   =  OUTPUT_VARIABLE(1);		              // Activations, cell state (pre tanh) [seqLen, bs, numUnits]
	auto f   =  OUTPUT_VARIABLE(2);                   // Output - forget gate activations [seqLen, bs, numUnits]
	auto o   =  OUTPUT_VARIABLE(3);                   // Output - output gate activations [seqLen, bs, numUnits]
	auto z   =  OUTPUT_VARIABLE(4);                   // Output - input gate activations [seqLen, bs, numUnits]
	auto h   =  OUTPUT_VARIABLE(5);                   // Cell state, post tanh [seqLen, bs, numUnits]
	auto y   =  OUTPUT_VARIABLE(6);                   // current cell output [seqLen, bS, numProj], time t
    
    const int peephole   = INT_ARG(0);                     // if 1, provide peephole connections
    const int dataFormat = INT_ARG(1);                     // 0=TNS=[seqLen,mb,size]; 1=NST=[mb,size,seqLen]; 2=NTS=[mb,seqLen,size]
	const double forgetBias   = T_ARG(0);
    const double clippingCellValue  = T_ARG(1);       // clipping value for ct, if it is not equal to zero, then cell state is clipped

    //TODO input validation

    helpers::lstmBlockTimeLoop(maxTSLength, xt, cLast, yLast,   W, Wci, Wcf, Wco, b,    i, c, f, o, z, h, y, {(double)peephole, forgetBias, clippingCellValue}, dataFormat);

    return Status::OK();
}

DECLARE_TYPES(lstmBlock) {
    getOpDescriptor()
            ->setAllowedInputTypes(nd4j::DataType::ANY)
            ->setAllowedOutputTypes({ALL_FLOATS});
}


DECLARE_SHAPE_FN(lstmBlock) {
    auto xt    = inputShape->at(1);                   // input [seqLen, bS, inSize] at time t
    auto cLast = inputShape->at(2);                   // prev cell state, [bS, numUnits]

    int bs = xt[2];     //[rank, seqLen, bs, nIn, ...]
    int t = xt[1];
    int nOut = cLast[2];    //rank, bs, nOut, ...]

    Nd4jLong *s(nullptr);
    ALLOCATE(s, block.getWorkspace(), shape::shapeInfoLength(3), Nd4jLong);      // [time, bS, nOut]

    s[0] = 3;
    s[1] = t;
    s[2] = bs;
    s[3] = nOut;
    ShapeUtils::updateStridesAndType(s, xt, 'c');

    //7 outputs
    return SHAPELIST(s, s, s, s, s, s, s);
}   

}
}

#endif