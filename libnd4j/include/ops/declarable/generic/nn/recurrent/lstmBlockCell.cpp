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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_lstmBlockCell)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstmBlock.h>

namespace sd {
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

    auto i   =  OUTPUT_VARIABLE(0);                   // Output - input modulation gate activations [bS, numUnits]
    auto c   =  OUTPUT_VARIABLE(1);                      // Activations, cell state (pre tanh) [bs, numUnits]
    auto f   =  OUTPUT_VARIABLE(2);                   // Output - forget gate activations [bs, numUnits]
    auto o   =  OUTPUT_VARIABLE(3);                   // Output - output gate activations [bs, numUnits]
    auto z   =  OUTPUT_VARIABLE(4);                   // Output - input gate activations [bs, numUnits]
    auto h   =  OUTPUT_VARIABLE(5);                   // Cell state, post tanh [bs, numUnits]
    auto y   =  OUTPUT_VARIABLE(6);                   // current cell output [bS, numProj], time t


    const int peephole   = INT_ARG(0);                // if 1, provide peephole connections

    const double forgetBias   = T_ARG(0);
    const double clippingCellValue  = T_ARG(1);       // clipping value for ct, if it is not equal to zero, then cell state is clipped

    REQUIRE_TRUE(xt->rankOf()==2 && cLast->rankOf()==2 && yLast->rankOf()==2, 0, "lstmBlockCell: Input ranks must be 2 for inputs 0/1/2 (x, cLast, outLast) - got %i, %i, %i", xt->rankOf(), cLast->rankOf(), yLast->rankOf());
    const int rank     = xt->rankOf();
    const int bS       = xt->sizeAt(0);
    const int inSize   = xt->sizeAt(1);
    const int numUnits = cLast->sizeAt(1);

    REQUIRE_TRUE(xt->sizeAt(0) == yLast->sizeAt(0) && xt->sizeAt(0) == cLast->sizeAt(0), 0, "lstmBlockCell: Input minibatch sizes (dimension 0) must be same for xt, cLast, yLast");
    REQUIRE_TRUE(W->rankOf()==2, 0, "lstmBlockCell: Weights array rank must be 2");
    REQUIRE_TRUE(W->sizeAt(0)==(inSize+numUnits), 0, "lstmBlockCell: Weights size(0) must be equal to inSize + numUnits, got %i", W->sizeAt(0));
    REQUIRE_TRUE(W->sizeAt(1)==(4*numUnits), 0, "lstmBlockCell: Weights size(1) must be equal to 4*numUnits, got %i", W->sizeAt(1));
    REQUIRE_TRUE(b->rankOf()==1 && b->sizeAt(0)==(4*numUnits), 0, "lstmBlockCell: Biases must be rank 1, size 4*numUnits");
    REQUIRE_TRUE(i->rankOf()==2 && c->rankOf()==2 && f->rankOf()==2 && o->rankOf()==2 && z->rankOf()==2 && h->rankOf()==2 && y->rankOf()==2 &&
                 i->sizeAt(0)==bS && c->sizeAt(0)==bS && f->sizeAt(0)==bS && o->sizeAt(0)==bS && z->sizeAt(0)==bS && h->sizeAt(0)==bS && y->sizeAt(0)==bS &&
                 i->sizeAt(1)==numUnits && c->sizeAt(1)==numUnits && f->sizeAt(1)==numUnits && o->sizeAt(1)==numUnits && z->sizeAt(1)==numUnits && h->sizeAt(1)==numUnits && y->sizeAt(1)==numUnits,
                 0, "lstmBlockCell: Output arrays must all be rank 2 with size(0) == batchSize and size(1) == numUnits");

    // calculations
    helpers::lstmBlockCell(xt, cLast, yLast, W, Wci, Wcf, Wco, b, i, c, f, o, z, h, y, {(double)peephole, forgetBias, clippingCellValue});

    return Status::OK();
}

DECLARE_TYPES(lstmBlockCell) {
    getOpDescriptor()
            ->setAllowedInputTypes(sd::DataType::ANY)
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

    REQUIRE_TRUE(shape::rank(xt)==2 && shape::rank(cLast)==2 && shape::rank(yLast)==2, 0, "lstmBlockCell: Input ranks must be 2 for inputs 0/1/2 (x, cLast, outLast) - got %i, %i, %i", shape::rank(xt), shape::rank(cLast), shape::rank(yLast));
    const int inSize = xt[2];
    const int numUnits = cLast[2];  //[rank, bS, nOut, ...]
    REQUIRE_TRUE(xt[1] == yLast[1] && xt[1] == cLast[1], 0, "lstmBlockCell: Input minibatch sizes (dimension 0) must be same for xt, cLast, yLast");
    REQUIRE_TRUE(shape::rank(W)==2, 0, "lstmBlockCell: Weights array rank must be rank 2, got %i", shape::rank(W));
    REQUIRE_TRUE(W[1]==(inSize+numUnits), 0, "lstmBlockCell: Weights size(0) must be equal to inSize + numUnits, got %i", W[1]);
    REQUIRE_TRUE(W[2]==(4*numUnits), 0, "lstmBlockCell: Weights size(1) must be equal to 4*numUnits, got %i", W[2]);
    REQUIRE_TRUE(shape::rank(b)==1 && b[1]==(4*numUnits), 0, "lstmBlockCell: Biases must be rank 1, size 4*numUnits");

    // evaluate output shapeInfos
    const int bS = xt[1];
    Nd4jLong *s(nullptr);
    ALLOCATE(s, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);      // [bS, numUnits]

    s[0] = 2;
    s[1] = bS;
    s[2] = numUnits;

    ShapeUtils::updateStridesAndType(s, xt, 'c');

    auto s1 = CONSTANT(s);

    //7 outputs, all same shape: z, i, f, o, h, c, y
    return SHAPELIST(s1, s1, s1, s1, s1, s1, s1);
}

}
}

#endif