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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.02.2018
//

// implementation of gated Recurrent Unit cell
// (cf. http://arxiv.org/abs/1406.1078).
// Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
// "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"


#include<ops/declarable/helpers/gru.h>
#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>
#include <MmulHelper.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
void gruCell(nd4j::LaunchContext * context, const NDArray* x, const NDArray* hLast, const NDArray* Wru, const NDArray* Wc,
             const NDArray* bru, const NDArray* bc,
             NDArray* r, NDArray* u, NDArray* c, NDArray* h) {

    //Inputs:
    // x        input [bS x inSize]
    // hLast    previous cell output [bS x numUnits],  that is at previous time step t-1
    // Wru      RU weights - [bS, 2*numUnits] - reset and update gates
    // Wc       C weights - [bS, numUnits] - cell gate
    // bru      r and u biases, [2*numUnits] - reset and update gates
    // bc       c biases, [numUnits] - cell gate

    //Outputs:
    // r        Reset gate output [bS, numUnits]
    // u        Update gate output [bS, numUnits]
    // c        Cell gate output [bS, numUnits]
    // h        current cell output [bS, numUnits]

    const int nIn = x->sizeAt(1);
    const int nU = hLast->sizeAt(1);                // number of units

    //Concat inputs: [x, yt-1]: concat([bs,nIn],[bs,nOut]) -> [bs, (nIn+nOut)]
    nd4j::ops::concat concatOp;
    std::vector<NDArray*> inputs;
    std::vector<double> targs;
    std::vector<Nd4jLong> iargs({1});   //Axis = 1
    std::vector<bool> bargs;
    inputs.emplace_back(const_cast<NDArray*>(x));
    inputs.emplace_back(const_cast<NDArray*>(hLast));

    auto result = concatOp.execute(inputs, targs, iargs, bargs);
    auto concatOut = result->at(0);

    //mmul/z for reset and update gates: (x * weight_ux + hLast * weight_xr + b_u)
    auto m = mmul(*concatOut, *Wru);    //mmul: [bs, (nIn+numUnits)]* [(inSize+numUnits), 2*numUnits] = [bs, 4*numUnits]
    m += (*bru);

    sigmoidInplace(m);  //sigmoid(rz) and sigmoid(uz)
    auto mr = m({0,0, 0, nU});
    auto mu = m({0,0, nU, 2*nU});

    r->assign(&mr);
    u->assign(&mu);

    //Concatenated inputs: [x, yt-1 .* r]
    auto yr = (*concatOut)({0,0, nIn, nIn+nU});
    yr *= (*r);

    //c = tanh(x * weight_cx + (hLast .* r) * weight_cr + b_c)
    MmulHelper::mmul(concatOut, const_cast<NDArray*>(Wc), c, 1.0, 0.0);       //c = 1.0 * concatOut * Wc + 0.0 * c
    *c += *bc;
    tanhInplace(*c);

    //Output: h = (1-u).*c + u .* hPrev
    //auto hResult = (*u) * (*hLast) + (1.0f - *u) * (*c); const_cast<NDArray*>(h)->assign(&hResult);
    u->applyPairwiseTransform(pairwise::Multiply, hLast, h, nullptr);        //h = u * hLast
    auto temp = (1.0f - *u);
    temp *= (*c);
    (*h) += temp;

    delete result;
}

//////////////////////////////////////////////////////////////////////////
void gruTimeLoop(nd4j::LaunchContext * context, const NDArray* x, const NDArray* h0, const NDArray* Wx, const NDArray* Wh, const NDArray* b, NDArray* h) {

}

//////////////////////////////////////////////////////////////////////////
void gruCellBP(nd4j::LaunchContext * context, const NDArray* x, const NDArray* h0, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* dLdh, const NDArray* dLdWx0,
               const NDArray* dLdWh0, const NDArray* dLdb0, NDArray* dLdx, NDArray* dLdh0, NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb) {

}


}
}
}

