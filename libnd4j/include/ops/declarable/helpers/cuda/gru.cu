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

    // x                        input [bS, iS]
    // h0                       previous cell output [bS, nU],  that is at previous time step t-1
    // Wx                       input-to-hidden  weights, [iS, 3*nU]
    // Wh                       hidden-to-hidden weights, [nU, 3*nU]
    // b                        biases, [3*nU]
    // dLdh                     gradient wrt output, [bS,nU], that is epsilon_next
    // dLdWx0                   gradient wrt Wx at previous time step, [iS, 3*nU]
    // dLdWh0                   gradient wrt Wh at previous time step, [nU, 3*nU]
    // dLdb0                    gradient wrt b at previous time step,  [3*nU]

    // dLdx                   gradient wrt x,  [bS, iS], that is epsilon
    // dLdh0                  gradient wrt h0, [bS, nU]
    // dLdWx                  gradient wrt Wx, [iS, 3*nU]
    // dLdWh                  gradient wrt Wh, [nU, 3*nU]
    // dLdb                   gradient wrt b at previous time step,  [3*nU]

    // h is current cell output [bS, nU], that is at current time step t

    const int nU = h0->sizeAt(1);

    // ***** feed forward step ***** //
    // gates = sigmoid(x*Wx + h0*Wh + b)
    auto gates = sigmoid(mmul(*x, (*Wx)({0,0, 0,2*nU})) + mmul(*h0, (*Wh)({0,0, 0,2*nU})) + (*b)({0,2*nU}));       // [bS, 2*nU] + [bS, 2*nU] + [1, 2*nU] = [bS, 2*nU]
    // reset gate
    auto r = gates({0,0, 0, nU});               // [bS, nU]
    // update gate
    auto u = gates({0,0, nU, 2*nU});            // [bS, nU]
    // ◦ means element-wise product or so called Hadamard product
    // n = tanh(x*Wx + (r◦h0)*Wh + b)
    auto n = tanh(mmul(*x, (*Wx)({0,0, 2*nU,3*nU})) + mmul((*h0)*r, (*Wh)({0,0, 2*nU,3*nU})) + (*b)({2*nU,3*nU}));     // [bS, nU]

    // ***** back prop step ***** //
    auto Wxr  = (*Wx)({0,0, 0,   nU});
    auto Wxu  = (*Wx)({0,0, nU,  2*nU});
    auto Wxn  = (*Wx)({0,0, 2*nU,3*nU});
    auto Whr  = (*Wh)({0,0, 0,   nU});
    auto Whu  = (*Wh)({0,0, nU,  2*nU});
    auto Whn  = (*Wh)({0,0, 2*nU,3*nU});
    auto WxrT = Wxr.transpose();
    auto WxuT = Wxu.transpose();
    auto WxnT = Wxn.transpose();
    auto WhrT = Whr.transpose();
    auto WhuT = Whu.transpose();
    auto WhnT = Whn.transpose();
    auto xT   = x->transpose();
    auto h0T  = h0->transpose();

    auto dLdWxr = (*dLdWx)({0,0, 0,     nU});
    auto dLdWxu = (*dLdWx)({0,0, nU,  2*nU});
    auto dLdWxn = (*dLdWx)({0,0, 2*nU,3*nU});

    auto dLdWhr = (*dLdWh)({0,0, 0,     nU});
    auto dLdWhu = (*dLdWh)({0,0, nU,  2*nU});
    auto dLdWhn = (*dLdWh)({0,0, 2*nU,3*nU});

    auto dLdbr = (*dLdb)({0,     nU});
    auto dLdbu = (*dLdb)({nU,  2*nU});
    auto dLdbn = (*dLdb)({2*nU,3*nU});

    auto dhdu   = *h0  - n;              // [bS, nU]
    auto dhdn   = 1.f - u;               // [bS, nU]
    auto dSigdu = u * (1.f - u);         // [bS, nU]
    auto dSigdr = r * (1.f - r);         // [bS, nU]
    auto dActdn = 1.f - n * n;           // [bS, nU]
    auto dndr   = mmul(dActdn * (*h0), WhnT);
    auto drdh0  = mmul(dSigdr, WhrT);

    auto dLdn = (*dLdh) * dhdn;
    auto dLdu = (*dLdh) * dhdu;
    auto dLdr = dLdn * dndr;

    dLdx->assign( mmul(dLdu * dSigdu, WxuT) + mmul(dLdr * dSigdr, WxrT) + mmul(dLdn * dActdn, WxnT) );      // [bS,iS]
    dLdh0->assign( mmul(dLdu * dSigdu, WhuT) + mmul(dLdn * dActdn * (r + drdh0), WhnT) + (*dLdh)*u );       // [bS,nU]

    dLdWxr.assign( mmul(xT, dSigdr * dLdr) );                                                               //  [iS,nU]
    dLdWhr.assign( mmul(h0T, dSigdr * dLdr) );                                                              //  [nU,nU]

    dLdWxu.assign( mmul(xT, dSigdu * dLdu) );                                                               //  [iS,nU]
    dLdWhu.assign( mmul(h0T, dSigdu * dLdu) );                                                              //  [nU,nU]

    dLdWxn.assign( mmul(xT, dActdn * dLdn) );                                                               //  [iS,nU]
    dLdWhn.assign( mmul((r*(*h0)).transpose(), dActdn * dLdn) );                                               //  [nU,nU]

    dLdbr.assign( (dSigdr * dLdr).reduceAlongDims(reduce::Sum, {0}));                          // [nU]
    dLdbu.assign( (dSigdu * dLdu).reduceAlongDims(reduce::Sum, {0}));                          // [nU]
    dLdbn.assign( (dActdn * dLdn).reduceAlongDims(reduce::Sum, {0}));                          // [nU]

    if(dLdWx0 != nullptr)
        *dLdWx += *dLdWx0;

    if(dLdWh0 != nullptr)
        *dLdWh += *dLdWh0;

    if(dLdb0 != nullptr)
        *dLdb += *dLdb0;

}


}
}
}

