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

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
static FORCEINLINE NDArray sigmoid(const NDArray& arr) {
    return (const_cast<NDArray&>(arr)).transform(transform::Sigmoid);
}

//////////////////////////////////////////////////////////////////////////
static FORCEINLINE NDArray activation(const NDArray& arr) {
    return (const_cast<NDArray&>(arr)).transform(transform::Tanh);
}


//////////////////////////////////////////////////////////////////////////
void gruCell(const std::vector<NDArray*>& inArrs, NDArray* h) {

    auto x  = inArrs[0];                   // input [bS, iS], bS - batch size, iS - input size
    auto h0 = inArrs[1];                   // previous cell output [bS, nU],  that is at previous time step t-1

    auto Wx = inArrs[2];                   // input-to-hidden  weights, [iS, 3*nU]
    auto Wh = inArrs[3];                   // hidden-to-hidden weights, [nU, 3*nU]
    auto b  = inArrs[4];                   // biases, [3*nU]
    
    // h is current cell output [bS, nU], that is at current time step t    

    const int nU = h0->sizeAt(1);                // number of units
    
    // gates = sigmoid(x*Wx + h0*Wh + b)
    auto gates = sigmoid(mmul(*x, (*Wx)({0,0, 0,2*nU})) + mmul(*h0, (*Wh)({0,0, 0,2*nU})) + (*b)({0,2*nU}));       // [bS, 2*nU] + [bS, 2*nU] + [1, 2*nU] = [bS, 2*nU]
    
    // reset gate
    auto r = gates({0,0, 0,nU});                     // [bS, nU]

    // update gate
    auto u = gates({0,0, nU,2*nU});            // [bS, nU]

    // ◦ means element-wise product or so called Hadamard product
    // n = activation(x*Wx + (r◦h0)*Wh + b)
    auto n = activation(mmul(*x, (*Wx)({0,0, 2*nU,3*nU})) + mmul((*h0)*r, (*Wh)({0,0, 2*nU,3*nU})) + (*b)({2*nU,3*nU}));     // [bS, nU]

    // current cell output
    h->assign( u * (*h0) + (1.f - u) * n );
}

//////////////////////////////////////////////////////////////////////////
void gruTimeLoop(const std::vector<NDArray*>& inArrs, NDArray* h) {

    auto x  = inArrs[0];                   // input [time, bS, iS]
    auto h0 = inArrs[1];                   // initial cell output (at time step = 0) [bS, nU]

    auto Wx = inArrs[2];                   // input-to-hidden  weights, [iS, 3*nU]
    auto Wh = inArrs[3];                   // hidden-to-hidden weights, [nU, 3*nU]
    auto b  = inArrs[4];                   // biases, [3*nU]
    
    // h is cell outputs at each time step [time, bS, nU]

    const int time = x->sizeAt(0);    

    NDArray ht_1(*h0);

    // loop through time steps
    for (int t = 0; t < time; ++t) {

        auto xt = (*x)({t,t+1, 0,0, 0,0});
        auto ht = (*h)({t,t+1, 0,0, 0,0});

        helpers::gruCell({&xt, &ht_1, Wx, Wh, b}, &ht);
        ht_1.assign(ht);    
    }
}

//////////////////////////////////////////////////////////////////////////
void gruCellBP(const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs) {

    auto x      = inArrs[0];                   // input [bS, iS]
    auto h0     = inArrs[1];                   // previous cell output [bS, nU],  that is at previous time step t-1
    auto Wx     = inArrs[2];                   // input-to-hidden  weights, [iS, 3*nU]
    auto Wh     = inArrs[3];                   // hidden-to-hidden weights, [nU, 3*nU]
    auto b      = inArrs[4];                   // biases, [3*nU]
    auto dLdh   = inArrs[5];                   // gradient wrt output, [bS,nU], that is epsilon_next
    auto dLdWx0 = inArrs[6];                   // gradient wrt Wx at previous time step, [iS, 3*nU]
    auto dLdWh0 = inArrs[7];                   // gradient wrt Wh at previous time step, [nU, 3*nU]
    auto dLdb0  = inArrs[8];                   // gradient wrt b at previous time step,  [3*nU]

    auto dLdx   = outArrs[0];                  // gradient wrt x,  [bS, iS], that is epsilon
    auto dLdh0  = outArrs[1];                  // gradient wrt h0, [bS, nU]
    auto dLdWx  = outArrs[2];                  // gradient wrt Wx, [iS, 3*nU]
    auto dLdWh  = outArrs[3];                  // gradient wrt Wh, [nU, 3*nU]
    auto dLdb   = outArrs[4];                  // gradient wrt b at previous time step,  [3*nU]
    
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
    // n = activation(x*Wx + (r◦h0)*Wh + b)
    auto n = activation(mmul(*x, (*Wx)({0,0, 2*nU,3*nU})) + mmul((*h0)*r, (*Wh)({0,0, 2*nU,3*nU})) + (*b)({2*nU,3*nU}));     // [bS, nU]

    // ***** back prop step ***** // 
    auto Wxr  = (*Wx)({0,0, 0,   nU});
    auto Wxu  = (*Wx)({0,0, nU,  2*nU});
    auto Wxn  = (*Wx)({0,0, 2*nU,3*nU});
    auto Whr  = (*Wh)({0,0, 0,   nU});
    auto Whu  = (*Wh)({0,0, nU,  2*nU});
    auto Whn  = (*Wh)({0,0, 2*nU,3*nU});
    auto WxrT = Wxr.transp();
    auto WxuT = Wxu.transp();
    auto WxnT = Wxn.transp();
    auto WhrT = Whr.transp();
    auto WhuT = Whu.transp();
    auto WhnT = Whn.transp();
    auto xT   = x->transp();
    auto h0T  = h0->transp();

    auto dLdWxr = (*dLdWx)({0,0, 0,     nU});
    auto dLdWxu = (*dLdWx)({0,0, nU,  2*nU});
    auto dLdWxn = (*dLdWx)({0,0, 2*nU,3*nU});

    auto dLdWhr = (*dLdWh)({0,0, 0,     nU});
    auto dLdWhu = (*dLdWh)({0,0, nU,  2*nU});
    auto dLdWhn = (*dLdWh)({0,0, 2*nU,3*nU});

    auto dLdbr = (*dLdb)({0,     nU});
    auto dLdbu = (*dLdb)({nU,  2*nU});
    auto dLdbn = (*dLdb)({2*nU,3*nU});

    auto dhdu   = *h0  - n;               // [bS, nU]
    auto dhdn   = 1.0f - u;               // [bS, nU]
    auto dSigdu = u * (1.0f - u);         // [bS, nU]
    auto dSigdr = r * (1.0f - r);         // [bS, nU]
    auto dActdn = 1.0f - n * n;           // [bS, nU]
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
    dLdWhn.assign( mmul((r*(*h0)).transp(), dActdn * dLdn) );                                               //  [nU,nU]
    
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

