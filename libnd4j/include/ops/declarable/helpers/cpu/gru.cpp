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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.02.2018, Alex Black
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
    // x        input [bS, nIn], nIn - input size
    // hLast    previous cell output [bS, nUn],  that is at previous time step t-1, nUn - number of units
    // Wru      RU weights - [nIn+nUn, 2*nUn] - reset and update gates
    // Wc       C weights - [nIn+nUn, nUn] - cell gate
    // bru      r and u biases, [2*nUn] - reset and update gates
    // bc       c biases, [nUn] - cell gate

    //Outputs:
    // r        Reset gate output [bS, nUn]
    // u        Update gate output [bS, nUn]
    // c        Cell gate output [bS, nUn]
    // h        current cell output [bS, nUn]

    /***************************************************************************************/
    /************************ THIS IS NOT OPTIMAZED CODE ***********************************/
    /** however it is more math-friendly and convenient for backprop formulas derivation) **/

    const int bS  = x->sizeAt(0);
    const int nIn = x->sizeAt(1);
    const int nUn = hLast->sizeAt(1);

    NDArray Wr = (*Wru)({0,nIn,       0,0});       // reset gates weights   [nIn, 2*nUn]
    NDArray Wu = (*Wru)({nIn,nIn+nUn, 0,0});       // updates gates weights [nUn, 2*nUn]

    NDArray Wcr = (*Wc)({0,nIn,       0,0});       // reset cell weights    [nIn, nUn]
    NDArray Wcu = (*Wc)({nIn,nIn+nUn, 0,0});       // updates cell weights  [nUn, nUn]

    // gates = sigmoid(x*Wr + hLast*Wu + br + bu)
    NDArray gates = mmul(*x, Wr) + mmul(*hLast, Wu) + *bru;    // [bS, nIn] * [nIn, 2*nUn] + [bS, nUn] * [nUn, 2*nUn] + [2*nUn] = [bS, 2*nUn]
    gates.applyTransform(transform::Sigmoid);

    // reset gate
    r->assign(gates({0,0, 0,nUn}));               // [bS, nUn]

    // update gate
    u->assign(gates({0,0, nUn,2*nUn}));            // [bS, nUn]

    // cell gate c = activation(x*Wcr + (r◦hlast)*Wcu + bc)
    c->assign(mmul(*x, Wcr) + mmul(*r * *hLast, Wcu) + *bc);    // [bS, nIn] * [nIn, nUn] + [bS, nUn] * [nUn, nUn] + [nUn] = [bS, nUn]
    c->applyTransform(transform::Tanh);

    // cell output
    h->assign(*u * *hLast + (1.f - *u) * *c);




    /***************************************************************************************/
    /********************** THIS MORE OPTIMAZED CODE (except concat ) **********************/
    /***************************************************************************************/
/*
    //Concat inputs: x + hLast : [bs, nIn + nUn]
    NDArray xhConcat(x->ordering(), {bS, nIn + nUn}, x->dataType(), context);  // concat([bs, nIn], [bs, nUn]) -> [bs, nIn + nUn]
    helpers::concat(context, {const_cast<NDArray*>(x), const_cast<NDArray*>(hLast)},  xhConcat, {1});

    //mmul for reset and update gates: (x * weight_ux + hLast * weight_xr + b_u)
    auto m = mmul(xhConcat, *Wru) + *bru ;    // [bs, nIn+nUn] * [nIn+nUn, 2*nUn] = [bs, 2*nUn]
    // m += *bru;

    sigmoidInplace(m);  //sigmoid(rz) and sigmoid(uz)

    r->assign(m({0,0, 0, nUn}));
    u->assign(m({0,0, nUn, 2*nUn}));

    // hLast = hLast * r
    xhConcat({0,0, nIn, nIn+nUn}) *= *r;

    //c = tanh(x * weight_cx + (hLast .* r) * weight_cr + b_c)
    MmulHelper::mmul(&xhConcat, Wc, c, 1.0, 0.0);       //c = 1.0 * xhConcat * Wc + 0.0 * c
    *c += *bc;
    tanhInplace(*c);

    //Output: h = (1-u).*c + u .* hPrev
    //auto hResult = (*u) * (*hLast) + (1.0f - *u) * (*c); const_cast<NDArray*>(h)->assign(&hResult);
    u->applyPairwiseTransform(pairwise::Multiply, hLast, h, nullptr);        //h = u * hLast
    auto temp = (1.0f - *u);
    temp *= (*c);
    (*h) += temp;
*/
}

//////////////////////////////////////////////////////////////////////////
void gruTimeLoop(nd4j::LaunchContext * context, const NDArray* x, const NDArray* h0, const NDArray* Wx, const NDArray* Wh, const NDArray* b, NDArray* h) {

    // x   input [time, bS, iS]
    // h0  initial cell output (at time step = 0) [bS, nUn]
    // Wx  input-to-hidden  weights, [iS, 3*nUn]
    // Wh  hidden-to-hidden weights, [nUn, 3*nUn]
    // b   biases, [3*nUn]

    // h is cell outputs at each time step [time, bS, nUn]

    const int time = x->sizeAt(0);

    NDArray ht_1(*h0);

    // loop through time steps
    for (int t = 0; t < time; ++t) {

        auto xt = (*x)({t,t+1, 0,0, 0,0});
        auto ht = (*h)({t,t+1, 0,0, 0,0});

        //helpers::gruCell(&xt, &ht_1, Wx, Wh, b, &ht);
        //ht_1.assign(ht);
    }
}

//////////////////////////////////////////////////////////////////////////
void gruCellBP(nd4j::LaunchContext * context, const NDArray* x, const NDArray* h0, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* dLdh, const NDArray* dLdWx0,
           const NDArray* dLdWh0, const NDArray* dLdb0, NDArray* dLdx, NDArray* dLdh0, NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb) {

    // x                        input [bS, iS]
    // h0                       previous cell output [bS, nUn],  that is at previous time step t-1
    // Wx                       input-to-hidden  weights, [iS, 3*nUn]
    // Wh                       hidden-to-hidden weights, [nUn, 3*nUn]
    // b                        biases, [3*nUn]
    // dLdh                     gradient wrt output, [bS,nUn], that is epsilon_next
    // dLdWx0                   gradient wrt Wx at previous time step, [iS, 3*nUn]
    // dLdWh0                   gradient wrt Wh at previous time step, [nUn, 3*nUn]
    // dLdb0                    gradient wrt b at previous time step,  [3*nUn]

    // dLdx                   gradient wrt x,  [bS, iS], that is epsilon
    // dLdh0                  gradient wrt h0, [bS, nUn]
    // dLdWx                  gradient wrt Wx, [iS, 3*nUn]
    // dLdWh                  gradient wrt Wh, [nUn, 3*nUn]
    // dLdb                   gradient wrt b at previous time step,  [3*nUn]

    // h is current cell output [bS, nUn], that is at current time step t

    const int nUn = h0->sizeAt(1);

    // ***** feed forward step ***** //
    // gates = sigmoid(x*Wx + h0*Wh + b)
    auto gates = sigmoid(mmul(*x, (*Wx)({0,0, 0,2*nUn})) + mmul(*h0, (*Wh)({0,0, 0,2*nUn})) + (*b)({0,2*nUn}));       // [bS, 2*nUn] + [bS, 2*nUn] + [1, 2*nUn] = [bS, 2*nUn]
    // reset gate
    auto r = gates({0,0, 0, nUn});               // [bS, nUn]
    // update gate
    auto u = gates({0,0, nUn, 2*nUn});            // [bS, nUn]
    // ◦ means element-wise product or so called Hadamard product
    // n = tanh(x*Wx + (r◦h0)*Wh + b)
    auto n = tanh(mmul(*x, (*Wx)({0,0, 2*nUn,3*nUn})) + mmul((*h0)*r, (*Wh)({0,0, 2*nUn,3*nUn})) + (*b)({2*nUn,3*nUn}));     // [bS, nUn]

    // ***** back prop step ***** //
    auto Wxr  = (*Wx)({0,0, 0,   nUn});
    auto Wxu  = (*Wx)({0,0, nUn,  2*nUn});
    auto Wxn  = (*Wx)({0,0, 2*nUn,3*nUn});
    auto Whr  = (*Wh)({0,0, 0,   nUn});
    auto Whu  = (*Wh)({0,0, nUn,  2*nUn});
    auto Whn  = (*Wh)({0,0, 2*nUn,3*nUn});
    auto WxrT = Wxr.transpose();
    auto WxuT = Wxu.transpose();
    auto WxnT = Wxn.transpose();
    auto WhrT = Whr.transpose();
    auto WhuT = Whu.transpose();
    auto WhnT = Whn.transpose();
    auto xT   = x->transpose();
    auto h0T  = h0->transpose();

    auto dLdWxr = (*dLdWx)({0,0, 0,     nUn});
    auto dLdWxu = (*dLdWx)({0,0, nUn,  2*nUn});
    auto dLdWxn = (*dLdWx)({0,0, 2*nUn,3*nUn});

    auto dLdWhr = (*dLdWh)({0,0, 0,     nUn});
    auto dLdWhu = (*dLdWh)({0,0, nUn,  2*nUn});
    auto dLdWhn = (*dLdWh)({0,0, 2*nUn,3*nUn});

    auto dLdbr = (*dLdb)({0,     nUn});
    auto dLdbu = (*dLdb)({nUn,  2*nUn});
    auto dLdbn = (*dLdb)({2*nUn,3*nUn});

    auto dhdu   = *h0  - n;              // [bS, nUn]
    auto dhdn   = 1.f - u;               // [bS, nUn]
    auto dSigdu = u * (1.f - u);         // [bS, nUn]
    auto dSigdr = r * (1.f - r);         // [bS, nUn]
    auto dActdn = 1.f - n * n;           // [bS, nUn]
    auto dndr   = mmul(dActdn * (*h0), WhnT);
    auto drdh0  = mmul(dSigdr, WhrT);

    auto dLdn = (*dLdh) * dhdn;
    auto dLdu = (*dLdh) * dhdu;
    auto dLdr = dLdn * dndr;

    dLdx->assign( mmul(dLdu * dSigdu, WxuT) + mmul(dLdr * dSigdr, WxrT) + mmul(dLdn * dActdn, WxnT) );      // [bS,iS]
    dLdh0->assign( mmul(dLdu * dSigdu, WhuT) + mmul(dLdn * dActdn * (r + drdh0), WhnT) + (*dLdh)*u );       // [bS,nUn]

    dLdWxr.assign( mmul(xT, dSigdr * dLdr) );                                                               //  [iS,nUn]
    dLdWhr.assign( mmul(h0T, dSigdr * dLdr) );                                                              //  [nUn,nUn]

    dLdWxu.assign( mmul(xT, dSigdu * dLdu) );                                                               //  [iS,nUn]
    dLdWhu.assign( mmul(h0T, dSigdu * dLdu) );                                                              //  [nUn,nUn]

    dLdWxn.assign( mmul(xT, dActdn * dLdn) );                                                               //  [iS,nUn]
    dLdWhn.assign( mmul((r*(*h0)).transpose(), dActdn * dLdn) );                                               //  [nUn,nUn]

    dLdbr.assign( (dSigdr * dLdr).reduceAlongDims(reduce::Sum, {0}));                          // [nUn]
    dLdbu.assign( (dSigdu * dLdu).reduceAlongDims(reduce::Sum, {0}));                          // [nUn]
    dLdbn.assign( (dActdn * dLdn).reduceAlongDims(reduce::Sum, {0}));                          // [nUn]

    if(dLdWx0 != nullptr)
        *dLdWx += *dLdWx0;

    if(dLdWh0 != nullptr)
        *dLdWh += *dLdWh0;

    if(dLdb0 != nullptr)
        *dLdb += *dLdb0;

}

// //////////////////////////////////////////////////////////////////////////
// FIXME - gruTimeLoopBP is not correct
// template <typename T>
// void gruTimeLoopBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs) {

//     NDArray<T>* x      = inArrs[0];                   // input [time, bS, iS]
//     NDArray<T>* hi     = inArrs[1];                   // previous/initial cell output [bS, nUn],  that is at previous time step t-1
//     NDArray<T>* Wx     = inArrs[2];                   // input-to-hidden  weights, [iS, 3*nUn]
//     NDArray<T>* Wh     = inArrs[3];                   // hidden-to-hidden weights, [nUn, 3*nUn]
//     NDArray<T>* b      = inArrs[4];                   // biases, [3*nUn]
//     NDArray<T>* dLdh   = inArrs[5];                   // gradient wrt output, [time, bS, nUn], that is epsilon_next

//     NDArray<T>* dLdx   = outArrs[0];                  // gradient wrt x,  [time, bS, iS], that is epsilon
//     NDArray<T>* dLdhi  = outArrs[1];                  // gradient wrt hi, [bS, nUn]
//     NDArray<T>* dLdWx  = outArrs[2];                  // gradient wrt Wx, [iS, 3*nUn]
//     NDArray<T>* dLdWh  = outArrs[3];                  // gradient wrt Wh, [nUn, 3*nUn]
//     NDArray<T>* dLdb   = outArrs[4];                  // gradient wrt b,  [3*nUn]

//     const Nd4jLong time = x->sizeAt(0);
//     const Nd4jLong bS   = x->sizeAt(1);
//     const Nd4jLong iS   = x->sizeAt(2);
//     const Nd4jLong nUn   = hi->sizeAt(1);

//     NDArray<T> h(hi->ordering(), {time, bS, nUn});      // feed forward output

//     // first step, time = 0, feed forward
//     NDArray<T> x0 = (*x)({{0,1}, {}, {}});
//     NDArray<T> h0 = h({{0,1}, {}, {}});
//     helpers::gruCell<T>({&x0, hi, Wx, Wh, b}, &h0);

//     // first step, time = 0, back prop
//     NDArray<T> dLdx0 = (*dLdx)({{0,1}, {}, {}});
//     NDArray<T> dLdh0 = (*dLdh)({{0,1}, {}, {}});
//     helpers::gruCellBP<T>({&x0, hi, Wx, Wh, b, &dLdh0, nullptr, nullptr, nullptr}, {&dLdx0, dLdhi, dLdWx, dLdWh, dLdb});

//     // loop through the rest time steps
//     for (Nd4jLong t = time-1; t > 0; --t) {
//     for (Nd4jLong t = 1; t < time; ++t) {

//         NDArray<T> xt    =    (*x)({{t,t+1}, {}, {}});
//         NDArray<T> ht    =       h({{t,t+1}, {}, {}});
//         NDArray<T> ht_1  =       h({{t-1,t}, {}, {}});
//         NDArray<T> dLdxt = (*dLdx)({{t,t+1}, {}, {}});
//         NDArray<T> dLdht = (*dLdh)({{t,t+1}, {}, {}});

//         NDArray<T> dLdWxt_1 = dLdWx;
//         NDArray<T> dLdWht_1 = dLdWh;
//         NDArray<T> dLdbt_1  = dLdb;

//         // feed forward, calculation of ht
//         helpers::gruCell<T>({&xt, &ht_1, Wx, Wh, b}, &ht);

//         // back prop
//         helpers::gruCellBP<T>({&xt, &ht_1, Wx, Wh, b, &dLdht, &dLdWxt_1, &dLdWht_1, &dLdbt_1}, {&dLdxt, nullptr, dLdWx, dLdWh, dLdb});
//     }
// }


}
}
}

