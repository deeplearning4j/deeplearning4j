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
// (cf. https://arxiv.org/abs/1406.1078).
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
void gruCell(nd4j::LaunchContext * context, const NDArray* x, const NDArray* hLast, const NDArray* W, const NDArray* Wc,
             const NDArray* b, const NDArray* bc,
             NDArray* r, NDArray* u, NDArray* c, NDArray* h) {

    //Inputs:
    // x        input [bS, iS], iS - input size
    // hLast    previous cell output [bS, nU],  that is at previous time step t-1, nU - number of units
    // W        RU weights - [iS+nU, 2*nU] - reset and update gates
    // Wc       C weights - [iS+nU, nU] - cell gate
    // b        r and u biases, [2*nU] - reset and update gates
    // bc       c biases, [nU] - cell gate

    //Outputs:
    // r        Reset gate output [bS, nU]
    // u        Update gate output [bS, nU]
    // c        Cell gate output [bS, nU]
    // h        current cell output [bS, nU]

    /***************************************************************************************/
    /************************ THIS IS NOT OPTIMAZED CODE ***********************************/
    /** however it is more math-friendly and convenient for backprop formulas derivation) **/

    const int bS  = x->sizeAt(0);
    const int iS = x->sizeAt(1);
    const int nU = hLast->sizeAt(1);

    NDArray Wrx = (*W)({0,iS,     0,nU});       // [iS, nU]
    NDArray Wux = (*W)({0,iS,     nU,2*nU});    // [iS, nU]
    NDArray Wrh = (*W)({iS,iS+nU, 0,nU});       // [nU, nU]
    NDArray Wuh = (*W)({iS,iS+nU, nU,2*nU});    // [nU, nU]

    NDArray Wcx = (*Wc)({0,iS,     0,0});       // reset cell weights    [iS, nU]
    NDArray Wch = (*Wc)({iS,iS+nU, 0,0});       // updates cell weights  [nU, nU]

    NDArray br = (*b)({0,  nU});                // [nU]
    NDArray bu = (*b)({nU, 2*nU});              // [nU]

    // × means matrix multipication
    // * means element-wise product or so called Hadamard product

    // reset gate
    r->assign(mmul(*x, Wrx) + mmul(*hLast, Wrh) + br);         // [bS, iS] × [iS, nU] + [bS, nU] × [nU, nU] + [nU] = [bS, nU]
    r->applyTransform(transform::Sigmoid, *r);

    // update gate
    u->assign(mmul(*x, Wux) + mmul(*hLast, Wuh) + bu);         // [bS, iS] × [iS, nU] + [bS, nU] × [nU, nU] + [nU] = [bS, nU]
    u->applyTransform(transform::Sigmoid, *u);

    // cell gate c = activation(x × Wcx + (r * hlast) × Wch + bc)
    c->assign(mmul(*x, Wcx) + mmul(*r * *hLast, Wch) + *bc);    // [bS, iS] × [iS, nU] + [bS, nU] × [nU, nU] + [nU] = [bS, nU]
    c->applyTransform(transform::Tanh, *c);

    NDArray temp = 1.f - *c * *c;

    // cell output
    h->assign(*u * *hLast + (1.f - *u) * *c);


    /***************************************************************************************/
    /*************** THIS IS MORE OPTIMAZED CODE (should think about concat) ***************/
    /***************************************************************************************/
/*
    //Concat inputs: x + hLast : [bs, iS + nU]
    NDArray xhConcat(x->ordering(), {bS, iS + nU}, x->dataType(), context);  // concat([bs, iS], [bs, nU]) -> [bs, iS + nU]
    helpers::concat(context, {const_cast<NDArray*>(x), const_cast<NDArray*>(hLast)},  xhConcat, {1});

    //mmul for reset and update gates: (x × weight_ux + hLast × weight_xr + b_u)
    auto m = mmul(xhConcat, *W) + *b ;    // [bs, iS+nU] * [iS+nU, 2*nU] = [bs, 2*nU]
    // m += *bru;

    m.applyTransform(transform::Sigmoid);  //sigmoid(rz) and sigmoid(uz)

    r->assign(m({0,0, 0, nU}));
    u->assign(m({0,0, nU, 2*nU}));

    // hLast = hLast * r
    xhConcat({0,0, iS, iS+nU}) *= *r;

    //c = tanh(x × weight_cx + (hLast * r) × weight_cr + b_c)
    MmulHelper::mmul(&xhConcat, Wc, c, 1.0, 0.0);       //c = 1.0 * xhConcat * Wc + 0.0 * c
    *c += *bc;
    c->applyTransform(transform::Tanh);

    //Output: h = (1-u).*c + u .* hPrev
    //auto hResult = (*u) * (*hLast) + (1.0f - *u) * (*c); const_cast<NDArray*>(h)->assign(&hResult);
    u->applyPairwiseTransform(pairwise::Multiply, hLast, h, nullptr);        //h = u * hLast
    auto temp = (1.0f - *u);
    temp *= (*c);
    (*h) += temp;
*/
}

//////////////////////////////////////////////////////////////////////////
void gruTimeLoop(nd4j::LaunchContext * context, const NDArray* x, const NDArray* hLast, const NDArray* Wx, const NDArray* Wh, const NDArray* b, NDArray* h) {

    // x   input [time, bS, iS]
    // hLast  initial cell output (at time step = 0) [bS, nU]
    // Wx  input-to-hidden  weights, [iS, 3*nU]
    // Wh  hidden-to-hidden weights, [nU, 3*nU]
    // b   biases, [3*nU]

    // h is cell outputs at each time step [time, bS, nU]

    const int time = x->sizeAt(0);

    NDArray ht_1(*hLast);

    // loop through time steps
    for (int t = 0; t < time; ++t) {

        auto xt = (*x)({t,t+1, 0,0, 0,0});
        auto ht = (*h)({t,t+1, 0,0, 0,0});

        // helpers::gruCell(&xt, &ht_1, Wx, Wh, b, &ht);
        // ht_1.assign(ht);
    }
}

//////////////////////////////////////////////////////////////////////////
void gruCellBP(nd4j::LaunchContext* context,
              const NDArray* x,    const NDArray* hLast,
              const NDArray* W,    const NDArray* Wc,        const NDArray* b,    const NDArray* bc,
              const NDArray* dLdr, const NDArray* dLdu,      const NDArray* dLdc, const NDArray* dLdh,
                    NDArray* dLdx,       NDArray* dLdhLast,
                    NDArray* dLdW,       NDArray* dLdWc,
                    NDArray* dLdb,       NDArray* dLdbc) {

    //Inputs:
    // x              input [bS, iS]
    // hLast          previous cell output [bS, nU],  that is at previous time step t-1
    // W              weights - [iS+nU, 2*nU] - reset and update gates
    // Wc             C weights - [iS+nU, nU] - cell gate
    // b              r and u biases, [2*nU] - reset and update gates
    // bc             c biases, [nU] - cell gate
    // dLdr           gradient wrt reset gate, [bS, nU]
    // dLdu           gradient wrt update gate, [bS, nU]
    // dLdc           gradient wrt cell state, [bS, nU]
    // dLdh           gradient wrt current cell output, [bS, nU]

    //Outputs:
    // dLdx           gradient wrt x,  [bS, iS],
    // dLdhLast       gradient wrt hLast, [bS, nU]
    // dLdW           gradient wrt W,  [iS+nU, 2*nU]
    // dLdWc          gradient wrt Wc, [iS+nU, nU]
    // dLdb           gradient wrt bru [2*nU]
    // dLdbc          gradient wrt bc  [nU]

    // * means element-wise product or so called Hadamard product
    // × means matrix multiplication

    /************************************************************************************************/
    /******************************* THIS IS NOT OPTIMAZED CODE *************************************/
    /*** aim is to have math-readable code in order to keep track of backprop formulas derivation ***/

    const int bS  = x->sizeAt(0);
    const int iS = x->sizeAt(1);
    const int nU = hLast->sizeAt(1);

    NDArray xT     = x->transpose();            // [iS, bS]
    NDArray hLastT = hLast->transpose();        // [nU, bS]

    NDArray Wrx = (*W)({0,iS,     0,nU});       // [iS, nU]
    NDArray Wux = (*W)({0,iS,     nU,2*nU});    // [iS, nU]
    NDArray Wrh = (*W)({iS,iS+nU, 0,nU});       // [nU, nU]
    NDArray Wuh = (*W)({iS,iS+nU, nU,2*nU});    // [nU, nU]

    NDArray Wcx = (*Wc)({0,iS,     0,0});       // reset cell weights    [iS, nU]
    NDArray Wch = (*Wc)({iS,iS+nU, 0,0});       // updates cell weights  [nU, nU]

    NDArray br = (*b)({0,  nU});                // [nU]
    NDArray bu = (*b)({nU, 2*nU});              // [nU]

    NDArray WrxT = Wrx.transpose();             // [nU, iS]
    NDArray WuxT = Wux.transpose();             // [nU, iS]
    NDArray WrhT = Wrh.transpose();             // [nU, nU]
    NDArray WuhT = Wuh.transpose();             // [nU, nU]

    NDArray WcxT = Wcx.transpose();             // [nU, iS]
    NDArray WchT = Wch.transpose();             // [nU, nU]

    NDArray dLdWrx = (*dLdW)({0,iS,     0,nU});     // [iS, nU]
    NDArray dLdWux = (*dLdW)({0,iS,     nU,2*nU});  // [iS, nU]
    NDArray dLdWrh = (*dLdW)({iS,iS+nU, 0,nU});     // [nU, nU]
    NDArray dLdWuh = (*dLdW)({iS,iS+nU, nU,2*nU});  // [nU, nU]

    NDArray dLdWcx = (*dLdWc)({0,iS,     0,0});     // [iS, nU]
    NDArray dLdWch = (*dLdWc)({iS,iS+nU, 0,0});     // [nU, nU]

    NDArray dLdbr = (*dLdb)({0,  nU});              // [nU]
    NDArray dLdbu = (*dLdb)({nU, 2*nU});            // [nU]


    // ***** feed forward step ***** //

    // reset gate
    NDArray r = mmul(*x, Wrx) + mmul(*hLast, Wrh) + br;         // [bS, iS] × [iS, nU] + [bS, nU] × [nU, nU] + [nU] = [bS, nU]
    r.applyTransform(transform::Sigmoid, r);

    // update gate
    NDArray u = mmul(*x, Wux) + mmul(*hLast, Wuh) + bu;         // [bS, iS] × [iS, nU] + [bS, nU] × [nU, nU] + [nU] = [bS, nU]
    u.applyTransform(transform::Sigmoid, u);

    // cell gate c = activation(x×Wcx + (r*hlast)×Wcu + bc)
    NDArray c = mmul(*x, Wcx) + mmul(r * *hLast, Wch) + *bc;    // [bS, iS] × [iS, nU] + [bS, nU] × [nU, nU] + [nU] = [bS, nU]
    c.applyTransform(transform::Tanh, c);

    // h = (1 - u) * c + u * hPrev


    // ***** back prop step ***** //

    // notations:
    // Zr = x × Wrx + hLast × Wrh + br
    // Zu = x × Wux + hLast × Wuh + bu
    // Sr = sigmoid(Zr)
    // Su = sigmoid(Zu)
    // Zc = x × Wcx + (r * hlast) × Wch + bc


    // dLdx = dLdh * dhdx = dLdh * (dhdu * dudx + dhdc * dcdx) = (dLdh * dhdu) * dudx + (dLdh * dhdc) * dcdx = dLdu * dudx + dLdc * dcdx
    //      = dLdx_u + dLdx_c
    // dLdx_u = dLdu * dudx = dLdu * dudZu * dZudx = |dZudx = ... × WuxT| = (dLdu * dudZu) × WuxT
    // dLdx_c = dLdc * dcdx = dLdc * dcdZc * (dZcdx + dZcdr * drdx) = dLdc * dcdZc * dZcdx + dLdc * dcdZc * dZcdr * drdx = dLdx_c0 + dLdx_c1
    // dLdx_c0 = dLdc * dcdZc * dZcdx = |dZcdx = ... × WcxT| = (dLdc * dcdZc) × WcxT
    // dZcdr = (... * hLast) × WchT
    // dLdc * dcdZc * dZcdr = dLdr = (dLdc * dcdZc * hLast) × WchT
    // drdx = drdZr * dZrdx
    // dZrdx = ... × WrxT
    // dLdx_c1 = dLdc * dcdZc * dZcdr * drdx = dLdr * drdx = (dLdr * drdZr) × WrxT
    // finally dLdx = dLdx_u + dLdx_c0 + dLdx_c1 = (dLdu * dudZu) × WuxT + (dLdc * dcdZc) × WcxT + (dLdr * drdZr) × WrxT


    // dLdhLast    = dLdh * (dhdhLast + dhdu * dudhLast + dhdc * dcdhLast) = dLdh * dhdhLast + dLdu * dudhLast + dLdc * dcdhLast
    //             = dLdhLast_h + dLdhLast_u + dLdhLast_c
    // dLdhLast_h  = dLdh * dhdhLas = dLdh * u
    // dLdhLast_u  = dLdu * dudhLast = |dudhLast = dudZu * dZudhLast , dZudhLast = ... × WuhT| = (dLdu * dudZu) × WuhT
    // dLdhLast_c  = dLdc * dcdhLast  = dLdc * (dcdZc * dZcdhLast + dcdZc * dZcdr * drdhLast) =
    //             = dLdc * dcdZc * dZcdhLast + dLdc * dcdZc * dZcdr * drdhLast =
    //             = dLdc * dcdZc * dZcdhLast + dLdr * drdhLast = dLdhLast_c0 + dLdhLast_c1
    // dLdhLast_c0 = dLdc * dcdZc * dZcdhLast = |dZcdhLast = (... * r) × WchT| = (dLdc * dcdZc * r) × WchT
    // dLdhLast_c1 = dLdr * drdhLast = |drdhLast  = drdZr * dZrdhLast, dZrdhLast = ... × WrhT| = (dLdr * drdZr) × WrhT
    // finally dLdhLast = dLdhLast_h + dLdhLast_u + dLdhLast_c0 + dLdhLast_c1 =
    //                  = dLdh * u + (dLdu * dudZu) × WuhT + (dLdc * dcdZc * r) × WchT + (dLdr * drdZr) × WrhT


    // dLdWrx = dLdh * dhdWrx = (dLdh * dhdc) * dcdWrx = dLdc * dcdZc * dZcdWrx = dLdc * dcdZc * dZcdr * drdWrx =
    //        = dLdc * dcdZc * dZcdr * drdZr * dZrdWrx = dLdr * drdZr * dZrdWrx
    // dZrdWrx = xT × ...
    // finally dLdWrx = xT × (dLdr * drdZr)


    // dLdWrh = dLdh * dhdWrh = (dLdh * dhdc) * dcdWrh = dLdc * dcdZc * dZcdWrh = dLdc * dcdZc * dZcdr * drdWrh =
    //        = dLdc * dcdZc * dZcdr * drdZr * dZrdWrh = dLdr * drdZr * dZrdWrh
    // dZrdWrh = hLastT × ...
    // finally dLdWrh = hLastT × (dLdr * drdZr)


    // dLdWux = dLdh * dhdWux = (dLdh * dhdu) * dudWux = dLdu * dudZu * dZudWux
    // dZudWux = xT × ...
    // dLdu * dudZu * dZudWux = xT × (dLdu * dudZu)


    // dLdWuh = dLdh * dhdWuh = (dLdh * dhdu) * dudWuh = dLdh * dhdu * dudZu * dZudWuh = dLdu * dudZu * dZudWuh
    // dZudWuh = hLastT × ...
    // finally dLdWuh = hLastT × (dLdu * dudZu)


    // dLdWcx = dLdh * dhdWcx = dLdh * dhdc * dcdWcx = (dLdh * dhdc) * dcdZc * dZcdWcx = dLdc * dcdZc * dZcdWcx
    // dZcdWcx = xT × ...
    // finally dLdWcx = xT × (dLdc * dcdZc)


    // dLdWch = dLdh * dhdWch = dLdh * dhdc * dcdWch = (dLdh * dhdc) * dcdZc * dZcdWch = dLdc * dcdZc * dZcdWch
    // dZcdWch = (r*hLast)^T × ...
    // finally dLdWch = (r*hLast)^T × (dLdc * dcdZc)


    // dLdbr = dLdh * dhdbr = (dLdh * dhdc) * dcdbr = dLdc * dcdbr = dLdc * dcdZc * dZcdbr = dLdc * dcdZc * dZcdr * drdbr =
    //       = dLdr * drdZr * dZrdbr
    // dZrdbr = 1
    // finally dLdbr = dLdr * drdZr


    // dLdbu = dLdh * dhdbu = (dLdh * dhdu) * dudbu = dLdu * dudZu * dZudbu
    // dZudbu = 1
    // finally dLdbu = dLdu * dudZu


    // dLdbc = dLdh * dhdbc = (dLdh * dhdc) * dcdbc = dLdc * dcdZc * dZcdbc
    // dZcdbc = 1
    // finally dLdbc = dLdc * dcdZc

    NDArray dhdc  = 1.f - u;           // [bS, nU]
    NDArray dhdu  = *hLast - c;        // [bS, nU]
    NDArray dudZu = u * dhdc;          // [bS, nU]
    NDArray drdZr = r * (1.f - r);     // [bS, nU]
    NDArray dcdZc = 1.f - c * c;       // [bS, nU]
    NDArray dLdZc = *dLdc * dcdZc;     // [bS, nU]
    NDArray dLdZu = *dLdu * dudZu;     // [bS, nU]
    NDArray dLdZr = *dLdr * drdZr;     // [bS, nU]

    // NDArray dLdc  = *dLdh * dhdc;                       // [bS, nU]
    // NDArray dLdu  = *dLdh * dhdu;                       // [bS, nU]
    // NDArray dLdr  = mmul(dLdc * dcdZc * *hLast, WchT);  // [bS, nU]

    dLdx->assign(mmul(dLdZu, WuxT) + mmul(dLdZc, WcxT) + mmul(dLdZr, WrxT));                        // [bS, iS]

    dLdhLast->assign(*dLdh * u + mmul(dLdZu, WuhT) + mmul(dLdZc * r, WchT) + mmul(dLdZr, WrhT));    // [bS, nU]

    dLdWrx.assign(mmul(xT,     dLdZr));     // [iS, bS] × [bS, nU] = [iS, nU]
    dLdWrh.assign(mmul(hLastT, dLdZr));     // [nU, bS] × [bS, nU] = [nU, nU]
    dLdWux.assign(mmul(xT,     dLdZu));     // [iS, bS] × [bS, nU] = [iS, nU]
    dLdWuh.assign(mmul(hLastT, dLdZu));     // [nU, bS] × [bS, nU] = [nU, nU]

    dLdWcx.assign(mmul(xT, dLdZc));                          // [iS, bS] × [bS, nU] = [iS, nU]
    dLdWch.assign(mmul((r * *hLast).transpose(), dLdZc));    // [nU, bS] × [bS, nU] = [nU, nU]

    dLdbr.assign(dLdZr.reduceAlongDimension(reduce::Sum, {0}));  // [nU]
    dLdbu.assign(dLdZu.reduceAlongDimension(reduce::Sum, {0}));  // [nU]

    dLdbc->assign(dLdZc.reduceAlongDimension(reduce::Sum, {0})); // [nU]
}


}
}
}

