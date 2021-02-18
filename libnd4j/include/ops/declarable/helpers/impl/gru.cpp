/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 15.02.2018, Alex Black
//

// implementation of gated Recurrent Unit cell
// (cf. https://arxiv.org/abs/1406.1078).
// Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
// "Learning Phrase Representations using RNN Encoder-Decoder for StatnIntical Machine Translation"


#include <ops/declarable/helpers/gru.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>
#include <helpers/MmulHelper.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
void gruCell(sd::LaunchContext * context, const NDArray* x, const NDArray* hI, const NDArray* W, const NDArray* Wc,
             const NDArray* b, const NDArray* bc,
             NDArray* r, NDArray* u, NDArray* c, NDArray* h) {

    //Inputs:
    // x        input [bS, nIn], nIn - input size
    // hI       previous cell output [bS, nOut],  that is at previous time step t-1, nOut - number of units
    // W        RU weights - [nIn+nOut, 2*nOut] - reset and update gates
    // Wc       C weights - [nIn+nOut, nOut] - cell gate
    // b        r and u biases, [2*nOut] - reset and update gates
    // bc       c biases, [nOut] - cell gate

    //Outputs:
    // r        Reset gate output [bS, nOut]
    // u        Update gate output [bS, nOut]
    // c        Cell gate output [bS, nOut]
    // h        current cell output [bS, nOut]

    /***************************************************************************************/
    /************************ THIS IS NOT OPTIMAZED CODE ***********************************/
    /** however it is more math-friendly and convenient for backprop formulas derivation) **/

    const int bS  = x->sizeAt(0);
    const int nIn = x->sizeAt(1);
    const int nOut = hI->sizeAt(1);

    NDArray Wrx = (*W)({0,nIn,     0,nOut});       // [nIn, nOut]
    NDArray Wux = (*W)({0,nIn,     nOut,2*nOut});    // [nIn, nOut]
    NDArray Wrh = (*W)({nIn,nIn+nOut, 0,nOut});       // [nOut, nOut]
    NDArray Wuh = (*W)({nIn,nIn+nOut, nOut,2*nOut});    // [nOut, nOut]

    NDArray Wcx = (*Wc)({0,nIn,     0,0});       // reset cell weights    [nIn, nOut]
    NDArray Wch = (*Wc)({nIn,nIn+nOut, 0,0});       // updates cell weights  [nOut, nOut]

    NDArray br = (*b)({0,  nOut});                // [nOut]
    NDArray bu = (*b)({nOut, 2*nOut});              // [nOut]

    // × means matrix multipication
    // * means element-wise product or so called Hadamard product

    // reset gate
    r->assign(mmul(*x, Wrx) + mmul(*hI, Wrh) + br);         // [bS, nIn] × [nIn, nOut] + [bS, nOut] × [nOut, nOut] + [nOut] = [bS, nOut]
    r->applyTransform(transform::Sigmoid, *r);

    // update gate
    u->assign(mmul(*x, Wux) + mmul(*hI, Wuh) + bu);         // [bS, nIn] × [nIn, nOut] + [bS, nOut] × [nOut, nOut] + [nOut] = [bS, nOut]
    u->applyTransform(transform::Sigmoid, *u);

    // cell gate c = activation(x × Wcx + (r * hlast) × Wch + bc)
    c->assign(mmul(*x, Wcx) + mmul(*r * *hI, Wch) + *bc);    // [bS, nIn] × [nIn, nOut] + [bS, nOut] × [nOut, nOut] + [nOut] = [bS, nOut]
    c->applyTransform(transform::Tanh, *c);

    // cell output
    h->assign(*u * *hI + (1.f - *u) * *c);
}

//////////////////////////////////////////////////////////////////////////
void gruCell(sd::LaunchContext * context, const NDArray* x, const NDArray* hI, const NDArray* Wx, const NDArray* Wh, const NDArray* b,
             NDArray* gates, NDArray* h) {

    //Inputs:
    // x        input [bS, nIn]
    // hI       previous cell output [bS, nOut],  that is at previous time step t-1
    // Wx       weights for x - [nIn, 3*nOut]
    // Wh       weights for h - [nOut, 3*nOut]
    // b        biases [3*nOut]

    // 3*nOut means following sequence: reset, update, cell

    //Outputs:
    // gates    [bS, 3*nOut] = reset gate [bS, nOut] + update gate [bS, nOut] + cell gate [bS, nOut]
    // h        current cell output [bS, nOut]

    // formulas:
    // zr = x × Wxr + hI × Whr + br
    // zu = x × Wxu + hI × Whu + bu
    // r = sigmoid(zr)
    // u = sigmoid(zu)
    // zc = x × Wxc + (r * hI) × Whc + bc
    // c = tanh(zc)
    // h = (1-u)*c + u*hI

    const int bS  = x->sizeAt(0);
    const int nIn = x->sizeAt(1);
    const int nOut = hI->sizeAt(1);

    NDArray temp = gates->ulike();
    MmulHelper::mmul(x, Wx, &temp);                 // [bS, nIn] × [nIn, 3*nOut] = [bS, 3*nOut]
    temp += *b;

    MmulHelper::mmul(hI, Wh, gates);                // [bS, nOut] × [nOut, 3*nOut] = [bS, 3*nOut]

    NDArray ru = (*gates)({0,0, 0,2*nOut});             // [bS, 2*nOut]

    NDArray r  = (*gates)({0,0, 0,nOut});               // [bS, nOut]
    NDArray u  = (*gates)({0,0, nOut,2*nOut});          // [bS, nOut]
    NDArray c  = (*gates)({0,0, 2*nOut,3*nOut});        // [bS, nOut]

    // reset and update gates
    ru += temp({0,0, 0,2*nOut});
    ru.applyTransform(transform::Sigmoid, ru);

    // cell gate
    c.assign(c*r + temp({0,0, 2*nOut, 3*nOut}));
    c.applyTransform(transform::Tanh, c);

    // cell output
    h->assign(u * *hI + (1.f - u) * c);
}

//////////////////////////////////////////////////////////////////////////
void gruTimeLoop(sd::LaunchContext * context, const NDArray* x, const NDArray* hI, const NDArray* Wx, const NDArray* Wh, const NDArray* b, NDArray* h) {

    // sL means time steps

    // x   input [sL, bS, nIn]
    // hI  initial cell output (at time step = 0) [bS, nOut]
    // Wx  input-to-hidden  weights, [nIn, 3*nOut]
    // Wh  hidden-to-hidden weights, [nOut, 3*nOut]
    // b   biases, [3*nOut]

    // h  cell outputs at each time step [sL, bS, nOut]

    const int sL   = x->sizeAt(0);
    const int bS   = x->sizeAt(1);
    const int nOut = hI->sizeAt(1);

    NDArray gates(h->ordering(), {bS, 3*nOut}, h->dataType(), context);

    auto xSet = x->allTensorsAlongDimension({1,2});       // sub-arrays with shape [bS, nIn]
    auto hSet = h->allTensorsAlongDimension({1,2});       // sub-arrays with shape [bS, nOut]

    // time loop
    for (int t = 0; t < sL; ++t)
        gruCell(context, xSet.at(t), t == 0 ? hI : hSet.at(t-1), Wx, Wh, b, &gates, hSet.at(t));
}

//////////////////////////////////////////////////////////////////////////
void gruCellBp(sd::LaunchContext* context,
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


//////////////////////////////////////////////////////////////////////////
void gruCellBp(sd::LaunchContext* context,
              const NDArray* x, const NDArray* hI, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* dLdh, const NDArray* gates,
              NDArray* dLdx, NDArray* dLdhI, NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb) {

    //Inputs:
    // x              input [bS, nIn]
    // hI             previous cell output [bS, nOut],  that nIn at previous time step t-1
    // Wx             input-to-hidden weights - [nIn, 3*nOut]
    // Wh             hidden-to-hidden weights - [nOut, 3*nOut]
    // b              biases, [3*nOut] - reset and update gates
    // dLdh           gradient vs. ff output, [bS, nOut]

    //Outputs:
    // dLdx           gradient vs. x,  [bS, nIn],
    // dLdhI          gradient vs. hI, [bS, nOut]
    // dLdWx          gradient vs. W,  [nIn, 3*nOut]
    // dLdWh          gradient vs. Wc, [nOut, 3*nOut]
    // dLdb           gradient vs. b   [3*nOut]

    // 3*nOut means following sequence: reset, update, cell

    // * means element-wnIne product or so called Hadamard product
    // × means matrix multiplication

    // formulas:
    // zr = x × Wxr + hI × Whr + br
    // zu = x × Wxu + hI × Whu + bu
    // r = sigmoid(zr)
    // u = sigmoid(zu)
    // zc = x × Wxc + (r * hI) × Whc + bc
    // c = tanh(zc)
    // h = (1-u)*c + u*hI

    // dLdhI += dLdh;                       [bS, nOut]


    // dhdc = 1 - u                                                             [bS, nOut]
    // dhdu = -c + hI                                                           [bS, nOut]

    // dcdzc = 1 - c*c;                                                         [bS, nOut]
    // dudzu = u*(1-u)                                                          [bS, nOut]
    // drdzr = r(1-r)                                                           [bS, nOut]

    // dzcdr = (...*hI × WhcT)                                                  [bS, nOut]

    // dLdzr = dLdh*dhdc*dcdzc*dzcdr*drdzr = (dLdzc*hI*r(1-r) × WhcT);          [bS, nOut]
    // dLdzu = dLdh*dhdu*dudzu = dLdh*(hI-c)*u*(1-u)                            [bS, nOut]
    // dLdzc = dLdh*dhdc*dcdzc = dLdh*(1-u)*(1-c*c)                             [bS, nOut]

    // dLdx  = dLdzr × WxrT + dLdzu × WxuT + dLdzc × WxcT,                      [bs, nOut] × [nOut, nIn] + ... =  [bS, nIn]

    // dLdhI = dLdzr × WhrT + dLdzu × WhuT + dLdzc × WhcT,                      [bs, nOut] × [nOut, nOut] + ... =  [bS, nOut]

    // dLdWxr = xT × dLdzr                          [nIn, bS] x [bS, nOut] = [nIn, nOut]
    // dLdWxu = xT × dLdzu                          [nIn, bS] x [bS, nOut] = [nIn, nOut]
    // dLdWxc = xT × dLdzc                          [nIn, bS] x [bS, nOut] = [nIn, nOut]

    // dLdWhr = xT × dLdzr                          [nOut, bS] x [bS, nOut] = [nOut, nOut]
    // dLdWhu = xT × dLdzu                          [nOut, bS] x [bS, nOut] = [nOut, nOut]
    // dLdWhc = (r*hI)T × dLdzc                     [nOut, bS] x [bS, nOut] = [nOut, nOut]

    // dLdbr = dLdzr.reduce_sum_along_0_axis        [bS, nOut] -> reduce -> [nOut]
    // dLdbu = dLdzu.reduce_sum_along_0_axis        [bS, nOut] -> reduce -> [nOut]
    // dLdbc = dLdzc.reduce_sum_along_0_axis        [bS, nOut] -> reduce -> [nOut]

    const int nOut = hI->sizeAt(1);

    NDArray dLdz = gates->ulike();                      // [bS, 3*nOut]

    NDArray dLdzru = dLdz({0,0, 0,2*nOut});             // [bS, 2*nOut]

    NDArray dLdzr  = dLdz({0,0, 0,nOut});               // [bS, nOut]
    NDArray dLdzu  = dLdz({0,0, nOut,2*nOut});          // [bS, nOut]
    NDArray dLdzc  = dLdz({0,0, 2*nOut,3*nOut});        // [bS, nOut]

    NDArray r  = (*gates)({0,0, 0,nOut});               // [bS, nOut]
    NDArray u  = (*gates)({0,0, nOut,2*nOut});          // [bS, nOut]
    NDArray c  = (*gates)({0,0, 2*nOut,3*nOut});        // [bS, nOut]

    NDArray WhcT  = (*Wh)({0,0, 2*nOut,3*nOut}).transpose();

    if(dLdh)
        *dLdhI += *dLdh;

    NDArray temp1 = 1 - u;                                  // [bS, nOut]

    // dLdzc
    dLdzc.assign(*dLdhI * temp1 * (1-c*c));                 // [bS, nOut]

    // dLdzu
    dLdzu.assign(*dLdhI * (*hI - c) * u * temp1);           // [bS, nOut]

    // dLdzr
    NDArray temp2 = dLdzc * (*hI) * r *(1-r);
    MmulHelper::mmul(&temp2, &WhcT, &dLdzr);                // [bS, nOut] x [nOut, nOut] = [bS, nOut]

    // dLdx
    NDArray WxT = Wx->transpose();
    MmulHelper::mmul(&dLdz, &WxT, dLdx);                    // [bS, 3*nOut] x [3*nOut, nIn] = [bS, nIn]

    // dLdWx
    *dLdWx += mmul(x->transpose(), dLdz);                   // [nIn, bS] x [bS, 3*nOut] = [nIn, 3*nOut]

    // dLdb
    *dLdb += dLdz.reduceAlongDimension(reduce::Sum, {0});   // [bS, 3*nOut] -> reduce -> [3*nOut];

    dLdzc *= r;

    // dLdhI
    NDArray WhT = Wh->transpose();
    dLdhI->assign(*dLdhI*u + mmul(dLdz, WhT));              // [bS, 3*nOut] x [3*nOut, nOut] = [bS, nOut]

    // dLdWr
    *dLdWh += mmul(hI->transpose(), dLdz);                  // [nOut, bS] x [bS, 3*nOut] = [nOut, 3*nOut]
}


//////////////////////////////////////////////////////////////////////////
void gruTimeLoopBp(sd::LaunchContext * context,
                    const NDArray* x, const NDArray* hI, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* dLdh,
                    NDArray* dLdx, NDArray* dLdhI, NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb) {
    // sL means time steps

    // x      input [sL, bS, nIn]
    // hI     initial cell output (at time step = 0) [bS, nOut]
    // Wx     input-to-hidden  weights, [nIn, 3*nOut]
    // Wh     hidden-to-hidden weights, [nOut, 3*nOut]
    // b      biases, [3*nOut]
    // dLdh   gradient vs. ff output, [sL, bS, nOut]

    // dLdx   gradient vs. x,  [sL, bS, nIn],
    // dLdhI  gradient vs. hI, [bS, nOut]
    // dLdWx  gradient vs. W,  [nIn, 3*nOut]
    // dLdWh  gradient vs. Wc, [nOut, 3*nOut]
    // dLdb   gradient vs. b   [3*nOut]

    const int sL = x->sizeAt(0);
    const int bS   = x->sizeAt(1);
    const int nOut = hI->sizeAt(1);

    NDArray gates(x->ordering(), {sL, bS, 3*nOut}, dLdh->dataType(), x->getContext());
    NDArray h(x->ordering(), {sL+1, bS, nOut}, dLdh->dataType(), x->getContext());

    auto xSet     = x->allTensorsAlongDimension({1,2});         // sub-arrays with shape [bS, nIn]
    auto dLdhSet  = dLdh->allTensorsAlongDimension({1,2});      // sub-arrays with shape [bS, nOut]
    auto hSet     = h.allTensorsAlongDimension({1,2});          // sub-arrays with shape [bS, nOut]
    auto gatesSet = gates.allTensorsAlongDimension({1,2});      // sub-arrays with shape [bS, nOut]
    auto dLdxSet  = dLdx->allTensorsAlongDimension({1,2});      // sub-arrays with shape [bS, nIn]

    hSet.at(0)->assign(hI);

    // forward time loop
    for (int t = 0; t < sL; ++t)
        gruCell(context, xSet.at(t), hSet.at(t), Wx, Wh, b, gatesSet.at(t), hSet.at(t+1));

    // backward time loop
    for (int t = sL-1; t >= 0; --t)
        gruCellBp(context, xSet.at(t), hSet.at(t), Wx, Wh, b, dLdhSet.at(t), gatesSet.at(t),
                  dLdxSet.at(t), dLdhI, dLdWx, dLdWh, dLdb);
}


}
}
}
