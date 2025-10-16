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

#include <system/op_boilerplate.h>


#if NOT_EXCLUDED(OP_gru)

// implementation of gated Recurrent Unit cell
// (cf. https://arxiv.org/abs/1406.1078).
// Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
// "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"

#include <helpers/MmulHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/gru.h>
#include <ops/declarable/helpers/transforms.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
void gruCell(sd::LaunchContext* context, NDArray* x, NDArray* hI, NDArray* W, NDArray* Wc,
             NDArray* b, NDArray* bc, NDArray* r, NDArray* u, NDArray* c, NDArray* h) {
  // Inputs:
  // x        input [bS, nIn], nIn - input size
  // hI       previous cell output [bS, nOut],  that is at previous time step t-1, nOut - number of units
  // W        RU weights - [nIn+nOut, 2*nOut] - reset and update gates
  // Wc       C weights - [nIn+nOut, nOut] - cell gate
  // b        r and u biases, [2*nOut] - reset and update gates
  // bc       c biases, [nOut] - cell gate

  // Outputs:
  // r        Reset gate output [bS, nOut]
  // u        Update gate output [bS, nOut]
  // c        Cell gate output [bS, nOut]
  // h        current cell output [bS, nOut]

  /***************************************************************************************/
  /************************ THIS IS NOT OPTIMIZED CODE ***********************************/
  /** however it is more math-friendly and convenient for backprop formulas derivation) **/

  const int bS = x->sizeAt(0);
  const int nIn = x->sizeAt(1);
  const int nOut = hI->sizeAt(1);

  NDArray *Wrx = (*W)({0, nIn, 0, nOut});                  // [nIn, nOut]
  NDArray *Wux = (*W)({0, nIn, nOut, 2 * nOut});           // [nIn, nOut]
  NDArray *Wrh = (*W)({nIn, nIn + nOut, 0, nOut});         // [nOut, nOut]
  NDArray *Wuh = (*W)({nIn, nIn + nOut, nOut, 2 * nOut});  // [nOut, nOut]

  NDArray *Wcx = (*Wc)({0, nIn, 0, 0});           // reset cell weights    [nIn, nOut]
  NDArray *Wch = (*Wc)({nIn, nIn + nOut, 0, 0});  // updates cell weights  [nOut, nOut]

  NDArray *br = (*b)({0, nOut});         // [nOut]
  NDArray *bu = (*b)({nOut, 2 * nOut});  // [nOut]

  // × means matrix multiplication
  // * means element-wise product or so called Hadamard product

  // r = sigmoid(x × Wrx + hI × Wrh + br)
  auto xWrx = mmul(*x, *Wrx);
  auto hIWrh = mmul(*hI, *Wrh);
  auto* sum1 = *xWrx + *hIWrh;
  auto* rAssign = (*sum1) + (*br);
  delete sum1;
  delete hIWrh;
  r->assign(rAssign);  // [bS, nIn] × [nIn, nOut] + [bS, nOut] × [nOut, nOut] + [nOut] = [bS, nOut]
  delete rAssign;
  r->applyTransform(transform::Sigmoid, r);

  // u = sigmoid(x × Wux + hI × Wuh + bu)
  auto xWux = mmul(*x, *Wux);
  auto hIWuh = mmul(*hI, *Wuh);
  auto* sum2 = *xWux + *hIWuh;
  auto* uAssign = (*sum2) + (*bu);
  delete sum2;
  delete xWux;
  u->assign(uAssign);  // [bS, nIn] × [nIn, nOut] + [bS, nOut] × [nOut, nOut] + [nOut] = [bS, nOut]
  delete uAssign;
  u->applyTransform(transform::Sigmoid, u);

  // c = tanh(x × Wcx + (r * hI) × Wch + bc)
  auto* rTimesHi = (*r) * (*hI);
  auto xWcx = mmul(*x, *Wcx);
  auto rTimesHiWch = mmul(*rTimesHi, *Wch);
  delete rTimesHi;
  auto* sum3 = *xWcx + *rTimesHiWch;
  auto* cAssign = (*sum3) + (*bc);
  delete sum3;
  delete xWcx;
  delete rTimesHiWch;
  c->assign(cAssign);  // [bS, nIn] × [nIn, nOut] + [bS, nOut] × [nOut, nOut] + [nOut] = [bS, nOut]
  delete cAssign;
  c->applyTransform(transform::Tanh, c);

  // h = (1 - u) * c + u * hI
  auto* uTimesHi = (*u) * (*hI);
  auto* oneMinusU = 1.f - (*u);
  auto* oneMinusUTimesC = (*oneMinusU) * (*c);
  delete oneMinusU;
  auto* hAssign = (*uTimesHi) + (*oneMinusUTimesC);
  delete uTimesHi;
  delete oneMinusUTimesC;
  h->assign(hAssign);
  delete hAssign;

  delete Wrx;
  delete Wux;
  delete Wrh;
  delete Wuh;
  delete Wcx;
  delete Wch;
  delete br;
  delete bu;
}

//////////////////////////////////////////////////////////////////////////
void gruCell(NDArray* x, NDArray* hI, NDArray* Wx, NDArray* Wh, NDArray* b,
             NDArray* gates, NDArray* h, bool linearBeforeReset) {

  if(linearBeforeReset) {
    THROW_EXCEPTION("GRU: Linear before reset not implemented. Please set to false.");
  }

  // Inputs:
  // x        input [bS, nIn]
  // hI       previous cell output [bS, nOut],  that is at previous time step t-1
  // Wx       weights for x - [nIn, 3*nOut]
  // Wh       weights for h - [nOut, 3*nOut]
  // b        biases [3*nOut]

  // 3*nOut means following sequence: reset, update, cell

  // Outputs:
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

  const int bS = x->sizeAt(0);
  const int nIn = x->sizeAt(1);
  const int nOut = hI->sizeAt(1);

  NDArray *gatesULike = gates->ulike();
  NDArray temp = *gatesULike;
  MmulHelper::mmul(x, Wx, &temp);  // [bS, nIn] × [nIn, 3*nOut] = [bS, 3*nOut]
  temp += *b;

  MmulHelper::mmul(hI, Wh, gates);  // [bS, nOut] × [nOut, 3*nOut] = [bS, 3*nOut]

  NDArray *ru = (*gates)({0, 0, 0, 2 * nOut});  // [bS, 2*nOut]

  NDArray *r = (*gates)({0, 0, 0, nOut});             // [bS, nOut]
  NDArray *u = (*gates)({0, 0, nOut, 2 * nOut});      // [bS, nOut]
  NDArray *c = (*gates)({0, 0, 2 * nOut, 3 * nOut});  // [bS, nOut]

  NDArray *tempView1 = temp({0, 0, 0, 2 * nOut});
  NDArray *tempView2 = temp({0, 0, 2 * nOut, 3 * nOut});

  // reset and update gates
  *ru += *tempView1;
  ru->applyTransform(transform::Sigmoid, ru);

  // cell gate
  auto* cTimesR = (*c) * (*r);
  auto* cAssign = (*cTimesR) + (*tempView2);
  delete cTimesR;
  c->assign(cAssign);
  delete cAssign;
  c->applyTransform(transform::Tanh, c);

  // h = (1-u)*c + u*hI
  auto* uTimesHi = (*u) * (*hI);
  auto* oneMinusU = 1.f - (*u);
  auto* oneMinusUTimesC = (*oneMinusU) * (*c);
  delete oneMinusU;
  auto* hAssign = (*uTimesHi) + (*oneMinusUTimesC);
  delete uTimesHi;
  delete oneMinusUTimesC;
  h->assign(hAssign);
  delete hAssign;

  delete gatesULike;
  delete ru;
  delete r;
  delete u;
  delete c;
  delete tempView1;
  delete tempView2;
}

//////////////////////////////////////////////////////////////////////////
void gruTimeLoop(sd::LaunchContext* context, NDArray* x, NDArray* hI, NDArray* Wx, NDArray* Wh,
                 NDArray* b, NDArray* h, bool linearBeforeReset) {
  // sL means time steps

  // x   input [sL, bS, nIn]
  // hI  initial cell output (at time step = 0) [bS, nOut]
  // Wx  input-to-hidden  weights, [nIn, 3*nOut]
  // Wh  hidden-to-hidden weights, [nOut, 3*nOut]
  // b   biases, [3*nOut]

  // h  cell outputs at each time step [sL, bS, nOut]

  const int sL = x->sizeAt(0);
  const int bS = x->sizeAt(1);
  const int nOut = hI->sizeAt(1);

  std::vector<LongType> shape = {bS, 3 * nOut};
  NDArray gates(h->ordering(), shape, h->dataType(), context);

  auto xSet = x->allTensorsAlongDimension({1, 2});  // sub-arrays with shape [bS, nIn]
  auto hSet = h->allTensorsAlongDimension({1, 2});  // sub-arrays with shape [bS, nOut]

  // time loop
  for (int t = 0; t < sL; ++t) {
    gruCell(xSet.at(t), t == 0 ? hI : hSet.at(t - 1), Wx, Wh, b, &gates, hSet.at(t), linearBeforeReset);
  }
}

//////////////////////////////////////////////////////////////////////////
void gruCellBp(sd::LaunchContext* context, NDArray* x, NDArray* hLast, NDArray* W, NDArray* Wc,
               NDArray* b, NDArray* bc, NDArray* dLdr, NDArray* dLdu, NDArray* dLdc,
               NDArray* dLdh, NDArray* dLdx, NDArray* dLdhLast, NDArray* dLdW, NDArray* dLdWc, NDArray* dLdb,
               NDArray* dLdbc) {
  // Inputs:
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

  // Outputs:
  // dLdx           gradient wrt x,  [bS, iS],
  // dLdhLast       gradient wrt hLast, [bS, nU]
  // dLdW           gradient wrt W,  [iS+nU, 2*nU]
  // dLdWc          gradient wrt Wc, [iS+nU, nU]
  // dLdb           gradient wrt bru [2*nU]
  // dLdbc          gradient wrt bc  [nU]

  // * means element-wise product or so called Hadamard product
  // × means matrix multiplication

  /************************************************************************************************/
  /******************************* THIS IS NOT OPTIMIZED CODE *************************************/
  /*** aim is to have math-readable code in order to keep track of backprop formulas derivation ***/

  const int bS = x->sizeAt(0);
  const int iS = x->sizeAt(1);
  const int nU = hLast->sizeAt(1);

  NDArray *xT = x->transpose();          // [iS, bS]
  NDArray *hLastT = hLast->transpose();  // [nU, bS]

  NDArray *Wrx = (*W)({0, iS, 0, nU});             // [iS, nU]
  NDArray *Wux = (*W)({0, iS, nU, 2 * nU});        // [iS, nU]
  NDArray *Wrh = (*W)({iS, iS + nU, 0, nU});       // [nU, nU]
  NDArray *Wuh = (*W)({iS, iS + nU, nU, 2 * nU});  // [nU, nU]

  NDArray *Wcx = (*Wc)({0, iS, 0, 0});        // reset cell weights    [iS, nU]
  NDArray *Wch = (*Wc)({iS, iS + nU, 0, 0});  // updates cell weights  [nU, nU]

  NDArray *br = (*b)({0, nU});       // [nU]
  NDArray *bu = (*b)({nU, 2 * nU});  // [nU]

  NDArray *WrxT = Wrx->transpose();  // [nU, iS]
  NDArray *WuxT = Wux->transpose();  // [nU, iS]
  NDArray *WrhT = Wrh->transpose();  // [nU, nU]
  NDArray *WuhT = Wuh->transpose();  // [nU, nU]

  NDArray *WcxT = Wcx->transpose();  // [nU, iS]
  NDArray *WchT = Wch->transpose();  // [nU, nU]

  NDArray *dLdWrx = (*dLdW)({0, iS, 0, nU});             // [iS, nU]
  NDArray *dLdWux = (*dLdW)({0, iS, nU, 2 * nU});        // [iS, nU]
  NDArray *dLdWrh = (*dLdW)({iS, iS + nU, 0, nU});       // [nU, nU]
  NDArray *dLdWuh = (*dLdW)({iS, iS + nU, nU, 2 * nU});  // [nU, nU]

  NDArray *dLdWcx = (*dLdWc)({0, iS, 0, 0});        // [iS, nU]
  NDArray *dLdWch = (*dLdWc)({iS, iS + nU, 0, 0});  // [nU, nU]

  NDArray *dLdbr = (*dLdb)({0, nU});       // [nU]
  NDArray *dLdbu = (*dLdb)({nU, 2 * nU});  // [nU]

  // ***** feed forward step ***** //

  // r = sigmoid(x × Wrx + hLast × Wrh + br)
  auto xWrx = mmul(*x, *Wrx);
  auto hLastWrh = mmul(*hLast, *Wrh);
  auto* sum1 = *xWrx + *hLastWrh;
  auto* rTemp = (*sum1) + (*br);
  delete sum1;
  NDArray r = *rTemp;  // [bS, iS] × [iS, nU] + [bS, nU] × [nU, nU] + [nU] = [bS, nU]
  delete rTemp;
  r.applyTransform(transform::Sigmoid, &r);

  // u = sigmoid(x × Wux + hLast × Wuh + bu)
  auto xWux = mmul(*x, *Wux);
  auto hLastWuh = mmul(*hLast, *Wuh);
  auto* sum2 = *xWux + *hLastWuh;
  auto* uTemp = (*sum2) + (*bu);
  delete sum2;
  NDArray u = *uTemp;  // [bS, iS] × [iS, nU] + [bS, nU] × [nU, nU] + [nU] = [bS, nU]
  delete uTemp;
  delete xWux;
  u.applyTransform(transform::Sigmoid, &u);

  // c = tanh(x × Wcx + (r * hLast) × Wch + bc)
  auto* rTimesHLast2 = r * (*hLast);
  auto xWcx = mmul(*x, *Wcx);
  auto rTimesHLast2Wch = mmul(*rTimesHLast2, *Wch);
  delete rTimesHLast2;
  auto* sum3 = *xWcx + *rTimesHLast2Wch;
  auto* cTemp = (*sum3) + (*bc);
  delete sum3;
  delete xWcx;
  delete rTimesHLast2Wch;
  NDArray c = *cTemp;  // [bS, iS] × [iS, nU] + [bS, nU] × [nU, nU] + [nU] = [bS, nU]
  delete cTemp;
  c.applyTransform(transform::Tanh, &c);

  // h = (1 - u) * c + u * hPrev

  // ***** back prop step ***** //

  auto* hLastMinusC = (*hLast) - c;
  auto* oneMinusU = 1.f - u;
  auto* dudZu = u * (*oneMinusU);
  delete oneMinusU;
  auto* oneMinusR = 1.f - r;
  auto* drdZr = r * (*oneMinusR);
  delete oneMinusR;
  auto* cSquared = c * c;
  auto* oneMinusCSquared = 1.f - (*cSquared);
  delete cSquared;
  auto dcdZc = *oneMinusCSquared;
  delete oneMinusCSquared;
  auto* dLdZc = (*dLdc) * dcdZc;
  auto* dLdZu = (*dLdu) * (*dudZu);
  delete dudZu;
  auto* dLdZr = (*dLdr) * (*drdZr);
  delete drdZr;

  NDArray *dhdc = 1.f - u;         // [bS, nU]
  NDArray dhdu = *hLastMinusC;      // [bS, nU]
  delete hLastMinusC;

  // dLdx = dLdZu × WuxT + dLdZc × WcxT + dLdZr × WrxT
  auto dLdZuWuxT = mmul(*dLdZu, *WuxT);
  auto dLdZcWcxT = mmul(*dLdZc, *WcxT);
  auto dLdZrWrxT = mmul(*dLdZr, *WrxT);
  auto* temp1 = *dLdZuWuxT + *dLdZcWcxT;
  auto* dLdxTemp = (*temp1) + *dLdZrWrxT;
  delete temp1;

  delete dLdZuWuxT;
  delete dLdZcWcxT;
  delete dLdZrWrxT;
  dLdx->assign(dLdxTemp);  // [bS, iS]
  delete dLdxTemp;

  // dldZTimeR = dLdZc * r
  auto* dldZTimeR = (*dLdZc) * r;

  // dLdhLast = dLdh * u + dLdZu × WuhT + dldZTimeR × WchT + dLdZr × WrhT
  auto* dLdhTimesU = (*dLdh) * u;
  auto dLdZuWuhT = mmul(*dLdZu, *WuhT);
  auto dldZTimeRWchT = mmul(*dldZTimeR, *WchT);
  auto dLdZrWrhT = mmul(*dLdZr, *WrhT);
  auto* temp2 = (*dLdhTimesU) + *dLdZuWuhT;
  delete dLdhTimesU;
  delete dLdZuWuhT;
  auto* temp3 = (*temp2) + *dldZTimeRWchT;
  delete temp2;
  delete dldZTimeRWchT;
  auto* dLdhLastTemp = (*temp3) + *dLdZrWrhT;
  delete temp3;
  delete dLdZrWrhT;
  dLdhLast->assign(dLdhLastTemp);  // [bS, nU]
  delete dLdhLastTemp;

  // dLdWrx = xT × dLdZr
  auto dLdWrxTemp = mmul(*xT, *dLdZr);
  dLdWrx->assign(dLdWrxTemp);  // [iS, bS] × [bS, nU] = [iS, nU]
  delete dLdWrxTemp;
  // dLdWrh = hLastT × dLdZr
  auto dLdWrhTemp = mmul(*hLastT, *dLdZr);
  dLdWrh->assign(dLdWrhTemp);  // [nU, bS] × [bS, nU] = [nU, nU]
  delete dLdWrhTemp;
  // dLdWux = xT × dLdZu
  auto dLdWuxTemp = mmul(*xT, *dLdZu);
  dLdWux->assign(dLdWuxTemp);  // [iS, bS] × [bS, nU] = [iS, nU]
  delete dLdWuxTemp;
  // dLdWuh = hLastT × dLdZu
  auto dLdWuhTemp = mmul(*hLastT, *dLdZu);
  dLdWuh->assign(dLdWuhTemp);  // [nU, bS] × [bS, nU] = [nU, nU]
  delete dLdWuhTemp;
  // dLdWcx = xT × dLdZc
  auto dLdWcxTemp = mmul(*xT, *dLdZc);
  dLdWcx->assign(dLdWcxTemp);  // [iS, bS] × [bS, nU] = [iS, nU]
  delete dLdWcxTemp;
  // dLdWch = (r * hLast)T × dLdZc
  auto* rTimesHLast = r * (*hLast);
  NDArray* rTimesHLastT = rTimesHLast->transpose();
  delete rTimesHLast;
  auto dLdWchTemp = mmul(*rTimesHLastT, *dLdZc);
  dLdWch->assign(dLdWchTemp);  // [nU, bS] × [bS, nU] = [nU, nU]
  delete dLdWchTemp;
  // Calculate reduction for bias gradients
  std::vector<sd::LongType> zeroVec = {0};
  auto* dLdbrTemp = dLdZr->reduceAlongDimension(reduce::Sum, &zeroVec);
  dLdbr->assign(dLdbrTemp);  // [nU]
  delete dLdbrTemp;

  auto* dLdbuTemp = dLdZu->reduceAlongDimension(reduce::Sum, &zeroVec);
  dLdbu->assign(dLdbuTemp);  // [nU]
  delete dLdbuTemp;

  auto* dLdbcTemp = dLdZc->reduceAlongDimension(reduce::Sum, &zeroVec);
  dLdbc->assign(dLdbcTemp);  // [nU]
  delete dLdbcTemp;

  delete dhdc;
  delete dLdZc;
  delete dLdZu;
  delete dLdZr;
  delete dldZTimeR;
  delete Wrx;
  delete Wux;
  delete Wrh;
  delete Wuh;
  delete Wcx;
  delete Wch;
  delete br;
  delete bu;
  delete WrxT;
  delete WuxT;
  delete WrhT;
  delete WuhT;
  delete WcxT;
  delete WchT;
  delete dLdWrx;
  delete dLdWux;
  delete dLdWrh;
  delete dLdWuh;
  delete dLdWcx;
  delete dLdWch;
  delete dLdbr;
  delete dLdbu;
  delete xT;
  delete hLastT;
  delete rTimesHLastT;
}

//////////////////////////////////////////////////////////////////////////
void gruCellBp(sd::LaunchContext* context, NDArray* x, NDArray* hI, NDArray* Wx, NDArray* Wh,
               NDArray* b, NDArray* dLdh, NDArray* gates, NDArray* dLdx, NDArray* dLdhI,
               NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb) {
  // Inputs:
  // x              input [bS, nIn]
  // hI             previous cell output [bS, nOut],  that nIn at previous time step t-1
  // Wx             input-to-hidden weights - [nIn, 3*nOut]
  // Wh             hidden-to-hidden weights - [nOut, 3*nOut]
  // b              biases, [3*nOut] - reset and update gates
  // dLdh           gradient vs. ff output, [bS, nOut]

  // Outputs:
  // dLdx           gradient vs. x,  [bS, nIn],
  // dLdhI          gradient vs. hI, [bS, nOut]
  // dLdWx          gradient vs. W,  [nIn, 3*nOut]
  // dLdWh          gradient vs. Wc, [nOut, 3*nOut]
  // dLdb           gradient vs. b   [3*nOut]

  // 3*nOut means following sequence: reset, update, cell

  // * means element-wise product or so called Hadamard product
  // × means matrix multiplication

  const int nOut = hI->sizeAt(1);

  NDArray *gatesULike = gates->ulike();
  NDArray dLdz = *gatesULike;  // [bS, 3*nOut]

  NDArray *dLdzru = dLdz({0, 0, 0, 2 * nOut});  // [bS, 2*nOut]

  NDArray *dLdzr = dLdz({0, 0, 0, nOut});             // [bS, nOut]
  NDArray *dLdzu = dLdz({0, 0, nOut, 2 * nOut});      // [bS, nOut]
  NDArray *dLdzc = dLdz({0, 0, 2 * nOut, 3 * nOut});  // [bS, nOut]

  NDArray *r = (*gates)({0, 0, 0, nOut});             // [bS, nOut]
  NDArray *u = (*gates)({0, 0, nOut, 2 * nOut});      // [bS, nOut]
  NDArray *c = (*gates)({0, 0, 2 * nOut, 3 * nOut});  // [bS, nOut]

  NDArray *WhView = (*Wh)({0, 0, 2 * nOut, 3 * nOut});
  NDArray *WhcT = WhView->transpose();

  if (dLdh) *dLdhI += *dLdh;

  auto* oneMinusU = 1 - (*u);  // [bS, nOut]

  // dLdzc = dLdhI * (1-u) * (1-c²)
  auto* cSquared = (*c) * (*c);
  auto* oneMinusCSquared = 1.f - (*cSquared);
  delete cSquared;
  auto* temp1 = (*dLdhI) * (*oneMinusU);
  auto* dLdzcTemp = (*temp1) * (*oneMinusCSquared);
  delete temp1;
  delete oneMinusCSquared;
  dLdzc->assign(dLdzcTemp);  // [bS, nOut]
  delete dLdzcTemp;

  // dLdzu = dLdhI * (hI - c) * u * (1-u)
  auto* hIMinusC = (*hI) - (*c);
  auto* uTimesOneMinusU = (*u) * (*oneMinusU);
  auto* temp2 = (*dLdhI) * (*hIMinusC);
  auto* dLdzuTemp = (*temp2) * (*uTimesOneMinusU);
  delete temp2;
  delete hIMinusC;
  delete uTimesOneMinusU;
  dLdzu->assign(dLdzuTemp);  // [bS, nOut]
  delete dLdzuTemp;
  delete oneMinusU;

  // dLdzr = (dLdzc * hI * r * (1-r)) × WhcT
  auto* oneMinusR = 1 - (*r);
  auto* rTimesOneMinusR = (*r) * (*oneMinusR);
  delete oneMinusR;
  auto* temp3 = (*dLdzc) * (*hI);
  auto* temp4 = (*temp3) * (*rTimesOneMinusR);
  delete temp3;
  delete rTimesOneMinusR;
  MmulHelper::mmul(temp4, WhcT, dLdzr);  // [bS, nOut] x [nOut, nOut] = [bS, nOut]
  delete temp4;

  // dLdx = dLdz × WxT
  NDArray *WxT = Wx->transpose();
  MmulHelper::mmul(&dLdz, WxT, dLdx);  // [bS, 3*nOut] x [3*nOut, nIn] = [bS, nIn]

  // dLdWx += xT × dLdz
  NDArray *xT = x->transpose();
  auto dLdWxAdd = mmul(*xT, dLdz);
  *dLdWx += *dLdWxAdd;  // [nIn, bS] x [bS, 3*nOut] = [nIn, 3*nOut]

  delete dLdWxAdd;
  // dLdb += sum(dLdz, axis=0)
  std::vector<sd::LongType> zeroVec = {0};
  auto* dLdbAdd = dLdz.reduceAlongDimension(reduce::Sum, &zeroVec);
  *dLdb += (*dLdbAdd);  // [bS, 3*nOut] -> reduce -> [3*nOut];
  delete dLdbAdd;

  *dLdzc *= (*r);

  // dLdhI = dLdhI * u + dLdz × WhT
  NDArray *WhT = Wh->transpose();
  auto* dLdhIU = (*dLdhI) * (*u);
  auto mmulResult = mmul(dLdz, *WhT);
  auto* dLdhIAssign = (*dLdhIU) + *mmulResult;
  delete dLdhIU;
  delete mmulResult;
  dLdhI->assign(dLdhIAssign);  // [bS, 3*nOut] x [3*nOut, nOut] = [bS, nOut]
  delete dLdhIAssign;

  // dLdWh += hIT × dLdz
  NDArray *hITranspose = hI->transpose();
  auto dLdWhAdd = mmul(*hITranspose, dLdz);
  *dLdWh += *dLdWhAdd;  // [nOut, bS] x [bS, 3*nOut] = [nOut, 3*nOut]
  delete dLdWhAdd;

  delete gatesULike;
  delete dLdzru;
  delete dLdzr;
  delete dLdzu;
  delete dLdzc;
  delete r;
  delete u;
  delete c;
  delete WhView;
  delete WhcT;
  delete hITranspose;
  delete WhT;
  delete xT;
  delete WxT;
}

//////////////////////////////////////////////////////////////////////////
void gruTimeLoopBp(sd::LaunchContext* context, NDArray* x, NDArray* hI, NDArray* Wx,
                   NDArray* Wh, NDArray* b, NDArray* dLdh, NDArray* dLdx, NDArray* dLdhI,
                   NDArray* dLdWx, NDArray* dLdWh, NDArray* dLdb) {
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
  const int bS = x->sizeAt(1);
  const int nOut = hI->sizeAt(1);

  std::vector<sd::LongType> shape = {bS, 3 * nOut};
  std::vector<sd::LongType> hShape = {sL + 1, bS, nOut};
  NDArray gates(x->ordering(), shape, dLdh->dataType(), x->getContext());
  NDArray h(x->ordering(), hShape, dLdh->dataType(), x->getContext());

  auto xSet = x->allTensorsAlongDimension({1, 2});         // sub-arrays with shape [bS, nIn]
  auto dLdhSet = dLdh->allTensorsAlongDimension({1, 2});   // sub-arrays with shape [bS, nOut]
  auto hSet = h.allTensorsAlongDimension({1, 2});          // sub-arrays with shape [bS, nOut]
  auto gatesSet = gates.allTensorsAlongDimension({1, 2});  // sub-arrays with shape [bS, nOut]
  auto dLdxSet = dLdx->allTensorsAlongDimension({1, 2});   // sub-arrays with shape [bS, nIn]

  hSet.at(0)->assign(hI);

  // forward time loop
  for (int t = 0; t < sL; ++t) gruCell(xSet.at(t), hSet.at(t), Wx, Wh, b, gatesSet.at(t), hSet.at(t + 1), false);

  // backward time loop
  for (int t = sL - 1; t >= 0; --t)
    gruCellBp(context, xSet.at(t), hSet.at(t), Wx, Wh, b, dLdhSet.at(t), gatesSet.at(t), dLdxSet.at(t), dLdhI, dLdWx,
              dLdWh, dLdb);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
