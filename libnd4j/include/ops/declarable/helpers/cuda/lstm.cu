/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma, created on 14.02.2018
//

// implementation of operation for LSTM cell with peep hole connections:
// http://www.bioinf.jku.at/publications/older/2604.pdf
// S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
// and
// https://research.google.com/pubs/archive/43905.pdf
// Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for
// large scale acoustic modeling." INTERSPEECH, 2014.

#include <array/NDArrayList.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/lstm.h>
#include <ops/declarable/helpers/lstmBlock.h>
#include <ops/declarable/helpers/transforms.h>

#include <iterator>


namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
void lstmCell(LaunchContext* context, NDArray* xt, NDArray* ht_1, NDArray* ct_1,
              NDArray* Wx, NDArray* Wh, NDArray* Wc, NDArray* Wp, NDArray* b, NDArray* ht,
              NDArray* ct, const std::vector<double>& params) {
  // xt   input [bS x nIn]
  // ht_1 previous cell output [bS x numProj],  that is at previous time step t-1, in case of projection=false ->
  // numProj=nOut!!! ct_1 previous cell state  [bS x nOut], that is at previous time step t-1

  // Wx   input-to-hidden  weights, [nIn  x 4*nOut]
  // Wh   hidden-to-hidden weights, [numProj x 4*nOut]
  // Wc   diagonal weights for peephole connections [3*nOut]
  // Wp   projection weights [nOut x numProj]
  // b    biases, [4*nOut]

  // ht  current cell output [bS x numProj], that is at current time step t
  // ct  current cell state  [bS x nOut], that is at current time step t

  const bool peephole = (bool)params[0];  // if true, provide peephole connections
  const bool projection =
      (bool)params[1];  // if true, then projection is performed, if false then numProj==nOut is mandatory!!!!
  double clippingCellValue =
      params[2];  // clipping value for ct, if it is not equal to zero, then cell state is clipped
  double clippingProjValue =
      params[3];  // clipping value for projected ht, if it is not equal to zero, then projected cell output is clipped
  const double forgetBias = params[4];

  const int bS = xt->sizeAt(0);
  const int nIn = xt->sizeAt(1);
  const int numProj = ht_1->sizeAt(1);
  const int nOut = ct_1->sizeAt(1);

  auto z = mmul(*xt, *Wx) + mmul(*ht_1, *Wh) + *b;  // [bS x 4*nOut] + [bS x 4*nOut] + [1 x 4*nOut] = [bS x 4*nOut]

  auto zit = z({0, 0, 0, nOut});             // z for input gate,  = mmul(Wxi,xt) + mmul(Whi,ht_1) + bi    = [bS x nOut]
  auto zft = z({0, 0, nOut, 2 * nOut});      // z for forget gate, = mmul(Wxf,xt) + mmul(Whf,ht_1) + bf    = [bS x nOut]
  auto zct = z({0, 0, 2 * nOut, 3 * nOut});  // z for cell state,  = mmul(Wxc,xt) + mmul(Whc,ht_1) + bc    = [bS x nOut]
  auto zot = z({0, 0, 3 * nOut, 4 * nOut});  // z for output gate, = mmul(Wxo,xt) + mmul(Who,ht_1) + bo    = [bS x nOut]

  if (peephole) {                              // add peephole connections: z  +  ct_1*Wc
    zit += (*ct_1) * (*Wc)({0, nOut});         // add peephole connections to input gate
    zft += (*ct_1) * (*Wc)({nOut, 2 * nOut});  // add peephole connections to forget gate
  }

  // current sell state = ft*ct_1 + it*tanh(mmul(Wxc,xt) + mmul(Whc,ht_1) + bc
  NDArray zftPlusForgetBias = zft + forgetBias;
  NDArray toAssign = sigmoid(zftPlusForgetBias) * (*ct_1) + sigmoid(zit) * tanh(zct);
  ct->assign(&toAssign);

  // if clipping value is provided then cell state is clipped by this value prior to the cell output activation
  if (clippingCellValue > 0.0) ct->applyScalar(scalar::LstmClip, clippingCellValue, ct);

  if (peephole) zot += (*ct) * (*Wc)({{2 * nOut, 3 * nOut}});  // add peephole connections to output gate zot + ct*Wc

  // current cell output = ot*tanh(ct)
  auto htNoPeepHole = sigmoid(zot) * tanh(*ct);  // = [bS x nOut]

  // apply projection
  if (projection) {
    NDArray restultOne = mmul(htNoPeepHole, *Wp);
    ht->assign(&restultOne);  // [bS x nOut] * [ nOut x numProj] = [bS x numProj]
    // if clipping projection is provided then projected cell output state is clipped by this value
    if (clippingProjValue != 0.) ht->applyScalar(scalar::LstmClip, clippingProjValue, ht);
  } else
    ht->assign(&htNoPeepHole);
}

void lstmBlockCell(NDArray* xt, NDArray* cLast, NDArray* yLast, NDArray* W, NDArray* Wci,
                   NDArray* Wcf, NDArray* Wco, NDArray* b, NDArray* i, NDArray* c, NDArray* f,
                   NDArray* o, NDArray* z, NDArray* h, NDArray* y, const std::vector<double>& params) {
  /* Input arrays:
   *    0: xt              - input [bS, nIn] at time t
   *    1: cLast (cs_prev) - previous cell state  [bS, nOut], time t-1
   *    2: yLast (h_prev)  - previous output [bS, nOut], time t-1
   *    3: W               - Weights - concatenated (input-to-hidden, hidden-to-hidden weights)  weights, [(nIn+nOut),
   * 4*nOut] 4: Wci             - weights - cell peephole (t-1) connections to input modulation gate, [nOut] 5: Wcf -
   * weights - cell peephole (t-1) connections to forget gate, [nOut] 6: Wco             - weights - cell peephole (t)
   * connections to output gate, [nOut] 7: b               - biases, [4*nOut]
   *
   *  Input integer arguments:
   *    0: if not zero, provide peephole connections
   *
   *  Input float arguments:
   *    0: the bias added to forget gates in order to reduce the scale of forgetting in the beginning of the training
   *    1: clipping value for cell state, if it is not equal to zero, then cell state is clipped
   *
   * Output arrays:
   *    0: i      - Input modulation gate activations [bS, nOut]
   *    1: c (cs) - Cell state (pre tanh) [bs, nOut] (cs)
   *    2: f      - Output - forget gate activations [bs, nOut]
   *    3: o      - Output - output gate activations [bs, nOut]
   *    4: z (ci) - Output - block input [bs, nOut]
   *    5: h (co) - Cell state, post tanh [bs, nOut]
   *    6: y (h)  - Current cell output [bS, nOut], time t
   */
  const bool peephole = (bool)params[0];  // if true, provide peephole connections
  const double forgetBias = params[1];
  const double clippingCellValue =
      params[2];  // clipping value for ct, if it is not equal to zero, then cell state is clipped

  const int bS = xt->sizeAt(0);
  const int nIn = xt->sizeAt(1);
  const int nOut = cLast->sizeAt(1);

  std::vector<sd::LongType> shape = {xt->sizeAt(0), xt->sizeAt(1) + yLast->sizeAt(1)};
  // Concat inputs: [xt, yt-1]: concat([bs,nIn],[bs,nOut]) -> [bs, (nIn+nOut)]
  NDArray concatOut(xt->ordering(), shape, xt->dataType(),
                    xt->getContext());
  concat(xt->getContext(), {const_cast<NDArray*>(xt), const_cast<NDArray*>(yLast)}, concatOut, {1});

  auto m = mmul(concatOut, *W);  // mmul: [bs, (nIn+nOut)] * [(nIn+nOut), 4*nOut] = [bs, 4*nOut]
  m += (*b);                     // addiRowVector

  // Note: weights are ordered [inputGate, blockInput, forgetGate, outputGate] to match TF (TF code comments state
  // [i,f,z/ci,o] but behaviour is [i,z,f,o])
  auto zi = m({0, 0, 0, nOut});             // z for input modulation gate, [bS, nOut]
  auto zz = m({0, 0, nOut, 2 * nOut});      // z for block input, [bS, nOut]
  auto zf = m({0, 0, 2 * nOut, 3 * nOut});  // z for forget gate, [bS, nOut]
  auto zo = m({0, 0, 3 * nOut, 4 * nOut});  // z for output gate, [bS, nOut]

  if (peephole) {             // add peephole connections: z  +  ct_1*Wc
    zi += (*cLast) * (*Wci);  // add peephole connections to input gate
    zf += (*cLast) * (*Wcf);  // add peephole connections to forget gate
  }

  // current sell state = ft*cLast + it*tanh(mmul(Wxc,xt) + mmul(Whc,ht_1) + bc
  if (forgetBias != 0.0) zf += forgetBias;

  zz.applyTransform(transform::Tanh, z);     // z = tanh(zz)
  zi.applyTransform(transform::Sigmoid, i);  // i = sigmoid(zi)
  zf.applyTransform(transform::Sigmoid, f);  // f = sigmoid(zf);

  // cell state = blockInput .* inputGate + prevCellState .* forgetGate
  z->applyPairwiseTransform(pairwise::Multiply, i, c);  // c = z * i
  auto temp = (*f) * (*cLast);
  *c += temp;                              // c = (i * z) + (zf * (*cLast))
  c->applyTransform(transform::Tanh, h);  // h = tanh(c)

  // if clipping value is provided then cell state is clipped by this value prior to the cell output activation
  if (clippingCellValue > 0.0) c->applyScalar(scalar::LstmClip, clippingCellValue, c);

  if (peephole) {
    // add peephole connections to output gate zot + ct*Wc
    auto prod = *c * (*Wco);
    zo += prod;
  }
  zo.applyTransform(transform::Sigmoid, o);  // o = sigmoid(zo)

  // current cell output = ot*tanh(ct)
  c->applyTransform(transform::Tanh, h);                 // h = tanh(c)
  o->applyPairwiseTransform(pairwise::Multiply, h, y);  // y = o * h
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
