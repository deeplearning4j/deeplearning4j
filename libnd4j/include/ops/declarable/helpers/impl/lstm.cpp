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
// @author Yurii Shyrma, created on 14.02.2018
//

// implementation of operation for LSTM cell with peep hole connections:
// http://www.bioinf.jku.at/publications/older/2604.pdf
// S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
// and
// https://research.google.com/pubs/archive/43905.pdf
// Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for
// large scale acoustic modeling." INTERSPEECH, 2014.

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_lstm)

#include <array/NDArrayList.h>
#include <graph/VariableSpace.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <ops/declarable/helpers/lstm.h>
#include <ops/declarable/helpers/transforms.h>

#include <iterator>

namespace sd {
namespace ops {
namespace helpers {

/////////////////////////////////////////////////////////////////////////////
void lstmBlockTimeLoop(NDArray* maxSeqLength, NDArray* xSeq, NDArray* c0, NDArray* y0,
                       NDArray* W, NDArray* Wci, NDArray* Wcf, NDArray* Wco, NDArray* b,
                       NDArray* iSeq, NDArray* cSeq, NDArray* fSeq, NDArray* oSeq,
                       NDArray* zSeq, NDArray* hSeq, NDArray* ySeq, const std::vector<double>& params,
                       const int dataFormat) {
  int seqLen, bS, nIn, nOut;

  if (dataFormat == 0) {
    seqLen = xSeq->sizeAt(0);
    bS = xSeq->sizeAt(1);
    nIn = xSeq->sizeAt(2);
    nOut = iSeq->sizeAt(2);
  } else if (dataFormat == 1) {
    seqLen = xSeq->sizeAt(2);
    bS = xSeq->sizeAt(0);
    nIn = xSeq->sizeAt(1);
    nOut = iSeq->sizeAt(1);
  } else if (dataFormat == 2) {
    seqLen = xSeq->sizeAt(1);
    bS = xSeq->sizeAt(0);
    nIn = xSeq->sizeAt(2);
    nOut = iSeq->sizeAt(2);
  }

  const std::vector<sd::LongType> inSliceShape({bS, nIn});
  const std::vector<sd::LongType> outSliceShape({bS, nOut});

  auto c_t1 = const_cast<NDArray*>(c0);
  auto y_t1 = const_cast<NDArray*>(y0);

  // loop through time steps
  for (int t = 0; t < seqLen; ++t) {
    auto xt = timeSubset(xSeq, t, dataFormat);

    auto it = timeSubset(iSeq, t, dataFormat);
    auto ct = timeSubset(cSeq, t, dataFormat);
    auto ft = timeSubset(fSeq, t, dataFormat);
    auto ot = timeSubset(oSeq, t, dataFormat);
    auto zt = timeSubset(zSeq, t, dataFormat);
    auto ht = timeSubset(hSeq, t, dataFormat);
    auto yt = timeSubset(ySeq, t, dataFormat);

    helpers::lstmBlockCell(&xt, c_t1, y_t1, W, Wci, Wcf, Wco, b, &it, &ct, &ft, &ot, &zt, &ht, &yt, params);

    if (t != 0) {
      delete c_t1;
      delete y_t1;
    }

    if (t < seqLen - 1) {
      c_t1 = new NDArray(std::move(ct));
      y_t1 = new NDArray(std::move(yt));
    }
  }
}

//////////////////////////////////////////////////////////////////////////
void lstmTimeLoop(sd::LaunchContext* context, NDArray* x, NDArray* h0, NDArray* c0, NDArray* Wx,
                  NDArray* Wh, NDArray* Wc, NDArray* Wp, NDArray* b, NDArray* h, NDArray* c,
                  const std::vector<double>& params) {
  // x  input [time x bS x nIn]
  // h0 initial cell output (at time step = 0) [bS x numProj], in case of projection=false -> numProj == numUnits !!!
  // c0 initial cell state  (at time step = 0) [bS x numUnits],

  // Wx input-to-hidden  weights, [nIn  x 4*numUnits]
  // Wh hidden-to-hidden weights, [numProj x 4*numUnits]
  // Wc diagonal weights for peephole connections [3*numUnits]
  // Wp projection weights [numUnits x numProj]
  // b  biases, [4*numUnits]

  // h cell outputs [time x bS x numProj], that is per each time step
  // c cell states  [time x bS x numUnits] that is per each time step

  const int time = x->sizeAt(0);

  NDArray currentH(*h0);
  NDArray currentC(*c0);

  // loop through time steps
  for (int t = 0; t < time; ++t) {
    auto xt = (*x)({t, t + 1, 0, 0, 0, 0});
    auto ht = (*h)({t, t + 1, 0, 0, 0, 0});
    auto ct = (*c)({t, t + 1, 0, 0, 0, 0});

    helpers::lstmCell(context, &xt, &currentH, &currentC, Wx, Wh, Wc, Wp, b, &ht, &ct, params);
    currentH.assign(&ht);
    currentC.assign(&ct);
  }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif