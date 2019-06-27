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
// @author Yurii Shyrma, created on 14.02.2018
//

// implementation of operation for LSTM cell with peep hole connections:
// http://www.bioinf.jku.at/publications/older/2604.pdf
// S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
// and
// https://research.google.com/pubs/archive/43905.pdf
// Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." INTERSPEECH, 2014.


#include <ops/declarable/helpers/lstm.h>
#include <VariableSpace.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <array/NDArrayList.h>
#include <iterator>
#include <MmulHelper.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

/////////////////////////////////////////////////////////////////////////////
            void lstmBlockTimeLoop(const NDArray* maxSeqLength, const NDArray* xSeq, const NDArray* c0, const NDArray* y0,
                                   const NDArray* W, const NDArray* Wci, const NDArray* Wcf, const NDArray* Wco, const NDArray* b,
                                   const NDArray* iSeq, const NDArray* cSeq, const NDArray* fSeq, const NDArray* oSeq, const NDArray* zSeq,
                                   const NDArray* hSeq, const NDArray* ySeq, const std::vector<double>& params, const int dataFormat){

                const int seqLen = xSeq->sizeAt(0);
                const int mb = xSeq->sizeAt(1);
                const int inSize = xSeq->sizeAt(2);
                const int outSize = iSeq->sizeAt(2);

                const std::vector<Nd4jLong> inSliceShape({mb,inSize});
                const std::vector<Nd4jLong> outSliceShape({mb,outSize});

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

                    if(t != 0) {
                        delete c_t1;
                        delete y_t1;
                    }

                    if(t < seqLen - 1) {
                        c_t1 = new NDArray(std::move(ct));
                        y_t1 = new NDArray(std::move(yt));
                    }
                }
            }



            //////////////////////////////////////////////////////////////////////////
            void lstmTimeLoop(nd4j::LaunchContext * context, const NDArray* x, const NDArray* h0, const NDArray* c0, const NDArray* Wx, const NDArray* Wh, const NDArray* Wc, const NDArray* Wp, const NDArray* b,
                              NDArray* h, NDArray* c, const std::vector<double>& params) {

                // x  input [time x bS x inSize]
                // h0 initial cell output (at time step = 0) [bS x numProj], in case of projection=false -> numProj == numUnits !!!
                // c0 initial cell state  (at time step = 0) [bS x numUnits],

                // Wx input-to-hidden  weights, [inSize  x 4*numUnits]
                // Wh hidden-to-hidden weights, [numProj x 4*numUnits]
                // Wc diagonal weights for peephole connections [3*numUnits]
                // Wp projection weights [numUnits x numProj]
                // b  biases, [4*numUnits]

                // h cell outputs [time x bS x numProj], that is per each time step
                // c cell states  [time x bS x numUnits] that is per each time step

                const int time  = x->sizeAt(0);

                NDArray currentH(*h0);
                NDArray currentC(*c0);

                // loop through time steps
                for (int t = 0; t < time; ++t) {
                    auto xt = (*x)({t,t+1, 0,0, 0,0});
                    auto ht = (*h)({t,t+1, 0,0, 0,0});
                    auto ct = (*c)({t,t+1, 0,0, 0,0});

                    helpers::lstmCell(context, &xt,&currentH,&currentC, Wx,Wh,Wc,Wp, b,   &ht, &ct,   params);
                    currentH.assign(ht);
                    currentC.assign(ct);
                }
            }


        }
    }
}