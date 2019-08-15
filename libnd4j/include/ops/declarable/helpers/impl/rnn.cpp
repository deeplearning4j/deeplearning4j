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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 16.04.2018
//

// function nnCell implements an Elman RNN cell: output = activation(Wx*x + bx  +  Wh*ht  + bh)

#include<ops/declarable/helpers/rnn.h>
#include <helpers/BlasHelper.h>


namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
void rnnCell(nd4j::LaunchContext * context, const NDArray* xt, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* hPrev, NDArray* ht) {

    // xt    input [bS x iS]
    // Wx    input-to-hidden weights, [iS  x nU]
    // Wh    hidden-to-hidden weights, [nU x nU]
    // b     biases, [2*nU]: {0, nU} are input-to-hidden biases and {nU, 2*nU} are hidden-to-hidden biases
    // hPrev previous cell output [bS x nU],  that is at previous time step t-1, in case of projection=false -> nU=nU!!!

    const int nU = hPrev->sizeAt(1);

    // ht is current cell output [bS x nU], that is at current time step t
    ht->assign(mmul(*xt, *Wx) + (*b)({{0, nU}})  +  mmul(*hPrev, *Wh) + (*b)({{nU, 2*nU}}));     // [bS x nU] + [nU]  +  [bS x nU] + [nU] = [bS x nU]
    ht->applyTransform(transform::Tanh);
}

//////////////////////////////////////////////////////////////////////////
void rnnTimeLoop(nd4j::LaunchContext * context, const NDArray* x, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* h0, const NDArray* maxTimeStep, NDArray* h, NDArray* hFinal) {

    // x   input [time x bS x iS]
	// Wx  input-to-hidden  weights, [iS  x nU]
    // Wh  hidden-to-hidden weights, [nU x nU]
	// b   biases for, [2*nU]

	// h0          initial cell output (at time step = 0) [bS x nU]
	// maxTimeStep vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

    const int time = x->sizeAt(0);
    const int bS   = x->sizeAt(1);

    // at first time step
    if(h0)
        hFinal->assign(h0);
    else
        *hFinal = 0.;

    BlasHelper::getInstance();          // to avoid memory leak in pragma parallel loops
    // loop through batch of inputs
    for (int e = 0; e < bS; ++e) {
        // loop through time steps
        for (int t = 0; t < time; ++t) {

            int maxStep = maxTimeStep ? maxTimeStep->e<int>(e) : time;

            auto xt   = (*x)({t,t+1, e,e+1, 0,0}, true);
            auto ht   = (*h)({t,t+1, e,e+1, 0,0}, true);
            auto hPrev = (*hFinal)({e,e+1, 0,0}, true);                       // previous state

            if(t >= maxStep) {
                ht = 0.;
                if(maxStep != 0)
                    hPrev.assign((*h)({maxStep-1,maxStep, e,e+1, 0,0}));
            }
            else {
                helpers::rnnCell(context, &xt, Wx, Wh, b, &hPrev, &ht);
                hPrev.assign(ht);
            }
        }
    }
}


}
}
}

