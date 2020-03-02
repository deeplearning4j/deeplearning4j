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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

// implementation of operation for LSTM cell with peep hole connections:
// http://www.bioinf.jku.at/publications/older/2604.pdf
// S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
// and
// https://research.google.com/pubs/archive/43905.pdf
// Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." INTERSPEECH, 2014.


#include <ops/declarable/helpers/lstmLayer.h>
#include <helpers/ShapeUtils.h>
// #include <VariableSpace.h>
// #include <ops/declarable/CustomOperations.h>
// #include<ops/declarable/helpers/transforms.h>
// #include <ops/declarable/helpers/legacy_helpers.h>
// #include <array/NDArrayList.h>
// #include <iterator>
// #include <helpers/MmulHelper.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
void lstmLayerCell(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                   const NDArray* b, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                   const std::vector<float>& params,
                   NDArray* h, NDArray* c) {


    /************************ THIS IS NOT OPTIMAZED CODE ***********************************/
    /** the objective is to provide math-readable code **/

    // equations (no peephole connections)
    // it  = σ(Wxi * xt  +  Wri * ht-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = ft ◦ ct-1 + it ◦ c't
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  bo)
    // ht  = ot ◦ tanh(ct)

    // equations (peephole connections are present)
    // it  = σ(Wxi * xt  +  Wri * ht-1  +  Wpi ◦ ct-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  Wpf ◦ ct-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = ft ◦ ct-1 + it ◦ c't
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  Wpo ◦ ct   +  bo)
    // ht  = ot ◦ tanh(ct)


    // IDs for activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus

    // params[0] - dataFormat, ignore
    // params[1] - directionMode, ignore
    // params[2] - cell clipping value, if it = 0 then do not apply clipping

    // params[3]  - activation ID for input (i), forget (f) and output (o) gates
    // params[4]  - alpha value for gates activation
    // params[5]  - beta value for gates activation

    // params[6]  - activation ID for cell state (c)
    // params[7]  - alpha value for cell state activation
    // params[8]  - beta value for cell state activation

    // params[9]  - activation ID for output (h)
    // params[10] - alpha value for output activation
    // params[11] - beta value for output activation

    // INPUTS:
    // x  - current input at time t, [bS, nIn] or [nIn] if seqLen != nullptr
    // Wx - input weights [nIn, 4*nOut]
    // Wr - recurrent weights [nOut, 4*nOut]
    // b  - biases [4*nOut], optional, may be nullptr
    // hI - previous (initial) output at time t-1, optional may be nullptr, [bS, nOut] or [nOut] if seqLen != nullptr
    // cI - previous (initial) cell state at time t-1, optional may be nullptr, [bS, nOut] or [nOut] if seqLen != nullptr
    // Wp - peephole weights [3*nOut], optional, may be nullptr

    // OUTPUTS:
    // h - current output, that is at current time step t, [bS, nOut] or [nOut] if seqLen != nullptr
    // c - current cell state,  that is at current time step t, [bS, nOut] or [nOut] if seqLen != nullptr

    // !!! dimension 4*nOut implies order it, ft, c't, ot
    // !!! dimension 3*nOut implies order it, ft, ot

    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    auto z = mmul(*x, *Wx) + mmul(*hI, *Wr);    //   [bs, nIn] * [nIn, 4*nOut] + [bs, nOut] * [nOut, 4*nOut] = [bS, 4*nOut]
                                                //or [nIn] * [nIn, 4*nOut] + [nOut] * [nOut, 4*nOut] = [4*nOut]

    // add biases if they are given
    if(b != nullptr)
        z += *b;                                    // broadcast [bS, 4*nOut] + [4*nOut] = [bS, 4*nOut]

    auto zi = x->rankOf() == 1 ? z({0,        nOut}) : z({0,0, 0,        nOut});         // input gate it, [bS, nOut]
    auto zf = x->rankOf() == 1 ? z({nOut,   2*nOut}) : z({0,0, nOut,   2*nOut});         // forget gate ft, [bS, nOut]
    auto zc = x->rankOf() == 1 ? z({2*nOut, 3*nOut}) : z({0,0, 2*nOut, 3*nOut});         // cell gate c't, [bS, nOut]
    auto zo = x->rankOf() == 1 ? z({3*nOut, 4*nOut}) : z({0,0, 3*nOut, 4*nOut});         // output gate ot, [bS, nOut]

    // peephole connections for input and forget gates
    if(Wp != nullptr) {
        zi += *cI * (*Wp)({0,      nOut});      // broadcast: [bS, nOut] + [bS, nOut] ◦ [nOut] = [bS, nOut]
        zf += *cI * (*Wp)({nOut, 2*nOut});      // broadcast: [bS, nOut] + [bS, nOut] ◦ [nOut] = [bS, nOut]
    }

    applyActivation(zi, params[3], params[4], params[5], zi);   // inplace
    applyActivation(zf, params[3], params[4], params[5], zf);   // inplace
    applyActivation(zc, params[6], params[7], params[8], zc);   // inplace

    c->assign(zf * *cI + zi * zc);          // [bS, nOut] ◦ [bS, nOut] + [bS, nOut] ◦ [bS, nOut] = [bS, nOut]

    // if clipping value is non-zero then cell state is clipped by this value prior to the cell output activation
    if(params[2] != 0)
        c->applyScalar(scalar::LstmClip, params[2], *c);

    // peephole connections for output gate
    if(Wp != nullptr)
        zo += *c * (*Wp)({2*nOut, 3*nOut});    // broadcast: [bS, nOut] + [nOut] ◦ [bS, nOut] = [bS, nOut]

    applyActivation(zo, params[3], params[4], params[5], zo);

    applyActivation(*c, params[9], params[10], params[11], *h);
    *h *= zo;                               // [bS, nOut] ◦ [bS, nOut]
}



//////////////////////////////////////////////////////////////////////////
void lstmLayerTimeLoop(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                       const NDArray* b, const NDArray* seqLen, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                       const std::vector<float>& params,
                       const bool forward,
                       NDArray* h, NDArray* hL, NDArray* cL) {

    // INPUTS:
    // x  - current input  [sL, bS, nIn],  [bS, sL, nIn],  [bS, nIn, sL],
    // Wx - input weights [nIn, 4*nOut]
    // Wr - recurrent weights [nOut, 4*nOut]
    // b  - biases [4*nOut], optional, may be nullptr
    // seqLen - [bS], optional, may be nullptr
    // hI - initial output  [bS, nOut], optional, may be nullptr
    // cI - initial cell state at time t-1 [bS, nOut], optional, may be nullptr
    // Wp - peephole weights [3*nOut], optional, may be nullptr

    // OUTPUTS:
    // h - output [sL, bS, nOut],  [bS, sL, nOut],  [bS, nOut, sL], optional, may be nullptr
    // hL - output at last step [bS, nOut], optional, may be nullptr
    // cL - cell state at last step [bS, nOut], optional, may be nullptr

    // params = {dataFormat, directionMode, cellClip, gateAct, gateAlpha, gateBeta, cellAct, cellAlpha, cellBeta, outAct, outAlpha, outBeta};
    // dataFormat: 0,3 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL]

    const int dataFormat    = params[0];
    const int directionMode = params[1];

    const Nd4jLong sL   = x->sizeAt(dataFormat);
    const Nd4jLong bS   = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(1);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    const std::vector<Nd4jLong> shapeOut = {bS, nOut};

    auto h0 = const_cast<NDArray*>(hI);
    if(!hI) {
        h0 = new NDArray(x->ordering(), shapeOut, x->dataType(), x->getContext());
        h0->nullify();
    }

    auto c0 = const_cast<NDArray*>(cI);
    if(!cI) {
        c0 = new NDArray(x->ordering(), shapeOut, x->dataType(), x->getContext());
        c0->nullify();
    }

    auto ct = cL;
    if(!cL)
        cL = new NDArray(x->ordering(), shapeOut, x->dataType(), x->getContext());

    auto ht = hL;
    if(!h && !hL)
        ht = new NDArray(x->ordering(), shapeOut, x->dataType(), x->getContext());

    // create sets of required (depends on seqLen presence) sub-arrays
    std::vector<int> dims;
    ResultSet *xSet(nullptr), *hSet(nullptr), *h0Set(nullptr), *c0Set(nullptr), *htSet(nullptr), *ctSet(nullptr);

    if(!seqLen) {

        dims = ShapeUtils::evalDimsToExclude(x->rankOf(), {dataFormat < 3 ? dataFormat : 0});    // points on bS and nIn/nOut axes

        xSet = new ResultSet(x->allTensorsAlongDimension(dims));       // sub-arrays with shape [bS, nIn]
        if(h)
            hSet = new ResultSet(h->allTensorsAlongDimension(dims));   // sub-arrays with shape [bS, nOut]
    }
    else {

        dims = dataFormat == 2 ? std::vector<int>({1}) : std::vector<int>({2});    // points on nIn/nOut axis

        xSet  = new ResultSet(x->allTensorsAlongDimension(dims));               //  sub-arrays with shape [nIn]
        h0Set = new ResultSet(h0->allTensorsAlongDimension({1}));              //  sub-arrays with shape [nOut]
        c0Set = new ResultSet(c0->allTensorsAlongDimension({1}));              //  sub-arrays with shape [nOut]
        ctSet = new ResultSet(ct->allTensorsAlongDimension({1}));              //  sub-arrays with shape [nOut]
        if(h)
            hSet = new ResultSet(h->allTensorsAlongDimension(dims));            //  sub-arrays with shape [nOut]
        if(ht)
            htSet = new ResultSet(ht->allTensorsAlongDimension({1}));          //  sub-arrays with shape [nOut]
    }

    // loops
    if(forward) {

        if(!seqLen) {

            if(!h) {    // seqLen and h are absent

                lstmLayerCell(xSet->at(0), Wx, Wr, b, h0, c0, Wp, params, ht, ct); // first time step
                for (Nd4jLong t = 1; t < sL; ++t)
                    lstmLayerCell(xSet->at(t), Wx, Wr, b, ht, ct, Wp, params, ht, ct); // rest time steps
            }
            else {      // seqLen is absent and h is present

                lstmLayerCell(xSet->at(0), Wx, Wr, b, h0, c0, Wp, params, hSet->at(0), ct); // first time step
                for (Nd4jLong t = 1; t < sL; ++t)
                    lstmLayerCell(xSet->at(t), Wx, Wr, b, hSet->at(t - 1), ct, Wp, params, hSet->at(t), ct); // rest time steps

                if(hL)
                    hL->assign(hSet->at(sL - 1));     // assign last output to hL if it is not nullptr
            }
        }
        else {

            if(!h) {    // seqLen is present and h is absent

                for (Nd4jLong e = 0; e < bS; ++e) {

                    const int limit = seqLen->e<int>(e);

                    if(limit == 0) {
                        if(cL)
                            ctSet->at(e)->nullify();
                        if(hL)
                            htSet->at(e)->nullify();
                        continue;
                    }

                    auto ind = getBatchTimeTotalIndex(dataFormat, sL, bS, 0, e);
                    lstmLayerCell(xSet->at(ind), Wx, Wr, b, h0Set->at(e), c0Set->at(e), Wp, params, htSet->at(e), ctSet->at(e)); // first time step

                    for (int t = 1; t < limit; ++t) {
                        ind = getBatchTimeTotalIndex(dataFormat, sL, bS, t, e);
                        lstmLayerCell(xSet->at(ind), Wx, Wr, b, htSet->at(e), ctSet->at(e), Wp, params, htSet->at(e), ctSet->at(e)); // rest time steps
                    }
                }
            }
            else { // seqLen and h are present

                for (Nd4jLong e = 0; e < bS; ++e) {

                    int limit = seqLen->e<int>(e);

                    if(limit == 0) {

                        tensorAlongTimeBatchDims(*h, dataFormat, 0,0, e,e+1).nullify();     // nullify for given e and whole time range

                        if(cL)
                            ctSet->at(e)->nullify();
                        if(hL)
                            htSet->at(e)->nullify();

                        continue;
                    }

                    auto indPrev = getBatchTimeTotalIndex(dataFormat, sL, bS, 0, e);
                    lstmLayerCell(xSet->at(indPrev), Wx, Wr, b, h0Set->at(e), c0Set->at(e), Wp, params, hSet->at(indPrev), ctSet->at(e)); // first time step

                    for (int t = 1; t < limit; ++t) {
                        auto indCurr = getBatchTimeTotalIndex(dataFormat, sL, bS, t, e);
                        lstmLayerCell(xSet->at(indCurr), Wx, Wr, b, hSet->at(indPrev), ctSet->at(e), Wp, params, hSet->at(indCurr), ctSet->at(e)); // rest time steps
                        indPrev = indCurr;
                    }

                    if(hL)
                        htSet->at(e)->assign(hSet->at(indPrev));     // assign last output to hL if hL is not nullptr

                    tensorAlongTimeBatchDims(*h, dataFormat, limit,sL, e,e+1).nullify();     // nullify for given e and time range [limit, sL)
                }
            }
        }
    }
    else {   // backward

         if(!seqLen) {

            if(!h) {    // seqLen and h are absent

                lstmLayerCell(xSet->at(sL - 1), Wx, Wr, b, h0, c0, Wp, params, ht, ct); // first time step
                for (Nd4jLong t = sL - 2; t >= 0; --t)
                    lstmLayerCell(xSet->at(t), Wx, Wr, b, ht, ct, Wp, params, ht, ct); // rest time steps
            }
            else {  // seqLen is absent and h is present

                lstmLayerCell(xSet->at(sL - 1), Wx, Wr, b, h0, c0, Wp, params, hSet->at(sL - 1), ct); // first time step
                for (Nd4jLong t = sL - 2; t >= 0; --t)
                    lstmLayerCell(xSet->at(t), Wx, Wr, b, hSet->at(t + 1), ct, Wp, params, hSet->at(t), ct); // rest time steps

                if(hL)
                    hL->assign(hSet->at(0));     // assign last output to hL if it is not nullptr
            }
        }
        else if(directionMode == 1) {   // only backward, no bidirectional mode

            if(!h) {    // h is absent and seqLen is present

                for (Nd4jLong e = 0; e < bS; ++e) {

                    const int limit = seqLen->e<int>(e);

                    if(limit == 0) {
                        if(cL)
                            ctSet->at(e)->nullify();
                        if(hL)
                            htSet->at(e)->nullify();
                        continue;
                    }

                    auto ind = getBatchTimeTotalIndex(dataFormat, sL, bS, sL - 1, e);
                    lstmLayerCell(xSet->at(ind), Wx, Wr, b, h0Set->at(e), c0Set->at(e), Wp, params, htSet->at(e), ctSet->at(e)); // first time step

                    for (Nd4jLong t = sL - 2; t >= sL - limit; --t) {
                        ind = getBatchTimeTotalIndex(dataFormat, sL, bS, t, e);
                        lstmLayerCell(xSet->at(ind), Wx, Wr, b, htSet->at(e), ctSet->at(e), Wp, params, htSet->at(e), ctSet->at(e)); // rest time steps
                    }
                }
            }
            else {  // seqLen and h are present

                for (Nd4jLong e = 0; e < bS; ++e) {

                    int limit = seqLen->e<int>(e);

                    if(limit == 0) {

                        tensorAlongTimeBatchDims(*h, dataFormat, 0,0, e,e+1).nullify();     // nullify for given e and whole time range

                        if(cL)
                            ctSet->at(e)->nullify();
                        if(hL)
                            htSet->at(e)->nullify();

                        continue;
                    }

                    auto indPrev = getBatchTimeTotalIndex(dataFormat, sL, bS, sL - 1, e);
                    lstmLayerCell(xSet->at(indPrev), Wx, Wr, b, h0Set->at(e), c0Set->at(e), Wp, params, hSet->at(indPrev), ctSet->at(e)); // first time step

                    for (Nd4jLong t = sL - 2; t >= sL - limit; --t) {
                        auto indCurr = getBatchTimeTotalIndex(dataFormat, sL, bS, t, e);
                        lstmLayerCell(xSet->at(indCurr), Wx, Wr, b, hSet->at(indPrev), ctSet->at(e), Wp, params, hSet->at(indCurr), ctSet->at(e)); // rest time steps
                        indPrev = indCurr;
                    }

                    if(hL)
                        htSet->at(e)->assign(hSet->at(indPrev));     // assign last output to hL if it is not nullptr

                    tensorAlongTimeBatchDims(*h, dataFormat, 0,sL-limit, e,e+1).nullify();    // nullify for given e and time range [limit, sL)
                }
            }
        }
        else {   // backward in bidirectional mode

            if(!h) {    // h is absent and seqLen is present

                for (Nd4jLong e = 0; e < bS; ++e) {

                    const int limit = seqLen->e<int>(e);

                    if(limit == 0) {
                        if(cL)
                            ctSet->at(e)->nullify();
                        if(hL)
                            htSet->at(e)->nullify();
                        continue;
                    }

                    auto ind = getBatchTimeTotalIndex(dataFormat, sL, bS, limit - 1, e);
                    lstmLayerCell(xSet->at(ind), Wx, Wr, b, h0Set->at(e), c0Set->at(e), Wp, params, htSet->at(e), ctSet->at(e)); // first time step

                    for (int t = limit - 2; t >= 0; --t) {
                        ind = getBatchTimeTotalIndex(dataFormat, sL, bS, t, e);
                        lstmLayerCell(xSet->at(ind), Wx, Wr, b, htSet->at(e), ctSet->at(e), Wp, params, htSet->at(e), ctSet->at(e)); // rest time steps
                    }
                }
            }
            else {  // seqLen and h are present

                for (Nd4jLong e = 0; e < bS; ++e) {

                    int limit = seqLen->e<int>(e);

                    if(limit == 0) {

                        tensorAlongTimeBatchDims(*h, dataFormat, 0,0, e,e+1).nullify();     // nullify for given e and whole time range

                        if(cL)
                            ctSet->at(e)->nullify();
                        if(hL)
                            htSet->at(e)->nullify();

                        continue;
                    }

                    auto indPrev = getBatchTimeTotalIndex(dataFormat, sL, bS, limit - 1, e);
                    lstmLayerCell(xSet->at(indPrev), Wx, Wr, b, h0Set->at(e), c0Set->at(e), Wp, params, hSet->at(indPrev), ctSet->at(e)); // first time step

                    for (int t = limit - 2; t >= 0; --t) {
                        auto indCurr = getBatchTimeTotalIndex(dataFormat, sL, bS, t, e);
                        lstmLayerCell(xSet->at(indCurr), Wx, Wr, b, hSet->at(indPrev), ctSet->at(e), Wp, params, hSet->at(indCurr), ctSet->at(e)); // rest time steps
                        indPrev = indCurr;
                    }

                    if(hL)
                        htSet->at(e)->assign(hSet->at(indPrev));     // assign last output to hL if it is not nullptr

                    tensorAlongTimeBatchDims(*h, dataFormat, limit,sL, e,e+1).nullify();    // nullify for given e and time range [limit, sL)
                }
            }
        }
    }

    delete xSet;
    delete hSet;
    delete h0Set;
    delete c0Set;
    delete htSet;
    delete ctSet;
}



}
}
}
