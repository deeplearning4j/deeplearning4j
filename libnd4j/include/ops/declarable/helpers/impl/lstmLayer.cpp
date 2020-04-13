/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
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
#include <execution/Threads.h>
#include <ops/declarable/helpers/activations.h>
#include <helpers/ShapeUtils.h>
#include <helpers/MmulHelper.h>
// #include <VariableSpace.h>
// #include <ops/declarable/CustomOperations.h>
// #include<ops/declarable/helpers/transforms.h>
// #include <ops/declarable/helpers/legacy_helpers.h>
// #include <array/NDArrayList.h>
// #include <iterator>


namespace sd 	  {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
static void applyActivation(const NDArray& x, const int opId, const float alpha, const float beta, NDArray& z) {

    switch (opId) {
        case 0:
            (const_cast<NDArray&>(x)).applyTransform(transform::Tanh, z);
            break;
        case 1:
            (const_cast<NDArray&>(x)).applyScalar<float>(scalar::RELU, 0, z);
            break;
        case 2:
            (const_cast<NDArray&>(x)).applyTransform(transform::Sigmoid, z);
            break;
        case 3: {
            ExtraArguments args({ static_cast<double>(alpha), static_cast<double>(beta)});
            (const_cast<NDArray&>(x)).applyTransform(transform::Affine, z, &args);
            break;
        }
        case 4:
            (const_cast<NDArray&>(x)).applyScalar<float>(scalar::LeakyRELU, alpha, z);
            break;
        case 5:
            thresholdRelu(x.getContext(), x, alpha, z);
            break;
        case 6: {
            ExtraArguments args({ static_cast<double>(alpha), static_cast<double>(beta)});
            (const_cast<NDArray&>(x)).applyTransform(transform::ScaledTanh, z, &args);
            break;
        }
        case 7:
            (const_cast<NDArray&>(x)).applyTransform(transform::HardSigmoid, z);
            break;
        case 8:
            (const_cast<NDArray&>(x)).applyScalar<float>(scalar::ELU, alpha, z);
            break;
        case 9:
            (const_cast<NDArray&>(x)).applyTransform(transform::SoftSign, z);
            break;
        case 10:
            (const_cast<NDArray&>(x)).applyTransform(transform::SoftPlus, z);
            break;
        default:
            throw std::invalid_argument("LSTM_LAYER operation: wrong id number of activation !");
    }
}

//////////////////////////////////////////////////////////////////////////
static void activationDeriv(const NDArray& x, const int opId, const float alpha, const float beta, NDArray& z) {

    switch (opId) {
        case 0:
            (const_cast<NDArray&>(x)).applyTransform(transform::TanhDerivative, z);
            break;
        case 1:
            (const_cast<NDArray&>(x)).applyScalar<float>(scalar::RELUDerivative, 0, z);
            break;
        case 2:
            (const_cast<NDArray&>(x)).applyTransform(transform::SigmoidDerivative, z);
            break;
        case 3: {
            z = alpha;
            break;
        }
        case 4:
            (const_cast<NDArray&>(x)).applyScalar<float>(scalar::LeakyRELUDerivative, alpha, z);
            break;
        case 5:
            (const_cast<NDArray&>(x)).applyScalar<float>(scalar::RELUDerivative, alpha, z);
            break;
        case 6: {
            auto func = PRAGMA_THREADS_FOR {
                for(Nd4jLong i = start; i < stop; ++i) {
                    auto val = beta * x.e<float>(i);
                    z.p<float>(i, alpha * beta * (1.f - sd::math::nd4j_tanh<float,float>(val) * sd::math::nd4j_tanh<float,float>(val)));
                }
            };
            samediff::Threads::parallel_for(func, 0, x.lengthOf());
            break;
        }
        case 7:
            (const_cast<NDArray&>(x)).applyTransform(transform::HardSigmoidDerivative, z);
            break;
        case 8:
            (const_cast<NDArray&>(x)).applyScalar<float>(scalar::ELUDerivative, alpha, z);
            break;
        case 9:
            (const_cast<NDArray&>(x)).applyTransform(transform::SoftSignDerivative, z);
            break;
        case 10: {
            auto func = PRAGMA_THREADS_FOR {
                for(Nd4jLong i = start; i < stop; ++i) {
                    auto val = sd::math::nd4j_exp<float, float>(x.e<float>(i));
                    z.p<float>(i, val / (1.f + val));
                }
            };
            samediff::Threads::parallel_for(func, 0, x.lengthOf());
            break;
        }
        default:
            throw std::invalid_argument("LSTM_LAYER operation: wrong id number of activation !");
    }
}

//////////////////////////////////////////////////////////////////////////
// FIXME - derivative undefined when not-clipped c has element/elements equal to -clipVal or clipVal
static void clipDeriv(const float clipVal, const NDArray& c, NDArray& z0, NDArray& z1, NDArray& z2, NDArray& z3) {

    if(clipVal == 0)
        return;

    auto func = PRAGMA_THREADS_FOR {
        for(Nd4jLong i = start; i < stop; ++i) {
            const auto val = c.e<float>(i);
            if(val == -clipVal || val == clipVal) {
                z0.p<float>(i, 0.f);
                z1.p<float>(i, 0.f);
                z2.p<float>(i, 0.f);
                z3.p<float>(i, 0.f);
            }
        }
    };
    samediff::Threads::parallel_for(func, 0, c.lengthOf());
}

//////////////////////////////////////////////////////////////////////////
static NDArray tensorAlongTimeBatchDims(const NDArray& arr, const int dataFormat, const int t1, const int t2, const int b1, const int b2) {

    if(dataFormat == 0 || dataFormat == 3)
        return arr({t1,t2, b1,b2, 0,0});    // TNS: [sL, bS, nIn]

    if(dataFormat == 1)
        return arr({b1,b2, t1,t2, 0,0});    // NTS: [bS, sL ,nIn]

    return arr({b1,b2, 0,0, t1,t2});        // NST: [bS, nIn, sL]
}

//////////////////////////////////////////////////////////////////////////
static FORCEINLINE int getBatchTimeTotalIndex(const int dataFormat, const int sL, const int bS, const int t, const int b) {

    if(dataFormat == 0 || dataFormat == 3)
        return t * bS + b;   // TNS: shape [sL, bS, nIn]

    return b * sL + t;       // NTS, NST: shape [bS, sL, nIn], [bS, nIn, sL]
}

//////////////////////////////////////////////////////////////////////////
// x{M,K} x y{K,N} = z{M,N}, dzdy{K,N,M,N} - Jacobian derivative  -> if x.rankOf() == 2
// x{K} x y{K,N} = z{N}, dzdy{K,N,N} - Jacobian derivative        -> if x.rankOf() == 1
static NDArray mmulJacobianWeightsDeriv(const int nOut, const NDArray& x) {

    std::vector<Nd4jLong> outShape = x.rankOf() == 1 ? std::vector<Nd4jLong>({x.sizeAt(0), nOut, nOut}) : std::vector<Nd4jLong>({x.sizeAt(1), nOut, x.sizeAt(0), nOut});

    NDArray dzdy(x.ordering(), outShape, x.dataType(), x.getContext());

    if(x.rankOf() == 1) {
        auto func = PRAGMA_THREADS_FOR_3D {

            for (auto i0 = start_x; i0 < stop_x; ++i0) {
                for (auto i1 = start_y; i1 < stop_y; ++i1) {
                    for (auto i2 = start_z; i2 < stop_z; ++i2) {
                        if(i1 == i2)
                            dzdy.p<double>(i0,i1,i2, x.e<double>(i0));
                        else
                            dzdy.p<double>(i0,i1,i2, 0);
                    }
                }
            }
        };

        samediff::Threads::parallel_for(func,  0,dzdy.sizeAt(0),1,  0,dzdy.sizeAt(1),1,  0,dzdy.sizeAt(2),1);
    }
    else {
        auto func = PRAGMA_THREADS_FOR_3D {

            for (auto i0 = start_x; i0 < stop_x; ++i0) {
                for (auto i1 = start_y; i1 < stop_y; ++i1) {
                    for (auto i2 = start_z; i2 < stop_z; ++i2) {
                        for (auto i3 = 0; i3 < dzdy.sizeAt(3); ++i3) {
                            if(i1 == i3)
                                dzdy.p<double>(i0,i1,i2,i3, x.e<double>(i2,i0));
                            else
                                dzdy.p<double>(i0,i1,i2,i3, 0);
                        }
                    }
                }
            }
        };

        samediff::Threads::parallel_for(func,  0,dzdy.sizeAt(0),1,  0,dzdy.sizeAt(1),1,  0,dzdy.sizeAt(2),1);
    }

    return dzdy;
}

//////////////////////////////////////////////////////////////////////////
void lstmLayerCell(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                   const NDArray* b, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                   const std::vector<float>& params,
                   NDArray* h, NDArray* c) {

    // * -> means element-wise multiplication
    // ^ -> means matrix multiplication

    /************************ THIS IS NOT OPTIMAZED CODE ***********************************/
    /** the objective is to provide math-readable code **/

    // equations (no peephole connections)
    // it  = σ(Wxi ^ xt  +  Wri ^ ht-1  +  bi)
    // ft  = σ(Wxf ^ xt  +  Wrf ^ ht-1  +  bf)
    // c't = tanh(Wxc ^ xt  +  Wrc ^ ht-1  +  bc)
    // ct  = ft * ct-1 + it * c't
    // ot  = σ(Wxo ^ xt  +  Wro ^ ht-1  +  bo)
    // ht  = ot * tanh(ct)

    // equations (peephole connections are present)
    // it  = σ(Wxi ^ xt  +  Wri ^ ht-1  +  Wpi * ct-1  +  bi)
    // ft  = σ(Wxf ^ xt  +  Wrf ^ ht-1  +  Wpf * ct-1  +  bf)
    // c't = tanh(Wxc ^ xt  +  Wrc ^ ht-1  +  bc)
    // ct  = ft * ct-1 + it * c't
    // ot  = σ(Wxo ^ xt  +  Wro ^ ht-1  +  Wpo * ct   +  bo)
    // ht  = ot * tanh(ct)


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
    // hI - (ht-1) previous (initial) output at time t-1, optional may be nullptr, [bS, nOut] or [nOut] if seqLen != nullptr
    // cI - (ct-1) previous (initial) cell state at time t-1, optional may be nullptr, [bS, nOut] or [nOut] if seqLen != nullptr
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
        z += *b;                                    // broadcast [bS, 4*nOut](or[4*nOut]) + [4*nOut] = [bS, 4*nOut]

    auto zi = x->rankOf() == 1 ? z({0,        nOut}) : z({0,0, 0,        nOut});         // input gate it, [bS, nOut](or[nOut])
    auto zf = x->rankOf() == 1 ? z({nOut,   2*nOut}) : z({0,0, nOut,   2*nOut});         // forget gate ft, [bS, nOut](or[nOut])
    auto zg = x->rankOf() == 1 ? z({2*nOut, 3*nOut}) : z({0,0, 2*nOut, 3*nOut});         // cell gate c't, [bS, nOut](or[nOut])
    auto zo = x->rankOf() == 1 ? z({3*nOut, 4*nOut}) : z({0,0, 3*nOut, 4*nOut});         // output gate ot, [bS, nOut](or[nOut])

    // peephole connections for input and forget gates
    if(Wp != nullptr) {
        zi += *cI * (*Wp)({0,      nOut});      // broadcast: [bS, nOut] + [bS, nOut] * [nOut] = [bS, nOut](or[nOut])
        zf += *cI * (*Wp)({nOut, 2*nOut});      // broadcast: [bS, nOut] + [bS, nOut] * [nOut] = [bS, nOut](or[nOut])
    }

    applyActivation(zi, params[3], params[4], params[5], zi);   // inplace
    applyActivation(zf, params[3], params[4], params[5], zf);   // inplace
    applyActivation(zg, params[6], params[7], params[8], zg);   // inplace

    c->assign(zf * *cI + zi * zg);          // [bS, nOut] * [bS, nOut] + [bS, nOut] * [bS, nOut] = [bS, nOut](or[nOut])

    // if clipping value is non-zero then cell state is clipped by this value prior to the cell output activation
    if(params[2] != 0)
        c->applyScalar(scalar::LstmClip, params[2], *c);

    // peephole connections for output gate
    if(Wp != nullptr)
        zo += *c * (*Wp)({2*nOut, 3*nOut});    // broadcast: [bS, nOut] + [bS, nOut] * [nOut] = [bS, nOut](or[nOut])

    applyActivation(zo, params[3], params[4], params[5], zo);

    applyActivation(*c, params[9], params[10], params[11], *h);
    *h *= zo;                               // [bS, nOut] * [bS, nOut](or[nOut])
}


//////////////////////////////////////////////////////////////////////////
// this auxiliary ff should be running before backprop
void lstmLayerCell(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                   const NDArray* b, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                   const std::vector<float>& params,
                   NDArray* z, NDArray* a, NDArray* h, NDArray* c) {

    // z - zi, zf, zg, zo
    // a - i, f, g, o

    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    z->assign(mmul(*x, *Wx) + mmul(*hI, *Wr));    //   [bs, nIn] * [nIn, 4*nOut] + [bs, nOut] * [nOut, 4*nOut] = [bS, 4*nOut]
                                                  //or [nIn] * [nIn, 4*nOut] + [nOut] * [nOut, 4*nOut] = [4*nOut]
    // add biases if they are given
    if(b != nullptr)
        *z += *b;                                    // broadcast [bS, 4*nOut](or[4*nOut]) + [4*nOut] = [bS, 4*nOut]

    auto zi = x->rankOf() == 1 ? (*z)({0,        nOut}) : (*z)({0,0, 0,        nOut});         // input gate it, [bS, nOut](or[nOut])
    auto zf = x->rankOf() == 1 ? (*z)({nOut,   2*nOut}) : (*z)({0,0, nOut,   2*nOut});         // forget gate ft, [bS, nOut](or[nOut])
    auto zg = x->rankOf() == 1 ? (*z)({2*nOut, 3*nOut}) : (*z)({0,0, 2*nOut, 3*nOut});         // cell gate c't, [bS, nOut](or[nOut])
    auto zo = x->rankOf() == 1 ? (*z)({3*nOut, 4*nOut}) : (*z)({0,0, 3*nOut, 4*nOut});         // output gate ot, [bS, nOut](or[nOut])

    auto i = x->rankOf() == 1 ? (*a)({0,        nOut}) : (*a)({0,0, 0,        nOut});         // input gate it, [bS, nOut](or[nOut])
    auto f = x->rankOf() == 1 ? (*a)({nOut,   2*nOut}) : (*a)({0,0, nOut,   2*nOut});         // forget gate ft, [bS, nOut](or[nOut])
    auto g = x->rankOf() == 1 ? (*a)({2*nOut, 3*nOut}) : (*a)({0,0, 2*nOut, 3*nOut});         // cell gate c't, [bS, nOut](or[nOut])
    auto o = x->rankOf() == 1 ? (*a)({3*nOut, 4*nOut}) : (*a)({0,0, 3*nOut, 4*nOut});         // output gate ot, [bS, nOut](or[nOut])

    // peephole connections for input and forget gates
    if(Wp != nullptr) {
        zi += *cI * (*Wp)({0,      nOut});      // broadcast: [bS, nOut] + [bS, nOut] * [nOut] = [bS, nOut](or[nOut])
        zf += *cI * (*Wp)({nOut, 2*nOut});      // broadcast: [bS, nOut] + [bS, nOut] * [nOut] = [bS, nOut](or[nOut])
    }

    applyActivation(zi, params[3], params[4], params[5], i);
    applyActivation(zf, params[3], params[4], params[5], f);
    applyActivation(zg, params[6], params[7], params[8], g);

    c->assign(f * *cI + i * g);          // [bS, nOut] * [bS, nOut] + [bS, nOut] * [bS, nOut] = [bS, nOut](or[nOut])

    // if clipping value is non-zero then cell state is clipped by this value prior to the cell output activation
    if(params[2] != 0)
        c->applyScalar(scalar::LstmClip, params[2], *c);

    // peephole connections for output gate
    if(Wp != nullptr)
        zo += *c * (*Wp)({2*nOut, 3*nOut});    // broadcast: [bS, nOut] + [bS, nOut] * [nOut] = [bS, nOut](or[nOut])

    applyActivation(zo, params[3], params[4], params[5], o);

    applyActivation(*c, params[9], params[10], params[11], *h);
    *h *= o;                               // [bS, nOut] * [bS, nOut](or[nOut])
}


//////////////////////////////////////////////////////////////////////////
void lstmLayerCellBp(const NDArray* x, const NDArray* Wx, const NDArray* Wr, const NDArray* b, const NDArray* hI, const NDArray* cI, const NDArray* Wp,
                     const NDArray* dLdh, const NDArray* dLdc,
                     const NDArray* z, const NDArray* a, const NDArray* c, const std::vector<float>& params,
                     NDArray* dLdx, NDArray* dLdWx, NDArray* dLdWr, NDArray* dLdhI, NDArray* dLdcI, NDArray* dLdb, NDArray* dLdWp) {

    /************************ THIS IS NOT OPTIMAZED CODE ***********************************/
    /** the objective is to provide math-readable code **/

    // equations (no peephole connections)
    // zi = x ^ Wxi + hI ^ Wri + bi
    // zf = x ^ Wxf + hI ^ Wrf + bf
    // zg = x ^ Wxg + hI ^ Wrg + bg
    // zo = x ^ Wxo + hI ^ Wro + bo
    // i = act(zi)
    // f = act(zf)
    // g = actC(zg)
    // o = act(zo)
    // c = clip(f * cI + i * g)
    // h = o * actH(c)

    // equations (peephole connections are present)
    // zi = x ^ Wxi + hI ^ Wri + cI * Wpi + bi
    // zf = x ^ Wxf + hI ^ Wrf + cI * Wpf + bf
    // zg = x ^ Wxg + hI ^ Wrg + bg
    // zo = x ^ Wxo + hI ^ Wro + c  * Wpo + bo
    // i = act(zi)
    // f = act(zf)
    // g = actC(zg)
    // o = act(zo)
    // c = clip(f * cI + i * g)
    // h = o * actH(c)

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
    // x    - current input at time t, [bS, nIn] or [nIn] if seqLen != nullptr
    // Wx   - input weights [nIn, 4*nOut]
    // Wr   - recurrent weights [nOut, 4*nOut]
    // b    - biases [4*nOut], optional, may be nullptr
    // hI   - (ht-1) previous (initial) output at time t-1, [bS, nOut] or [nOut] if seqLen != nullptr
    // cI   - (ct-1) previous (initial) cell state at time t-1, [bS, nOut] or [nOut] if seqLen != nullptr
    // Wp   - peephole weights [3*nOut], optional, may be nullptr
    // dLdh - loss derivative with respect to h, [bS, nOut] or [nOut] if seqLen != nullptr
    // dLdc - loss derivative with respect to c, [bS, nOut] or [nOut] if seqLen != nullptr
    // z    - zi,zf,zg,zo taken from ff outputs to reduce amount of calculations in bp, [bS, 4*nOut]
    // a    - i,f,g,o taken from ff outputs to reduce amount of calculations in bp, [bS, 4*nOut]
    // c    - taken from ff outputs to reduce amount of calculations in bp, [bS, nOut]

    // OUTPUTS:
    // dLdx  - loss derivative with respect to x, [bS, nIn] or [nIn] if seqLen != nullptr
    // dLdWx - loss derivative with respect to Wx, [nIn, 4*nOut]
    // dLdWr - loss derivative with respect to Wr, [nOut, 4*nOut]
    // dLdb  - loss derivative with respect to b, optional, may be nullptr, [4*nOut]
    // dLdhI - loss derivative with respect to hI, optional may be nullptr, [bS, nOut] or [nOut] if seqLen != nullptr
    // dLdcI - loss derivative with respect to cI, optional may be nullptr, [bS, nOut] or [nOut] if seqLen != nullptr
    // dLdWp - loss derivative with respect to Wp, optional, may be nullptr,  [3*nOut]

    // !!! dimension 4*nOut implies order i, f, g, o
    // !!! dimension 3*nOut implies order i, f, o

    // dhdc  = o*tanhDeriv + Wp ? tanh(c)*dodzo*dzodc : 0       [bS, nOut]
    // dcdcI = f + Wp ? dcdzi*dzidcI + dcdzf*dzfdcI : 0         [bS, nOut]

    // dLdhI += dLdh;                       [bS, nOut]
    // dLdcI += dLdhI * dhdc;               [bS, nOut]

    // dLdzi = dLdcI*dcdi*didzi;            [bS, nOut](or[nOut])
    // dLdzf = dLdcI*dcdf*dfdzf;            [bS, nOut](or[nOut])
    // dLdzg = dLdcI*dcdg*dgdzg;            [bS, nOut](or[nOut])
    // dLdzo = dLdhI*dhdo*dodzo;            [bS, nOut](or[nOut])

    // dLdx  = dLdzi^WxiT + dLdzf^WxfT + dLdzg^WxgT + dLdzo^WxoT,   [bS, nIn]
    // dLdhI = dLdzi^WriT + dLdzf^WrfT + dLdzg^WrgT + dLdzo^WroT,   [bS, nOut]
    // dLdcI = dLdcI*dcdcI,                                         [bS, nOut]

    // dLdWxi = xT^dLdzi                            [nIn, bS] x [bS, nOut] = [nIn, nOut]
    // dLdWxf = xT^dLdzf                            [nIn, bS] x [bS, nOut] = [nIn, nOut]
    // dLdWxg = xT^dLdzg                            [nIn, bS] x [bS, nOut] = [nIn, nOut]
    // dLdWxo = xT^dLdzo                            [nIn, bS] x [bS, nOut] = [nIn, nOut]

    // dLdWri = hIT^dLdzi                           [nOut, bS] x [bS, nOut] = [nOut, nOut]
    // dLdWrf = hIT^dLdzf                           [nOut, bS] x [bS, nOut] = [nOut, nOut]
    // dLdWrg = hIT^dLdzg                           [nOut, bS] x [bS, nOut] = [nOut, nOut]
    // dLdWro = hIT^dLdzo                           [nOut, bS] x [bS, nOut] = [nOut, nOut]

    //  dLdbi = dLdzi.reduce_sum_along_0_axis       [bS, nOut] -> reduce -> [nOut]
    //  dLdbf = dLdzf.reduce_sum_along_0_axis       [bS, nOut] -> reduce -> [nOut]
    //  dLdbg = dLdzg.reduce_sum_along_0_axis       [bS, nOut] -> reduce -> [nOut]
    //  dLdbo = dLdzo.reduce_sum_along_0_axis       [bS, nOut] -> reduce -> [nOut]

    //  dLdWpi = (dLdzi*cI).reduce_sum_along_0_axis       [bS, nOut] -> reduce -> [nOut]
    //  dLdWpf = (dLdzf*cI).reduce_sum_along_0_axis       [bS, nOut] -> reduce -> [nOut]
    //  dLdWpo = (dLdzo*c) .reduce_sum_along_0_axis       [bS, nOut] -> reduce -> [nOut]

    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;
    const Nd4jLong nIn  = x->sizeAt(-1);

    NDArray zi = x->rankOf() == 1 ? (*z)({0,        nOut}) : (*z)({0,0, 0,        nOut});         // input gate i, [bS, nOut](or[nOut])
    NDArray zf = x->rankOf() == 1 ? (*z)({nOut,   2*nOut}) : (*z)({0,0, nOut,   2*nOut});         // forget gate f, [bS, nOut](or[nOut])
    NDArray zg = x->rankOf() == 1 ? (*z)({2*nOut, 3*nOut}) : (*z)({0,0, 2*nOut, 3*nOut});         // cell gate g, [bS, nOut](or[nOut])
    NDArray zo = x->rankOf() == 1 ? (*z)({3*nOut, 4*nOut}) : (*z)({0,0, 3*nOut, 4*nOut});         // output gate o, [bS, nOut](or[nOut])

    NDArray i = x->rankOf() == 1 ? (*a)({0,        nOut}) : (*a)({0,0, 0,        nOut});         // input gate i, [bS, nOut](or[nOut])
    NDArray f = x->rankOf() == 1 ? (*a)({nOut,   2*nOut}) : (*a)({0,0, nOut,   2*nOut});         // forget gate f, [bS, nOut](or[nOut])
    NDArray g = x->rankOf() == 1 ? (*a)({2*nOut, 3*nOut}) : (*a)({0,0, 2*nOut, 3*nOut});         // cell gate g, [bS, nOut](or[nOut])
    NDArray o = x->rankOf() == 1 ? (*a)({3*nOut, 4*nOut}) : (*a)({0,0, 3*nOut, 4*nOut});         // output gate o, [bS, nOut](or[nOut])

    NDArray dLdz = z->ulike();                                                                  // [bS, 4*nOut](or[4*nOut])
    NDArray dLdzi = x->rankOf() == 1 ? dLdz({0,        nOut}) : dLdz({0,0, 0,        nOut});
    NDArray dLdzf = x->rankOf() == 1 ? dLdz({nOut,   2*nOut}) : dLdz({0,0, nOut,   2*nOut});
    NDArray dLdzg = x->rankOf() == 1 ? dLdz({2*nOut, 3*nOut}) : dLdz({0,0, 2*nOut, 3*nOut});
    NDArray dLdzo = x->rankOf() == 1 ? dLdz({3*nOut, 4*nOut}) : dLdz({0,0, 3*nOut, 4*nOut});

    // dcdzi = dcdi*didzi, [bS, nOut](or[nOut])
    activationDeriv(zi, params[3], params[4], params[5], dLdzi);     // didzi, inplace
    dLdzi *= g;                                                      // dcdi = g*clipDeriv

    // dcdzf = dcdf*dfdzf, [bS, nOut](or[nOut])
    activationDeriv(zf, params[3], params[4], params[5], dLdzf);     // dfdzf, inplace
    dLdzf *= *cI;                                                    // dcdf = cI*clipDeriv

    // dcdzg = dcde*dedzg, [bS, nOut](or[nOut])
    activationDeriv(zg, params[6], params[7], params[8], dLdzg);    // dgdzg, inplace
    dLdzg *= i;                                                     // dcdf = i*clipDeriv

    // dhdzo = dhdo*dodzo = actH(c)*dodzo, [bS, nOut](or[nOut])
    activationDeriv(zo, params[3], params[4], params[5], dLdzo);
    NDArray temp = dLdzo.ulike();
    applyActivation(*c, params[9], params[10], params[11], temp);     // actH(c), inplace
    dLdzo *= temp;

    // dcdcI
    NDArray dcdcI = f.dup();                                           // dcdcI = f*clipDeriv [bS, nOut](or[nOut])

    // take into account possible deposit from clipping derivative
    clipDeriv(params[2], *c, dLdzi, dLdzf, dLdzg, dcdcI);

    // dhdc
    NDArray dhdc = c->ulike();
    activationDeriv(*c, params[9], params[10], params[11], dhdc);    // [bS, nOut]
    dhdc *= o;

    if(Wp) {
        dhdc  += dLdzo*(*Wp)({2*nOut, 3*nOut});
        dcdcI += dLdzi*(*Wp)({0, nOut}) + dLdzf*(*Wp)({nOut, 2*nOut});     // broadcast [bS, nOut] * nOut + ...
    }

    if(dLdh)
        *dLdhI += *dLdh;
    if(dLdc)
        *dLdcI += *dLdc;
    else
        *dLdcI += *dLdhI * dhdc;

    dLdzi *= *dLdcI;     // [bS, nOut](or[nOut])
    dLdzf *= *dLdcI;     // [bS, nOut](or[nOut])
    dLdzg *= *dLdcI;     // [bS, nOut](or[nOut])
    dLdzo *= *dLdhI;     // [bS, nOut](or[nOut])

    // dLdx
    NDArray WxT = Wx->transpose();
    MmulHelper::mmul(&dLdz, &WxT, dLdx);        // [bS, 4*nOut] x [4*nOut, nIn] (or [4*nOut] x [4*nOut, nIn]) = [bS, nIn] ( or[nIn] )

    // dLdhI
    NDArray WrT = Wr->transpose();
    MmulHelper::mmul(&dLdz, &WrT, dLdhI);       // [bS, 4*nOut] x [4*nOut, nOut] (or [4*nOut] x [4*nOut, nOut]) = [bS, nOut] ( or[nOut] )

    // dLdcI
    dLdcI->assign(*dLdcI*dcdcI);                                            // [bS, nOut](or[nOut])

    if(x->rankOf() == 1) {

        NDArray xT  = x->reshape(x->ordering(),{nIn, 1});               // [nIn]  -> [nIn, 1]
        NDArray hIT = hI->reshape(hI->ordering(),{nOut, 1});            // [nOut] -> [nOut, 1]
        NDArray dLdzR = dLdz.reshape(dLdz.ordering(), {1, 4*nOut});     // [nOut] -> [1, 4*nOut]

        // dLdWx
        *dLdWx += mmul(xT, dLdzR);        // [nIn, 1] x [1, 4*nOut] = [nIn, 4*nOut]

        // dLdWr
        *dLdWr += mmul(hIT, dLdzR);        // [nOut, 1] x [1, 4*nOut] = [nOut, 4*nOut]
    }
    else {

        // dLdWx
        *dLdWx += mmul(x->transpose(), dLdz);         // [nIn, bS] x [bS, 4*nOut] = [nIn, 4*nOut]

        // dLdWr
        *dLdWr += mmul(hI->transpose(), dLdz);        // [nOut, bS] x [bS, 4*nOut] = [nOut, 4*nOut]
    }

    // dLdb
    if(b && x->rankOf() == 1)
        *dLdb += dLdz;                                          // [4*nOut]
    else if(b)
        *dLdb += dLdz.reduceAlongDimension(reduce::Sum, {0});   // [bS, 4*nOut] -> reduce -> [4*nOut];

    // dLdWp
    if(Wp && x->rankOf() == 1) {
        (*dLdWp)({       0,nOut}) += std::move(dLdzi)*(*cI);          // [nOut]
        (*dLdWp)({  nOut,2*nOut}) += std::move(dLdzf)*(*cI);          // [nOut]
        (*dLdWp)({2*nOut,3*nOut}) += std::move(dLdzo)*(*c);           // [nOut]
    }
    else if(Wp) {
        NDArray temp(Wp->ordering(), {nOut}, Wp->dataType(), Wp->getContext());
        (std::move(dLdzi)*(*cI)).reduceAlongDimension(reduce::Sum, temp, {0});     // [bS, nOut] -> reduce -> [nOut]
        (*dLdWp)({0,nOut}) += temp;
        (std::move(dLdzf)*(*cI)).reduceAlongDimension(reduce::Sum, temp, {0});     // [bS, nOut] -> reduce -> [nOut]
        (*dLdWp)({nOut,2*nOut}) += temp;
        (std::move(dLdzo)*(*c)).reduceAlongDimension(reduce::Sum, temp, {0});     // [bS, nOut] -> reduce -> [nOut]
        (*dLdWp)({2*nOut,3*nOut}) += temp;
    }
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

    const Nd4jLong sL   = dataFormat == 3 ?  x->sizeAt(0) : x->sizeAt(dataFormat);
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
        ct = new NDArray(x->ordering(), shapeOut, x->dataType(), x->getContext());

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

                    if(limit != sL)
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

                    if(limit != sL)
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

                    if(limit != sL)
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

    if(!hI)
        delete h0;
    if(!cI)
        delete c0;
    if(!cL)
        delete ct;
    if(!h && !hL)
        delete ht;
}


//////////////////////////////////////////////////////////////////////////
void lstmLayerTimeLoopBp(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                         const NDArray* b, const NDArray* seqLen, NDArray* hI, NDArray* cI, const NDArray* Wp,
                         const NDArray* dLdh, const NDArray* dLdhL, const NDArray* dLdcL,
                         const std::vector<float>& params, const bool forward,
                         NDArray* dLdx, NDArray* dLdWx, NDArray* dLdWr, NDArray* dLdb, NDArray* dLdhI, NDArray* dLdcI, NDArray* dLdWp) {

    // INPUTS:
    // x  - current input  [sL, bS, nIn],  [bS, sL, nIn],  [bS, nIn, sL],
    // Wx - input weights [nIn, 4*nOut]
    // Wr - recurrent weights [nOut, 4*nOut]
    // b  - biases [4*nOut], optional, may be nullptr
    // seqLen - [bS], optional, may be nullptr
    // hI - initial output  [bS, nOut], optional, may be nullptr
    // cI - initial cell state at time t-1 [bS, nOut], optional, may be nullptr
    // Wp - peephole weights [3*nOut], optional, may be nullptr
    // dLdh - gradient vs. output [sL, bS, nOut],  [bS, sL, nOut],  [bS, nOut, sL], optional, may be nullptr
    // dLdhL - gradient vs. output at last time step [bS, nOut], optional, may be nullptr
    // dLdcL - gradient vs. cell state at last time step [bS, nOut], optional, may be nullptr

    // OUTPUTS:
    // dLdx  - gradient vs. input [sL, bS, nIn],  [bS, sL, nIn],  [bS, nIn, sL]
    // dLdWx - gradient vs. input weights [nIn, 4*nOut]
    // dLdWr - gradient vs. recurrent weights [nOut, 4*nOut]
    // dLdb  - gradient vs. biases [4*nOut], optional, may be nullptr
    // dLdhI - gradient vs. initial output  [bS, nOut], optional, may be nullptr
    // dLdcI - gradient vs. initial cell state at time t-1 [bS, nOut], optional, may be nullptr
    // dLdWp - gradient vs. peephole weights [3*nOut], optional, may be nullptr

    // params = {dataFormat, directionMode, cellClip, gateAct, gateAlpha, gateBeta, cellAct, cellAlpha, cellBeta, outAct, outAlpha, outBeta};
    // dataFormat: 0,3 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL]

    const int dataFormat    = params[0];
    const int directionMode = params[1];

    const int sL = dataFormat == 3 ?  x->sizeAt(0) : x->sizeAt(dataFormat);
    const int bS = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(1);
    const int nOut = Wx->sizeAt(-1) / 4;

    auto dLdh0 = dLdhI;
    if(!hI)
        dLdh0 = new NDArray(x->ordering(), {bS, nOut}, x->dataType(), x->getContext());     // this constructor nullifies array automatically

    auto dLdc0 = dLdcI;
    if(!cI)
        dLdc0 = new NDArray(x->ordering(), {bS, nOut}, x->dataType(), x->getContext());     // this constructor nullifies array automatically

    NDArray z(x->ordering(), {sL, bS, 4*nOut}, x->dataType(), x->getContext());
    NDArray a = z.ulike();
    NDArray h(x->ordering(), {sL+1, bS, nOut}, x->dataType(), x->getContext());
    NDArray c = h.ulike();

    // create sets of required (depends on seqLen presence) sub-arrays
    std::vector<int> dims;
    ResultSet *xSet(nullptr), *dLdxSet(nullptr), *hSet(nullptr), *cSet(nullptr), *zSet(nullptr), *aSet(nullptr), *dLdhSet(nullptr),
              *dLdh0Set(nullptr), *dLdc0Set(nullptr), *dLdhLSet(nullptr), *dLdcLSet(nullptr), *hISet(nullptr), *cISet(nullptr);

    if(!seqLen) {

        dims = ShapeUtils::evalDimsToExclude(x->rankOf(), {dataFormat < 3 ? dataFormat : 0});    // points on [bS, nIn/nOut]

        xSet    = new ResultSet(x->allTensorsAlongDimension(dims));         // sub-arrays with shape [bS, nIn]
        dLdxSet = new ResultSet(dLdx->allTensorsAlongDimension(dims));      // sub-arrays with shape [bS, nIn]
        hSet    = new ResultSet(h.allTensorsAlongDimension({1, 2}));        // sub-arrays with shape [bS, nOut]
        cSet    = new ResultSet(c.allTensorsAlongDimension({1, 2}));        // sub-arrays with shape [bS, nOut]
        zSet    = new ResultSet(z.allTensorsAlongDimension({1, 2}));        // sub-arrays with shape [bS, 4*nOut]
        aSet    = new ResultSet(a.allTensorsAlongDimension({1, 2}));        // sub-arrays with shape [bS, 4*nOut]
        if(dLdh)
            dLdhSet = new ResultSet(dLdh->allTensorsAlongDimension(dims));  // sub-arrays with shape [bS, nOut]
    }
    else {

        dims = dataFormat == 2 ? std::vector<int>({1}) : std::vector<int>({2});    // points on nIn/nOut axis

        xSet    = new ResultSet(x->allTensorsAlongDimension(dims));         // sub-arrays with shape [nIn]
        dLdxSet = new ResultSet(dLdx->allTensorsAlongDimension(dims));      // sub-arrays with shape [nIn]
        hSet    = new ResultSet(h.allTensorsAlongDimension({2}));           // sub-arrays with shape [nOut]
        cSet    = new ResultSet(c.allTensorsAlongDimension({2}));           // sub-arrays with shape [nOut]
        zSet    = new ResultSet(z.allTensorsAlongDimension({2}));           // sub-arrays with shape [4*nOut]
        aSet    = new ResultSet(a.allTensorsAlongDimension({2}));           // sub-arrays with shape [4*nOut]

        if(hI)
            hISet = new ResultSet(hI->allTensorsAlongDimension({1}));      // sub-arrays with shape [nOut]
        if(cI)
            cISet = new ResultSet(cI->allTensorsAlongDimension({1}));      // sub-arrays with shape [nOut]

        dLdh0Set = new ResultSet(dLdh0->allTensorsAlongDimension({1}));      // sub-arrays with shape [nOut]
        dLdc0Set = new ResultSet(dLdc0->allTensorsAlongDimension({1}));      // sub-arrays with shape [nOut]

        if(dLdh)
            dLdhSet = new ResultSet(dLdh->allTensorsAlongDimension(dims));  // sub-arrays with shape [nOut]
        if(!dLdh && dLdhL)
            dLdhLSet = new ResultSet(dLdhL->allTensorsAlongDimension({1}));  // sub-arrays with shape [nOut]
        if(!dLdh && !dLdhL)
            dLdcLSet = new ResultSet(dLdcL->allTensorsAlongDimension({1}));  // sub-arrays with shape [nOut]
    }


    // loops
    if(forward) {

        if(!seqLen) {   // seqLen is absent

            if(hI)
                h({0,1, 0,0, 0,0}).assign(hI);
            else
                h({0,1, 0,0, 0,0}).nullify();
            if(cI)
                c({0,1, 0,0, 0,0}).assign(cI);
            else
                c({0,1, 0,0, 0,0}).nullify();

            // ff
            for (int t = 0; t < sL; ++t)
                lstmLayerCell(xSet->at(t), Wx, Wr, b, hSet->at(t), cSet->at(t), Wp, params, zSet->at(t), aSet->at(t), hSet->at(t+1), cSet->at(t+1));

            // bp
            for (int t = sL-1; t >= 0; --t) {
                const NDArray* dLdhh = dLdh  ? dLdhSet->at(t) : (t == sL-1 ? dLdhL : nullptr);
                const NDArray* dLdcc = dLdhh ? nullptr : (t == sL-1 ? dLdcL : nullptr);
                lstmLayerCellBp(xSet->at(t), Wx, Wr, b, hSet->at(t), cSet->at(t), Wp, dLdhh, dLdcc,
                                zSet->at(t), aSet->at(t), cSet->at(t+1), params, dLdxSet->at(t), dLdWx, dLdWr, dLdh0, dLdc0, dLdb, dLdWp);
            }
        }
        else {  // seqLen is present

            for (int e = 0; e < bS; ++e) {

                const int limit = seqLen->e<int>(e);

                if(limit == 0) {
                    tensorAlongTimeBatchDims(*dLdx, dataFormat, 0,0, e,e+1).nullify();     // nullify for given e and whole time range
                    continue;
                }

                if(hI)
                    h({0,1, e,e+1, 0,0}).assign(hISet->at(e));
                else
                    h({0,1, e,e+1, 0,0}).nullify();
                if(cI)
                    c({0,1, e,e+1, 0,0}).assign(cISet->at(e));
                else
                    c({0,1, e,e+1, 0,0}).nullify();

                // ff
                for (int t = 0; t < limit; ++t)
                    lstmLayerCell(xSet->at(getBatchTimeTotalIndex(dataFormat, sL, bS, t, e)), Wx, Wr, b, hSet->at(t*bS + e), cSet->at(t*bS + e), Wp, params,
                                  zSet->at(t*bS + e), aSet->at(t*bS + e), hSet->at((t+1)*bS + e), cSet->at((t+1)*bS + e));

                // bp
                for (int t = limit-1; t >= 0; --t) {
                    const auto ind = getBatchTimeTotalIndex(dataFormat, sL, bS, t, e);
                    const NDArray* dLdhh = dLdh  ? dLdhSet->at(ind) : (t == limit-1 && dLdhL ? dLdhLSet->at(e) : nullptr);
                    const NDArray* dLdcc = dLdhh ? nullptr : (t == limit-1 ? dLdcLSet->at(e) : nullptr);
                    lstmLayerCellBp(xSet->at(ind), Wx, Wr, b, hSet->at(t*bS + e), cSet->at(t*bS + e), Wp, dLdhh, dLdcc,
                                    zSet->at(t*bS + e), aSet->at(t*bS + e), cSet->at((t+1)*bS + e), params, dLdxSet->at(ind), dLdWx, dLdWr,
                                    dLdh0Set->at(e), dLdc0Set->at(e), dLdb, dLdWp);
                }

               if(limit != sL)
                    tensorAlongTimeBatchDims(*dLdx, dataFormat, limit,sL, e,e+1).nullify();     // nullify for given e and time range [limit, sL)
            }
        }
    }
    else {   // backward or bidirectional

        if(!seqLen) {     // backward or bidirectional, seqLen is absent

            if(hI)
                h({sL,sL+1, 0,0, 0,0}).assign(hI);
            else
                h({sL,sL+1, 0,0, 0,0}).nullify();
            if(cI)
                c({sL,sL+1, 0,0, 0,0}).assign(cI);
            else
                c({sL,sL+1, 0,0, 0,0}).nullify();

            // ff
            for (int t = sL-1; t >= 0; --t)
                lstmLayerCell(xSet->at(t), Wx, Wr, b, hSet->at(t+1), cSet->at(t+1), Wp, params, zSet->at(t), aSet->at(t), hSet->at(t), cSet->at(t));

            // bp
            for (int t = 0; t < sL; ++t) {
                const NDArray* dLdhh = dLdh  ? dLdhSet->at(t) : (t == 0 ? dLdhL : nullptr);
                const NDArray* dLdcc = dLdhh ? nullptr : (t == 0 ? dLdcL : nullptr);
                lstmLayerCellBp(xSet->at(t), Wx, Wr, b, hSet->at(t+1), cSet->at(t+1), Wp, dLdhh, dLdcc,
                                zSet->at(t), aSet->at(t), cSet->at(t), params, dLdxSet->at(t), dLdWx, dLdWr, dLdh0, dLdc0, dLdb, dLdWp);
            }
        }
        else if(directionMode == 1) {   // backward, seqLen is present

            for (int e = 0; e < bS; ++e) {

                const int limit = seqLen->e<int>(e);

                if(limit == 0) {
                    tensorAlongTimeBatchDims(*dLdx, dataFormat, 0,0, e,e+1).nullify();     // nullify for given e and whole time range
                    continue;
                }

                if(hI)
                    h({sL,sL+1, e,e+1, 0,0}).assign(hISet->at(e));
                else
                    h({sL,sL+1, e,e+1, 0,0}).nullify();
                if(cI)
                    c({sL,sL+1, e,e+1, 0,0}).assign(cISet->at(e));
                else
                    c({sL,sL+1, e,e+1, 0,0}).nullify();

                // ff
                for (int t = sL - 1; t >= sL-limit; --t)
                    lstmLayerCell(xSet->at(getBatchTimeTotalIndex(dataFormat, sL, bS, t, e)), Wx, Wr, b, hSet->at((t+1)*bS + e), cSet->at((t+1)*bS + e), Wp, params,
                                  zSet->at(t*bS + e), aSet->at(t*bS + e), hSet->at(t*bS + e), cSet->at(t*bS + e));

                // bp
                for (int t = sL-limit; t < sL; ++t) {
                    const auto ind = getBatchTimeTotalIndex(dataFormat, sL, bS, t, e);
                    const NDArray* dLdhh = dLdh  ? dLdhSet->at(ind) : (t == sL-limit && dLdhL ? dLdhLSet->at(e) : nullptr);
                    const NDArray* dLdcc = dLdhh ? nullptr : (t == sL-limit ? dLdcLSet->at(e) : nullptr);
                    lstmLayerCellBp(xSet->at(ind), Wx, Wr, b, hSet->at((t+1)*bS + e), cSet->at((t+1)*bS + e), Wp, dLdhh, dLdcc,
                                    zSet->at(t*bS + e), aSet->at(t*bS + e), cSet->at(t*bS + e), params, dLdxSet->at(ind), dLdWx, dLdWr,
                                    dLdh0Set->at(e), dLdc0Set->at(e), dLdb, dLdWp);
                }

                if(limit != sL)
                    tensorAlongTimeBatchDims(*dLdx, dataFormat, 0,sL-limit, e,e+1).nullify();    // nullify for given e and time range [limit, sL)
            }
        }
        else {   // bidirectional mode, seqLen is present

            for (int e = 0; e < bS; ++e) {

                const int limit = seqLen->e<int>(e);

                if(limit == 0) {
                    tensorAlongTimeBatchDims(*dLdx, dataFormat, 0,0, e,e+1).nullify();     // nullify for given e and whole time range
                    continue;
                }

                if(hI)
                    h({limit,limit+1, e,e+1, 0,0}).assign(hISet->at(e));
                else
                    h({limit,limit+1, e,e+1, 0,0}).nullify();
                if(cI)
                    c({limit,limit+1, e,e+1, 0,0}).assign(cISet->at(e));
                else
                    c({limit,limit+1, e,e+1, 0,0}).nullify();

                // ff
                for (int t = limit - 1; t >= 0; --t)
                    lstmLayerCell(xSet->at(getBatchTimeTotalIndex(dataFormat, sL, bS, t, e)), Wx, Wr, b, hSet->at((t+1)*bS + e), cSet->at((t+1)*bS + e), Wp, params,
                                  zSet->at(t*bS + e), aSet->at(t*bS + e), hSet->at(t*bS + e), cSet->at(t*bS + e));

                // bp
                for (int t = 0; t < limit; ++t) {
                    const auto ind = getBatchTimeTotalIndex(dataFormat, sL, bS, t, e);
                    const NDArray* dLdhh = dLdh  ? dLdhSet->at(ind) : (t == 0 && dLdhL ? dLdhLSet->at(e) : nullptr);
                    const NDArray* dLdcc = dLdhh ? nullptr : (t == 0 ? dLdcLSet->at(e) : nullptr);
                    lstmLayerCellBp(xSet->at(ind), Wx, Wr, b, hSet->at((t+1)*bS + e), cSet->at((t+1)*bS + e), Wp, dLdhh, dLdcc,
                                    zSet->at(t*bS + e), aSet->at(t*bS + e), cSet->at(t*bS + e), params, dLdxSet->at(ind), dLdWx, dLdWr,
                                    dLdh0Set->at(e), dLdc0Set->at(e), dLdb, dLdWp);
                }

                if(limit != sL)
                    tensorAlongTimeBatchDims(*dLdx, dataFormat, limit,sL, e,e+1).nullify();    // nullify for given e and time range [limit, sL)
            }
        }
    }

    delete xSet; delete dLdxSet; delete hSet; delete cSet; delete aSet; delete zSet;
    delete dLdhSet; delete dLdh0Set; delete dLdc0Set; delete dLdhLSet; delete dLdcLSet; delete hISet; delete cISet;

    if(!hI)
        delete dLdh0;
    if(!cI)
        delete dLdc0;
}


}
}
}



//////////////////////////////////////////////////////////////////////////
// void lstmLayerCellBp(const NDArray* x,    const NDArray* Wx, const NDArray* Wr,
//                      const NDArray* b,          NDArray* hI,       NDArray* cI, const NDArray* Wp, const NDArray* dLdh,
//                      const std::vector<float>& params, const bool firstIter,

//                      NDArray* dhIdcI, NDArray* dhIdWx, NDArray* dcIdWx, NDArray* dhIdWr, NDArray* dcIdWr,
//                      NDArray* dhIdb, NDArray* dcIdb, NDArray* dhIdWp, NDArray* dcIdWp,
//                      NDArray* dLdx, NDArray* dLdWx, NDArray* dLdWr, NDArray* dLdhI, NDArray* dLdcI, NDArray* dLdb, NDArray* dLdWp) {

//     /************************ THIS IS NOT OPTIMAZED CODE ***********************************/
//     /** the objective is to provide math-readable code **/

//     // equations (no peephole connections)
//     // zi = x ^ Wxi + hI ^ Wri + bi
//     // zf = x ^ Wxf + hI ^ Wrf + bf
//     // zg = x ^ Wxg + hI ^ Wrg + bg
//     // zo = x ^ Wxo + hI ^ Wro + bo
//     // i = act(zi)
//     // f = act(zf)
//     // g = actC(zg)
//     // o = act(zo)
//     // c = clip(f * cI + i * g)
//     // h = o * actH(c)

//     // equations (peephole connections are present)
//     // zi = x ^ Wxi + hI ^ Wri + cI * Wpi + bi
//     // zf = x ^ Wxf + hI ^ Wrf + cI * Wpf + bf
//     // zg = x ^ Wxg + hI ^ Wrg + bg
//     // zo = x ^ Wxo + hI ^ Wro + c  * Wpo + bo
//     // i = act(zi)
//     // f = act(zf)
//     // g = actC(zg)
//     // o = act(zo)
//     // c = clip(f * cI + i * g)
//     // h = o * actH(c)

//     // IDs for activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus

//     // params[0] - dataFormat, ignore
//     // params[1] - directionMode, ignore
//     // params[2] - cell clipping value, if it = 0 then do not apply clipping

//     // params[3]  - activation ID for input (i), forget (f) and output (o) gates
//     // params[4]  - alpha value for gates activation
//     // params[5]  - beta value for gates activation

//     // params[6]  - activation ID for cell state (c)
//     // params[7]  - alpha value for cell state activation
//     // params[8]  - beta value for cell state activation

//     // params[9]  - activation ID for output (h)
//     // params[10] - alpha value for output activation
//     // params[11] - beta value for output activation

//     // INPUTS:
//     // x    - current input at time t, [bS, nIn] or [nIn] if seqLen != nullptr
//     // Wx   - input weights [nIn, 4*nOut]
//     // Wr   - recurrent weights [nOut, 4*nOut]
//     // b    - biases [4*nOut], optional, may be nullptr
//     // hI   - (ht-1) previous (initial) output at time t-1, [bS, nOut] or [nOut] if seqLen != nullptr
//     // cI   - (ct-1) previous (initial) cell state at time t-1, [bS, nOut] or [nOut] if seqLen != nullptr
//     // Wp   - peephole weights [3*nOut], optional, may be nullptr
//     // dLdh - loss derivative with respect to h, [bS, nOut] or [nOut] if seqLen != nullptr
//     // dhIdcI - derivative from previous time step, [bS, nOut] or [nOut] if seqLen != nullptr
//     // dhIdWx - derivative from previous time step (Jacobian), [nIn, 4*nOut, bS, nOut] or [nIn, 4*nOut, nOut] if seqLen != nullptr
//     // dcIdWx - derivative from previous time step (Jacobian), [nIn, 4*nOut, bS, nOut] or [nIn, 4*nOut, nOut] if seqLen != nullptr
//     // dhIdWr - derivative from previous time step (Jacobian), [nOut, 4*nOut, bS, nOut] or [nOut, 4*nOut, nOut] if seqLen != nullptr
//     // dcIdWr - derivative from previous time step (Jacobian), [nOut, 4*nOut, bS, nOut] or [nOut, 4*nOut, nOut] if seqLen != nullptr
//     // dcIdWp - derivative from previous time step, [3*nOut], optional, may be nullptr
//     // dhIdWp - derivative from previous time step, [3*nOut], optional, may be nullptr
//     // dcIdb  - derivative from previous time step, [4*nOut], optional, may be nullptr
//     // dhIdb  - derivative from previous time step, [4*nOut], optional, may be nullptr

//     // OUTPUTS:
//     // dLdx  - loss derivative with respect to x, [bS, nIn] or [nIn] if seqLen != nullptr
//     // dLdWx - loss derivative with respect to Wx, [nIn, 4*nOut]
//     // dLdWr - loss derivative with respect to Wr, [nOut, 4*nOut]
//     // dLdb  - loss derivative with respect to b, optional, may be nullptr, [4*nOut]
//     // dLdhI - loss derivative with respect to hI, optional may be nullptr, [bS, nOut] or [nOut] if seqLen != nullptr
//     // dLdcI - loss derivative with respect to cI, optional may be nullptr, [bS, nOut] or [nOut] if seqLen != nullptr
//     // dLdWp - loss derivative with respect to Wp, optional, may be nullptr,  [3*nOut]

//     // !!! dimension 4*nOut implies order i, f, g, o
//     // !!! dimension 3*nOut implies order i, f, o

//     // dcdzi = dcdi*didzi
//     // dcdzf = dcdf*dfdzf
//     // dcdzg = dcdg*dgdzg
//     // dhdzo = dhdo*dodzo

//     // dhdc = dhdc + Wp ? dhdzo*dzodc : 0           [bS, nOut]
//     // factor = dLdh*dhdc                           [bS, nOut]
//     // iFactor = factor*dcdzi                       [bS, nOut]
//     // fFactor = factor*dcdzf                       [bS, nOut]
//     // eFactor = factor*dcdzg                       [bS, nOut]
//     // oFactor = *dLdh*dhdzo                        [bS, nOut]

//     // tempC   = dcdcI + Wp ? dcdzi*dzidcI + dcdzf*dzfdcI : 0;
//     // tempIFE = dcdzi^WriT + dcdzf^WrfT + dcdzg^WrgT
//     // tempO   = dhdzo^WroT

//     // dhIdcI = dhdc_from_previous_time_step

//     // dLdx   = iFactor^WxiT + fFactor^WxfT + eFactor^WxgT + oFactor^WxoT,      [bS, nIn]
//     // dLdhI  = iFactor^WriT + fFactor^WrfT + eFactor^WrgT + oFactor^WroT,      [bS, nOut]
//     // dLdcI  = factor*tempC + dLdhI * dhIdcI, dhIdcI=0 if firstIter,           [bS, nOut]

//     // dcdWxi(dcIdWxi) = dcdzi*dzidWxi + tempIFE*dhIdWxi + tempC*dcIdWxi,           dcIdWxi=dhIdWxi= 0 if firstIter, [nIn, nOut, bS, nOut]
//     // dcdWxf(dcIdWxf) = dcdzf*dzfdWxf + tempIFE*dhIdWxf + tempC*dcIdWxf,           dcIdWxf=dhIdWxf= 0 if firstIter, [nIn, nOut, bS, nOut]
//     // dcdWxg(dcIdWxg) = dcdzg*dzgdWxg + tempIFE*dhIdWxg + tempC*dcIdWxg,           dcIdWxg=dhIdWxg= 0 if firstIter, [nIn, nOut, bS, nOut]
//     // dcdWxo(dcIdWxo) =       0       + tempIFE*dhIdWxo + tempC*dcIdWxo;           dcIdWxo=dhIdWxo= 0 if firstIter, [nIn, nOut, bS, nOut]

//     // dhdWxi(dhIdWxi) =       0       + dhdc*dcdWxi + tempO*dhIdWxi,           dhIdWxi= 0 if firstIter, [nIn, nOut, bS, nOut]
//     // dhdWxf(dhIdWxf) =       0       + dhdc*dcdWxf + tempO*dhIdWxf,           dhIdWxf= 0 if firstIter, [nIn, nOut, bS, nOut]
//     // dhdWxg(dhIdWxg) =       0       + dhdc*dcdWxg + tempO*dhIdWxg,           dhIdWxg= 0 if firstIter, [nIn, nOut, bS, nOut]
//     // dhdWxo(dhIdWxo) = dhdzo*dzodWxo + dhdc*dcdWxo + tempO*dhIdWxo,           dhIdWxo= 0 if firstIter, [nIn, nOut, bS, nOut]

//     // dhdWri(dhIdWri) =       0       + dhdc*dcdWri + tempO*dhIdWri,           dhIdWri= 0 if firstIter, [nOut, nOut, bS, nOut]
//     // dhdWrf(dhIdWrf) =       0       + dhdc*dcdWrf + tempO*dhIdWrf,           dhIdWrf= 0 if firstIter, [nOut, nOut, bS, nOut]
//     // dhdWrg(dhIdWrg) =       0       + dhdc*dcdWrg + tempO*dhIdWrg,           dhIdWrg= 0 if firstIter, [nOut, nOut, bS, nOut]
//     // dhdWro(dhIdWro) = dhdzo*dzodWro + dhdc*dcdWro + tempO*dhIdWro,           dhIdWro= 0 if firstIter, [nOut, nOut, bS, nOut]

//     // dcdWri(dcIdWri) = dcdzi*dzidWri + tempIFE*dhIdWri + tempC*dcIdWri,           dcIdWri=dhIdWri= 0 if firstIter, [nOut, nOut, bS, nOut]
//     // dcdWrf(dcIdWrf) = dcdzf*dzfdWrf + tempIFE*dhIdWrf + tempC*dcIdWrf,           dcIdWri=dhIdWri= 0 if firstIter, [nOut, nOut, bS, nOut]
//     // dcdWrg(dcIdWrg) = dcdzg*dzgdWrg + tempIFE*dhIdWrg + tempC*dcIdWrg,           dcIdWri=dhIdWri= 0 if firstIter, [nOut, nOut, bS, nOut]
//     // dcdWro(dcIdWro) =       0       + tempIFE*dhIdWro + tempC*dcIdWro;           dcIdWro=dhIdWro= 0 if firstIter, [nOut, nOut, bS, nOut]

//     // dcIdWpi = (dcdzi*cI + tempIFE*dhIdWpi + tempC*dcIdWpi).reduceALongFirstDim,  dcIdWpi=dhIdWpi= 0 if firstIter, [bS, nOut]->reduce->[bS]
//     // dcIdWpf = (dcdzf*cI + tempIFE*dhIdWpf + tempC*dcIdWpf).reduceALongFirstDim,  dcIdWpf=dhIdWpf= 0 if firstIter, [bS, nOut]->reduce->[bS]
//     // dcIdWpo = (0        + tempIFE*dhIdWpo + tempC*dcIdWpo).reduceALongFirstDim,  dcIdWpo=dhIdWpo= 0 if firstIter, [bS, nOut]->reduce->[bS]

//     // dhdWpi(dhIdWpi) =(   0    + dhdc*dcdWpi + tempO*dhIdWpi).reduceALongFirstDim,           dhIdWpi= 0 if firstIter, [bS, nOut]->reduce->[bS]
//     // dhdWpf(dhIdWpf) =(   0    + dhdc*dcdWpf + tempO*dhIdWpf).reduceALongFirstDim,           dhIdWpf= 0 if firstIter, [bS, nOut]->reduce->[bS]
//     // dhdWpo(dhIdWpo) =(dhdzo*c + dhdc*dcdWpo + tempO*dhIdWpo).reduceALongFirstDim,           dhIdWpo= 0 if firstIter, [bS, nOut]->reduce->[bS]

//     // dcdbi(dcIdbi) = (dcdzi + tempIFE*dhIdbi + tempC*dcIdbi).reduceALongFirstDim,           dcIdbi=dhIdbi= 0 if firstIter, [bS, nOut]->reduce->[bS]
//     // dcdbf(dcIdbf) = (dcdzf + tempIFE*dhIdbf + tempC*dcIdbf).reduceALongFirstDim,           dcIdbf=dhIdbf= 0 if firstIter, [bS, nOut]->reduce->[bS]
//     // dcdbg(dcIdbg) = (dcdzg + tempIFE*dhIdbg + tempC*dcIdbg).reduceALongFirstDim,           dcIdbg=dhIdbg= 0 if firstIter, [bS, nOut]->reduce->[bS]
//     // dcdbo(dcIdbo) = (  0   + tempIFE*dhIdbo + tempC*dcIdbo).reduceALongFirstDim;           dcIdbo=dhIdbo= 0 if firstIter, [bS, nOut]->reduce->[bS]

//     // dhdbi(dhIdbi) = (  0   + dhdc*dcdbi + tempO*dhIdbi).reduceALongFirstDim,           dhIdbi= 0 if firstIter, [bS, nOut]->reduce->[bS]
//     // dhdbf(dhIdbf) = (  0   + dhdc*dcdbf + tempO*dhIdbf).reduceALongFirstDim,           dhIdbf= 0 if firstIter, [bS, nOut]->reduce->[bS]
//     // dhdbg(dhIdbg) = (  0   + dhdc*dcdbg + tempO*dhIdbg).reduceALongFirstDim,           dhIdbg= 0 if firstIter, [bS, nOut]->reduce->[bS]
//     // dhdbo(dhIdbo) = (dhdzo + dhdc*dcdbo + tempO*dhIdbo).reduceALongFirstDim,           dhIdbo= 0 if firstIter, [bS, nOut]->reduce->[bS]

//     const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

//     NDArray *Wpi(nullptr), *Wpf(nullptr), *Wpo(nullptr), *dcIdWpi(nullptr), *dcIdWpf(nullptr), *dcIdWpo(nullptr), *dhIdWpi(nullptr), *dhIdWpf(nullptr), *dhIdWpo(nullptr);
//     if(Wp) {
//         Wpi = new NDArray((*Wp)({0, nOut}));
//         Wpf = new NDArray((*Wp)({nOut, 2*nOut}));
//         Wpo = new NDArray((*Wp)({2*nOut, 3*nOut}));
//         dhIdWpi = new NDArray((*dhIdWp)({0, nOut}));
//         dhIdWpf = new NDArray((*dhIdWp)({nOut, 2*nOut}));
//         dhIdWpo = new NDArray((*dhIdWp)({2*nOut, 3*nOut}));
//         dcIdWpi = new NDArray((*dcIdWp)({0, nOut}));
//         dcIdWpf = new NDArray((*dcIdWp)({nOut, 2*nOut}));
//         dcIdWpo = new NDArray((*dcIdWp)({2*nOut, 3*nOut}));
//     }

//     NDArray *dcIdbi(nullptr), *dcIdbf(nullptr), *dcIdbg(nullptr), *dcIdbo(nullptr), *dhIdbi(nullptr), *dhIdbf(nullptr), *dhIdbg(nullptr), *dhIdbo(nullptr);
//     if(b) {
//         dhIdbi = new NDArray((*dhIdb)({0, nOut}));
//         dhIdbf = new NDArray((*dhIdb)({nOut, 2*nOut}));
//         dhIdbg = new NDArray((*dhIdb)({2*nOut, 3*nOut}));
//         dhIdbo = new NDArray((*dhIdb)({3*nOut, 4*nOut}));
//         dcIdbi = new NDArray((*dcIdb)({0, nOut}));
//         dcIdbf = new NDArray((*dcIdb)({nOut, 2*nOut}));
//         dcIdbg = new NDArray((*dcIdb)({2*nOut, 3*nOut}));
//         dcIdbo = new NDArray((*dcIdb)({3*nOut, 4*nOut}));
//     }

//     NDArray dhIdWxi = x->rankOf() == 1 ? (*dhIdWx)({0,0,  0,nOut,        0,0}) : (*dhIdWx)({0,0,  0,nOut,        0,0, 0,0});    // [nIn, nOut, bS, nOut] or [nIn, nOut, nOut] if seqLen != nullptr
//     NDArray dhIdWxf = x->rankOf() == 1 ? (*dhIdWx)({0,0,  nOut,2*nOut,   0,0}) : (*dhIdWx)({0,0,  nOut,2*nOut,   0,0, 0,0});    // [nIn, nOut, bS, nOut] or [nIn, nOut, nOut] if seqLen != nullptr
//     NDArray dhIdWxg = x->rankOf() == 1 ? (*dhIdWx)({0,0,  2*nOut,3*nOut, 0,0}) : (*dhIdWx)({0,0,  2*nOut,3*nOut, 0,0, 0,0});    // [nIn, nOut, bS, nOut] or [nIn, nOut, nOut] if seqLen != nullptr
//     NDArray dhIdWxo = x->rankOf() == 1 ? (*dhIdWx)({0,0,  3*nOut,4*nOut, 0,0}) : (*dhIdWx)({0,0,  3*nOut,4*nOut, 0,0, 0,0});    // [nIn, nOut, bS, nOut] or [nIn, nOut, nOut] if seqLen != nullptr

//     NDArray dhIdWri = x->rankOf() == 1 ? (*dhIdWr)({0,0,  0,nOut,        0,0}) : (*dhIdWr)({0,0,  0,nOut,        0,0, 0,0});    // [nOut, nOut, bS, nOut] or [nOut, nOut, nOut] if seqLen != nullptr
//     NDArray dhIdWrf = x->rankOf() == 1 ? (*dhIdWr)({0,0,  nOut,2*nOut,   0,0}) : (*dhIdWr)({0,0,  nOut,2*nOut,   0,0, 0,0});    // [nOut, nOut, bS, nOut] or [nOut, nOut, nOut] if seqLen != nullptr
//     NDArray dhIdWrg = x->rankOf() == 1 ? (*dhIdWr)({0,0,  2*nOut,3*nOut, 0,0}) : (*dhIdWr)({0,0,  2*nOut,3*nOut, 0,0, 0,0});    // [nOut, nOut, bS, nOut] or [nOut, nOut, nOut] if seqLen != nullptr
//     NDArray dhIdWro = x->rankOf() == 1 ? (*dhIdWr)({0,0,  3*nOut,4*nOut, 0,0}) : (*dhIdWr)({0,0,  3*nOut,4*nOut, 0,0, 0,0});    // [nOut, nOut, bS, nOut] or [nOut, nOut, nOut] if seqLen != nullptr

//     NDArray dcIdWxi = x->rankOf() == 1 ? (*dcIdWx)({0,0,  0,nOut,        0,0}) : (*dcIdWx)({0,0,  0,nOut,        0,0, 0,0});    // [nIn, nOut, bS, nOut] or [nIn, nOut, nOut] if seqLen != nullptr
//     NDArray dcIdWxf = x->rankOf() == 1 ? (*dcIdWx)({0,0,  nOut,2*nOut,   0,0}) : (*dcIdWx)({0,0,  nOut,2*nOut,   0,0, 0,0});    // [nIn, nOut, bS, nOut] or [nIn, nOut, nOut] if seqLen != nullptr
//     NDArray dcIdWxg = x->rankOf() == 1 ? (*dcIdWx)({0,0,  2*nOut,3*nOut, 0,0}) : (*dcIdWx)({0,0,  2*nOut,3*nOut, 0,0, 0,0});    // [nIn, nOut, bS, nOut] or [nIn, nOut, nOut] if seqLen != nullptr
//     NDArray dcIdWxo = x->rankOf() == 1 ? (*dcIdWx)({0,0,  3*nOut,4*nOut, 0,0}) : (*dcIdWx)({0,0,  3*nOut,4*nOut, 0,0, 0,0});    // [nIn, nOut, bS, nOut] or [nIn, nOut, nOut] if seqLen != nullptr

//     NDArray dcIdWri = x->rankOf() == 1 ? (*dcIdWr)({0,0,  0,nOut,        0,0}) : (*dcIdWr)({0,0,  0,nOut,        0,0, 0,0});    // [nOut, nOut, bS, nOut] or [nOut, nOut, nOut] if seqLen != nullptr
//     NDArray dcIdWrf = x->rankOf() == 1 ? (*dcIdWr)({0,0,  nOut,2*nOut,   0,0}) : (*dcIdWr)({0,0,  nOut,2*nOut,   0,0, 0,0});    // [nOut, nOut, bS, nOut] or [nOut, nOut, nOut] if seqLen != nullptr
//     NDArray dcIdWrg = x->rankOf() == 1 ? (*dcIdWr)({0,0,  2*nOut,3*nOut, 0,0}) : (*dcIdWr)({0,0,  2*nOut,3*nOut, 0,0, 0,0});    // [nOut, nOut, bS, nOut] or [nOut, nOut, nOut] if seqLen != nullptr
//     NDArray dcIdWro = x->rankOf() == 1 ? (*dcIdWr)({0,0,  3*nOut,4*nOut, 0,0}) : (*dcIdWr)({0,0,  3*nOut,4*nOut, 0,0, 0,0});    // [nOut, nOut, bS, nOut] or [nOut, nOut, nOut] if seqLen != nullptr

//     NDArray WxiT = (*Wx)({0,0,  0,     nOut}).transpose();              // [nOut, nIn]
//     NDArray WxfT = (*Wx)({0,0,  nOut,  2*nOut}).transpose();            // [nOut, nIn]
//     NDArray WxgT = (*Wx)({0,0,  2*nOut,3*nOut}).transpose();            // [nOut, nIn]
//     NDArray WxoT = (*Wx)({0,0,  3*nOut,4*nOut}).transpose();            // [nOut, nIn]

//     NDArray WriT = (*Wr)({0,0,  0,     nOut}).transpose();              // [nOut, nOut]
//     NDArray WrfT = (*Wr)({0,0,  nOut,  2*nOut}).transpose();            // [nOut, nOut]
//     NDArray WrgT = (*Wr)({0,0,  2*nOut,3*nOut}).transpose();            // [nOut, nOut]
//     NDArray WroT = (*Wr)({0,0,  3*nOut,4*nOut}).transpose();            // [nOut, nOut]

//     // ***** feed forward step ***** //

//     auto z = mmul(*x, *Wx) + mmul(*hI, *Wr);    //   [bs, nIn] * [nIn, 4*nOut] + [bs, nOut] * [nOut, 4*nOut] = [bS, 4*nOut]
//                                                 //or [nIn] * [nIn, 4*nOut] + [nOut] * [nOut, 4*nOut] = [4*nOut]
//     // add biases if they are given
//     if(b)
//         z += *b;                                    // broadcast [bS, 4*nOut] + [4*nOut] = [bS, 4*nOut](or[4*nOut])

//     auto zi = x->rankOf() == 1 ? z({0,        nOut}) : z({0,0, 0,        nOut});         // input gate i, [bS, nOut](or[nOut])
//     auto zf = x->rankOf() == 1 ? z({nOut,   2*nOut}) : z({0,0, nOut,   2*nOut});         // forget gate f, [bS, nOut](or[nOut])
//     auto zg = x->rankOf() == 1 ? z({2*nOut, 3*nOut}) : z({0,0, 2*nOut, 3*nOut});         // cell gate g, [bS, nOut](or[nOut])
//     auto zo = x->rankOf() == 1 ? z({3*nOut, 4*nOut}) : z({0,0, 3*nOut, 4*nOut});         // output gate o, [bS, nOut](or[nOut])

//     // peephole connections for input and forget gates
//     if(Wp) {
//         zi += *cI * *Wpi;      // broadcast: [bS, nOut] + [bS, nOut] * [nOut] = [bS, nOut](or[nOut])
//         zf += *cI * *Wpf;      // broadcast: [bS, nOut] + [bS, nOut] * [nOut] = [bS, nOut](or[nOut])
//     }

//     NDArray i = zi.ulike();        // [bS, nOut]
//     NDArray f = zf.ulike();        // [bS, nOut]
//     NDArray g = zg.ulike();        // [bS, nOut]
//     applyActivation(zi, params[3], params[4], params[5], i);
//     applyActivation(zf, params[3], params[4], params[5], f);
//     applyActivation(zg, params[6], params[7], params[8], g);

//     NDArray c = f * *cI + i * g;          // [bS, nOut] * [bS, nOut] + [bS, nOut] * [bS, nOut] = [bS, nOut](or[nOut])

//     // if clipping value is non-zero then cell state is clipped by this value prior to the cell output activation
//     if(params[2] != 0)
//         c.applyScalar(scalar::LstmClip, params[2], c);

//     // peephole connections for output gate
//     if(Wp)
//         zo += c * *Wpo;    // broadcast: [bS, nOut] + [bS, nOut] * [nOut] = [bS, nOut](or[nOut])

//     NDArray o = zo.ulike();     // [bS, nOut](or[nOut])
//     applyActivation(zo, params[3], params[4], params[5], o);

//     // ***** back prop step ***** //

//     NDArray dWxJacobian = mmulJacobianWeightsDeriv(nOut, *x);       // [nIn,  nOut, bS, nOut] (or [nIn,  nOut, nOut])
//     NDArray dWrJacobian = mmulJacobianWeightsDeriv(nOut, *hI);      // [nOut, nOut, bS, nOut] (or [nOut, nOut, nOut])

//     // dodzo
//     NDArray dodzo = zo.ulike();                                              // [bS, nOut](or[nOut])
//     activationDeriv(zo, params[3], params[4], params[5], dodzo);

//     // dhdzo = dhdo*dodzo = actH(c)*dodzo
//     NDArray dhdzo = zo.ulike();                                              // [bS, nOut](or[nOut])
//     applyActivation(c, params[9], params[10], params[11], dhdzo);            // actH(c)
//     hI->assign(o*dhdzo);
//     dhdzo *= dodzo;

//     // dcdzi = dcdi*didzi
//     NDArray dcdzi = zi.ulike();                                             // [bS, nOut](or[nOut])
//     activationDeriv(zi, params[3], params[4], params[5], dcdzi);            // didzi
//     dcdzi *= g;                                                             // dcdi = g*clipDeriv

//     // dcdzf = dcdf*dfdzf
//     NDArray dcdzf = zf.ulike();                                             // [bS, nOut](or[nOut])
//     activationDeriv(zf, params[3], params[4], params[5], dcdzf);            // dfdzf
//     dcdzf *= *cI;                                                           // dcdf = cI*clipDeriv

//     // dcdzg = dcde*dedzg
//     NDArray dcdzg = zg.ulike();                                             // [bS, nOut](or[nOut])
//     activationDeriv(zg, params[6], params[7], params[8], dcdzg);            // dedzg
//     dcdzg *= i;                                                             // dcdf = i*clipDeriv

//     // dcdcI
//     NDArray dcdcI = f.dup();                                                // [bS, nOut](or[nOut])

//     // take into account possible deposit from clipping derivative
//     clipDeriv(params[2], c, dcdzi, dcdzf, dcdzg, dcdcI);

//     // dzodc
//     NDArray* dzodc = Wpo;                           // [nOut], should be [bS, nOut] actually, however it will be broadcasted appropriately in future calcus (element-wise multiplication)

//     // dzidcI
//     NDArray* dzidcI = Wpi;                          // [nOut], should be [bS, nOut] actually, however it will be broadcasted appropriately in future calcus (element-wise multiplication)

//     // dzfdcI
//     NDArray* dzfdcI = Wpf;                          // [nOut], should be [bS, nOut] actually, however it will be broadcasted appropriately in future calcus (element-wise multiplication)

//     // dhdc
//     NDArray dhdc = c.ulike();
//     activationDeriv(c, params[9], params[10], params[11], dhdc);    // [bS, nOut]
//     dhdc *= o;
//     if(Wp)
//         dhdc += dhdzo* *dzodc;

//     NDArray factor = *dLdh * dhdc;

//     NDArray iFactor = factor*dcdzi;     // [bS, nOut](or[nOut])
//     NDArray fFactor = factor*dcdzf;     // [bS, nOut](or[nOut])
//     NDArray eFactor = factor*dcdzg;     // [bS, nOut](or[nOut])
//     NDArray oFactor = *dLdh *dhdzo;     // [bS, nOut](or[nOut])

//     NDArray tempC = dcdcI;
//     if(Wp)
//         tempC += dcdzi*(*dzidcI) + dcdzf*(*dzfdcI);

//     // dLdx
//     dLdx->assign(mmul(iFactor, WxiT) + mmul(fFactor, WxfT) + mmul(eFactor, WxgT) + mmul(oFactor, WxoT));    // [bS, nIn](or[nOut])
//     // NDArray temp = c.ulike();
//     // applyActivation(c, params[9], params[10], params[11], temp);            // actH(c)
//     // dLdx->assign(mmul(o*(1-temp*temp)*g*i*(1-i), WxiT) + mmul(o*(1-temp*temp)*(*cI)*f*(1-f), WxfT) + mmul(o*(1-temp*temp)*i*g*(1-g), WxgT) + mmul(temp*o*(1-o), WxoT));    // [bS, nIn](or[nOut])

//     // dLdhI
//     NDArray* dLdhII = dLdhI;
//     if(dLdcI && !dLdhI)
//         dLdhII = new NDArray(dLdcI->ulike());
//     dLdhII->assign(mmul(iFactor, WriT) + mmul(fFactor, WrfT) + mmul(eFactor, WrgT) + mmul(oFactor, WroT));   // [bS, nOut](or[nOut])

//     if(firstIter) {

//         // dLdcI
//         if(dLdcI)
//             dLdcI->assign(factor*tempC);                        // [bS, nOut](or[nOut])

//         // dcIdWxi(dcdWxi)
//         dcIdWxi.assign(dcdzi*dWxJacobian);      // broadcast [bS, nOut] * [nIn, nOut, bS, nOut] (or [nOut] * [nIn, nOut, nOut]);
//         // dcIdWxf(dcdWxf)
//         dcIdWxf.assign(dcdzf*dWxJacobian);
//         // dcIdWxg(dcdWxg)
//         dcIdWxg.assign(dcdzg*dWxJacobian);
//         // dcIdWxo(dcdWxo) = 0
//         dcIdWxo.nullify();

//         // dhIdWxi
//         dhIdWxi.assign(dhdc*dcIdWxi);           // broadcast [bS, nOut] * [nIn, nOut, bS, nOut] (or [nOut] * [nIn, nOut, nOut]);
//         // dhIdWxf
//         dhIdWxf.assign(dhdc*dcIdWxf);
//         // dhIdWxg
//         dhIdWxg.assign(dhdc*dcIdWxg);
//         // dhIdWxo
//         dhIdWxo.assign(dhdzo*dWxJacobian /*+ 0 */);

//         // dcIdWri(dcdWri)
//         dcIdWri.assign(dcdzi*dWrJacobian);      // broadcast [bS, nOut] * [nOut, nOut, bS, nOut](or [nOut] * [nIn, nOut, nOut]);;
//         // dcIdWrf(dcdWrf)
//         dcIdWrf.assign(dcdzf*dWrJacobian);
//         // dcIdWrg(dcdWrg)
//         dcIdWrg.assign(dcdzg*dWrJacobian);
//         // dcIdWro(dcdWro) = 0
//         dcIdWro.nullify();

//         // dhIdWri
//         dhIdWri.assign(dhdc*dcIdWri);           // broadcast [bS, nOut] * [nIn, nOut, bS, nOut] (or [nOut] * [nIn, nOut, nOut]);
//         // dhIdWrf
//         dhIdWrf.assign(dhdc*dcIdWrf);
//         // dhIdWrg
//         dhIdWrg.assign(dhdc*dcIdWrg);
//         // dhIdWro
//         dhIdWro.assign(dhdzo*dWrJacobian /*+ 0 */);

//         if(Wp && x->rankOf() == 1) {
//             // dcIdWpi
//             dcIdWpi->assign(dcdzi*(*cI));       // [nOut] * [nOut]
//             // dcIdWpf
//             dcIdWpf->assign(dcdzf*(*cI));       // [nOut] * [nOut]
//             // dcIdWpo
//             dcIdWpo->nullify();                 // [nOut]

//             // dhdWpi
//             dhIdWpi->assign(dhdc*(*dcIdWpi));     // [nOut] * [nOut]
//             // dhdWpf
//             dhIdWpf->assign(dhdc*(*dcIdWpf));     // [nOut] * [nOut]
//             // dhdWpo
//             dhIdWpo->assign(dhdzo*c /* +0*/);     // [nOut] * [nOut]
//         }
//         else if(Wp) {
//             // dcIdWpi
//             (dcdzi*(*cI)).reduceAlongDimension(reduce::Sum, *dcIdWpi, {0});       // [bS, nOut]->reduce->[nOut]
//             // dcIdWpf
//             (dcdzf*(*cI)).reduceAlongDimension(reduce::Sum, *dcIdWpf, {0});       // [bS, nOut]->reduce->[nOut]
//             // dcIdWpo
//             dcIdWpo->nullify();                                                   // [nOut]

//             // dhIdWpi
//             (*dLdh*dhdc*(dcdzi*(*cI))).reduceAlongDimension(reduce::Sum, *dhIdWpi, {0});     // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // dhIdWpf
//             (*dLdh*dhdc*(dcdzf*(*cI))).reduceAlongDimension(reduce::Sum, *dhIdWpf, {0});     // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // dhIdWpo
//             (*dLdh*dhdzo*c /* +0*/).reduceAlongDimension(reduce::Sum, *dhIdWpo, {0});     // ([bS, nOut] * [nOut])->reduce->[nOut]
//         }

//         if(b && x->rankOf() == 1) {
//             // dcIdbi
//             dcIdbi->assign(dcdzi);           // [nOut]
//             // dcIdbf
//             dcIdbf->assign(dcdzf);           // [nOut]
//             // dcIdbg
//             dcIdbg->assign(dcdzg);           // [nOut]
//             // dcIdbo
//             dcIdbo->nullify();               // [nOut]

//             //dhIdbi
//             dhIdbi->assign(dhdc*(*dcIdbi));     // [nOut]
//             //dhIdbf
//             dhIdbf->assign(dhdc*(*dcIdbf));     // [nOut]
//             //dhIdbg
//             dhIdbg->assign(dhdc*(*dcIdbg));     // [nOut]
//             //dhIdbo
//             dhIdbo->assign(dhdzo);              // [nOut]

//         }
//         else if(b) {
//             // dcIdbi
//             dcdzi.reduceAlongDimension(reduce::Sum, *dcIdbi, {0});       // [bS, nOut]->reduce->[nOut]
//             // dcIdbf
//             dcdzf.reduceAlongDimension(reduce::Sum, *dcIdbf, {0});       // [bS, nOut]->reduce->[nOut]
//             // dcIdbg
//             dcdzg.reduceAlongDimension(reduce::Sum, *dcIdbg, {0});       // [bS, nOut]->reduce->[nOut]
//             // dcIdbo
//             dcIdbo->nullify();                  // [nOut]

//             //dhIdbi
//             (*dLdh*dhdc*dcdzi).reduceAlongDimension(reduce::Sum, *dhIdbi, {0});     // ([bS, nOut] * [nOut])->reduce->[nOut]
//             //dhIdbf
//             (*dLdh*dhdc*dcdzf).reduceAlongDimension(reduce::Sum, *dhIdbf, {0});     // ([bS, nOut] * [nOut])->reduce->[nOut]
//             //dhIdbg
//             (*dLdh*dhdc*(*dcIdbg)).reduceAlongDimension(reduce::Sum, *dhIdbg, {0});     // ([bS, nOut] * [nOut])->reduce->[nOut]
//             //dhIdbo
//             (*dLdh*dhdzo).reduceAlongDimension(reduce::Sum, *dhIdbo, {0});     // ([bS, nOut] * [nOut])->reduce->[nOut]

//         }
//     }
//     else {

//         NDArray tempIFE = mmul(dcdzi, WriT) + mmul(dcdzf, WrfT) + mmul(dcdzg, WrgT);
//         NDArray tempO   = mmul(dhdzo, WroT);

//         // dLdcI
//         if(dLdcI)
//             dLdcI->assign(factor*tempC + (*dLdhII)*(*dhIdcI));

//         // dcIdWxi(dcdWxi)
//         dcIdWxi.assign(dcdzi*dWxJacobian + tempIFE*dhIdWxi + tempC*dcIdWxi);       // broadcast [bS, nOut] * [nIn, nOut, bS, nOut](or [nOut] * [nIn, nOut, nOut]);
//         // dcIdWxf(dcdWxf)
//         dcIdWxf.assign(dcdzf*dWxJacobian + tempIFE*dhIdWxf + tempC*dcIdWxf);
//         // dcIdWxg(dcdWxg)
//         dcIdWxg.assign(dcdzg*dWxJacobian + tempIFE*dhIdWxg + tempC*dcIdWxg);
//         // dcIdWxo(dcdWxo)
//         dcIdWxo.assign(/* 0 + */tempIFE * dhIdWxo + tempC*dcIdWxo);

//          // dhIdWxi
//         dhIdWxi.assign(dhdc*dcIdWxi + tempO*dhIdWxi);                           // broadcast [bS, nOut] * [nIn, nOut, bS, nOut](or [nOut] * [nIn, nOut, nOut]);
//         // dhIdWxf
//         dhIdWxf.assign(dhdc*dcIdWxf + tempO*dhIdWxf);
//         // dhIdWxg
//         dhIdWxg.assign(dhdc*dcIdWxg + tempO*dhIdWxg);
//         // dhIdWxo
//         dhIdWxo.assign(dhdzo*dWxJacobian + dhdc*dcIdWxo + tempO*dhIdWxo);

//         // dcIdWri(dcdWri)
//         dcIdWri.assign(dcdzi*dWrJacobian + tempIFE*dhIdWri + tempC*dcIdWri);      // broadcast [bS, nOut] * [nOut, nOut, bS, nOut](or [nOut] * [nIn, nOut, nOut]);
//         // dcIdWrf(dcdWrf)
//         dcIdWrf.assign(dcdzf*dWrJacobian + tempIFE*dhIdWrf + tempC*dcIdWrf);
//         // dcIdWrg(dcdWrg)
//         dcIdWrg.assign(dcdzg*dWrJacobian + tempIFE*dhIdWrg + tempC*dcIdWrg);
//         // dcIdWro(dcdWro)
//         dcIdWro.assign(/* 0 + */tempIFE * dhIdWro + tempC*dcIdWro);

//         // dhIdWri
//         dhIdWri.assign(dhdc*dcIdWri + tempO*dhIdWri);                           // broadcast [bS, nOut] * [nOut, nOut, bS, nOut](or [nOut] * [nIn, nOut, nOut]);
//         // dhIdWrf
//         dhIdWrf.assign(dhdc*dcIdWrf + tempO*dhIdWrf);
//         // dhIdWrg
//         dhIdWrg.assign(dhdc*dcIdWrg + tempO*dhIdWrg);
//         // dhIdWro
//         dhIdWro.assign(dhdzo*dWrJacobian + dhdc*dcIdWro + tempO*dhIdWro);

//         if(Wp && x->rankOf() == 1) {
//             // dcIdWpi
//             dcIdWpi->assign(dcdzi*(*cI) + tempIFE*(*dhIdWpi) + tempC*(*dcIdWpi));       // [nOut] * [nOut]
//             // dcIdWpf
//             dcIdWpf->assign(dcdzf*(*cI) + tempIFE*(*dhIdWpf) + tempC*(*dcIdWpf));       // [nOut] * [nOut]
//             // dcIdWpo
//             dcIdWpo->assign(/*  0 +  */   tempIFE*(*dhIdWpo) + tempC*(*dcIdWpo));       // [nOut] * [nOut]

//             // dhdWpi
//             dhIdWpi->assign(dhdc*(*dcIdWpi) + tempO*(*dhIdWpi));                // [nOut] * [nOut]
//             // dhdWpf
//             dhIdWpf->assign(dhdc*(*dcIdWpf) + tempO*(*dhIdWpf));                // [nOut] * [nOut]
//             // dhdWpo
//             dhIdWpo->assign(dhdzo*c + dhdc*(*dcIdWpo) + tempO*(*dhIdWpo));      // [nOut] * [nOut]
//         }
//         else if(Wp) {
//             // dcIdWpi
//             (dcdzi*(*cI) + tempIFE*(*dhIdWpi) + tempC*(*dcIdWpi)).reduceAlongDimension(reduce::Sum, *dcIdWpi, {0});   // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // dcIdWpf
//             (dcdzf*(*cI) + tempIFE*(*dhIdWpf) + tempC*(*dcIdWpf)).reduceAlongDimension(reduce::Sum, *dcIdWpf, {0});   // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // dcIdWpo
//             (/*  0 +  */   tempIFE*(*dhIdWpo) + tempC*(*dcIdWpo)).reduceAlongDimension(reduce::Sum, *dcIdWpo, {0});   // ([bS, nOut] * [nOut])->reduce->[nOut]

//             // dhIdWpi
//             (dhdc*(*dcIdWpi) + tempO*(*dhIdWpi)).reduceAlongDimension(reduce::Sum, *dhIdWpi, {0});              // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // dhIdWpf
//             (dhdc*(*dcIdWpf) + tempO*(*dhIdWpf)).reduceAlongDimension(reduce::Sum, *dhIdWpf, {0});              // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // dhIdWpo
//             (dhdzo*c + dhdc*(*dcIdWpo) + tempO*(*dhIdWpo)).reduceAlongDimension(reduce::Sum, *dhIdWpo, {0});    // ([bS, nOut] * [nOut])->reduce->[nOut]
//         }

//         if(b && x->rankOf() == 1) {
//             // dcIdbi
//             dcIdbi->assign(dcdzi  + tempIFE*(*dhIdbi) + tempC*(*dcIdbi));           // [nOut]
//             // dcIdbf
//             dcIdbf->assign(dcdzf  + tempIFE*(*dhIdbf) + tempC*(*dcIdbf));           // [nOut]
//             // dcIdbg
//             dcIdbg->assign(dcdzg  + tempIFE*(*dhIdbg) + tempC*(*dcIdbg));           // [nOut]
//             // dcIdbo
//             dcIdbo->assign(/*0+*/   tempIFE*(*dhIdbo) + tempC*(*dcIdbo));           // [nOut]

//             //dhIdbi
//             dhIdbi->assign(dhdc*(*dcIdbi) + tempO*(*dhIdbi));           // [nOut]
//             //dhIdbf
//             dhIdbf->assign(dhdc*(*dcIdbf) + tempO*(*dhIdbf));           // [nOut]
//             //dhIdbg
//             dhIdbg->assign(dhdc*(*dcIdbg) + tempO*(*dhIdbg));           // [nOut]
//             //dhIdbo
//             dhIdbo->assign(dhdzo + dhdc*(*dcIdbo) + tempO*(*dhIdbo));   // [nOut]

//         }
//         else if(b) {
//             // dcIdbi
//             (dcdzi  + tempIFE*(*dhIdbi) + tempC*(*dcIdbi)).reduceAlongDimension(reduce::Sum, *dcIdbi, {0});       // [bS, nOut]->reduce->[nOut]
//             // dcIdbf
//             (dcdzf  + tempIFE*(*dhIdbf) + tempC*(*dcIdbf)).reduceAlongDimension(reduce::Sum, *dcIdbf, {0});       // [bS, nOut]->reduce->[nOut]
//             // dcIdbg
//             (dcdzg  + tempIFE*(*dhIdbg) + tempC*(*dcIdbg)).reduceAlongDimension(reduce::Sum, *dcIdbg, {0});       // [bS, nOut]->reduce->[nOut]
//             // dcIdbo
//             (/*0+*/   tempIFE*(*dhIdbo) + tempC*(*dcIdbo)).reduceAlongDimension(reduce::Sum, *dcIdbo, {0});       // [bS, nOut]->reduce->[nOut]

//             //dhIdbi
//             (dhdc*(*dcIdbi) + tempO*(*dhIdbi)).reduceAlongDimension(reduce::Sum, *dhIdbi, {0});         // ([bS, nOut] * [nOut])->reduce->[nOut]
//             //dhIdbf
//             (dhdc*(*dcIdbf) + tempO*(*dhIdbf)).reduceAlongDimension(reduce::Sum, *dhIdbf, {0});         // ([bS, nOut] * [nOut])->reduce->[nOut]
//             //dhIdbg
//             (dhdc*(*dcIdbg) + tempO*(*dhIdbg)).reduceAlongDimension(reduce::Sum, *dhIdbg, {0});         // ([bS, nOut] * [nOut])->reduce->[nOut]
//             //dhIdbo
//             (dhdzo + dhdc*(*dcIdbo) + tempO*(*dhIdbo)).reduceAlongDimension(reduce::Sum, *dhIdbo, {0}); // ([bS, nOut] * [nOut])->reduce->[nOut]

//         }
//     }

//     const std::vector<int> dimsToExclude = x->rankOf() == 1 ? std::vector<int>({2}) : std::vector<int>({2, 3});

//     // dLdWxi, dLdWxf, dLdWxg, dLdWxo
//     (*dLdh*(*dhIdWx)).reduceAlongDimension(reduce::Sum, *dLdWx, dimsToExclude);

//     // dLdWri, dLdWrf, dLdWrg, dLdWro
//     (*dLdh*(*dhIdWr)).reduceAlongDimension(reduce::Sum, *dLdWr, dimsToExclude);

//     // dLdWpi, dLdWpf, dLdWpo
//     if(Wp) {
//         if(x->rankOf() == 1) {
//             (*dLdWp)({0, nOut}).assign(*dLdh*(*dhIdWpi));          // [nOut] * [nOut]
//             (*dLdWp)({nOut, 2*nOut}).assign(*dLdh*(*dhIdWpf));     // [nOut] * [nOut]
//             (*dLdWp)({2*nOut, 3*nOut}).assign(*dLdh*(*dhIdWpo));   // [nOut] * [nOut]
//         }
//         else {
//             // NDArray temp1 = (*dLdWp)({0, nOut});
//             // NDArray temp2 = (*dLdWp)({nOut, 2*nOut});
//             // NDArray temp3 = (*dLdWp)({2*nOut, 3*nOut});
//             // dhIdWpi->reduceAlongDimension(reduce::Sum, temp1, {0});       // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // dhIdWpf->reduceAlongDimension(reduce::Sum, temp2, {0});       // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // dhIdWpo->reduceAlongDimension(reduce::Sum, temp3, {0});       // ([bS, nOut] * [nOut])->reduce->[nOut]
//             (*dLdWp)({0, nOut}).assign(dhIdWpi);
//             (*dLdWp)({nOut, 2*nOut}).assign(dhIdWpf);
//             (*dLdWp)({2*nOut, 3*nOut}).assign(dhIdWpo);
//         }
//     }

//     // dLdbi, dLdbf, dLdbg, dLdbo
//     if(b) {
//         if(x->rankOf() == 1) {
//             (*dLdb)({0, nOut}).assign(*dLdh*(*dhIdbi));          // [nOut] * [nOut]
//             (*dLdb)({nOut, 2*nOut}).assign(*dLdh*(*dhIdbf));     // [nOut] * [nOut]
//             (*dLdb)({2*nOut, 3*nOut}).assign(*dLdh*(*dhIdbg));   // [nOut] * [nOut]
//             (*dLdb)({3*nOut, 4*nOut}).assign(*dLdh*(*dhIdbo));   // [nOut] * [nOut]
//         }
//         else {
//             // NDArray temp1 = (*dLdb)({0, nOut});
//             // NDArray temp2 = (*dLdb)({nOut, 2*nOut});
//             // NDArray temp3 = (*dLdb)({2*nOut, 3*nOut});
//             // NDArray temp4 = (*dLdb)({3*nOut, 4*nOut});
//             // (*dLdh*(*dhIdbi)).reduceAlongDimension(reduce::Sum, temp1, {0});       // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // (*dLdh*(*dhIdbf)).reduceAlongDimension(reduce::Sum, temp2, {0});       // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // (*dLdh*(*dhIdbg)).reduceAlongDimension(reduce::Sum, temp3, {0});       // ([bS, nOut] * [nOut])->reduce->[nOut]
//             // (*dLdh*(*dhIdbo)).reduceAlongDimension(reduce::Sum, temp3, {0});       // ([bS, nOut] * [nOut])->reduce->[nOut]
//             (*dLdb)({0, nOut}).assign(dhIdbi);
//             (*dLdb)({nOut, 2*nOut}).assign(dhIdbf);
//             (*dLdb)({2*nOut, 3*nOut}).assign(dhIdbg);
//             (*dLdb)({3*nOut, 4*nOut}).assign(dhIdbo);
//         }
//     }

//     //dhIdcI
//     if(dLdcI)
//         dhIdcI->assign(dhdc);

//     cI->assign(c);

//     if(dLdcI && !dLdhI)
//         delete dLdhII;
//     if(Wp) {
//         delete Wpi; delete Wpf; delete Wpo; delete dcIdWpi; delete dcIdWpf; delete dcIdWpo; delete dhIdWpi; delete dhIdWpf; delete dhIdWpo;
//     }
//     if(b) {
//         delete dcIdbi; delete dcIdbf; delete dcIdbg; delete dcIdbo; delete dhIdbi; delete dhIdbf; delete dhIdbg; delete dhIdbo;
//     }
// }
