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
// (cf. http://arxiv.org/abs/1406.1078).
// Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
// "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation"


#include<ops/declarable/helpers/gru.h>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> sigmoid(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Sigmoid<T>>();    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static FORCEINLINE NDArray<T> activation(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void gruCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h) {

    NDArray<T>* x  = inArrs[0];                   // input [bS, iS], bS - batch size, iS - input size
    NDArray<T>* h0 = inArrs[1];                   // previous cell output [bS, nU],  that is at previous time step t-1

    NDArray<T>* Wx = inArrs[2];                   // input-to-hidden  weights, [iS, 3*nU] 
    NDArray<T>* Wh = inArrs[3];                   // hidden-to-hidden weights, [nU, 3*nU]     
    NDArray<T>* b  = inArrs[4];                   // biases, [3*nU] 
    
    // h is current cell output [bS, nU], that is at current time step t    

    const int nU = h0->sizeAt(1);                // number of units
    
    // gates = sigmoid(x*Wx + h0*Wh + b)
    NDArray<T> gates = sigmoid<T>(mmul(*x, (*Wx)({{},{0,2*nU}})) + mmul(*h0, (*Wh)({{},{0,2*nU}})) + (*b)({{0,2*nU}}));       // [bS, 2*nU] + [bS, 2*nU] + [1, 2*nU] = [bS, 2*nU]    
    
    // reset gate
    NDArray<T> r = gates({{}, {0, nU}});                     // [bS, nU]

    // update gate
    NDArray<T> u = gates({{}, {nU, 2*nU}});            // [bS, nU]

    // ◦ means element-wise product or so called Hadamard product
    // n = activation(x*Wx + (r◦h0)*Wh + b)
    NDArray<T> n = activation<T>(mmul(*x, (*Wx)({{},{2*nU, 3*nU}})) + mmul((*h0)*r, (*Wh)({{},{2*nU,3*nU}})) + (*b)({{2*nU,3*nU}}));     // [bS, nU]

    // current cell output
    h->assign( u * (*h0) + ((T)1. - u) * n );
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void gruTimeLoop(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h) {

    NDArray<T>* x  = inArrs[0];                   // input [time, bS, iS]
    NDArray<T>* h0 = inArrs[1];                   // initial cell output (at time step = 0) [bS, nU]

    NDArray<T>* Wx = inArrs[2];                   // input-to-hidden  weights, [iS, 3*nU] 
    NDArray<T>* Wh = inArrs[3];                   // hidden-to-hidden weights, [nU, 3*nU]     
    NDArray<T>* b  = inArrs[4];                   // biases, [3*nU] 
    
    // h is cell outputs at each time step [time, bS, nU]

    const int time = x->sizeAt(0);    

    NDArray<T> ht_1(*h0);

    // loop through time steps
    for (int t = 0; t < time; ++t) {

        NDArray<T> xt = (*x)({{t,t+1}, {}, {}});
        NDArray<T> ht = (*h)({{t,t+1}, {}, {}});

        helpers::gruCell<T>({&xt, &ht_1, Wx, Wh, b}, &ht);
        ht_1.assign(ht);    
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void gruCellBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs) {

    NDArray<T>* x      = inArrs[0];                   // input [bS, iS]
    NDArray<T>* h0     = inArrs[1];                   // previous cell output [bS, nU],  that is at previous time step t-1
    NDArray<T>* Wx     = inArrs[2];                   // input-to-hidden  weights, [iS, 3*nU] 
    NDArray<T>* Wh     = inArrs[3];                   // hidden-to-hidden weights, [nU, 3*nU]     
    NDArray<T>* b      = inArrs[4];                   // biases, [3*nU]     
    NDArray<T>* dLdh   = inArrs[5];                   // gradient wrt output, [bS,nU], that is epsilon_next
    NDArray<T>* dLdWx0 = inArrs[6];                   // gradient wrt Wx at previous time step, [iS, 3*nU]
    NDArray<T>* dLdWh0 = inArrs[7];                   // gradient wrt Wh at previous time step, [nU, 3*nU]
    NDArray<T>* dLdb0  = inArrs[8];                   // gradient wrt b at previous time step,  [3*nU]

    NDArray<T>* dLdx   = outArrs[0];                  // gradient wrt x,  [bS, iS], that is epsilon
    NDArray<T>* dLdh0  = outArrs[1];                  // gradient wrt h0, [bS, nU]
    NDArray<T>* dLdWx  = outArrs[2];                  // gradient wrt Wx, [iS, 3*nU]
    NDArray<T>* dLdWh  = outArrs[3];                  // gradient wrt Wh, [nU, 3*nU]
    NDArray<T>* dLdb   = outArrs[4];                  // gradient wrt b at previous time step,  [3*nU]
    
    // h is current cell output [bS, nU], that is at current time step t    

    const int nU = h0->sizeAt(1);
    // ***** feed forward step ***** //    
    // gates = sigmoid(x*Wx + h0*Wh + b)
    NDArray<T> gates = sigmoid<T>(mmul(*x, (*Wx)({{},{0,2*nU}})) + mmul(*h0, (*Wh)({{},{0,2*nU}})) + (*b)({{0,2*nU}}));       // [bS, 2*nU] + [bS, 2*nU] + [1, 2*nU] = [bS, 2*nU]    
    // reset gate
    NDArray<T> r = gates({{}, {0, nU}});               // [bS, nU]
    // update gate
    NDArray<T> u = gates({{}, {nU, 2*nU}});            // [bS, nU]
    // ◦ means element-wise product or so called Hadamard product
    // n = activation(x*Wx + (r◦h0)*Wh + b)
    NDArray<T> n = activation<T>(mmul(*x, (*Wx)({{},{2*nU, 3*nU}})) + mmul((*h0)*r, (*Wh)({{},{2*nU,3*nU}})) + (*b)({{2*nU,3*nU}}));     // [bS, nU]


    // ***** back prop step ***** // 
    NDArray<T> Wxr  = (*Wx)({{}, {0,   nU}});
    NDArray<T> Wxu  = (*Wx)({{}, {nU,  2*nU}});
    NDArray<T> Wxn  = (*Wx)({{}, {2*nU,3*nU}});
    NDArray<T> Whr  = (*Wh)({{}, {0,   nU}});
    NDArray<T> Whu  = (*Wh)({{}, {nU,  2*nU}});
    NDArray<T> Whn  = (*Wh)({{}, {2*nU,3*nU}});
    NDArray<T> WxrT = Wxr.transp();
    NDArray<T> WxuT = Wxu.transp();
    NDArray<T> WxnT = Wxn.transp();
    NDArray<T> WhrT = Whr.transp();
    NDArray<T> WhuT = Whu.transp();
    NDArray<T> WhnT = Whn.transp();
    NDArray<T> xT   = x->transp();
    NDArray<T> h0T  = h0->transp();

    NDArray<T> dLdWxr = (*dLdWx)({{}, {0,     nU}});
    NDArray<T> dLdWxu = (*dLdWx)({{}, {nU,  2*nU}});
    NDArray<T> dLdWxn = (*dLdWx)({{}, {2*nU,3*nU}});

    NDArray<T> dLdWhr = (*dLdWh)({{}, {0,     nU}});    
    NDArray<T> dLdWhu = (*dLdWh)({{}, {nU,  2*nU}});    
    NDArray<T> dLdWhn = (*dLdWh)({{}, {2*nU,3*nU}});

    NDArray<T> dldbr = (*dLdb)({{0,     nU}});
    NDArray<T> dldbu = (*dLdb)({{nU,  2*nU}});
    NDArray<T> dldbn = (*dLdb)({{2*nU,3*nU}});

    NDArray<T> dhdu   = *h0  - n;               // [bS, nU]
    NDArray<T> dhdn   = (T)1 - u;               // [bS, nU]    
    NDArray<T> dSigdu = u * ((T)1 - u);         // [bS, nU]
    NDArray<T> dSigdr = r * ((T)1 - r);         // [bS, nU]
    NDArray<T> dActdn = (T)1 - n * n;           // [bS, nU]
    NDArray<T> dndr   = mmul(dActdn * (*h0), WhnT);
    NDArray<T> drdh0  = mmul(dSigdr, WhrT);     

    NDArray<T> dLdn = (*dLdh) * dhdn;
    NDArray<T> dLdu = (*dLdh) * dhdu;
    NDArray<T> dLdr = dLdn * dndr;


    dLdx->assign( mmul(dLdu * dSigdu, WxuT) + mmul(dLdr * dSigdr, WxrT) + mmul(dLdn * dActdn, WxnT) );      // [bS,iS]

    dLdh0->assign( mmul(dLdu * dSigdu, WhuT) + mmul(dLdn * dActdn * (r + drdh0), WhnT) + (*dLdh)*u );       // [bS,nU]

    dLdWxr.assign( mmul(xT, dSigdr * dLdr) );                                                               //  [iS,nU]
    if(dLdWx0 != nullptr)
        dLdWxr += (*dLdWx0)({{}, {0,   nU}});

    dLdWhr.assign( mmul(h0T, dSigdr * dLdr) );                                                              //  [nU,nU]
    if(dLdWh0 != nullptr)
        dLdWhr += (*dLdWh0)({{}, {0,   nU}});

    dLdWxu.assign( mmul(xT, dSigdu * dLdu) );                                                               //  [iS,nU]
    if(dLdWx0 != nullptr)
        dLdWxu += (*dLdWx0)({{}, {nU,2*nU}});

    dLdWhu.assign( mmul(h0T, dSigdu * dLdu) );                                                               //  [nU,nU]
    if(dLdWh0 != nullptr)
        dLdWhu += (*dLdWh0)({{}, {nU,2*nU}});

    dLdWxn.assign( mmul(xT, dActdn * dLdn) );                                                               //  [iS,nU]
    if(dLdWx0 != nullptr)
        dLdWxn += (*dLdWx0)({{}, {2*nU,3*nU}});

    dLdWhn.assign( mmul((r*(*h0)).transp(), dActdn * dLdn) );                                               //  [nU,nU]
    if(dLdWh0 != nullptr)
        dLdWhn += (*dLdWh0)({{}, {2*nU,3*nU}});

    dldbr.assign( (dSigdr * dLdr).template reduceAlongDims<simdOps::Sum<T>>({0}));                          // [nU]
    if(dLdb0 != nullptr)
        dldbr += (*dLdb0)({{0, nU}});

    dldbu.assign( (dSigdu * dLdu).template reduceAlongDims<simdOps::Sum<T>>({0}));                          // [nU]
    if(dLdb0 != nullptr)
        dldbu += (*dLdb0)({{nU, 2*nU}});

    dldbn.assign( (dActdn * dLdn).template reduceAlongDims<simdOps::Sum<T>>({0}));                          // [nU]
    if(dLdb0 != nullptr)
        dldbn += (*dLdb0)({{2*nU, 3*nU}});    
}



template void gruCell<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>* h);
template void gruCell<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* h);
template void gruCell<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>* h);

template void gruTimeLoop<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>* h);
template void gruTimeLoop<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* h);
template void gruTimeLoop<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>* h);

template void gruCellBP<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs);
template void gruCellBP<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs);
template void gruCellBP<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs);


}
}
}

