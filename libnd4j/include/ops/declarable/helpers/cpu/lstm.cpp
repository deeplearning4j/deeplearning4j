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
// @author Yurii Shyrma, created on 14.02.2018
//

// implementation of operation for LSTM cell with peep hole connections:
// http://www.bioinf.jku.at/publications/older/2604.pdf
// S. Hochreiter and J. Schmidhuber. "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.
// and 
// https://research.google.com/pubs/archive/43905.pdf
// Hasim Sak, Andrew Senior, and Francoise Beaufays. "Long short-term memory recurrent neural network architectures for large scale acoustic modeling." INTERSPEECH, 2014.


#include<ops/declarable/helpers/lstm.h>

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
static void clipping(NDArray<T>* arr, T limit) {    
    
    if(limit < (T)0.)
        limit *= (T)(-1.);

    auto clip = LAMBDA_T(value, limit) {
        if(value < -limit || value > limit)
            value = limit;
        return value; 
    };

    arr->applyLambda(clip);    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void lstmCell(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<T>& params) {

    NDArray<T>* xt   = inArrs[0];                   // input [bS x inSize]
    NDArray<T>* ht_1 = inArrs[1];                   // previous cell output [bS x numProj],  that is at previous time step t-1, in case of projection=false -> numProj=numUnits!!! 
    NDArray<T>* ct_1 = inArrs[2];                   // previous cell state  [bS x numUnits], that is at previous time step t-1   

    NDArray<T>* Wx   = inArrs[3];                   // input-to-hidden  weights, [inSize  x 4*numUnits] 
    NDArray<T>* Wh   = inArrs[4];                   // hidden-to-hidden weights, [numProj x 4*numUnits] 
    NDArray<T>* Wc   = inArrs[5];                   // diagonal weights for peephole connections [3*numUnits] 
    NDArray<T>* Wp   = inArrs[6];                   // projection weights [numUnits x numProj] 
    NDArray<T>* b    = inArrs[7];                   // biases, [4*numUnits] 
    
    NDArray<T>* ht   =  outArrs[0];                 // current cell output [bS x numProj], that is at current time step t
    NDArray<T>* ct   =  outArrs[1];                 // current cell state  [bS x numUnits], that is at current time step t
    
    const bool peephole   = (bool)params[0];        // if true, provide peephole connections
    const bool projection = (bool)params[1];        // if true, then projection is performed, if false then numProj==numUnits is mandatory!!!!
    T clippingCellValue   = params[2];              // clipping value for ct, if it is not equal to zero, then cell state is clipped
    T clippingProjValue   = params[3];              // clipping value for projected ht, if it is not equal to zero, then projected cell output is clipped
    const T forgetBias    = params[4];

    const int bS   = xt->sizeAt(0);
    const int inSize      = xt->sizeAt(1);
    const int numProj     = ht_1->sizeAt(1);
    const int numUnits    = ct_1->sizeAt(1);    
    
    NDArray<T> z = mmul(*xt, *Wx) + mmul(*ht_1, *Wh) + *b;      // [bS x 4*numUnits] + [bS x 4*numUnits] + [1 x 4*numUnits] = [bS x 4*numUnits]    
    
    NDArray<T> zit = z({{},{0,            numUnits}});      	// z for input gate,  = mmul(Wxi,xt) + mmul(Whi,ht_1) + bi    = [bS x numUnits]
    NDArray<T> zft = z({{},{numUnits,   2*numUnits}});      	// z for forget gate, = mmul(Wxf,xt) + mmul(Whf,ht_1) + bf    = [bS x numUnits]
    NDArray<T> zct = z({{},{2*numUnits, 3*numUnits}});      	// z for cell state,  = mmul(Wxc,xt) + mmul(Whc,ht_1) + bc    = [bS x numUnits]     
    NDArray<T> zot = z({{},{3*numUnits, 4*numUnits}});      	// z for output gate, = mmul(Wxo,xt) + mmul(Who,ht_1) + bo    = [bS x numUnits] 

    if(peephole) {                                              // add peephole connections: z  +  ct_1*Wc
        zit += (*ct_1) * (*Wc)({{0,          numUnits}});       // add peephole connections to input gate
        zft += (*ct_1) * (*Wc)({{numUnits, 2*numUnits}});       // add peephole connections to forget gate
    }

    // current sell state = ft*ct_1 + it*activation(mmul(Wxc,xt) + mmul(Whc,ht_1) + bc
    ct->assign( sigmoid<T>(zft + forgetBias) * (*ct_1) + sigmoid<T>(zit) * activation<T>(zct) );
    
    // if clipping value is provided then cell state is clipped by this value prior to the cell output activation
    if(clippingCellValue != (T)0.)
        clipping(ct, clippingCellValue);

    if(peephole) 
        zot += (*ct) * (*Wc)({{2*numUnits, 3*numUnits}});            // add peephole connections to output gate zot + ct*Wc

    // current cell output = ot*activation(ct)   
    NDArray<T> htNoPeepHole = sigmoid<T>(zot) * activation<T>(*ct);      // = [bS x numUnits]

    // apply projection
    if(projection) {
        ht->assign( mmul(htNoPeepHole, *Wp) );                           // [bS x numUnits] * [ numUnits x numProj] = [bS x numProj]
        // if clipping projection is provided then projected cell output state is clipped by this value 
        if(clippingProjValue != (T)0.)
            clipping(ht, clippingProjValue);
    }
    else
        ht->assign(&htNoPeepHole);     
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void lstmTimeLoop(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs, const std::vector<T>& params) {
    
    NDArray<T>* x  = inArrs[0];                    // input [time x bS x inSize]
    NDArray<T>* h0 = inArrs[1];                    // initial cell output (at time step = 0) [bS x numProj], in case of projection=false -> numProj == numUnits !!!
    NDArray<T>* c0 = inArrs[2];                    // initial cell state  (at time step = 0) [bS x numUnits],  

    NDArray<T>* Wx  = inArrs[3];                   // input-to-hidden  weights, [inSize  x 4*numUnits] 
    NDArray<T>* Wh  = inArrs[4];                   // hidden-to-hidden weights, [numProj x 4*numUnits] 
    NDArray<T>* Wc  = inArrs[5];                   // diagonal weights for peephole connections [3*numUnits] 
    NDArray<T>* Wp  = inArrs[6];                   // projection weights [numUnits x numProj] 
    NDArray<T>* b   = inArrs[7];                   // biases, [4*numUnits] 
    
    NDArray<T>* h   =  outArrs[0];                 // cell outputs [time x bS x numProj], that is per each time step
    NDArray<T>* c   =  outArrs[1];                 // cell states  [time x bS x numUnits] that is per each time step    

    const int time  = x->sizeAt(0);

    NDArray<T> currentH(*h0);
    NDArray<T> currentC(*c0);

    // loop through time steps
    for (int t = 0; t < time; ++t) {
        
        NDArray<T> xt = (*x)({{t,t+1}, {}, {}});
        NDArray<T> ht = (*h)({{t,t+1}, {}, {}});
        NDArray<T> ct = (*c)({{t,t+1}, {}, {}});

        helpers::lstmCell<T>({&xt,&currentH,&currentC, Wx,Wh,Wc,Wp, b},   {&ht, &ct},   params);
        currentH.assign(ht);
        currentC.assign(ct);
    }    
}


template void clipping<float>(NDArray<float>* arr, float limit);
template void clipping<float16>(NDArray<float16>* arr, float16 limit);
template void clipping<double>(NDArray<double>* arr, double limit);
template void clipping<int>(NDArray<int>* arr, int limit);
template void clipping<Nd4jLong>(NDArray<Nd4jLong>* arr, Nd4jLong limit);

template void lstmCell<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs, const std::vector<float>& params);
template void lstmCell<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs, const std::vector<float16>& params);
template void lstmCell<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs, const std::vector<double>& params);
template void lstmCell<int>(const std::vector<NDArray<int>*>& inArrs, const std::vector<NDArray<int>*>& outArrs, const std::vector<int>& params);
template void lstmCell<Nd4jLong>(const std::vector<NDArray<Nd4jLong>*>& inArrs, const std::vector<NDArray<Nd4jLong>*>& outArrs, const std::vector<Nd4jLong>& params);

template void lstmTimeLoop<float>(const std::vector<NDArray<float>*>& inArrs, const std::vector<NDArray<float>*>& outArrs, const std::vector<float>& params);
template void lstmTimeLoop<float16>(const std::vector<NDArray<float16>*>& inArrs, const std::vector<NDArray<float16>*>& outArrs, const std::vector<float16>& params);
template void lstmTimeLoop<double>(const std::vector<NDArray<double>*>& inArrs, const std::vector<NDArray<double>*>& outArrs, const std::vector<double>& params);
template void lstmTimeLoop<int>(const std::vector<NDArray<int>*>& inArrs, const std::vector<NDArray<int>*>& outArrs, const std::vector<int>& params);
template void lstmTimeLoop<Nd4jLong>(const std::vector<NDArray<Nd4jLong>*>& inArrs, const std::vector<NDArray<Nd4jLong>*>& outArrs, const std::vector<Nd4jLong>& params);


}
}
}

