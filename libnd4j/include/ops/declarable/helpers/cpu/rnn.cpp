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
template <typename T>
static FORCEINLINE NDArray<T> activation(const NDArray<T>& arr) {    
    
    return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();    
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void rnnCell(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* ht) {

    NDArray<T>* xt   = inArrs[0];                   // input [bS x iS]    
    NDArray<T>* Wx   = inArrs[1];                   // input-to-hidden weights, [iS  x nU] 
    NDArray<T>* Wh   = inArrs[2];                   // hidden-to-hidden weights, [nU x nU] 
    NDArray<T>* b    = inArrs[3];                   // biases, [2*nU]: {0, nU} are input-to-hidden biases and {nU, 2*nU} are hidden-to-hidden biases    
    NDArray<T>* ht_1 = inArrs[4];                   // previous cell output [bS x nU],  that is at previous time step t-1, in case of projection=false -> nU=nU!!!     

    const int nU  = ht_1->sizeAt(1);
    
    // ht is current cell output [bS x nU], that is at current time step t        
    ht->assign(activation<T>(mmul(*xt, *Wx) + (*b)({{0, nU}})  +  mmul(*ht_1, *Wh) + (*b)({{nU, 2*nU}})));      // [bS x nU] + [nU]  +  [bS x nU] + [nU] = [bS x nU]    

}



//////////////////////////////////////////////////////////////////////////
template <typename T>
void rnnTimeLoop(const std::vector<NDArray<T>*>& inArrs, NDArray<T>* h, NDArray<T>* hFinal) {

    NDArray<T>* x  = inArrs[0];               	// input [N x bS x iS]
	NDArray<T>* Wx = inArrs[1];               	// input-to-hidden  weights, [iS  x nU] 	
    NDArray<T>* Wh = inArrs[2];               	// hidden-to-hidden weights, [nU x nU]         
	NDArray<T>* b  = inArrs[3];               	// biases for, [2*nU] 

	NDArray<T>* hi          = inArrs[4];		// initial cell output (at time step = 0) [bS x nU]	
	NDArray<T>* maxTimeStep = inArrs[5];     	// vector [bS] containing integer values within [0,N), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep
    
    const Nd4jLong N  = x->sizeAt(0);
    const Nd4jLong bS = x->sizeAt(1);        
    
    // at first time step
    if(hi)
        hFinal->assign(hi);
    else 
        *hFinal = 0.;   

    BlasHelper::getInstance();          // to avoid memory leak in pragma parallel loops
// #pragma omp parallel for schedule(guided) collapse(2) if(bS > Environment::getInstance()->elementwiseThreshold())  
    // loop through batch of inputs           
    for (Nd4jLong e = 0; e < bS; ++e) {                  
        // loop through time steps
        for (Nd4jLong t = 0; t < N; ++t) {                                 

            int maxStep = maxTimeStep ? (int)(*maxTimeStep)(e) : N;

            NDArray<T> xt   = (*x)({t,t+1, e,e+1, 0,0}, true);
            NDArray<T> ht   = (*h)({t,t+1, e,e+1, 0,0}, true);
            NDArray<T> ht_1 = (*hFinal)({e,e+1, 0,0}, true);                       // previous state 
            
            if(t >= maxStep) {
                ht = 0.;
                if(maxStep != 0)                    
                    ht_1.assign((*h)({maxStep-1,maxStep, e,e+1, 0,0}));
            }
            else {
                helpers::rnnCell<T>({&xt, Wx, Wh, b, &ht_1}, &ht);
                ht_1.assign(ht);
            }
        }
    }    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void rnnTimeLoopBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs) {

    NDArray<T>* x           = inArrs[0];         // input [N x bS x iS]
    NDArray<T>* Wx          = inArrs[1];         // input-to-hidden weights, [iS  x nU] 
    NDArray<T>* Wh          = inArrs[2];         // hidden-to-hidden weights, [nU x nU] 
    NDArray<T>* b           = inArrs[3];         // biases, [2*nU]: {0, nU} are input-to-hidden biases and {nU, 2*nU} are hidden-to-hidden biases        
    NDArray<T>* hi          = inArrs[4];         // initial cell output [bS x nU]
    NDArray<T>* maxTimeStep = inArrs[5];         // vector [bS] containing integer values within [0,N), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep
    NDArray<T>* dLdh        = inArrs[6];         // set of derivatives dL_{t}/dh_{t}, [N x bS x nU], epsilon    
          
    NDArray<T>* dLdx  = outArrs[0];          // set of derivatives dL/dx_{t}, [N x bS x iS]
    NDArray<T>* dLdWx = outArrs[1];          // derivative dL/dWx, [iS x nU]
    NDArray<T>* dLdWh = outArrs[2];          // derivative dL/dWh, [nU x nU]
    NDArray<T>* dLdb  = outArrs[3];          // derivative dL/db, [2*nU]
    // NDArray<T>* dLdhtP1  = outArrs[6];         // derivative dL/dh_{t+1}, [bS x nU], for example for t=0 it is equal to: dL_0/dh_0 + dL_1/dh_1°dh_1/dh_0 + dL_2/dh_2°dh_2/dh_1°dh_1/dh_0 + dL_3/dh_3°dh_3/dh_2°dh_2/dh_1°dh_1/dh_0 + ..., [bS x nU]

    *dLdWx = T(0);
    *dLdWh = T(0);
    *dLdb  = T(0);

    const Nd4jLong N  = x->sizeAt(0);
    const Nd4jLong bS = x->sizeAt(1);
    const Nd4jLong nU = Wx->sizeAt(1);

    NDArray<T> h(hi->ordering(), {N, bS, nU});          // set of cell outputs, [N x bS x nU]
    NDArray<T> hFinal(hi->ordering(), {bS, nU}); 

    // feed forward time loop
    rnnTimeLoop({x, Wx, Wb, b, hi, maxTimeStep}, &h, &hFinal);    // use dhtdhtM1 as temporary array here 
   
    NDArray<T> WxT = Wx->transp();
    NDArray<T> WhT = Wh->transp();
    NDArray<T>* xT = x->permute({0,2,1});                       // [N x bS x iS] -> [N x iS x bS]
    NDArray<T>* hT = h->permute({0,2,1});                       // [N x bS x nU] -> [N x nU x bS]

    // backprop time loop
    for(Nd4jLong t = N-1; i >= 0; --t) {
                
        NDArray<T> dLtdht = (*dLdh)({t,t+1, 0,0, 0,0});                 // derivative dL_{t}/dh_{t}, epsilon, [bS x nU]           
        NDArray<T> dhtP1dht = dLtdht;                                   // dh_{t+1}/dh_{t}, [bS x nU], choose initial value to be equal dLtdht

        for(Nd4jLong tt = t; tt >= 0; --tt) {

            NDArray<T> ht  =  (*h)({tt,tt+1, 0,0, 0,0});
            NDArray<T> xtT = (*xT)({tt,tt+1, 0,0, 0,0});
            NDArray<T> htT = (*hT)({tt,tt+1, 0,0, 0,0});
            
            NDArray<T> dhtdzt = T(1) - ht * ht;                             // derivative dh_{t}/dz_{t}, [bS x nU]
            *dLdWx += mmul(xtT, dhtdzt * dhtP1dht);
            *dLdWh += mmul(htT, dhtdzt * dhtP1dht);
            *dLdb  += mmul(dhtdzt.sum(), dhtdzt * dhtP1dht);

            dhtP1dht *= mmul(dhtdzt, WhT);                                  // get series of derivatives products, for example: dL_3/dh_3°dh_3/dh_2, dL_3/dh_3°dh_3/dh_2°dh_2/dh_1, dL_3/dh_3°dh_3/dh_2°dh_2/dh_1°dh_1/dh_0

        }
        
    }

    delete xT;
    delete hT;
    
    // back propagation    
                         // dh_{t}/dz_{t}, [bS x nU]
    

    dLdht->assign(*dLtdht + *dLdhtP1);
    
    dLdhtP1->assign(dLdht * dhtdhtM1);

    dLdxt->assign(mmul(*dLdht * dhtdzt, Wx->transp()));     // [bS x iS]

    for(int i = 0; i < t; ++t)

    // dLdhtP1->assign(dLdht);
    }
}


template void rnnCell<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>* ht);
template void rnnCell<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* ht);
template void rnnCell<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>* ht);

template void rnnTimeLoop<float>  (const std::vector<NDArray<float>*>&   inArrs, NDArray<float>*   h, NDArray<float>*   hFinal);
template void rnnTimeLoop<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>* h, NDArray<float16>* hFinal);
template void rnnTimeLoop<double> (const std::vector<NDArray<double>*>&  inArrs, NDArray<double>*  h, NDArray<double>*  hFinal);


}
}
}


