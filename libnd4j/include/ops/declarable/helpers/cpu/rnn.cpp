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
static FORCEINLINE NDArray activation(const NDArray& arr) {

    return (const_cast<NDArray&>(arr)).transform(transform::Tanh);
}


//////////////////////////////////////////////////////////////////////////
void rnnCell(const NDArray* xt, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* ht_1, NDArray* ht) {

    // xt   input [bS x inSize]
    // Wx   input-to-hidden weights, [inSize  x numUnits]
    // Wh   hidden-to-hidden weights, [numUnits x numUnits]
    // b    biases, [2*numUnits]: {0, numUnits} are input-to-hidden biases and {numUnits, 2*numUnits} are hidden-to-hidden biases
    // ht_1 previous cell output [bS x numUnits],  that is at previous time step t-1, in case of projection=false -> numUnits=numUnits!!!

    const int numUnits  = ht_1->sizeAt(1);
    
    // ht is current cell output [bS x numUnits], that is at current time step t        
    NDArray arr = (*b)({{numUnits, 2*numUnits}});
    arr.printShapeInfo();
    arr.printIndexedBuffer();    

    NDArray temp = mmul(*xt, *Wx) + (*b)({{0, numUnits}})  +  mmul(*ht_1, *Wh) + (*b)({{numUnits, 2*numUnits}});
    NDArray temp2 = activation(temp);
    // temp2.printShapeInfo();
    // ht->printShapeInfo();
    printf("!!!!!!!!!!!!\n");
    ht->assign(temp2);      // [bS x numUnits] + [numUnits]  +  [bS x numUnits] + [numUnits] = [bS x numUnits]        
}


//////////////////////////////////////////////////////////////////////////
void rnnTimeLoop(const NDArray* x, const NDArray* Wx, const NDArray* Wh, const NDArray* b, const NDArray* h0, const NDArray* maxTimeStep, NDArray* h, NDArray* hFinal) {

    // x   input [time x bS x inSize]
	// Wx  input-to-hidden  weights, [inSize  x numUnits]
    // Wh  hidden-to-hidden weights, [numUnits x numUnits]
	// b   biases for, [2*numUnits]

	// h0          initial cell output (at time step = 0) [bS x numUnits]
	// maxTimeStep vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep
    
    const int time     = x->sizeAt(0);
    const int bS       = x->sizeAt(1);        
    
    // at first time step
    if(h0)
        hFinal->assign(h0);
    else 
        *hFinal = 0.;   

    BlasHelper::getInstance();          // to avoid memory leak in pragma parallel loops
// #pragma omp parallel for schedule(guided) collapse(2) if(bS > Environment::getInstance()->elementwiseThreshold())  
    // loop through batch of inputs           
    for (int e = 0; e < bS; ++e) {                  
        // loop through time steps
        for (int t = 0; t < time; ++t) {

            int maxStep = maxTimeStep ? maxTimeStep->e<int>(e) : time;

            auto xt   = (*x)({t,t+1, e,e+1, 0,0}, true);
            auto ht   = (*h)({t,t+1, e,e+1, 0,0}, true);
            auto ht_1 = (*hFinal)({e,e+1, 0,0}, true);                       // previous state
            
            if(t >= maxStep) {
                ht = 0.;
                if(maxStep != 0)                    
                    ht_1.assign((*h)({maxStep-1,maxStep, e,e+1, 0,0}));
            }
            else {
                helpers::rnnCell(&xt, Wx, Wh, b, &ht_1, &ht);
                ht_1.assign(ht);
            }
        }
    }    
}


}
}
}

