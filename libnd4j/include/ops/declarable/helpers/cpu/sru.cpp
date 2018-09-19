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
// implementation of operations for Simple Recurrent Unit: arXiv:1709.02755v2 [cs.CL] 12 Sep 2017
//
//  @author Yurii Shyrma, created on 05.12.2017
//

#include<ops/declarable/helpers/sru.h>
#include <NDArrayFactory.h>

namespace nd4j    {
namespace ops     {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
static FORCEINLINE NDArray activation(const NDArray& arr) {
    
    // return (const_cast<NDArray<T>&>(arr)).template transform<simdOps::Tanh<T>>();    
    auto result = NDArrayFactory::_create(&arr, false, arr.getWorkspace());
    (const_cast<NDArray&>(arr)).applyTransform(transform::Tanh, &result);
    return result;
}


//////////////////////////////////////////////////////////////////////////
static FORCEINLINE NDArray sigmoid(const NDArray& arr) {
    return (const_cast<NDArray&>(arr)).transform(transform::Sigmoid);
}


//////////////////////////////////////////////////////////////////////////
void sruCell(const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs) {

    auto x  = inArrs[0];               // input [bS x inSize], bS - batch size, inSize - number of features
    auto c0 = inArrs[1];               // previous cell state c  [bS x inSize], that is at previous time step t-1
    auto w  = inArrs[2];               // weights [inSize x 3*inSize]
    auto b  = inArrs[3];               // biases [2*inSize]

    auto h  = outArrs[0];              // current cell output [bS x inSize], that is at current time step t
    auto c  = outArrs[1];              // current cell state  [bS x inSize], that is at current time step t

    const int inSize = x->sizeAt(1);           // inSize - number of features
            
    auto z = mmul(*x, *w);               //  [bS x 3*inSize]

    // forget gate = sigmoid(x*Wf + bf)
    auto f = sigmoid(z({0,0, inSize,   2*inSize}) + (*b)({0, inSize}));
    
    // reset gate = sigmoid(x*Wr + br)
    auto r = sigmoid(z({0,0, 2*inSize, 3*inSize}) + (*b)({inSize, 2*inSize}));

    // ◦ means element-wise product or so called Hadamard product
    // current sell state = f◦c0 + (1 - f)◦(x*Wc)
    c->assign( f*(*c0) + (1.f - f) * z({0,0 ,0, inSize}) );
    // *c = f*(*c0 - z({},{0, inSize})) + z({{},{0, inSize}});

    // current cell output = r◦activation(c) + (1 - r)◦x
    h->assign( r*activation(*c) + (1.f - r) * (*x) );
    // *h = r * (activation<T>(c) - *x) + *x;        
}

//////////////////////////////////////////////////////////////////////////
// template <typename T>
// void sruCellBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs) {

//     NDArray<T>* x    = inArrs[0];               // input [bS x inSize], bS - batch size, inSize - number of features
//     NDArray<T>* c0   = inArrs[1];               // previous cell state c  [bS x inSize], that is at previous time step t-1   
//     NDArray<T>* w    = inArrs[2];               // weights [inSize x 3*inSize]
//     NDArray<T>* b    = inArrs[3];               // biases [2*inSize]
//     NDArray<T>* dLdC = inArrs[4];               // gradient of the loss func with respect to cell output [bS x inSize]
//     NDArray<T>* dLdH = inArrs[5];               // gradient of the loss func with respect to cell state  [bS x inSize]

//     NDArray<T>* dLdX  = outArrs[0];             // gradient of the loss func with respect to input [bS x inSize], so called epsilon
//     NDArray<T>* dLdW  = outArrs[1];             // gradient of the loss func with respect to weights [inSize x 3*inSize]
//     NDArray<T>* dLdB  = outArrs[2];             // gradient of the loss func with respect to biases [2*inSize]
//     NDArray<T>* dLdC0 = outArrs[3];             // gradient of the loss func with respect to previous cell state [bS, inSize]

//     const int inSize = x->sizeAt(1);           // inSize - number of features
            
//     //*********** feed forward ***********//
//     NDArray<T> z = mmul(*x, *w);               //  [bS x 3*inSize]    

//     // forget gate = sigmoid(x*Wf + bf)
//     NDArray<T> f = sigmoid<T>(z({{},{inSize,   2*inSize}}) + (*b)({{0, inSize}}));             // [bS, inSize]    
//     NDArray<T> oneMinusF = 1. - f;
    
//     // reset gate = sigmoid(x*Wr + br)
//     NDArray<T> r = sigmoid<T>(z({{},{2*inSize, 3*inSize}}) + (*b)({{inSize, 2*inSize}}));      // [bS, inSize]    
//     NDArray<T> oneMinusR = 1. - r;

//     // current sell state = f◦c0 + (1 - f)◦(x*Wc)             --->  c->assign( f*(*c0) + ((T)1. - f) * z({{},{0, inSize}}) );
//     // current cell output = r◦activation(c) + (1 - r)◦x      --->  h->assign( r*activation<T>(*c) + ((T)1. - r) * (*x) );    


//     //*********** back propagation ***********//
//     // dCdC0 = f;    
//     // dFdX = Wf    
//     // dRdX = Wr

//     NDArray<T> tanh = activation<T>(*c);
//     NDArray<T> dFdBf = f * oneMinusF;
//     NDArray<T> dRdBr = r * oneMinusR;   
//     NDArray<T> dHdR = tanh - *x;    
//     // dCdF = c0 - x*Wc;    
//     NDArray<T> dCdF = *c0 - z({{},{0, inSize}});    
//     // dHdC = r * (1 - tanh*tanh)
//     NDArray<T> dHdC = r * (1. - tanh * tanh);
//     // dCdX = dCdX + dCdF*dFdX = (1-f)*Wc +  dCdF*Wf
//     NDArray<T> dCdX = oneMinusF * (*w)({{},{0, inSize}}) + dCdF * (*w)({{},{inSize, 2*inSize}});


//     // dLdC0 =  dLdC * dCdC0 = dLdC * f
//     dLdC0->assign((*dLdC) * f);

    
//     // dLdBf = dLdH*dHdBf + dLdC*dCdBf = dLdH*dHdC*dCdBf + dLdC*dCdF*dFdBf =  dLdH*dHdC*dCdF*dFdBf + dLdC*dCdF*dFdBf = (dLdH*dHdC + dLdC)*dCdF*dFdBf
//     (*dLdB)({{0, inSize}}).assign(((*dLdH) * dHdC + *dLdC) * dCdF * dFdBf);
//     // dLdBr = dLdH * dHdR * dRdBr 
//     (*dLdB)({{inSize, 2*inSize}}).assign((*dLdH) * dHdR * dRdBr)
    
    
//     // dLdWc = dLdH*dHdWc + dLdC*dCdWc = dLdH*dHdC*dCdWc + dLdC*dCdWc = (dLdH*dHdC + dLdC) * dCdWc = (dLdH*dHdC + dLdC) * (1-f)*x
//     (*dLdW)({{}, {0, inSize}}).assign(((*dLdH) * dHdC + *dLdC) * oneMinusF * (*x));
//     // dLdWf = dLdBf * x
//     (*dLdW)({{}, {inSize, 2*inSize}}).assign((*dLdB)({{0, inSize}}) * (*x));
//     // dLdWr = dLdBr * x
//     (*dLdW)({{}, {2*inSize, 3*inSize}}).assign((*dLdB)({{inSize, 2*inSize}}) * (*x));    
    

//     // dLdX = dLdH*dHdX + dLdC*dCdX = dLdH*(dHdX + dHdR*dRdX + dHdC*dCdX) + dLdC*dCdF*dFdX = dLdH*(1 - r + dHdR*dRdX + dHdC*dCdX) + dLdC*dCdX
//     dLdX->assign((*dLdH) * (oneMinusR + dHdR * (*w)({{},{2*inSize, 3*inSize}}) + dHdC * dCdX) + (*dLdC) * dCdX);   
// }

//////////////////////////////////////////////////////////////////////////
void sruTimeLoop(const std::vector<NDArray*>& inArrs, const std::vector<NDArray*>& outArrs) {
    
    auto x  = inArrs[0];                     // input [bS x inSize x time]
    auto c0 = inArrs[1];                     // initial cell state  (at time step = 0) [bS x inSize],
    auto w  = inArrs[2];                     // weights, [3*inSize x inSize]
    auto b  = inArrs[3];                     // biases,  [2*inSize]
    
    auto h  = outArrs[0];                    // cell outputs [bS x inSize x time]
    auto c  = outArrs[1];                    // cell states  [bS x inSize x time]

    w = w->transpose();                             // [3*inSize x inSize] -> [inSize x 3*inSize] 

    const int time  = x->sizeAt(2);

    NDArray ct_1(*c0);

    // loop through time steps
    for (int t = 0; t < time; ++t) {

        auto xt = (*x)({0,0, 0,0, t,t+1});
        auto ht = (*h)({0,0, 0,0, t,t+1});
        auto ct = (*c)({0,0, 0,0, t,t+1});

        helpers::sruCell({&xt, &ct_1, w, b},  {&ht, &ct});
        ct_1.assign(ct);
    }    

    delete w;
}

}
}
}