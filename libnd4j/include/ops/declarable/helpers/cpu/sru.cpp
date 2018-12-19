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
    auto result = NDArray(&arr, false, arr.getWorkspace());
    (const_cast<NDArray&>(arr)).applyTransform(transform::Tanh, &result);
    return result;
}


//////////////////////////////////////////////////////////////////////////
static FORCEINLINE NDArray sigmoid(const NDArray& arr) {
    return (const_cast<NDArray&>(arr)).transform(transform::Sigmoid);
}


//////////////////////////////////////////////////////////////////////////
void sruCell(const NDArray* x, const NDArray* c0, const NDArray* w, const NDArray* b, NDArray* h, NDArray* c) {

    // x   input [bS x inSize], bS - batch size, inSize - number of features
    // c0  previous cell state c  [bS x inSize], that is at previous time step t-1
    // w   weights [inSize x 3*inSize]
    // b   biases [2*inSize]

    // h   current cell output [bS x inSize], that is at current time step t
    // c   current cell state  [bS x inSize], that is at current time step t

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
void sruTimeLoop(const NDArray* x, const NDArray* c0, const NDArray* w, const NDArray* b, NDArray* h, NDArray* c) {
    
    // x   input [bS x inSize x time]
    // c0  initial cell state  (at time step = 0) [bS x inSize],
    // w   weights, [3*inSize x inSize]
    // b   biases,  [2*inSize]
    
    // h   cell outputs [bS x inSize x time]
    // c   cell states  [bS x inSize x time]

    w = w->transpose();                             // [3*inSize x inSize] -> [inSize x 3*inSize] 

    const int time  = x->sizeAt(2);

    NDArray ct_1(*c0);

    // loop through time steps
    for (int t = 0; t < time; ++t) {

        auto xt = (*x)({0,0, 0,0, t,t+1});
        auto ht = (*h)({0,0, 0,0, t,t+1});
        auto ct = (*c)({0,0, 0,0, t,t+1});

        helpers::sruCell(&xt, &ct_1, w, b,  &ht, &ct);
        ct_1.assign(ct);
    }    

    delete w;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void sruBI_(NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* mask, NDArray* ht, NDArray* ct) {

    // x     input 3d tensor [time x bS x 2*inSize], time - number of time steps, bS - batch size, inSize - number of features
    // w     2d tensor of weights [2*inSize x 6*inSize]
    // b     row of biases with twice length [1 × 4*inSize]
    // c0    2d tensor of initial state [bS x 2*inSize] at time t=0
    // mask  optional, 2d tensor of dropout mask [bS x 2*inSize]

    // ht  [time x bS x 2K]
    // ct  [time x bS x 2K]
    
    const Nd4jLong time   = x->sizeAt(0);                     // time - number of time steps
    const Nd4jLong bS     = x->sizeAt(1);                     // bS - batch size
    const Nd4jLong inSize = x->sizeAt(2) / 2;                 // inSize - number of features
  
    //  x = x * mask
    if(mask)
        x->applyBroadcast(broadcast::Multiply, {1, 2}, mask, x, nullptr);             // apply mask
    
    // U = x * w
    NDArray wi = mmul(*x, *w);                    //  U [time x bS x 6K]

    const Nd4jLong d2      = 2*inSize;
    const Nd4jLong ncols   = bS*d2;     
    const Nd4jLong ncolsWi = 3*ncols;    

    T* pI    = x->bufferAsT<T>();
    T* pWi   = wi.bufferAsT<T>();
    T* pBias = const_cast<NDArray*>(b)->bufferAsT<T>();
    T* pInit = const_cast<NDArray*>(c0)->bufferAsT<T>();
    T* pMask = mask ? const_cast<NDArray*>(mask)->bufferAsT<T>() : nullptr;
    T* pHt   = ht->bufferAsT<T>();
    T* pCt   = ct->bufferAsT<T>();

    Nd4jLong ncolsRev, ncolsWiRev;                   // for reverse direction
    T maskVal, cur, bF, bR, ft, rt, val;
    T *pIVal(nullptr), *pWiVal(nullptr), *pHtVal(nullptr), *pCtVal(nullptr);
    bool flip = false;

    for (Nd4jLong col = 0; col < ncols; ++col) {           
        
        flip       = (col % d2) >= inSize;
        maskVal    = mask ? *(pMask + col) : T(1);
        cur        = *(pInit + col);
        bF         = *(pBias + col%d2);
        bR         = *(pBias + col%d2 + d2);
        pWiVal     = pWi     + 3*col;
        pIVal      = pI      + col;
        pHtVal     = pHt     + col;
        pCtVal     = pCt     + col;

        if (flip) {
            pIVal  += (time-1)*ncols;
            pWiVal += (time-1)*ncolsWi;
            pHtVal += (time-1)*ncols;
            pCtVal += (time-1)*ncols;
        }

        ncolsRev   = flip ? -ncols   : ncols;
        ncolsWiRev = flip ? -ncolsWi : ncolsWi;

        for (Nd4jLong t = 0; t < time; ++t) {
            // evaluate sigmoids
            ft = (1.)/(1. + nd4j::math::nd4j_exp<T, T>(-(*(pWiVal + 1) + bF)));
            rt = (1.)/(1. + nd4j::math::nd4j_exp<T, T>(-(*(pWiVal + 2) + bR)));

            cur = (cur - *pWiVal)*ft + *pWiVal;
            *pCtVal = cur;
            val = nd4j::math::nd4j_tanh<T, T>(cur);
            *pHtVal = (val*maskVal - *pIVal)*rt + *pIVal;

            pIVal  += ncolsRev;
            pWiVal += ncolsWiRev;
            pCtVal += ncolsRev;
            pHtVal += ncolsRev;
        }
    }    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void sruBIBP_(NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* ct, const NDArray* inGradC0, const NDArray* inGradHt, const NDArray* mask,    
                     NDArray* gradI, NDArray* gradW, NDArray* gradB, NDArray* gradC0) {
    
    // x  input 3d tensor [time x bS x 2*inSize], time - number of time steps, bS - batch size, inSize - number of features
    // w  2d tensor of weights [2*inSize x 6*inSize]
    // b  row of biases with twice length [1 × 4*inSize]
    // c0 2d tensor of initial state [bS x 2*inSize] at time t=0
    // ct [time x bS x 2*inSize]
    // inGradC0 [bS x 2*inSize]
    // inGradHt  [time x bS x 2*inSize]
    // mask optional,  2d tensor of dropout mask [bS x 2*inSize]

    // gradI  [time x bS x 2*inSize]
    // gradW  [time x 2*inSize x 6*inSize]
    // gradB  [1 x 4*inSize]
    // gradC0 [bS x 2*inSize]

    const Nd4jLong time   = x->sizeAt(0);                     // time - number of time steps
    const Nd4jLong bS     = x->sizeAt(1);
    const Nd4jLong inSize = x->sizeAt(2) / 2;

    //  x = x * mask
    if(mask)
        x->applyBroadcast(broadcast::Multiply, {1, 2}, mask, x, nullptr);             // apply mask        

    // U = x * w
    NDArray wi = mmul(*x, *w);                    //  [time x bS x 2*inSize] * [2*inSize x 6*inSize] = [time x bS x 6*inSize]
    NDArray gradBias(x->ordering(), {bS, 4*inSize}, x->dataType(), x->getWorkspace());
    NDArray gradWi  (x->ordering(), {time, bS, 6*inSize}, x->dataType(), x->getWorkspace());
    
    const Nd4jLong d2      = 2*inSize;
    const Nd4jLong ncols   = bS*d2;     
    const Nd4jLong ncolsWi = 3*ncols;    
    T* pInput     = x->bufferAsT<T>();
    T* pWi        = wi.bufferAsT<T>();
    T* pBias      = const_cast<NDArray*>(b)->bufferAsT<T>();
    T* pInit      = const_cast<NDArray*>(c0)->bufferAsT<T>();
    T* pMask      = mask ? const_cast<NDArray*>(mask)->bufferAsT<T>() : nullptr;
    T* pState     = const_cast<NDArray*>(ct)->bufferAsT<T>();
    T* pInGradCt  = const_cast<NDArray*>(inGradC0)->bufferAsT<T>();
    T* pInGradHt  = const_cast<NDArray*>(inGradHt)->bufferAsT<T>();
    T* pGradWi    = gradWi.bufferAsT<T>();
    T* pGradInput = gradI->bufferAsT<T>();
    T* pGradBias  = gradBias.bufferAsT<T>();
    T* pGradInit  = gradC0->bufferAsT<T>();
    
    Nd4jLong ncolsRev, ncolsWiRev;                   // for reverse direction
    T gbF, gbR, cur, maskVal, bF, bR, ft, rt, val, prevVal, gft, grt, gradSateVal;
    bool flip = false;
    T *pInputVal(nullptr), *pWiVal(nullptr),  *pStateVal(nullptr), *pInGradHtVal(nullptr), *pGradWiVal(nullptr), *pGradInputVal(nullptr); 
    
    for (Nd4jLong col = 0; col < ncols; ++col) {           
        gbF = gbR = (T)0.;
        flip          = (col%d2) >= inSize;
        maskVal       = mask ? *(pMask + col) : T(1.);
        cur           = *(pInGradCt + col);
        bF            = *(pBias     + col%d2);
        bR            = *(pBias     + col%d2 + d2);
        pWiVal        = pWi         + 3*col;
        pInputVal     = pInput      + col;
        pStateVal     = pState      + col;
        pInGradHtVal  = pInGradHt    + col;
        pGradWiVal    = pGradWi     + 3*col;
        pGradInputVal = pGradInput  + col;                    
        if (!flip) {
            pInputVal     += (time-1)*ncols;
            pWiVal        += (time-1)*ncolsWi;
            pStateVal     += (time-1)*ncols;
            pInGradHtVal  += (time-1)*ncols;
            pGradWiVal    += (time-1)*ncolsWi;
            pGradInputVal += (time-1)*ncols;
        }
        ncolsRev   = flip ? -ncols   : ncols;
        ncolsWiRev = flip ? -ncolsWi : ncolsWi;
        
        for (Nd4jLong t = 0; t < time; ++t) {
            // evaluate sigmoids 
            ft = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T,T>(-(*(pWiVal + 1) + bF)));
            rt = ((T)1.)/((T)1. + nd4j::math::nd4j_exp<T,T>(-(*(pWiVal + 2) + bR)));
            
            val     = nd4j::math::nd4j_tanh<T,T>(*pStateVal);            
            prevVal = (t < time-1) ? (*(pStateVal - ncolsRev)) : (*(pInit + col));
            // grad wrt input
            *pGradInputVal = *pInGradHtVal - (*pInGradHtVal)*rt ;
            // grad wrt rt, wiR and bR
            grt = (*pInGradHtVal) * (val*maskVal - *pInputVal) * (rt - rt*rt);
            *(pGradWiVal + 2) = grt;
            gbR += grt;
            // grad wrt state          
            gradSateVal = (*pInGradHtVal) * maskVal * (rt - rt*val*val) + cur;
            // grad wrt wi0
            *pGradWiVal = gradSateVal - gradSateVal*ft;
            // grad wrt ft, wi1, and bF
            gft = gradSateVal * (prevVal - *pWiVal) * (ft - ft*ft);
            *(pGradWiVal + 1) = gft;
            gbF += gft;
            // grad wrt c_previous
            cur = gradSateVal * ft;
            pInputVal     -= ncolsRev;
            pWiVal        -= ncolsWiRev;
            pStateVal     -= ncolsRev;
            pGradWiVal    -= ncolsWiRev;
            pGradInputVal -= ncolsRev;
            pInGradHtVal  -= ncolsRev;            
        } 
        *(pGradBias + col) = gbF;
        *(pGradBias + col + ncols) = gbR;
        *(pGradInit + col) = cur;
    }

    // gradB    
    gradBias.reduceAlongDimension(reduce::Sum, gradB, {0}, false, true);    // [1 x 4K]    
    
    // gradW     
    x->permutei({0, 2, 1});                                             // [time x bS x 2*inSize] -> [time x 2*inSize x bS]
    *gradW = mmul(*x, gradWi);                                    // [time x 2*inSize x bS ] * [time x bS x 6*inSize] = [time x 2*inSize x 6*inSize]
}


void sruBI(NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* mask, NDArray* ht, NDArray* ct) {
    BUILD_SINGLE_SELECTOR(x->dataType(), sruBI_, (x, w, b, c0, mask, ht, ct), FLOAT_TYPES);
}
void sruBIBP(NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* ct, const NDArray* inGradC0, const NDArray* inGradH, const NDArray* mask, NDArray* gradI, NDArray* gradW, NDArray* gradB, NDArray* gradC0) {
    BUILD_SINGLE_SELECTOR(x->dataType(), sruBIBP_, (x, w, b, c0, ct, inGradC0, inGradH, mask, gradI, gradW, gradB, gradC0), FLOAT_TYPES);
}


BUILD_SINGLE_TEMPLATE(template void sruBI_,   (NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* mask, NDArray* ht, NDArray* ct), FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void sruBIBP_, (NDArray* x, const NDArray* w, const NDArray* b, const NDArray* c0, const NDArray* ct, const NDArray* inGradC0, const NDArray* inGradH, const NDArray* mask, NDArray* gradI, NDArray* gradW, NDArray* gradB, NDArray* gradC0), FLOAT_TYPES);


}
}
}