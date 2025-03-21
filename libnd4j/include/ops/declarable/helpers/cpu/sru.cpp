/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/sru.h>
#if NOT_EXCLUDED(OP_sru)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
static SD_INLINE NDArray activation(NDArray& arr) {
  auto result = NDArray(&arr, false, arr.getContext());
  (const_cast<NDArray&>(arr)).applyTransform(transform::Tanh, &result);
  return result;
}

//////////////////////////////////////////////////////////////////////////
static SD_INLINE NDArray sigmoid(NDArray& arr) {
  return (const_cast<NDArray&>(arr)).transform(transform::Sigmoid);
}

//////////////////////////////////////////////////////////////////////////
void sruCell(sd::LaunchContext* context, NDArray* x, NDArray* c0, NDArray* w, NDArray* b,
             NDArray* h, NDArray* c) {
  // x   input [bS x inSize], bS - batch size, inSize - number of features
  // c0  previous cell state c  [bS x inSize], that is at previous time step t-1
  // w   weights [inSize x 3*inSize]
  // b   biases [2*inSize]

  // h   current cell output [bS x inSize], that is at current time step t
  // c   current cell state  [bS x inSize], that is at current time step t

  const int inSize = x->sizeAt(1);  // inSize - number of features

  auto z = mmul(*x, *w);  //  [bS x 3*inSize]

  // forget gate = sigmoid(x*Wf + bf)
  NDArray result = z({0, 0, inSize, 2 * inSize}) + (*b)({0, inSize});
  auto f = sigmoid(result);

  // reset gate = sigmoid(x*Wr + br)
  NDArray sigmoidIn = z({0, 0, 2 * inSize, 3 * inSize}) + (*b)({inSize, 2 * inSize});
  auto r = sigmoid(sigmoidIn);

  // ◦ means element-wise product or so called Hadamard product
  // current sell state = f◦c0 + (1 - f)◦(x*Wc)
  c->assign(f * (*c0) + (1.f - f) * z({0, 0, 0, inSize}));
  // *c = f*(*c0 - z({},{0, inSize})) + z({{},{0, inSize}});

  // current cell output = r◦activation(c) + (1 - r)◦x
  h->assign(r * activation(*c) + (1.f - r) * (*x));
  // *h = r * (activation<T>(c) - *x) + *x;
}

//////////////////////////////////////////////////////////////////////////
void sruTimeLoop(sd::LaunchContext* context, NDArray* x, NDArray* c0, NDArray* w, NDArray* b,
                 NDArray* h, NDArray* c) {
  // x   input [bS x inSize x time]
  // c0  initial cell state  (at time step = 0) [bS x inSize],
  // w   weights, [3*inSize x inSize]
  // b   biases,  [2*inSize]

  // h   cell outputs [bS x inSize x time]
  // c   cell states  [bS x inSize x time]

  auto wT = w->transpose();  // [3*inSize x inSize] -> [inSize x 3*inSize]

  const int time = x->sizeAt(2);

  NDArray ct_1(*c0);

  // loop through time steps
  for (int t = 0; t < time; ++t) {
    auto xt = (*x)({0, 0, 0, 0, t, t + 1});
    auto ht = (*h)({0, 0, 0, 0, t, t + 1});
    auto ct = (*c)({0, 0, 0, 0, t, t + 1});

    helpers::sruCell(context, &xt, &ct_1, &wT, b, &ht, &ct);
    ct_1.assign(ct);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void sruBI_(NDArray* x, NDArray* w, NDArray* b, NDArray* c0, NDArray* mask, NDArray* ht,
                   NDArray* ct) {
  // x     input 3d tensor [time x bS x 2*K], time - number of time steps, bS - batch size, K - number of features
  // w     2d tensor of weights [2*K x 6*K]
  // b     row of biases with twice length [4*K]
  // c0    2d tensor of initial state [bS x 2*K] at time t=0
  // mask  optional, 2d tensor of dropout mask [bS x 2*K]

  // ht  [time x bS x 2*K]
  // ct  [time x bS x 2*K]

  const sd::LongType time = x->sizeAt(0);   // time - number of time steps
  const sd::LongType bS = x->sizeAt(1);     // bS - batch size
  const sd::LongType K = x->sizeAt(2) / 2;  // K - number of features
  std::vector<sd::LongType> dims2 = {1, 2};
  //  x = x * mask
  if (mask) x->applyBroadcast(broadcast::Multiply, &dims2, mask, x);  // apply mask

  // U = x * w
  NDArray wi = mmul(*x, *w);  //  U [time x bS x 6*K]

  const sd::LongType d2 = 2 * K;
  const sd::LongType ncols = bS * d2;
  const sd::LongType ncolsWi = 3 * ncols;

  T* pI = x->bufferAsT<T>();
  T* pWi = wi.bufferAsT<T>();
  T* pBias = const_cast<NDArray*>(b)->bufferAsT<T>();
  T* pInit = const_cast<NDArray*>(c0)->bufferAsT<T>();
  T* pMask = mask ? const_cast<NDArray*>(mask)->bufferAsT<T>() : nullptr;
  T* pHt = ht->bufferAsT<T>();
  T* pCt = ct->bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    for (auto col = start; col < stop; col++) {
      const auto colNum = col % d2;
      bool flip = colNum >= K;
      T maskVal = mask ? *(pMask + col) : T(1);
      T cur = *(pInit + col);
      T bF = *(pBias + colNum);
      T bR = *(pBias + colNum + d2);
      T* pWiVal = pWi + 3 * col;
      T* pIVal = pI + col;
      T* pHtVal = pHt + col;
      T* pCtVal = pCt + col;

      if (flip) {
        const auto step = (time - 1) * ncols;
        pIVal += step;
        pHtVal += step;
        pCtVal += step;
        pWiVal += (time - 1) * ncolsWi;
      }

      auto ncolsRev = flip ? -ncols : ncols;
      auto ncolsWiRev = flip ? -ncolsWi : ncolsWi;

      for (sd::LongType t = 0; t < time; ++t) {
        // evaluate sigmoids
        T ft = (1.) / (1. + sd::math::sd_exp<T, T>(-(pWiVal[1] + bF)));
        T rt = (1.) / (1. + sd::math::sd_exp<T, T>(-(pWiVal[2] + bR)));

        cur = (cur - *pWiVal) * ft + *pWiVal;
        *pCtVal = cur;
        T val = sd::math::sd_tanh<T, T>(cur);
        *pHtVal = (val * maskVal - *pIVal) * rt + *pIVal;

        pIVal += ncolsRev;
        pWiVal += ncolsWiRev;
        pCtVal += ncolsRev;
        pHtVal += ncolsRev;
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, ncols);
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void sruBIBP_(NDArray* x, NDArray* w, NDArray* b, NDArray* c0, NDArray* ct,
                     NDArray* inGradC0, NDArray* inGradHt, NDArray* mask, NDArray* gradI,
                     NDArray* gradW, NDArray* gradB, NDArray* gradC0) {
  // x  input 3d tensor [time x bS x 2*K], time - number of time steps, bS - batch size, K - number of features
  // w  2d tensor of weights [2*K x 6*K]
  // b  row of biases with twice length 4*K]
  // c0 2d tensor of initial state [bS x 2*K] at time t=0
  // ct [time x bS x 2*K]
  // inGradC0 [bS x 2*K]
  // inGradHt  [time x bS x 2*K]
  // mask optional,  2d tensor of dropout mask [bS x 2*K]

  // gradI  [time x bS x 2*K]
  // gradW  [time x 2*K x 6*K]
  // gradB  [4*K]
  // gradC0 [bS x 2*K]

  const sd::LongType time = x->sizeAt(0);  // time - number of time steps
  const sd::LongType bS = x->sizeAt(1);
  const sd::LongType K = x->sizeAt(2) / 2;
  std::vector<sd::LongType> dims2 = {1, 2};

  //  x = x * mask
  if (mask) x->applyBroadcast(broadcast::Multiply, &dims2, mask, x);  // apply mask

  // U = x * w
  NDArray wi = mmul(*x, *w);  //  [time x bS x 2*K] * [2*K x 6*K] = [time x bS x 6*K]
  std::vector<sd::LongType> biasShape = {bS, 4 * K};
  std::vector<sd::LongType> wShape = {time, bS, 6 * K};
  NDArray gradBias(x->ordering(), biasShape, x->dataType(), x->getContext());
  NDArray gradWi(x->ordering(), wShape, x->dataType(), x->getContext());

  const sd::LongType d2 = 2 * K;
  const sd::LongType ncols = bS * d2;
  const sd::LongType ncolsWi = 3 * ncols;
  T* pInput = x->bufferAsT<T>();
  T* pWi = wi.bufferAsT<T>();
  T* pBias = const_cast<NDArray*>(b)->bufferAsT<T>();
  T* pInit = const_cast<NDArray*>(c0)->bufferAsT<T>();
  T* pMask = mask ? const_cast<NDArray*>(mask)->bufferAsT<T>() : nullptr;
  T* pState = const_cast<NDArray*>(ct)->bufferAsT<T>();
  T* pInGradCt = const_cast<NDArray*>(inGradC0)->bufferAsT<T>();
  T* pInGradHt = const_cast<NDArray*>(inGradHt)->bufferAsT<T>();
  T* pGradWi = gradWi.bufferAsT<T>();
  T* pGradInput = gradI->bufferAsT<T>();
  T* pGradBias = gradBias.bufferAsT<T>();
  T* pGradInit = gradC0->bufferAsT<T>();

  auto func = PRAGMA_THREADS_FOR {
    for (auto col = start; col < stop; col++) {
      T gbF = 0.f;
      T gbR = 0.f;
      const auto colNum = col % d2;
      const bool flip = colNum >= K;
      T maskVal = mask ? *(pMask + col) : T(1.);
      T cur = *(pInGradCt + col);
      T bF = *(pBias + colNum);
      T bR = *(pBias + colNum + d2);
      T* pWiVal = pWi + 3 * col;
      T* pInputVal = pInput + col;
      T* pStateVal = pState + col;
      T* pInGradHtVal = pInGradHt + col;
      T* pGradWiVal = pGradWi + 3 * col;
      T* pGradInputVal = pGradInput + col;

      if (!flip) {
        const auto stepI = (time - 1) * ncols;
        const auto stepW = (time - 1) * ncolsWi;
        pInputVal += stepI;
        pStateVal += stepI;
        pInGradHtVal += stepI;
        pGradInputVal += stepI;
        pWiVal += stepW;
        pGradWiVal += stepW;
      }

      sd::LongType ncolsRev = flip ? -ncols : ncols;
      sd::LongType ncolsWiRev = flip ? -ncolsWi : ncolsWi;

      for (sd::LongType t = 0; t < time; ++t) {
        // evaluate sigmoids
        T ft = ((T)1.) / ((T)1. + sd::math::sd_exp<T, T>(-(*(pWiVal + 1) + bF)));
        T rt = ((T)1.) / ((T)1. + sd::math::sd_exp<T, T>(-(*(pWiVal + 2) + bR)));

        T val = sd::math::sd_tanh<T, T>(*pStateVal);
        T prevVal = (t < time - 1) ? (*(pStateVal - ncolsRev)) : (*(pInit + col));
        // grad wrt input
        *pGradInputVal = *pInGradHtVal - (*pInGradHtVal) * rt;
        // grad wrt rt, wiR and bR
        T grt = (*pInGradHtVal) * (val * maskVal - *pInputVal) * (rt - rt * rt);
        *(pGradWiVal + 2) = grt;
        gbR += grt;
        // grad wrt state
        T gradSateVal = (*pInGradHtVal) * maskVal * (rt - rt * val * val) + cur;
        // grad wrt wi0
        *pGradWiVal = gradSateVal - gradSateVal * ft;
        // grad wrt ft, wi1, and bF
        T gft = gradSateVal * (prevVal - *pWiVal) * (ft - ft * ft);
        *(pGradWiVal + 1) = gft;
        gbF += gft;
        // grad wrt c_previous
        cur = gradSateVal * ft;

        pInputVal -= ncolsRev;
        pWiVal -= ncolsWiRev;
        pStateVal -= ncolsRev;
        pGradWiVal -= ncolsWiRev;
        pGradInputVal -= ncolsRev;
        pInGradHtVal -= ncolsRev;
      }
      *(pGradBias + col) = gbF;
      *(pGradBias + col + ncols) = gbR;
      *(pGradInit + col) = cur;
    }
  };

  samediff::Threads::parallel_tad(func, 0, ncols);

  // gradB

  std::vector<sd::LongType> dims = {0};
  gradBias.reduceAlongDimension(reduce::Sum, gradB, &dims);  // [4*K]

  // gradW
  x->permutei({0, 2, 1}, 0, false);                       // [time x bS x 2*K] -> [time x 2*K x bS]
  MmulHelper::mmul(x, &gradWi, gradW, 1., 0.);  // [time x 2*K x bS ] * [time x bS x 6*K] = [time x 2*K x 6*K]
}

void sruBI(sd::LaunchContext* context, NDArray* x, NDArray* w, NDArray* b, NDArray* c0,
           NDArray* mask, NDArray* ht, NDArray* ct) {
  BUILD_SINGLE_SELECTOR(x->dataType(), sruBI_, (x, w, b, c0, mask, ht, ct), SD_FLOAT_TYPES);
}
void sruBIBP(sd::LaunchContext* context, NDArray* x, NDArray* w, NDArray* b, NDArray* c0,
             NDArray* ct, NDArray* inGradC0, NDArray* inGradH, NDArray* mask, NDArray* gradI,
             NDArray* gradW, NDArray* gradB, NDArray* gradC0) {
  BUILD_SINGLE_SELECTOR(x->dataType(), sruBIBP_,
                        (x, w, b, c0, ct, inGradC0, inGradH, mask, gradI, gradW, gradB, gradC0), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void sruBI_,
                      (NDArray * x, NDArray* w, NDArray* b, NDArray* c0, NDArray* mask,
                       NDArray* ht, NDArray* ct),
                      SD_FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE(template void sruBIBP_,
                      (NDArray * x, NDArray* w, NDArray* b, NDArray* c0, NDArray* ct,
                       NDArray* inGradC0, NDArray* inGradH, NDArray* mask, NDArray* gradI,
                       NDArray* gradW, NDArray* gradB, NDArray* gradC0),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

//////////////////////////////////////////////////////////////////////////
// template <typename T>
// void sruCellBP(const std::vector<NDArray<T>*>& inArrs, const std::vector<NDArray<T>*>& outArrs) {

//     NDArray<T>* x    = inArrs[0];               // input [bS x inSize], bS - batch size, inSize - number of features
//     NDArray<T>* c0   = inArrs[1];               // previous cell state c  [bS x inSize], that is at previous time
//     step t-1 NDArray<T>* w    = inArrs[2];               // weights [inSize x 3*inSize] NDArray<T>* b    = inArrs[3];
//     // biases [2*inSize] NDArray<T>* dLdC = inArrs[4];               // gradient of the loss func with respect to
//     cell output [bS x inSize] NDArray<T>* dLdH = inArrs[5];               // gradient of the loss func with respect
//     to cell state  [bS x inSize]

//     NDArray<T>* dLdX  = outArrs[0];             // gradient of the loss func with respect to input [bS x inSize], so
//     called epsilon NDArray<T>* dLdW  = outArrs[1];             // gradient of the loss func with respect to weights
//     [inSize x 3*inSize] NDArray<T>* dLdB  = outArrs[2];             // gradient of the loss func with respect to
//     biases [2*inSize] NDArray<T>* dLdC0 = outArrs[3];             // gradient of the loss func with respect to
//     previous cell state [bS, inSize]

//     const int inSize = x->sizeAt(1);           // inSize - number of features

//     //*********** feed forward ***********//
//     NDArray<T> z = mmul(*x, *w);               //  [bS x 3*inSize]

//     // forget gate = sigmoid(x*Wf + bf)
//     NDArray<T> f = sigmoid<T>(z({{},{inSize,   2*inSize}}) + (*b)({{0, inSize}}));             // [bS, inSize]
//     NDArray<T> oneMinusF = 1. - f;

//     // reset gate = sigmoid(x*Wr + br)
//     NDArray<T> r = sigmoid<T>(z({{},{2*inSize, 3*inSize}}) + (*b)({{inSize, 2*inSize}}));      // [bS, inSize]
//     NDArray<T> oneMinusR = 1. - r;

//     // current sell state = f◦c0 + (1 - f)◦(x*Wc)             --->  c->assign( f*(*c0) + ((T)1. - f) * z({{},{0,
//     inSize}}) );
//     // current cell output = r◦activation(c) + (1 - r)◦x      --->  h->assign( r*activation<T>(*c) + ((T)1. - r) *
//     (*x) );

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

//     // dLdBf = dLdH*dHdBf + dLdC*dCdBf = dLdH*dHdC*dCdBf + dLdC*dCdF*dFdBf =  dLdH*dHdC*dCdF*dFdBf + dLdC*dCdF*dFdBf
//     = (dLdH*dHdC + dLdC)*dCdF*dFdBf
//     (*dLdB)({{0, inSize}}).assign(((*dLdH) * dHdC + *dLdC) * dCdF * dFdBf);
//     // dLdBr = dLdH * dHdR * dRdBr
//     (*dLdB)({{inSize, 2*inSize}}).assign((*dLdH) * dHdR * dRdBr)

//     // dLdWc = dLdH*dHdWc + dLdC*dCdWc = dLdH*dHdC*dCdWc + dLdC*dCdWc = (dLdH*dHdC + dLdC) * dCdWc = (dLdH*dHdC +
//     dLdC) * (1-f)*x
//     (*dLdW)({{}, {0, inSize}}).assign(((*dLdH) * dHdC + *dLdC) * oneMinusF * (*x));
//     // dLdWf = dLdBf * x
//     (*dLdW)({{}, {inSize, 2*inSize}}).assign((*dLdB)({{0, inSize}}) * (*x));
//     // dLdWr = dLdBr * x
//     (*dLdW)({{}, {2*inSize, 3*inSize}}).assign((*dLdB)({{inSize, 2*inSize}}) * (*x));

//     // dLdX = dLdH*dHdX + dLdC*dCdX = dLdH*(dHdX + dHdR*dRdX + dHdC*dCdX) + dLdC*dCdF*dFdX = dLdH*(1 - r + dHdR*dRdX
//     + dHdC*dCdX) + dLdC*dCdX dLdX->assign((*dLdH) * (oneMinusR + dHdR * (*w)({{},{2*inSize, 3*inSize}}) + dHdC *
//     dCdX) + (*dLdC) * dCdX);
// }
#endif