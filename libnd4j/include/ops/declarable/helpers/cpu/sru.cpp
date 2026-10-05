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
#include <ops/op_types.h>

#include <vector>
#if NOT_EXCLUDED(OP_sru)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
static SD_INLINE NDArray activation(NDArray& arr) {
  // an array of its own: NDArray(NDArray*, bool, context) wraps arr's buffer from its start, so tanh(c) of a time
  // step's view of the cell states was written over the first cell states
  NDArray result(arr.shapeInfo(), false, arr.getContext());
  arr.applyTransform(transform::Tanh, &result);
  return result;
}

//////////////////////////////////////////////////////////////////////////
static SD_INLINE NDArray* sigmoid(NDArray& arr) {
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

  NDArray *z = mmul(*x, *w);  //  [bS x 3*inSize]

  // forget gate = sigmoid(x*Wf + bf)
  NDArray *zView1 = (*z)({0, 0, inSize, 2 * inSize});
  NDArray *bView1 = (*b)({0, inSize});
  NDArray *addResult1 = (*zView1) + (*bView1);
  NDArray *f = sigmoid(*addResult1);
  delete addResult1;

  // reset gate = sigmoid(x*Wr + br)
  NDArray *zView2 = (*z)({0, 0, 2 * inSize, 3 * inSize});
  NDArray *bView2 = (*b)({inSize, 2 * inSize});
  NDArray *addResult2 = (*zView2) + (*bView2);
  NDArray *r = sigmoid(*addResult2);
  delete addResult2;

  // ◦ means element-wise product or so called Hadamard product
  // current sell state = f◦c0 + (1 - f)◦(x*Wc)
  NDArray *zView3 = (*z)({0, 0, 0, inSize});
  NDArray *fMulC0 = (*f) * (*c0);
  NDArray *oneMinusF = 1.f - (*f);
  NDArray *oneMinusFMulZ = (*oneMinusF) * (*zView3);
  NDArray *assignOne = (*fMulC0) + (*oneMinusFMulZ);
  c->assign(assignOne);
  delete fMulC0;
  delete oneMinusFMulZ;
  delete assignOne;
  delete oneMinusF;
  // *c = f*(*c0 - z({},{0, inSize})) + z({{},{0, inSize}});

  // current cell output = r◦activation(c) + (1 - r)◦x
  NDArray activationC = activation(*c);
  NDArray *rMulActivation = (*r) * activationC;
  NDArray *oneMinusR = 1.f - (*r);
  NDArray *oneMinusRMulX = *oneMinusR * (*x);
  NDArray *assign2 = (*rMulActivation) + (*oneMinusRMulX);
  h->assign(assign2);
  delete rMulActivation;
  delete oneMinusRMulX;
  delete assign2;
  delete oneMinusR;
  // *h = r * (activation<T>(c) - *x) + *x;

  delete z;
  delete zView1;
  delete bView1;
  delete f;
  delete zView2;
  delete bView2;
  delete r;
  delete zView3;
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

  NDArray *wT = w->transpose();  // [3*inSize x inSize] -> [inSize x 3*inSize]

  const int time = x->sizeAt(2);

  // the previous cell state starts as a copy of c0: the loop overwrites it every step
  NDArray* ct_1 = c0->dup();

  // loop through time steps
  for (int t = 0; t < time; ++t) {
    NDArray *xt = (*x)({0, 0, 0, 0, t, t + 1});
    NDArray *ht = (*h)({0, 0, 0, 0, t, t + 1});
    NDArray *ct = (*c)({0, 0, 0, 0, t, t + 1});

    helpers::sruCell(context, xt, ct_1, wT, b, ht, ct);
    ct_1->assign(ct);

    delete xt;
    delete ht;
    delete ct;
  }

  delete ct_1;
  delete wT;
}

//////////////////////////////////////////////////////////////////////////
// x * mask, the mask [bS x 2*K] broadcast over time, in an array of its own: the ops must leave their input as it is
// (the mask was multiplied into x itself, so the caller's array changed and a second call applied the mask again).
// nullptr without a mask. The mask is small: the ops use a dense copy of it, which is what the broadcast reads.
static NDArray* maskedInput(NDArray* x, NDArray* mask) {
  if (mask == nullptr) return nullptr;

  std::vector<sd::LongType> shape = {x->sizeAt(0), x->sizeAt(1), x->sizeAt(2)};
  NDArray* masked = new NDArray('c', shape, x->dataType(), x->getContext());
  std::vector<sd::LongType> dims = {1, 2};
  x->applyBroadcast(broadcast::Multiply, &dims, mask, masked);
  return masked;
}

//////////////////////////////////////////////////////////////////////////
// sruBI_ and sruBIBP_ run a thread for each (batch, feature) column of x [time x bS x 2*K], the features from K on
// running backwards in time. Every array is addressed through its own strides, so any layout (F order, a view) is read
// and written as it is: U = x * w comes out in whatever order the matrix product gives it.
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

  using AccT = typename simdOps::AggregateType<T>::type;

  const sd::LongType time = x->sizeAt(0);
  const sd::LongType bS = x->sizeAt(1);
  const sd::LongType d2 = x->sizeAt(2);  // 2*K
  const sd::LongType K = d2 / 2;
  const sd::LongType ncols = bS * d2;

  //  xm = x * mask
  NDArray* denseMask = mask != nullptr ? mask->dup('c') : nullptr;
  mask = denseMask;
  NDArray* xMasked = maskedInput(x, mask);
  NDArray* xm = xMasked != nullptr ? xMasked : x;

  // U = xm * w, [time x bS x 6*K]: the pre-activations of feature k at time t are U[t, b, 3*k] (the candidate),
  // U[t, b, 3*k + 1] (forget gate) and U[t, b, 3*k + 2] (reset gate)
  std::vector<sd::LongType> wiShape = {time, bS, 6 * K};
  NDArray* wi = new NDArray('c', wiShape, x->dataType(), x->getContext());
  MmulHelper::matmul(xm, w, wi, false, false, 1.0, 0.0);

  const T* pX = xm->bufferAsT<T>();
  const T* pWi = wi->bufferAsT<T>();
  const T* pBias = b->bufferAsT<T>();
  const T* pInit = c0->bufferAsT<T>();
  const T* pMask = mask != nullptr ? mask->bufferAsT<T>() : nullptr;
  T* pHt = ht->bufferAsT<T>();
  T* pCt = ct->bufferAsT<T>();

  const sd::LongType* xStride = xm->stridesOf();
  const sd::LongType* wiStride = wi->stridesOf();
  const sd::LongType biasStride = b->stridesOf()[0];
  const sd::LongType* initStride = c0->stridesOf();
  const sd::LongType* maskStride = mask != nullptr ? mask->stridesOf() : nullptr;
  const sd::LongType* htStride = ht->stridesOf();
  const sd::LongType* ctStride = ct->stridesOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto col = start; col < stop; col++) {
      const sd::LongType batch = col / d2;
      const sd::LongType k = col % d2;
      const bool flip = k >= K;  // the second half of the features runs backwards in time

      const AccT maskVal =
          pMask != nullptr ? static_cast<AccT>(pMask[batch * maskStride[0] + k * maskStride[1]]) : static_cast<AccT>(1);
      AccT cur = static_cast<AccT>(pInit[batch * initStride[0] + k * initStride[1]]);
      const AccT bF = static_cast<AccT>(pBias[k * biasStride]);
      const AccT bR = static_cast<AccT>(pBias[(k + d2) * biasStride]);

      // the first time step of the column, and the step from one time step to the next
      const sd::LongType first = flip ? time - 1 : 0;
      const sd::LongType dir = flip ? -1 : 1;
      sd::LongType xOffset = first * xStride[0] + batch * xStride[1] + k * xStride[2];
      sd::LongType wiOffset = first * wiStride[0] + batch * wiStride[1] + 3 * k * wiStride[2];
      sd::LongType htOffset = first * htStride[0] + batch * htStride[1] + k * htStride[2];
      sd::LongType ctOffset = first * ctStride[0] + batch * ctStride[1] + k * ctStride[2];
      const sd::LongType xStep = dir * xStride[0];
      const sd::LongType wiStep = dir * wiStride[0];
      const sd::LongType htStep = dir * htStride[0];
      const sd::LongType ctStep = dir * ctStride[0];

      for (sd::LongType t = 0; t < time; ++t) {
        const AccT u0 = static_cast<AccT>(pWi[wiOffset]);
        const AccT u1 = static_cast<AccT>(pWi[wiOffset + wiStride[2]]);
        const AccT u2 = static_cast<AccT>(pWi[wiOffset + 2 * wiStride[2]]);
        const AccT xVal = static_cast<AccT>(pX[xOffset]);

        // evaluate sigmoids
        const AccT ft = static_cast<AccT>(1) / (static_cast<AccT>(1) + sd::math::sd_exp<AccT, AccT>(-(u1 + bF)));
        const AccT rt = static_cast<AccT>(1) / (static_cast<AccT>(1) + sd::math::sd_exp<AccT, AccT>(-(u2 + bR)));

        cur = (cur - u0) * ft + u0;
        pCt[ctOffset] = static_cast<T>(cur);
        const AccT val = sd::math::sd_tanh<AccT, AccT>(cur);
        pHt[htOffset] = static_cast<T>((val * maskVal - xVal) * rt + xVal);

        xOffset += xStep;
        wiOffset += wiStep;
        htOffset += htStep;
        ctOffset += ctStep;
      }
    }
  };

  samediff::Threads::parallel_for(func, 0, ncols);

  delete wi;
  delete xMasked;
  delete denseMask;
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
  // inGradC0 [bS x 2*K], the gradient with respect to the state a column ends in
  // inGradHt  [time x bS x 2*K]
  // mask optional,  2d tensor of dropout mask [bS x 2*K]

  // gradI  [time x bS x 2*K]
  // gradW  [time x 2*K x 6*K], the gradient of the weights at each time step: their sum is the gradient of w
  // gradB  [4*K]
  // gradC0 [bS x 2*K]

  using AccT = typename simdOps::AggregateType<T>::type;

  const sd::LongType time = x->sizeAt(0);  // time - number of time steps
  const sd::LongType bS = x->sizeAt(1);
  const sd::LongType d2 = x->sizeAt(2);  // 2*K
  const sd::LongType K = d2 / 2;
  const sd::LongType ncols = bS * d2;

  //  xm = x * mask
  NDArray* denseMask = mask != nullptr ? mask->dup('c') : nullptr;
  mask = denseMask;
  NDArray* xMasked = maskedInput(x, mask);
  NDArray* xm = xMasked != nullptr ? xMasked : x;

  // U = xm * w
  std::vector<sd::LongType> wiShape = {time, bS, 6 * K};
  NDArray* wi = new NDArray('c', wiShape, x->dataType(), x->getContext());
  MmulHelper::matmul(xm, w, wi, false, false, 1.0, 0.0);  // [time x bS x 2*K] * [2*K x 6*K]
  std::vector<sd::LongType> biasShape = {bS, 4 * K};
  std::vector<sd::LongType> wShape = {time, bS, 6 * K};
  // the gradients of the gates' biases of each batch row (the first 2*K are the forget gates', the rest the reset
  // gates') and of U
  NDArray gradBias('c', biasShape, x->dataType(), x->getContext());
  NDArray gradWi('c', wShape, x->dataType(), x->getContext());

  const T* pX = xm->bufferAsT<T>();
  const T* pWi = wi->bufferAsT<T>();
  const T* pBias = b->bufferAsT<T>();
  const T* pInit = c0->bufferAsT<T>();
  const T* pMask = mask != nullptr ? mask->bufferAsT<T>() : nullptr;
  const T* pState = ct->bufferAsT<T>();
  const T* pInGradCt = inGradC0->bufferAsT<T>();
  const T* pInGradHt = inGradHt->bufferAsT<T>();
  T* pGradWi = gradWi.bufferAsT<T>();
  T* pGradInput = gradI->bufferAsT<T>();
  T* pGradBias = gradBias.bufferAsT<T>();
  T* pGradInit = gradC0->bufferAsT<T>();

  const sd::LongType* xStride = xm->stridesOf();
  const sd::LongType* wiStride = wi->stridesOf();
  const sd::LongType biasStride = b->stridesOf()[0];
  const sd::LongType* initStride = c0->stridesOf();
  const sd::LongType* maskStride = mask != nullptr ? mask->stridesOf() : nullptr;
  const sd::LongType* stateStride = ct->stridesOf();
  const sd::LongType* inGradCtStride = inGradC0->stridesOf();
  const sd::LongType* inGradHtStride = inGradHt->stridesOf();
  const sd::LongType* gradWiStride = gradWi.stridesOf();
  const sd::LongType* gradInputStride = gradI->stridesOf();
  const sd::LongType* gradBiasStride = gradBias.stridesOf();
  const sd::LongType* gradInitStride = gradC0->stridesOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto col = start; col < stop; col++) {
      const sd::LongType batch = col / d2;
      const sd::LongType k = col % d2;
      const bool flip = k >= K;  // the second half of the features runs backwards in time

      AccT gbF = static_cast<AccT>(0);
      AccT gbR = static_cast<AccT>(0);
      const AccT maskVal =
          pMask != nullptr ? static_cast<AccT>(pMask[batch * maskStride[0] + k * maskStride[1]]) : static_cast<AccT>(1);
      AccT cur = static_cast<AccT>(pInGradCt[batch * inGradCtStride[0] + k * inGradCtStride[1]]);
      const AccT bF = static_cast<AccT>(pBias[k * biasStride]);
      const AccT bR = static_cast<AccT>(pBias[(k + d2) * biasStride]);

      // the sweep goes back through the time steps of the forward pass: from the last one it handled, in the direction
      // opposite to the one it went
      const sd::LongType first = flip ? 0 : time - 1;
      const sd::LongType dir = flip ? 1 : -1;
      sd::LongType xOffset = first * xStride[0] + batch * xStride[1] + k * xStride[2];
      sd::LongType wiOffset = first * wiStride[0] + batch * wiStride[1] + 3 * k * wiStride[2];
      sd::LongType stateOffset = first * stateStride[0] + batch * stateStride[1] + k * stateStride[2];
      sd::LongType inGradHtOffset = first * inGradHtStride[0] + batch * inGradHtStride[1] + k * inGradHtStride[2];
      sd::LongType gradInputOffset = first * gradInputStride[0] + batch * gradInputStride[1] + k * gradInputStride[2];
      sd::LongType gradWiOffset = first * gradWiStride[0] + batch * gradWiStride[1] + 3 * k * gradWiStride[2];
      const sd::LongType xStep = dir * xStride[0];
      const sd::LongType wiStep = dir * wiStride[0];
      const sd::LongType stateStep = dir * stateStride[0];
      const sd::LongType inGradHtStep = dir * inGradHtStride[0];
      const sd::LongType gradInputStep = dir * gradInputStride[0];
      const sd::LongType gradWiStep = dir * gradWiStride[0];

      for (sd::LongType t = 0; t < time; ++t) {
        const AccT u0 = static_cast<AccT>(pWi[wiOffset]);
        const AccT u1 = static_cast<AccT>(pWi[wiOffset + wiStride[2]]);
        const AccT u2 = static_cast<AccT>(pWi[wiOffset + 2 * wiStride[2]]);
        const AccT xVal = static_cast<AccT>(pX[xOffset]);
        const AccT gradHt = static_cast<AccT>(pInGradHt[inGradHtOffset]);

        // evaluate sigmoids
        const AccT ft = static_cast<AccT>(1) / (static_cast<AccT>(1) + sd::math::sd_exp<AccT, AccT>(-(u1 + bF)));
        const AccT rt = static_cast<AccT>(1) / (static_cast<AccT>(1) + sd::math::sd_exp<AccT, AccT>(-(u2 + bR)));

        const AccT val = sd::math::sd_tanh<AccT, AccT>(static_cast<AccT>(pState[stateOffset]));
        // the state before this time step: the one the sweep reaches next, c0 after the last
        const AccT prevVal = (t < time - 1) ? static_cast<AccT>(pState[stateOffset + stateStep])
                                            : static_cast<AccT>(pInit[batch * initStride[0] + k * initStride[1]]);
        // grad wrt input: the highway connection (the gradient through U is added to it below)
        pGradInput[gradInputOffset] = static_cast<T>(gradHt - gradHt * rt);
        // grad wrt rt, wiR and bR
        const AccT grt = gradHt * (val * maskVal - xVal) * (rt - rt * rt);
        pGradWi[gradWiOffset + 2 * gradWiStride[2]] = static_cast<T>(grt);
        gbR += grt;
        // grad wrt state
        const AccT gradStateVal = gradHt * maskVal * (rt - rt * val * val) + cur;
        // grad wrt wi0
        pGradWi[gradWiOffset] = static_cast<T>(gradStateVal - gradStateVal * ft);
        // grad wrt ft, wi1, and bF
        const AccT gft = gradStateVal * (prevVal - u0) * (ft - ft * ft);
        pGradWi[gradWiOffset + gradWiStride[2]] = static_cast<T>(gft);
        gbF += gft;
        // grad wrt c_previous
        cur = gradStateVal * ft;

        xOffset += xStep;
        wiOffset += wiStep;
        stateOffset += stateStep;
        inGradHtOffset += inGradHtStep;
        gradInputOffset += gradInputStep;
        gradWiOffset += gradWiStep;
      }
      pGradBias[batch * gradBiasStride[0] + k * gradBiasStride[1]] = static_cast<T>(gbF);
      pGradBias[batch * gradBiasStride[0] + (k + d2) * gradBiasStride[1]] = static_cast<T>(gbR);
      pGradInit[batch * gradInitStride[0] + k * gradInitStride[1]] = static_cast<T>(cur);
    }
  };

  samediff::Threads::parallel_for(func, 0, ncols);

  // the input also reaches the cells through U = xm * w: gradI += gradWi * w^T, and the mask scales what flows back to x
  NDArray* wT = w->transpose();  // [2*K x 6*K] -> [6*K x 2*K]
  std::vector<sd::LongType> viaUShape = {time, bS, 2 * K};
  NDArray* viaU = new NDArray('c', viaUShape, x->dataType(), x->getContext());
  MmulHelper::matmul(&gradWi, wT, viaU, false, false, 1.0, 0.0);  // [time x bS x 6*K] * [6*K x 2*K]
  gradI->applyPairwiseTransform(pairwise::Add, viaU, gradI);
  if (mask != nullptr) {
    std::vector<sd::LongType> dims = {1, 2};
    gradI->applyBroadcast(broadcast::Multiply, &dims, mask, gradI);
  }

  // gradB
  std::vector<sd::LongType> sumDims = {0};
  gradBias.reduceAlongDimension(reduce::Sum, gradB, &sumDims);  // [bS x 4*K] -> [4*K]

  // gradW, a view of xm as [time x 2*K x bS] leaves xm as it is
  std::vector<sd::LongType> permutation = {0, 2, 1};
  NDArray* xT = xm->permute(permutation, false, false);         // [time x bS x 2*K] -> [time x 2*K x bS]
  MmulHelper::mmul(xT, &gradWi, gradW, 1., 0.);  // [time x 2*K x bS ] * [time x bS x 6*K] = [time x 2*K x 6*K]

  delete xT;
  delete viaU;
  delete wT;
  delete wi;
  delete xMasked;
  delete denseMask;
}

void sruBI(sd::LaunchContext* context, NDArray* x, NDArray* w, NDArray* b, NDArray* c0,
           NDArray* mask, NDArray* ht, NDArray* ct) {
  NDArray::preparePrimaryUse({ht, ct}, {x, w, b, c0, mask});
  BUILD_SINGLE_SELECTOR(x->dataType(), sruBI_, (x, w, b, c0, mask, ht, ct), SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({ht, ct}, {x, w, b, c0, mask});
}
void sruBIBP(sd::LaunchContext* context, NDArray* x, NDArray* w, NDArray* b, NDArray* c0,
             NDArray* ct, NDArray* inGradC0, NDArray* inGradH, NDArray* mask, NDArray* gradI,
             NDArray* gradW, NDArray* gradB, NDArray* gradC0) {
  NDArray::preparePrimaryUse({gradI, gradW, gradB, gradC0}, {x, w, b, c0, ct, inGradC0, inGradH, mask});
  BUILD_SINGLE_SELECTOR(x->dataType(), sruBIBP_,
                        (x, w, b, c0, ct, inGradC0, inGradH, mask, gradI, gradW, gradB, gradC0), SD_FLOAT_TYPES);
  NDArray::registerPrimaryUse({gradI, gradW, gradB, gradC0}, {x, w, b, c0, ct, inGradC0, inGradH, mask});
}
BUILD_SINGLE_TEMPLATE( void sruBI_,
                       (NDArray * x, NDArray* w, NDArray* b, NDArray* c0, NDArray* mask,
                           NDArray* ht, NDArray* ct),
                       SD_FLOAT_TYPES);
BUILD_SINGLE_TEMPLATE( void sruBIBP_,
                       (NDArray * x, NDArray* w, NDArray* b, NDArray* c0, NDArray* ct,
                           NDArray* inGradC0, NDArray* inGradH, NDArray* mask, NDArray* gradI,
                           NDArray* gradW, NDArray* gradB, NDArray* gradC0),
                       SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

//////////////////////////////////////////////////////////////////////////

#endif
