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
#include <helpers/DebugHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/sru.h>
#include <ops/op_types.h>

#include <vector>

#include "execution/cuda/LaunchDims.h"


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
void sruCell(LaunchContext* context, NDArray* x, NDArray* c0, NDArray* w, NDArray* b,
             NDArray* h, NDArray* c) {
  // x   input [bS x inSize], bS - batch size, inSize - number of features
  // c0  previous cell state c  [bS x inSize], that is at previous time step t-1
  // w   weights [inSize x 3*inSize]
  // b   biases [2*inSize]

  // h   current cell output [bS x inSize], that is at current time step t
  // c   current cell state  [bS x inSize], that is at current time step t

  const int inSize = x->sizeAt(1);  // inSize - number of features

  auto z = mmul(*x, *w);  //  [bS x 3*inSize]

  // Get subarray slices - operator() returns NDArray*
  NDArray* zSlice1 = (*z)({0, 0, inSize, 2 * inSize});
  NDArray* bSlice1 = (*b)({0, inSize});

  // forget gate = sigmoid(x*Wf + bf)
  NDArray* fInPtr = *zSlice1 + *bSlice1;
  auto f = sigmoid(*fInPtr);

  NDArray* zSlice2 = (*z)({0, 0, 2 * inSize, 3 * inSize});
  NDArray* bSlice2 = (*b)({inSize, 2 * inSize});

  NDArray* rInPtr = *zSlice2 + *bSlice2;
  // reset gate = sigmoid(x*Wr + br)
  auto r = sigmoid(*rInPtr);

  // ◦ means element-wise product or so called Hadamard product
  // current sell state = f◦c0 + (1 - f)◦(x*Wc)
  NDArray* zSlice3 = (*z)({0, 0, 0, inSize});
  NDArray* oneMinusFPtr = 1.f - (*f);
  NDArray* fMulC0 = (*f) * (*c0);
  NDArray* oneMinusFMulZ = (*oneMinusFPtr) * (*zSlice3);
  NDArray* cAssignPtr = *fMulC0 + *oneMinusFMulZ;
  c->assign(cAssignPtr);
  // *c = f*(*c0 - z({},{0, inSize})) + z({{},{0, inSize}});

  // current cell output = r◦activation(c) + (1 - r)◦x
  NDArray* oneMinusRPtr = 1.f - (*r);
  NDArray activationC = activation(*c);
  NDArray* rMulAct = (*r) * activationC;
  NDArray* oneMinusRMulX = (*oneMinusRPtr) * (*x);
  NDArray* resultPtr = *rMulAct + *oneMinusRMulX;
  h->assign(resultPtr);
  // *h = r * (activation<T>(c) - *x) + *x;

  delete zSlice1;
  delete bSlice1;
  delete zSlice2;
  delete bSlice2;
  delete zSlice3;
  delete fInPtr;
  delete rInPtr;
  delete oneMinusFPtr;
  delete fMulC0;
  delete oneMinusFMulZ;
  delete cAssignPtr;
  delete oneMinusRPtr;
  delete rMulAct;
  delete oneMinusRMulX;
  delete resultPtr;
  delete z;
  delete f;
  delete r;
}

//////////////////////////////////////////////////////////////////////////
void sruTimeLoop(LaunchContext* context, NDArray* x, NDArray* c0, NDArray* w, NDArray* b,
                 NDArray* h, NDArray* c) {
  // x   input [bS x inSize x time]
  // c0  initial cell state  (at time step = 0) [bS x inSize],
  // w   weights, [3*inSize x inSize]
  // b   biases,  [2*inSize]

  // h   cell outputs [bS x inSize x time]
  // c   cell states  [bS x inSize x time]

  auto wT = w->transpose();  // [3*inSize x inSize] -> [inSize x 3*inSize]

  const int time = x->sizeAt(2);

  // the previous cell state starts as a copy of c0: the loop overwrites it every step
  NDArray* ct_1 = c0->dup();

  // loop through time steps
  for (int t = 0; t < time; ++t) {
    auto xt = (*x)({0, 0, 0, 0, t, t + 1});
    auto ht = (*h)({0, 0, 0, 0, t, t + 1});
    auto ct = (*c)({0, 0, 0, 0, t, t + 1});

    sruCell(context, xt, ct_1, wT, b, ht, ct);
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

  std::vector<LongType> shape = {x->sizeAt(0), x->sizeAt(1), x->sizeAt(2)};
  NDArray* masked = new NDArray('c', shape, x->dataType(), x->getContext());
  std::vector<LongType> dims = {1, 2};
  x->applyBroadcast(broadcast::Multiply, &dims, mask, masked);
  return masked;
}

//////////////////////////////////////////////////////////////////////////
// sruBICuda and sruBIBPCuda run a thread for each (batch, feature) column of x [time, bS, 2*K], the features from K on
// running backwards in time. Every array is addressed through its own strides, so any layout (F order, a view) is read
// and written as it is: U = x * w comes out in whatever order the matrix product gives it. The columns are walked with
// a grid stride, so any launch covers them all.
template <typename T>
SD_KERNEL static void sruBICuda(const void* vx, const LongType* xShapeInfo, const void* vwi,
                                 const LongType* wiShapeInfo, const void* vb, const LongType* bShapeInfo,
                                 const void* vc0, const LongType* c0ShapeInfo, const void* vmask,
                                 const LongType* maskShapeInfo, void* vht, const LongType* htShapeInfo,
                                 void* vct, const LongType* ctShapeInfo) {
  // Inputs:
  // x     [time, bS, 2*K]
  // wi    [time, bS, 6*K], wi = mmul(x, weights): the pre-activations of feature k are wi[., ., 3*k + 0, 1, 2]
  // b     [4*K]
  // c0    [bS, 2*K]
  // mask  [bS, 2*K], optional

  // Outputs:
  // ht  [time, bS, 2*K]
  // ct  [time, bS, 2*K]

  using AccT = typename simdOps::AggregateType<T>::type;

  const T* x = reinterpret_cast<const T*>(vx);
  const T* wi = reinterpret_cast<const T*>(vwi);
  const T* b = reinterpret_cast<const T*>(vb);
  const T* c0 = reinterpret_cast<const T*>(vc0);
  const T* mask = reinterpret_cast<const T*>(vmask);
  T* ht = reinterpret_cast<T*>(vht);
  T* ct = reinterpret_cast<T*>(vct);

  const LongType* xShape = shape::shapeOf(xShapeInfo);
  const LongType* xStride = shape::stride(xShapeInfo);
  const LongType* wiStride = shape::stride(wiShapeInfo);
  const LongType biasStride = shape::stride(bShapeInfo)[0];
  const LongType* c0Stride = shape::stride(c0ShapeInfo);
  const LongType* maskStride = vmask != nullptr ? shape::stride(maskShapeInfo) : nullptr;
  const LongType* htStride = shape::stride(htShapeInfo);
  const LongType* ctStride = shape::stride(ctShapeInfo);

  const LongType time = xShape[0];
  const LongType bS = xShape[1];
  const LongType d2 = xShape[2];  // 2*K
  const LongType K = d2 / 2;
  const LongType ncols = bS * d2;

  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType col = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; col < ncols; col += step) {
    const LongType batch = col / d2;
    const LongType k = col % d2;
    const bool flip = k >= K;  // the second half of the features runs backwards in time

    const AccT maskVal =
        vmask != nullptr ? static_cast<AccT>(mask[batch * maskStride[0] + k * maskStride[1]]) : static_cast<AccT>(1);
    AccT cur = static_cast<AccT>(c0[batch * c0Stride[0] + k * c0Stride[1]]);
    const AccT bF = static_cast<AccT>(b[k * biasStride]);
    const AccT bR = static_cast<AccT>(b[(k + d2) * biasStride]);

    // the first time step of the column, and the step from one time step to the next
    const LongType first = flip ? time - 1 : 0;
    const LongType dir = flip ? -1 : 1;
    LongType xOffset = first * xStride[0] + batch * xStride[1] + k * xStride[2];
    LongType wiOffset = first * wiStride[0] + batch * wiStride[1] + 3 * k * wiStride[2];
    LongType htOffset = first * htStride[0] + batch * htStride[1] + k * htStride[2];
    LongType ctOffset = first * ctStride[0] + batch * ctStride[1] + k * ctStride[2];
    const LongType xStep = dir * xStride[0];
    const LongType wiStep = dir * wiStride[0];
    const LongType htStep = dir * htStride[0];
    const LongType ctStep = dir * ctStride[0];

    for (LongType t = 0; t < time; ++t) {
      const AccT u0 = static_cast<AccT>(wi[wiOffset]);
      const AccT u1 = static_cast<AccT>(wi[wiOffset + wiStride[2]]);
      const AccT u2 = static_cast<AccT>(wi[wiOffset + 2 * wiStride[2]]);
      const AccT xVal = static_cast<AccT>(x[xOffset]);

      // evaluate sigmoids
      const AccT ft = static_cast<AccT>(1) / (static_cast<AccT>(1) + math::sd_exp<AccT, AccT>(-(u1 + bF)));
      const AccT rt = static_cast<AccT>(1) / (static_cast<AccT>(1) + math::sd_exp<AccT, AccT>(-(u2 + bR)));

      cur = (cur - u0) * ft + u0;
      ct[ctOffset] = static_cast<T>(cur);
      const AccT val = math::sd_tanh<AccT, AccT>(cur);
      ht[htOffset] = static_cast<T>((val * maskVal - xVal) * rt + xVal);

      xOffset += xStep;
      wiOffset += wiStep;
      htOffset += htStep;
      ctOffset += ctStep;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void sruBICudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                              const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo,
                              const void* vwi,
                              const LongType* wiShapeInfo, const void* vb, const LongType* bShapeInfo, const void* vc0,
                              const LongType* c0ShapeInfo,
                              const void* vmask, const LongType* maskShapeInfo, void* vht,
                              const LongType* htShapeInfo, void* vct, const LongType* ctShapeInfo) {
  sruBICuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vwi, wiShapeInfo, vb, bShapeInfo,
                                                                       vc0, c0ShapeInfo, vmask, maskShapeInfo, vht,
                                                                       htShapeInfo, vct, ctShapeInfo);
  if (!DebugHelper::inGraphCapture(const_cast<cudaStream_t*>(stream))) DebugHelper::checkGlobalErrorCode("sruBICuda failed");
}

//////////////////////////////////////////////////////////////////////////
void sruBI(LaunchContext* context, NDArray* x, NDArray* w, NDArray* b, NDArray* c0,
           NDArray* mask, NDArray* ht, NDArray* ct) {
  //  xm = x * mask
  NDArray* denseMask = mask != nullptr ? mask->dup('c') : nullptr;
  mask = denseMask;
  NDArray* xMasked = maskedInput(x, mask);
  NDArray* xm = xMasked != nullptr ? xMasked : x;

  // U = xm * w
  std::vector<LongType> wiShape = {x->sizeAt(0), x->sizeAt(1), 3 * x->sizeAt(2)};
  NDArray* wi = new NDArray('c', wiShape, x->dataType(), context);
  MmulHelper::matmul(xm, w, wi, false, false, 1.0, 0.0);  // U [time x bS x 6*K]

  PointersManager manager(context, "sru_bi");

  // a thread for each (batch, feature) of the last two dimensions of x: bS * 2K
  dim3 sruBiDims2 = sruBiDims(x->sizeAt(1) * x->sizeAt(2),x->rankOf());
  NDArray::prepareSpecialUse({ht, ct}, {xm, wi, b, c0, mask});
  BUILD_SINGLE_SELECTOR(
      x->dataType(), sruBICudaLauncher,
      (sruBiDims2.y,sruBiDims2.x, 0, context->getCudaStream(), xm->specialBuffer(), xm->specialShapeInfo(),
          wi->specialBuffer(), wi->specialShapeInfo(), b->specialBuffer(), b->specialShapeInfo(), c0->specialBuffer(),
          c0->specialShapeInfo(), mask ? mask->specialBuffer() : nullptr, mask ? mask->specialShapeInfo() : nullptr,
          ht->specialBuffer(), ht->specialShapeInfo(), ct->specialBuffer(), ct->specialShapeInfo()),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({ht, ct}, {xm, wi, b, c0, mask});

  // the temporary arrays are released once the kernel is done with them
  manager.synchronize();

  delete wi;
  delete xMasked;
  delete denseMask;
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL static void sruBIBPCuda(const void* vx, const LongType* xShapeInfo, const void* vwi,
                                   const LongType* wiShapeInfo, const void* vb, const LongType* bShapeInfo,
                                   const void* vc0, const LongType* c0ShapeInfo, const void* vmask,
                                   const LongType* maskShapeInfo, const void* vct, const LongType* ctShapeInfo,
                                   const void* vgradHt, const LongType* gradHtShapeInfo, const void* vgradCt,
                                   const LongType* gradCtShapeInfo, void* vgradI, const LongType* gradIShapeInfo,
                                   void* vgradWi, const LongType* gradWiShapeInfo, void* vgradB,
                                   const LongType* gradBShapeInfo, void* vgradC0, const LongType* gradC0ShapeInfo) {
  // Inputs:
  // x      [time, bS, 2*K]
  // wi     [time, bS, 6*K], wi = mmul(x, weights);
  // b      [4*K]
  // c0     [bS, 2*K]
  // mask   [bS, 2*K], optional
  // ct     [time, bS, 2*K]
  // gradHt [time, bS, 2*K]
  // gradCt [bS, 2*K]

  // Outputs:
  // gradI   [time, bS, 2*K], the highway part of the gradient of the input
  // gradWi  [time, bS, 6*K]
  // gradB   [bS, 4*K], the first 2*K of each row are the forget gates' biases' gradients, the rest the reset gates'
  // gradC0  [bS, 2*K]

  using AccT = typename simdOps::AggregateType<T>::type;

  const T* x = reinterpret_cast<const T*>(vx);
  const T* wi = reinterpret_cast<const T*>(vwi);
  const T* b = reinterpret_cast<const T*>(vb);
  const T* c0 = reinterpret_cast<const T*>(vc0);
  const T* mask = reinterpret_cast<const T*>(vmask);
  const T* ct = reinterpret_cast<const T*>(vct);
  const T* gradHt = reinterpret_cast<const T*>(vgradHt);
  const T* gradCt = reinterpret_cast<const T*>(vgradCt);

  T* gradI = reinterpret_cast<T*>(vgradI);
  T* gradWi = reinterpret_cast<T*>(vgradWi);
  T* gradB = reinterpret_cast<T*>(vgradB);
  T* gradC0 = reinterpret_cast<T*>(vgradC0);

  const LongType* xShape = shape::shapeOf(xShapeInfo);
  const LongType* xStride = shape::stride(xShapeInfo);
  const LongType* wiStride = shape::stride(wiShapeInfo);
  const LongType biasStride = shape::stride(bShapeInfo)[0];
  const LongType* c0Stride = shape::stride(c0ShapeInfo);
  const LongType* maskStride = vmask != nullptr ? shape::stride(maskShapeInfo) : nullptr;
  const LongType* ctStride = shape::stride(ctShapeInfo);
  const LongType* gradHtStride = shape::stride(gradHtShapeInfo);
  const LongType* gradCtStride = shape::stride(gradCtShapeInfo);
  const LongType* gradIStride = shape::stride(gradIShapeInfo);
  const LongType* gradWiStride = shape::stride(gradWiShapeInfo);
  const LongType* gradBStride = shape::stride(gradBShapeInfo);
  const LongType* gradC0Stride = shape::stride(gradC0ShapeInfo);

  const LongType time = xShape[0];
  const LongType bS = xShape[1];
  const LongType d2 = xShape[2];  // 2*K
  const LongType K = d2 / 2;
  const LongType ncols = bS * d2;

  const LongType step = static_cast<LongType>(gridDim.x) * blockDim.x;
  for (LongType col = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x; col < ncols; col += step) {
    const LongType batch = col / d2;
    const LongType k = col % d2;
    const bool flip = k >= K;  // the second half of the features runs backwards in time

    AccT gbF = static_cast<AccT>(0);
    AccT gbR = static_cast<AccT>(0);
    const AccT maskVal =
        vmask != nullptr ? static_cast<AccT>(mask[batch * maskStride[0] + k * maskStride[1]]) : static_cast<AccT>(1);
    AccT cur = static_cast<AccT>(gradCt[batch * gradCtStride[0] + k * gradCtStride[1]]);
    const AccT bF = static_cast<AccT>(b[k * biasStride]);
    const AccT bR = static_cast<AccT>(b[(k + d2) * biasStride]);

    // the sweep goes back through the time steps of the forward pass: from the last one it handled, in the direction
    // opposite to the one it went
    const LongType first = flip ? 0 : time - 1;
    const LongType dir = flip ? 1 : -1;
    LongType xOffset = first * xStride[0] + batch * xStride[1] + k * xStride[2];
    LongType wiOffset = first * wiStride[0] + batch * wiStride[1] + 3 * k * wiStride[2];
    LongType ctOffset = first * ctStride[0] + batch * ctStride[1] + k * ctStride[2];
    LongType gradHtOffset = first * gradHtStride[0] + batch * gradHtStride[1] + k * gradHtStride[2];
    LongType gradIOffset = first * gradIStride[0] + batch * gradIStride[1] + k * gradIStride[2];
    LongType gradWiOffset = first * gradWiStride[0] + batch * gradWiStride[1] + 3 * k * gradWiStride[2];
    const LongType xStep = dir * xStride[0];
    const LongType wiStep = dir * wiStride[0];
    const LongType ctStep = dir * ctStride[0];
    const LongType gradHtStep = dir * gradHtStride[0];
    const LongType gradIStep = dir * gradIStride[0];
    const LongType gradWiStep = dir * gradWiStride[0];

    for (LongType t = 0; t < time; ++t) {
      const AccT u0 = static_cast<AccT>(wi[wiOffset]);
      const AccT u1 = static_cast<AccT>(wi[wiOffset + wiStride[2]]);
      const AccT u2 = static_cast<AccT>(wi[wiOffset + 2 * wiStride[2]]);
      const AccT xVal = static_cast<AccT>(x[xOffset]);
      const AccT gradHtVal = static_cast<AccT>(gradHt[gradHtOffset]);

      // evaluate sigmoids
      const AccT ft = static_cast<AccT>(1) / (static_cast<AccT>(1) + math::sd_exp<AccT, AccT>(-(u1 + bF)));
      const AccT rt = static_cast<AccT>(1) / (static_cast<AccT>(1) + math::sd_exp<AccT, AccT>(-(u2 + bR)));

      const AccT val = math::sd_tanh<AccT, AccT>(static_cast<AccT>(ct[ctOffset]));
      // the state before this time step: the one the sweep reaches next, c0 after the last
      const AccT prevVal = (t < time - 1) ? static_cast<AccT>(ct[ctOffset + ctStep])
                                          : static_cast<AccT>(c0[batch * c0Stride[0] + k * c0Stride[1]]);

      // grad wrt input: the highway connection (the gradient through U is added to it by the host)
      gradI[gradIOffset] = static_cast<T>(gradHtVal - gradHtVal * rt);

      // grad wrt rt, wiR and bR
      const AccT grt = gradHtVal * (val * maskVal - xVal) * (rt - rt * rt);
      gradWi[gradWiOffset + 2 * gradWiStride[2]] = static_cast<T>(grt);
      gbR += grt;

      // grad wrt state
      const AccT gradStateVal = gradHtVal * maskVal * (rt - rt * val * val) + cur;

      // grad wrt wi0
      gradWi[gradWiOffset] = static_cast<T>(gradStateVal - gradStateVal * ft);

      // grad wrt ft, wi1, and bF
      const AccT gft = gradStateVal * (prevVal - u0) * (ft - ft * ft);
      gradWi[gradWiOffset + gradWiStride[2]] = static_cast<T>(gft);
      gbF += gft;

      // grad wrt c_previous
      cur = gradStateVal * ft;

      xOffset += xStep;
      wiOffset += wiStep;
      ctOffset += ctStep;
      gradHtOffset += gradHtStep;
      gradIOffset += gradIStep;
      gradWiOffset += gradWiStep;
    }

    // write the accumulated gradients
    gradB[batch * gradBStride[0] + k * gradBStride[1]] = static_cast<T>(gbF);
    gradB[batch * gradBStride[0] + (k + d2) * gradBStride[1]] = static_cast<T>(gbR);
    gradC0[batch * gradC0Stride[0] + k * gradC0Stride[1]] = static_cast<T>(cur);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void sruBIBPCudaLauncher(
    const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t* stream, const void* vx, const LongType* xShapeInfo, const void* vwi,
    const LongType* wiShapeInfo, const void* vb, const LongType* bShapeInfo, const void* vc0, const LongType* c0ShapeInfo, const void* vmask,
    const LongType* maskShapeInfo, const void* vct, const LongType* ctShapeInfo, const void* vgradHt, const LongType* gradHtShapeInfo, const void* vgradCt,
    const LongType* gradCtShapeInfo, void* vgradI, const LongType* gradIShapeInfo, void* vgradWi, const LongType* gradWiShapeInfo, void* vgradB,
    const LongType* gradBShapeInfo, void* vgradC0, const LongType* gradC0ShapeInfo) {
  sruBIBPCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
      vx, xShapeInfo, vwi, wiShapeInfo, vb, bShapeInfo, vc0, c0ShapeInfo, vmask, maskShapeInfo, vct, ctShapeInfo,
      vgradHt, gradHtShapeInfo, vgradCt, gradCtShapeInfo, vgradI, gradIShapeInfo, vgradWi, gradWiShapeInfo, vgradB,
      gradBShapeInfo, vgradC0, gradC0ShapeInfo);
  if (!DebugHelper::inGraphCapture(const_cast<cudaStream_t*>(stream))) DebugHelper::checkGlobalErrorCode("sruBIBPCuda failed");
}
BUILD_SINGLE_TEMPLATE( void sruBIBPCudaLauncher,
                      (const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                          const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, const void* vwi,
                          const sd::LongType* wiShapeInfo, const void* vb, const sd::LongType* bShapeInfo, const void* vc0,
                          const sd::LongType* c0ShapeInfo, const void* vmask, const sd::LongType* maskShapeInfo,
                          const void* vct, const sd::LongType* ctShapeInfo, const void* vgradHt,
                          const sd::LongType* gradHtShapeInfo, const void* vgradCt, const sd::LongType* gradCtShapeInfo,
                          void* vgradI, const sd::LongType* gradIShapeInfo, void* vgradWi,
                          const sd::LongType* gradWiShapeInfo, void* vgradB, const sd::LongType* gradBShapeInfo,
                          void* vgradC0, const sd::LongType* gradC0ShapeInfo),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
void sruBIBP(LaunchContext* context, NDArray* x, NDArray* w, NDArray* b, NDArray* c0,
             NDArray* ct, NDArray* gradCt, NDArray* gradHt, NDArray* mask, NDArray* gradI,
             NDArray* gradW, NDArray* gradB, NDArray* gradC0) {
  const LongType time = x->sizeAt(0);
  const LongType bS = x->sizeAt(1);
  const LongType K = x->sizeAt(2) / 2;

  //  xm = x * mask
  NDArray* denseMask = mask != nullptr ? mask->dup('c') : nullptr;
  mask = denseMask;
  NDArray* xMasked = maskedInput(x, mask);
  NDArray* xm = xMasked != nullptr ? xMasked : x;

  // U = xm * w
  std::vector<LongType> wiShape = {time, bS, 6 * K};
  NDArray* wi = new NDArray('c', wiShape, x->dataType(), context);
  MmulHelper::matmul(xm, w, wi, false, false, 1.0, 0.0);  // U [time x bS x 6*K]

  // the gradients of the gates' biases of each batch row (the first 2*K are the forget gates', the rest the reset
  // gates') and of U
  std::vector<LongType> gradBiasShape = {bS, 4 * K};
  std::vector<LongType> gradWiShape = {time, bS, 6 * K};
  NDArray gradBias('c', gradBiasShape, x->dataType(), context);
  NDArray gradWi('c', gradWiShape, x->dataType(), context);

  PointersManager manager(context, "sru_bi_bp");

  // a thread for each (batch, feature) of the last two dimensions of x: bS * 2K
  dim3 sruBiBpDims = sruBiDims(x->sizeAt(1) * x->sizeAt(2), x->rankOf());
  NDArray::prepareSpecialUse({gradI, &gradWi, &gradBias, gradC0}, {xm, wi, b, c0, ct, gradCt, gradHt, mask});
  BUILD_SINGLE_SELECTOR(
      x->dataType(), sruBIBPCudaLauncher,
      (sruBiBpDims.y, sruBiBpDims.x, 0, context->getCudaStream(), xm->specialBuffer(), xm->specialShapeInfo(),
          wi->specialBuffer(), wi->specialShapeInfo(), b->specialBuffer(), b->specialShapeInfo(), c0->specialBuffer(),
          c0->specialShapeInfo(), mask ? mask->specialBuffer() : nullptr, mask ? mask->specialShapeInfo() : nullptr,
          ct->specialBuffer(), ct->specialShapeInfo(), gradHt->specialBuffer(), gradHt->specialShapeInfo(),
          gradCt->specialBuffer(), gradCt->specialShapeInfo(), gradI->specialBuffer(), gradI->specialShapeInfo(),
          gradWi.specialBuffer(), gradWi.specialShapeInfo(), gradBias.specialBuffer(), gradBias.specialShapeInfo(),
          gradC0->specialBuffer(), gradC0->specialShapeInfo()),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({gradI, &gradWi, &gradBias, gradC0}, {xm, wi, b, c0, ct, gradCt, gradHt, mask});

  // the input also reaches the cells through U = xm * w: gradI += gradWi * w^T, and the mask scales what flows back to x
  NDArray* wT = w->transpose();  // [2*K x 6*K] -> [6*K x 2*K]
  std::vector<LongType> viaUShape = {time, bS, 2 * K};
  NDArray* viaU = new NDArray('c', viaUShape, x->dataType(), context);
  MmulHelper::matmul(&gradWi, wT, viaU, false, false, 1.0, 0.0);  // [time x bS x 6*K] * [6*K x 2*K]
  gradI->applyPairwiseTransform(pairwise::Add, viaU, gradI);
  if (mask != nullptr) {
    std::vector<LongType> dims = {1, 2};
    gradI->applyBroadcast(broadcast::Multiply, &dims, mask, gradI);
  }

  // gradB
  std::vector<LongType> sumDims = {0};
  gradBias.reduceAlongDimension(reduce::Sum, gradB, &sumDims);  // [bS x 4*K] -> [4*K]

  // gradW, a view of xm as [time x 2*K x bS] leaves xm as it is
  std::vector<LongType> permutation = {0, 2, 1};
  NDArray* xT = xm->permute(permutation, false, false);  // [time, bS, 2*K] -> [time, 2*K,  bS]
  MmulHelper::mmul(xT, &gradWi, gradW, 1., 0.);  // [time, 2*K, bS] x [time, bS , 6*K] = [time, 2*K, 6*K]

  // the temporary arrays are released once everything that reads them is done
  manager.synchronize();

  delete xT;
  delete viaU;
  delete wT;
  delete wi;
  delete xMasked;
  delete denseMask;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
