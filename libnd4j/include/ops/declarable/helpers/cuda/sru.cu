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
#include <helpers/MmulHelper.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/sru.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
static SD_INLINE NDArray activation(NDArray& arr) {
  auto result = NDArray(&arr, false, arr.getContext());
  arr.applyTransform(transform::Tanh,&result);
  return result;
}

//////////////////////////////////////////////////////////////////////////
static SD_INLINE NDArray sigmoid(NDArray& arr) {
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

  // forget gate = sigmoid(x*Wf + bf)
  NDArray fIn = z({0, 0, inSize, 2 * inSize}) + (*b)({0, inSize});
  auto f = sigmoid(fIn);

  NDArray rIn = z({0, 0, 2 * inSize, 3 * inSize}) + (*b)({inSize, 2 * inSize});
  // reset gate = sigmoid(x*Wr + br)
  auto r = sigmoid(rIn);

  // ◦ means element-wise product or so called Hadamard product
  // current sell state = f◦c0 + (1 - f)◦(x*Wc)
  c->assign(f * (*c0) + (1.f - f) * z({0, 0, 0, inSize}));
  // *c = f*(*c0 - z({},{0, inSize})) + z({{},{0, inSize}});

  // current cell output = r◦activation(c) + (1 - r)◦x
  h->assign(r * activation(*c) + (1.f - r) * (*x));
  // *h = r * (activation<T>(c) - *x) + *x;
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

  NDArray ct_1(*c0);

  // loop through time steps
  for (int t = 0; t < time; ++t) {
    auto xt = (*x)({0, 0, 0, 0, t, t + 1});
    auto ht = (*h)({0, 0, 0, 0, t, t + 1});
    auto ct = (*c)({0, 0, 0, 0, t, t + 1});

    sruCell(context, &xt, &ct_1, &wT, b, &ht, &ct);
    ct_1.assign(ct);
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void sruBICuda(const void* vx, const LongType* xShapeInfo, const void* vwi,
                                 const LongType* wiShapeInfo, const void* vb, const LongType* bShapeInfo,
                                 const void* vc0, const LongType* c0ShapeInfo, const void* vmask,
                                 const LongType* maskShapeInfo, void* vht, const LongType* htShapeInfo,
                                 void* vct, const LongType* ctShapeInfo) {
  // Inputs:
  // x     [time, bS, 2*K]
  // wi    [time, bS, 6*K], wi = mmul(x, weights);
  // b     [4*K]
  // c0    [bS, 2*K]
  // mask  [bS, 2*K], optional

  // Outputs:
  // ht  [time, bS, 2*K]
  // ct  [time, bS, 2*K]

  // Reinterpret inputs and outputs
  const T* x = reinterpret_cast<const T*>(vx);
  const T* wi = reinterpret_cast<const T*>(vwi);
  const T* b = reinterpret_cast<const T*>(vb);
  const T* c0 = reinterpret_cast<const T*>(vc0);
  const T* mask = reinterpret_cast<const T*>(vmask);
  T* ht = reinterpret_cast<T*>(vht);
  T* ct = reinterpret_cast<T*>(vct);

  const int rank = 3; // Assuming 3D tensors

  // Shared memory for caching shape information and other variables
  extern __shared__ unsigned char shmem[];
  // Pointers to shared memory segments
  LongType* sharedMem = reinterpret_cast<LongType*>(shmem);

  // Shared variables
  __shared__ LongType shared_time;
  __shared__ LongType shared_bS;
  __shared__ LongType shared_K;
  __shared__ LongType shared_len;
  __shared__ LongType shared_totalThreads;

  // Cached shape and stride pointers
  __shared__ const LongType* shared_xShape;
  __shared__ const LongType* shared_wiShape;
  __shared__ const LongType* shared_bShape;
  __shared__ const LongType* shared_c0Shape;
  __shared__ const LongType* shared_maskShape;
  __shared__ const LongType* shared_htShape;
  __shared__ const LongType* shared_ctShape;

  if (threadIdx.x == 0) {
    // Cache shape pointers
    shared_xShape = shape::shapeOf(xShapeInfo);
    shared_wiShape = shape::shapeOf(wiShapeInfo);
    shared_bShape = shape::shapeOf(bShapeInfo);
    shared_c0Shape = shape::shapeOf(c0ShapeInfo);
    shared_maskShape = shape::shapeOf(maskShapeInfo);
    shared_htShape = shape::shapeOf(htShapeInfo);
    shared_ctShape = shape::shapeOf(ctShapeInfo);

    // Cache time, bS, and K
    shared_time = shared_xShape[0];  // time
    shared_bS = shared_xShape[1];    // batch size (bS)
    shared_K = shared_xShape[2] / 2; // Assuming xShapeInfo[2] = 2*K

    // Calculate len = 2*K * bS
    shared_len = 2 * shared_K * shared_bS;

    // Calculate total number of threads across all blocks
    shared_totalThreads = gridDim.x * blockDim.x;
  }

  // Ensure all threads have access to the cached values
  __syncthreads();

  // Calculate the global thread ID
  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate space in shared memory for coordinates
  LongType* coords = sharedMem + threadIdx.x * (rank - 1); // Only last two dimensions {bS, 2*K}

  if (tid >= shared_len) return;

  // Convert linear index to multi-dimensional coordinates {bS, 2*K}
  INDEX2COORDS(tid, rank - 1, shared_xShape, coords); // coords[0] = bS, coords[1] = 2*K

  // Calculate necessary offsets
  LongType maskOffset = 0, c0Offset = 0, bFOffset = 0, bROffset = 0;

  if (vmask != nullptr) {
    COORDS2INDEX(rank - 1, shape::stride(maskShapeInfo), coords, maskOffset);
  }
  COORDS2INDEX(rank - 1, shape::stride(c0ShapeInfo), coords, c0Offset);
  COORDS2INDEX(rank - 1, shape::stride(bShapeInfo), coords + 1, bFOffset);
  bROffset = bFOffset + 2 * shared_K * shared_bShape[2]; // 2*K*b_stride

  // Fetch values
  const T maskVal = (vmask != nullptr) ? mask[maskOffset] : static_cast<T>(1);
  const T bF = b[bFOffset];
  const T bR = b[bROffset];
  T c0Val = c0[c0Offset];

  // Determine flip condition
  const bool flip = coords[1] >= shared_K;

  // Initialize coordinates for time iteration
  if (flip)
    coords[0] = shared_time - 1;
  else
    coords[0] = 0;

  // Calculate offsets for x, ht, ct
  LongType xOffset = 0, htOffset = 0, ctOffset = 0;
  COORDS2INDEX(rank, shape::stride(xShapeInfo), coords, xOffset);
  COORDS2INDEX(rank, shape::stride(htShapeInfo), coords, htOffset);
  COORDS2INDEX(rank, shape::stride(ctShapeInfo), coords, ctOffset);

  // Adjust coords for wi and gradWi
  coords[1] *= 3; // 6*K corresponds to 3 * 2*K

  // Calculate wi offsets
  LongType wiOffset0 = 0, wiOffset1 = 0, wiOffset2 = 0;
  COORDS2INDEX(rank, shape::stride(wiShapeInfo), coords, wiOffset0);
  wiOffset1 = wiOffset0 + shared_wiShape[rank]; // Add stride for wi1
  wiOffset2 = wiOffset1 + shared_wiShape[rank]; // Add stride for wi2

  // Iterate over the time steps
  for (LongType t = 0; t < shared_time; ++t) {
    // Evaluate sigmoids
    T ft = static_cast<T>(1) / (static_cast<T>(1) + math::sd_exp<T, T>(- (wi[wiOffset1] + bF)));
    T rt = static_cast<T>(1) / (static_cast<T>(1) + math::sd_exp<T, T>(- (wi[wiOffset2] + bR)));

    // Update c0Val and ct
    c0Val = (c0Val - wi[wiOffset0]) * ft + wi[wiOffset0];
    ct[ctOffset] = c0Val;

    // Compute tanh activation
    T val = math::sd_tanh<T, T>(c0Val);

    // Fetch x value
    T xVal = x[xOffset];

    // Compute ht
    ht[htOffset] = (val * maskVal - xVal) * rt + xVal;

    // Update offsets based on flip condition
    if (flip) {
      xOffset -= shape::stride(xShapeInfo)[0];        // time step stride
      htOffset -= shape::stride(htShapeInfo)[0];
      ctOffset -= shape::stride(ctShapeInfo)[0];
      wiOffset0 -= shape::stride(wiShapeInfo)[0];
      wiOffset1 -= shape::stride(wiShapeInfo)[0];
      wiOffset2 -= shape::stride(wiShapeInfo)[0];
    } else {
      xOffset += shape::stride(xShapeInfo)[0];        // time step stride
      htOffset += shape::stride(htShapeInfo)[0];
      ctOffset += shape::stride(ctShapeInfo)[0];
      wiOffset0 += shape::stride(wiShapeInfo)[0];
      wiOffset1 += shape::stride(wiShapeInfo)[0];
      wiOffset2 += shape::stride(wiShapeInfo)[0];
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
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "sruBICuda failed");

}

//////////////////////////////////////////////////////////////////////////
void sruBI(LaunchContext* context, NDArray* x, NDArray* w, NDArray* b, NDArray* c0,
           NDArray* mask, NDArray* ht, NDArray* ct) {
  //  x = x * mask
  std::vector<LongType> dims = {1,2};
  if (mask) x->applyBroadcast(broadcast::Multiply, &dims, mask, x);  // apply mask

  // U = x * w
  NDArray wi = mmul(*x, *w);  //  U [time x bS x 6*K]

  PointersManager manager(context, "sru_bi");

  dim3 sruBiDims2 = sruBiDims(x->sizeAt(1) * x->sizeAt(2),x->rankOf());
  NDArray::prepareSpecialUse({ht, ct}, {x, &wi, b, c0, mask});
  BUILD_SINGLE_SELECTOR(
      x->dataType(), sruBICudaLauncher,
      (sruBiDims2.y,sruBiDims2.x, sruBiDims2.z, context->getCudaStream(), x->specialBuffer(), x->specialShapeInfo(),
          wi.specialBuffer(), wi.specialShapeInfo(), b->specialBuffer(), b->specialShapeInfo(), c0->specialBuffer(),
          c0->specialShapeInfo(), mask ? mask->specialBuffer() : nullptr, mask ? mask->specialShapeInfo() : nullptr,
          ht->specialBuffer(), ht->specialShapeInfo(), ct->specialBuffer(), ct->specialShapeInfo()),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({ht, ct}, {x, &wi, b, c0, mask});

  manager.synchronize();
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
__global__ static void sruBIBPCuda(const void* vx, const LongType* xShapeInfo, const void* vwi,
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
  // gradI   [time, bS, 2*K]
  // gradWi  [time, 2*K, 6*K]
  // gradB   [bS, 4*K]
  // gradC0  [bS, 2*K]

  // Reinterpret inputs and outputs
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

  const int rank = 3; // Assuming 3D tensors

  // Shared memory for caching shape information
  extern __shared__ unsigned char shmem[];
  LongType* sharedMem = reinterpret_cast<LongType*>(shmem);

  __shared__ LongType shared_time;
  __shared__ LongType shared_K;
  __shared__ LongType shared_len;
  __shared__ LongType shared_totalThreads;

  // Cached shape pointers
  __shared__ const LongType* shared_xShape;
  __shared__ const LongType* shared_wiShape;
  __shared__ const LongType* shared_bShape;
  __shared__ const LongType* shared_c0Shape;
  __shared__ const LongType* shared_maskShape;
  __shared__ const LongType* shared_ctShape;
  __shared__ const LongType* shared_gradHtShape;
  __shared__ const LongType* shared_gradCtShape;
  __shared__ const LongType* shared_gradIShape;
  __shared__ const LongType* shared_gradWiShape;
  __shared__ const LongType* shared_gradBShape;
  __shared__ const LongType* shared_gradC0Shape;

  if (threadIdx.x == 0) {
    // Cache ranks, shapes, and strides
    shared_xShape = shape::shapeOf(xShapeInfo);
    shared_wiShape = shape::shapeOf(wiShapeInfo);
    shared_bShape = shape::shapeOf(bShapeInfo);
    shared_c0Shape = shape::shapeOf(c0ShapeInfo);
    shared_maskShape = shape::shapeOf(maskShapeInfo);
    shared_ctShape = shape::shapeOf(ctShapeInfo);
    shared_gradHtShape = shape::shapeOf(gradHtShapeInfo);
    shared_gradCtShape = shape::shapeOf(gradCtShapeInfo);
    shared_gradIShape = shape::shapeOf(gradIShapeInfo);
    shared_gradWiShape = shape::shapeOf(gradWiShapeInfo);
    shared_gradBShape = shape::shapeOf(gradBShapeInfo);
    shared_gradC0Shape = shape::shapeOf(gradC0ShapeInfo);

    // Cache time and K
    shared_time = shared_xShape[0];
    shared_K = shared_xShape[2] / 2; // Assuming xShapeInfo[2] = 2*K

    // Calculate len = 2*K * bS
    LongType bS = shared_xShape[1];
    shared_len = 2 * shared_K * bS;

    // Total threads across all blocks
    shared_totalThreads = gridDim.x * blockDim.x;
  }

  // Ensure all threads have access to the cached values
  __syncthreads();

  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Allocate space in shared memory for coordinates
  LongType* coords = sharedMem + threadIdx.x * rank;

  if (tid >= shared_len) return;

  // Convert linear index to coordinates {bS, 2*K}
  INDEX2COORDS(tid, rank - 1, shared_xShape, coords + 1); // Skipping the time dimension

  // Calculate necessary offsets
  LongType maskOffset = 0, c0Offset = 0, gradCtOffset = 0, gradC0Offset = 0;
  LongType bFOffset = 0, bROffset = 0, gradBFOffset = 0, gradBROffset = 0;

  if (vmask != nullptr) {
    COORDS2INDEX(rank - 1, shape::stride(maskShapeInfo), coords + 1, maskOffset);
  }
  COORDS2INDEX(rank - 1, shape::stride(c0ShapeInfo), coords + 1, c0Offset);
  COORDS2INDEX(rank - 1, shape::stride(gradCtShapeInfo), coords + 1, gradCtOffset);
  COORDS2INDEX(rank - 1, shape::stride(gradC0ShapeInfo), coords + 1, gradC0Offset);
  COORDS2INDEX(rank - 1, shape::stride(bShapeInfo), coords + 2, bFOffset);
  bROffset = bFOffset + 2 * shared_K * shared_bShape[2]; // 2*K*b_stride
  gradBFOffset = coords[1] * shared_gradBShape[3] / 2 + coords[2] * shared_gradBShape[4];
  gradBROffset = gradBFOffset + shared_gradBShape[3];

  const bool flip = coords[2] >= shared_K;

  if (flip)
    coords[0] = 0;
  else
    coords[0] = shared_time - 1;

  // Calculate offsets for x, ct, gradI, gradHt
  LongType xOffset = 0, ctOffset = 0, gradIOffset = 0, gradHtOffset = 0;
  COORDS2INDEX(rank, shape::stride(xShapeInfo), coords, xOffset);
  COORDS2INDEX(rank, shape::stride(ctShapeInfo), coords, ctOffset);
  COORDS2INDEX(rank, shape::stride(gradIShapeInfo), coords, gradIOffset);
  COORDS2INDEX(rank, shape::stride(gradHtShapeInfo), coords, gradHtOffset);

  // Adjust coords for wi and gradWi
  coords[2] *= 3;
  LongType gradWiOffset0 = 0, gradWiOffset1 = 0, gradWiOffset2 = 0;
  LongType wiOffset0 = 0, wiOffset1 = 0, wiOffset2 = 0;

  COORDS2INDEX(rank, shape::stride(gradWiShapeInfo), coords, gradWiOffset0);
  gradWiOffset1 = gradWiOffset0 + shared_gradWiShape[rank + 3]; // add last stride
  gradWiOffset2 = gradWiOffset1 + shared_gradWiShape[rank + 3]; // add last stride

  COORDS2INDEX(rank, shape::stride(wiShapeInfo), coords, wiOffset0);
  wiOffset1 = wiOffset0 + shared_wiShape[rank + 3]; // add last stride
  wiOffset2 = wiOffset1 + shared_wiShape[rank + 3]; // add last stride

  // Fetch values
  const T xVal = x[xOffset];
  const T maskVal = (vmask != nullptr) ? mask[maskOffset] : static_cast<T>(1);
  const T c0Val = c0[c0Offset];
  const T bF = b[bFOffset];
  const T bR = b[bROffset];
  T gradCtVal = gradCt[gradCtOffset];
  T gbF = static_cast<T>(0);
  T gbR = static_cast<T>(0);

  // Iterate over the time steps
  for (LongType t = 0; t < shared_time; ++t) {
    // Evaluate sigmoids
    T ft = static_cast<T>(1) / (static_cast<T>(1) + math::sd_exp<T, T>(- (wi[wiOffset1] + bF)));
    T rt = static_cast<T>(1) / (static_cast<T>(1) + math::sd_exp<T, T>(- (wi[wiOffset2] + bR)));

    T val = math::sd_tanh<T, T>(ct[ctOffset]);

    T prevVal;
    if (t < shared_time - 1)
      prevVal = ct[ctOffset += (flip ? shared_ctShape[rank + 1] : -shared_ctShape[rank + 1])];
    else
      prevVal = c0Val;

    // Gradient with respect to input
    gradI[gradIOffset] = gradHt[gradHtOffset] - gradHt[gradHtOffset] * rt;

    // Gradient with respect to rt, wiR, and bR
    T grt = gradHt[gradHtOffset] * (val * maskVal - x[xOffset]) * (rt - rt * rt);
    gradWi[gradWiOffset2] = grt;
    gbR += grt;

    // Gradient with respect to state
    T gradC0Val = gradHt[gradHtOffset] * maskVal * (rt - rt * val * val) + gradCtVal;

    // Gradient with respect to wi0
    gradWi[gradWiOffset0] = gradC0Val - gradC0Val * ft;

    // Gradient with respect to ft, wi1, and bF
    T gft = gradC0Val * (prevVal - wi[wiOffset0]) * (ft - ft * ft);
    gradWi[gradWiOffset1] = gft;
    gbF += gft;

    // Gradient with respect to c_previous
    gradCtVal = gradC0Val * ft;

    // Update offsets based on flip
    if (flip) {
      xOffset += shared_xShape[rank + 1]; // first stride, corresponds to time step
      gradHtOffset += shared_gradHtShape[rank + 1];
      gradIOffset += shared_gradIShape[rank + 1];
      wiOffset0 += shared_wiShape[rank + 1];
      wiOffset1 += shared_wiShape[rank + 1];
      wiOffset2 += shared_wiShape[rank + 1];
      gradWiOffset0 += shared_gradWiShape[rank + 1];
      gradWiOffset1 += shared_gradWiShape[rank + 1];
      gradWiOffset2 += shared_gradWiShape[rank + 1];
    }
    else {
      xOffset -= shared_xShape[rank + 1]; // first stride, corresponds to time step
      gradHtOffset -= shared_gradHtShape[rank + 1];
      gradIOffset -= shared_gradIShape[rank + 1];
      wiOffset0 -= shared_wiShape[rank + 1];
      wiOffset1 -= shared_wiShape[rank + 1];
      wiOffset2 -= shared_wiShape[rank + 1];
      gradWiOffset0 -= shared_gradWiShape[rank + 1];
      gradWiOffset1 -= shared_gradWiShape[rank + 1];
      gradWiOffset2 -= shared_gradWiShape[rank + 1];
    }
  }

  // Write accumulated gradients to output
  gradB[gradBFOffset] = gbF;
  gradB[gradBROffset] = gbR;
  gradC0[gradC0Offset] = gradCtVal;
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
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t *>(stream), "sruBIBPCuda failed");

}
BUILD_SINGLE_TEMPLATE(template void sruBIBPCudaLauncher,
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
  //  x = x * mask
  std::vector<LongType> dims = {1, 2};
  if (mask) x->applyBroadcast(broadcast::Multiply, &dims, mask, x);  // apply mask

  // U = x * w
  NDArray wi = mmul(*x, *w);  //  U [time x bS x 6*K]

  const int time = x->sizeAt(0);
  const int bS = x->sizeAt(1);
  const int K = x->sizeAt(2) / 2;

  std::vector<sd::LongType> gradBiasShape = {bS, 4 * K};
  std::vector<sd::LongType> gradWiShape = {time, bS, 6 * K};
  NDArray gradBias(x->ordering(), gradBiasShape, x->dataType(), context);
  NDArray gradWi(x->ordering(), gradWiShape, x->dataType(), context);

  PointersManager manager(context, "sru_bi_bp");

  const int threadsPerBlock = SD_MAX_NUM_THREADS / 4;
  const int blocksPerGrid = (x->sizeAt(1) * x->sizeAt(2) + threadsPerBlock - 1) /
                            threadsPerBlock;  // loop through last two dimensions of x array -> bS, 2*K
  const int sharedMem = threadsPerBlock * sizeof(LongType) * x->rankOf() + 128;
  dim3 sruBiBpDims = sruBiDims(x->sizeAt(1) + x->sizeAt(2),x->rankOf());
  NDArray::prepareSpecialUse({gradI, &gradWi, &gradBias, gradC0}, {x, &wi, b, c0, ct, gradCt, gradHt, mask});
  BUILD_SINGLE_SELECTOR(
      x->dataType(), sruBIBPCudaLauncher,
      (sruBiBpDims.y, sruBiBpDims.x,sruBiBpDims.z, context->getCudaStream(), x->specialBuffer(), x->specialShapeInfo(),
          wi.specialBuffer(), wi.specialShapeInfo(), b->specialBuffer(), b->specialShapeInfo(), c0->specialBuffer(),
          c0->specialShapeInfo(), mask ? mask->specialBuffer() : nullptr, mask ? mask->specialShapeInfo() : nullptr,
          ct->specialBuffer(), ct->specialShapeInfo(), gradHt->specialBuffer(), gradHt->specialShapeInfo(),
          gradCt->specialBuffer(), gradCt->specialShapeInfo(), gradI->specialBuffer(), gradI->specialShapeInfo(),
          gradWi.specialBuffer(), gradWi.specialShapeInfo(), gradBias.specialBuffer(), gradBias.specialShapeInfo(),
          gradC0->specialBuffer(), gradC0->specialShapeInfo()),
      SD_FLOAT_TYPES);
  NDArray::registerSpecialUse({gradI, &gradWi, &gradBias, gradC0}, {x, &wi, b, c0, ct, gradCt, gradHt, mask});

  manager.synchronize();


  std::vector<LongType> dims2 = {0};
  // gradB
  gradBias.reduceAlongDimension(reduce::Sum, gradB, &dims2);  // [4*K]

  // gradW
  x->permutei({0, 2, 1}, false, false);                       // [time, bS, 2*K] -> [time, 2*K,  bS]
  MmulHelper::mmul(x, &gradWi, gradW, 1., 0.);  // [time, 2*K, bS] x [time, bS , 6*K] = [time, 2*K, 6*K]
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
