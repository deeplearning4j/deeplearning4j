/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "vednnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

PLATFORM_IMPL(matmul, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

  uint64_t bGemm = 1;
  for (int i = 0; i < x->rankOf() - 2; i++) {
    bGemm = bGemm * x->sizeAt(i);
  }
  const uint64_t outDim = z->sizeAt(-1);
  const uint64_t nBatch = z->sizeAt(-2);
  const uint64_t inDim = x->sizeAt(-1);
#if !defined(HAVE_VEDA)
  if (bGemm == 1) {
    vednnLinearForward(inDim, outDim, nBatch, 1, x->buffer(), y->buffer(), z->buffer());
  } else {
    // because of the bgemm did not work as expected, we will manually parallelize over bGemm
    int xStride = x->rankOf() > 2 ? x->sizeAt(-1) * x->sizeAt(-2) : 0;
    int yStride = y->rankOf() > 2 ? y->sizeAt(-1) * y->sizeAt(-2) : 0;
    int zStride = z->rankOf() > 2 ? z->sizeAt(-1) * z->sizeAt(-2) : 0;

#pragma omp parallel for
    for (int i = 0; i < bGemm; i++) {
      float *xPtr = x->bufferAsT<float>() + i * xStride;
      float *yPtr = y->bufferAsT<float>() + i * yStride;
      float *zPtr = z->bufferAsT<float>() + i * zStride;
      vednnLinearForward(inDim, outDim, nBatch, 1, xPtr, yPtr, zPtr);
    }
  }
#else

  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
  auto func = handle.getFunctionByConstPtrName("vedaVednnLinearForwardExF32");

  VEDAdeviceptr vX, vY, vZ;
  const uint64_t xStride = x->rankOf() > 2 ? x->sizeAt(-1) * x->sizeAt(-2) : 0;
  const uint64_t yStride = y->rankOf() > 2 ? y->sizeAt(-1) * y->sizeAt(-2) : 0;
  const uint64_t zStride = z->rankOf() > 2 ? z->sizeAt(-1) * z->sizeAt(-2) : 0;

  vX = (VEDAdeviceptr)x->specialBuffer();
  vY = (VEDAdeviceptr)y->specialBuffer();
  vZ = (VEDAdeviceptr)z->specialBuffer();

  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, bGemm, inDim, outDim, nBatch, vX, xStride, vY, yStride, vZ, zStride));

#endif
  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(matmul, ENGINE_CPU) {
  auto input0 = INPUT_VARIABLE(0);
  auto input1 = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  auto alpha = block.numT() > 0 ? T_ARG(0) : 1.0;
  auto beta = block.numT() > 1 ? T_ARG(1) : 0.0;
  int transX = block.numI() > 0 ? INT_ARG(0) : 0;
  int transY = block.numI() > 1 ? INT_ARG(1) : 0;
  const int transZ = block.numI() > 2 ? INT_ARG(2) : 0;

  Requirements req("VEDNN MATMUL OP");
  // input related constraints
  req.expectEq(makeInfoVariable(alpha, "alpha"), 1.0) && req.expectEq(makeInfoVariable(beta, "beta"), 0.0) &&
      req.expectEq(makeInfoVariable(transX, "transX"), 0) && req.expectEq(makeInfoVariable(transY, "transY"), 0) &&
      req.expectEq(makeInfoVariable(transZ, "transZ"), 0) &&
      req.expectEq(makeInfoVariable(input0->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input0->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(input0->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectEq(makeInfoVariable(input1->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input1->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(input1->ews(), EWS_MSG_INPUT1), 1) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1);
  // matrix checks
  req.expectGreater(makeInfoVariable(input0->rankOf(), RANK_MSG_INPUT0), 0) &&
      req.expectGreater(makeInfoVariable(input1->rankOf(), RANK_MSG_INPUT1), 0) &&
      req.expectGreater(makeInfoVariable(output->rankOf(), RANK_MSG_OUTPUT), 0) &&
      req.expectTrue(makeInfoVariable(
          [input0, input1, output] {
            int i0Rank = input0->rankOf();
            int i1Rank = input1->rankOf();
            int outRank = output->rankOf();
            int maxRank = i0Rank > i1Rank ? i0Rank : i1Rank;
            maxRank = outRank > maxRank ? outRank : maxRank;

            for (int j = -maxRank; j <= -3; j++) {
              int bGemm0 = i0Rank >= -j ? input0->sizeAt(j) : 1;
              int bGemm1 = i1Rank >= -j ? input1->sizeAt(j) : 1;
              // if(bGemm0 != bGemm1){
              //     //if one of the ranks is below 3 we will allow it
              //     if(i0Rank <=2 ) bGemm0 = bGemm1;
              //     else if(i1Rank > 2 ) return false;
              // }
              int bGemmOut = outRank >= -j ? output->sizeAt(j) : 1;
              if (bGemm0 != bGemm1 || bGemmOut != bGemm0) {
                return false;
              }
            }
            return true;
          },
          "batch gemm constraints check")) &&
      req.expectTrue(makeInfoVariable(
          [input0, input1, output] {
            int inDimA = input0->sizeAt(-1);
            int nBatchB = input0->rankOf() >= 2 ? input0->sizeAt(-2) : 1;
            int inDimB = input1->rankOf() >= 2 ? input1->sizeAt(-2) : 1;
            int outDimB = input1->sizeAt(-1);
            int outDimC = output->sizeAt(-1);
            int nBatchC = output->rankOf() >= 2 ? output->sizeAt(-2) : 1;
            return nBatchB == nBatchC && inDimA == inDimB && outDimB == outDimC;
          },
          "matrix multiplication constraints check"));

  req.logTheSuccess();

  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
