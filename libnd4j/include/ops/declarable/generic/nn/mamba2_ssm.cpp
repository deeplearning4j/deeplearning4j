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

//
// @author Adam Gibson
//
// mamba2_ssm - Mamba-2 State Space Model (SSD) recurrence
//
// Implements the Mamba-2 SSD recurrence with head-structured state:
//   A_bar = exp(A[h] * dt[b,t,h])            (per-head scalar decay)
//   state[b,h,n,p] = A_bar * state + B[b,t,n] * dt * x[b,t,h*P+p]
//   y[b,t,h*P+p] = sum_n(C[b,t,n] * state[b,h,n,p]) + D[h] * x[b,t,h*P+p]
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_mamba2_ssm)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/mamba2_ssm.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(mamba2_ssm, 5, 2, false, 0, 3) {
  auto x  = INPUT_VARIABLE(0);   // [B, L, D]  where D = H * P
  auto A  = INPUT_VARIABLE(1);   // [H]
  auto B  = INPUT_VARIABLE(2);   // [B, L, N]
  auto C  = INPUT_VARIABLE(3);   // [B, L, N]
  auto dt = INPUT_VARIABLE(4);   // [B, L, H]

  auto y        = OUTPUT_VARIABLE(0);  // [B, L, D]
  auto stateOut = OUTPUT_VARIABLE(1);  // [B, H, N, P]

  const int H = INT_ARG(0);  // num_heads
  const int P = INT_ARG(1);  // head_dim
  const int N = INT_ARG(2);  // state_dim

  // Optional inputs
  NDArray* D_skip  = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
  NDArray* stateIn = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;

  // Validate shapes
  REQUIRE_TRUE(x->rankOf() == 3, 0,
               "mamba2_ssm: x must be rank 3 [B, L, D], got rank %lld", x->rankOf());
  REQUIRE_TRUE(A->rankOf() == 1, 0,
               "mamba2_ssm: A must be rank 1 [H], got rank %lld", A->rankOf());
  REQUIRE_TRUE(B->rankOf() == 3, 0,
               "mamba2_ssm: B must be rank 3 [B, L, N], got rank %lld", B->rankOf());
  REQUIRE_TRUE(C->rankOf() == 3, 0,
               "mamba2_ssm: C must be rank 3 [B, L, N], got rank %lld", C->rankOf());
  REQUIRE_TRUE(dt->rankOf() == 3, 0,
               "mamba2_ssm: dt must be rank 3 [B, L, H], got rank %lld", dt->rankOf());

  REQUIRE_TRUE(A->sizeAt(0) == H, 0,
               "mamba2_ssm: A[0] must equal num_heads=%d, got %lld", H, A->sizeAt(0));
  REQUIRE_TRUE(x->sizeAt(2) == H * P, 0,
               "mamba2_ssm: x[2] must equal H*P=%d, got %lld", H * P, x->sizeAt(2));
  REQUIRE_TRUE(B->sizeAt(2) == N, 0,
               "mamba2_ssm: B[2] must equal state_dim=%d, got %lld", N, B->sizeAt(2));
  REQUIRE_TRUE(C->sizeAt(2) == N, 0,
               "mamba2_ssm: C[2] must equal state_dim=%d, got %lld", N, C->sizeAt(2));
  REQUIRE_TRUE(dt->sizeAt(2) == H, 0,
               "mamba2_ssm: dt[2] must equal num_heads=%d, got %lld", H, dt->sizeAt(2));

  if (D_skip != nullptr) {
    REQUIRE_TRUE(D_skip->rankOf() == 1, 0,
                 "mamba2_ssm: D must be rank 1 [H], got rank %lld", D_skip->rankOf());
    REQUIRE_TRUE(D_skip->sizeAt(0) == H, 0,
                 "mamba2_ssm: D[0] must equal num_heads=%d, got %lld", H, D_skip->sizeAt(0));
  }

  if (stateIn != nullptr) {
    REQUIRE_TRUE(stateIn->rankOf() == 4, 0,
                 "mamba2_ssm: state_in must be rank 4 [B, H, N, P], got rank %lld", stateIn->rankOf());
    REQUIRE_TRUE(stateIn->sizeAt(1) == H && stateIn->sizeAt(2) == N && stateIn->sizeAt(3) == P, 0,
                 "mamba2_ssm: state_in shape mismatch, expected [B,%d,%d,%d]", H, N, P);
  }

  helpers::mamba2Ssm(block.launchContext(), x, A, B, C, dt, D_skip, stateIn,
                     y, stateOut, H, P, N);

  return sd::Status::OK;
}

DECLARE_TYPES(mamba2_ssm) {
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS})
      ->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(mamba2_ssm) {
  auto xShape = inputShape->at(0);  // [B, L, D]

  const int H = INT_ARG(0);
  const int P = INT_ARG(1);
  const int N = INT_ARG(2);

  auto Batch = shape::sizeAt(xShape, 0);
  auto L     = shape::sizeAt(xShape, 1);
  auto D     = shape::sizeAt(xShape, 2);

  // Output 0: y [B, L, D]
  auto yShape = ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(xShape), 'c', {Batch, L, D});

  // Output 1: state_out [B, H, N, P]
  auto stateShape = ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(xShape), 'c',
      {Batch, static_cast<LongType>(H), static_cast<LongType>(N), static_cast<LongType>(P)});

  return SHAPELIST(yShape, stateShape);
}

}  // namespace ops
}  // namespace sd

#endif
