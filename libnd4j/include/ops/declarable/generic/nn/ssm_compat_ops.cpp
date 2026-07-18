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
// @author Eclipse Deeplearning4j
//
// llama.cpp-compat Mamba/SSM ops. ssm_conv delegates to causal_conv1d;
// ssm_scan performs ZOH discretization (A_bar = exp(dt*A), B_bar = dt*B) then
// delegates to selective_scan. Both use this codebase's standard [B,L,*]
// layout (not ggml's state-prefixed conv buffer). No new kernels.
//

#include <system/op_boilerplate.h>

#include <helpers/ConstantShapeHelper.h>
#include <ops/BroadcastOpsTuple.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>

#include <vector>

namespace sd {
namespace ops {

// ─── ssm_conv: depthwise causal 1D conv (adapter over causal_conv1d) ────────
#if NOT_EXCLUDED(OP_ssm_conv)
CUSTOM_OP_IMPL(ssm_conv, 2, 1, false, 0, 0) {
    auto input = INPUT_VARIABLE(0);   // [B, L, D]
    auto conv = INPUT_VARIABLE(1);    // [D, K]
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(input->rankOf() == 3, 0, "ssm_conv: input must be rank 3 [B,L,D], got %i", input->rankOf());
    REQUIRE_TRUE(conv->rankOf() == 2, 0, "ssm_conv: conv weight must be rank 2 [D,K], got %i", conv->rankOf());
    if (input->isEmpty()) return Status::OK;

    sd::ops::causal_conv1d op;
    auto res = op.evaluate({input, conv}, {}, {0, 0});  // iArgs: activation=none, wFormat=[D,K]
    REQUIRE_TRUE(res.status() == Status::OK, 0, "ssm_conv: delegation to causal_conv1d failed");
    output->assign(res.at(0));  // discard the conv state_out
    return Status::OK;
}
DECLARE_TYPES(ssm_conv) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(ssm_conv) {
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(0))->primary());
}
#endif

// ─── ssm_scan: ZOH discretization + selective_scan ──────────────────────────
// Inputs: 0 x [B,L,dim], 1 dt [B,L,state], 2 A [B,L,state] (continuous),
//         3 B [B,L,state], 4 C [B,L,state], 5 s [B,dim,state] initial state (optional).
// A_bar = exp(dt * A); B_bar = dt * B; y = selective_scan(x, A_bar, B_bar, C, D=0, h0=s).
#if NOT_EXCLUDED(OP_ssm_scan)
CUSTOM_OP_IMPL(ssm_scan, 5, 1, false, 0, 0) {
    auto x = INPUT_VARIABLE(0);
    auto dt = INPUT_VARIABLE(1);
    auto A = INPUT_VARIABLE(2);
    auto B = INPUT_VARIABLE(3);
    auto C = INPUT_VARIABLE(4);
    NDArray* s = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(x->rankOf() == 3, 0, "ssm_scan: x must be rank 3 [B,L,dim], got %i", x->rankOf());
    REQUIRE_TRUE(dt->isSameShape(A) && A->isSameShape(B) && B->isSameShape(C), 0,
                 "ssm_scan: dt, A, B, C must all share shape [B,L,state]");
    if (x->isEmpty()) { output->nullify(); return Status::OK; }

    // ZOH discretization (elementwise, selective_scan's diagonal convention)
    std::vector<LongType> stateShape(A->rankOf());
    for (int i = 0; i < A->rankOf(); i++) stateShape[i] = A->sizeAt(i);
    NDArray aBar('c', stateShape, x->dataType(), block.launchContext());
    dt->applyPairwiseTransform(pairwise::Multiply, A, &aBar, nullptr);
    aBar.applyTransform(transform::Exp, &aBar);

    NDArray bBar('c', stateShape, x->dataType(), block.launchContext());
    dt->applyPairwiseTransform(pairwise::Multiply, B, &bBar, nullptr);

    // D skip = zeros [dim]
    std::vector<LongType> dShape = {x->sizeAt(2)};
    NDArray dZero('c', dShape, x->dataType(), block.launchContext());
    dZero.nullify();

    sd::ops::selective_scan scan;
    Status status;
    if (s != nullptr) {
        status = scan.execute({x, &aBar, &bBar, C, &dZero, s}, {output}, {}, {}, {});
    } else {
        status = scan.execute({x, &aBar, &bBar, C, &dZero}, {output}, {}, {}, {});
    }
    REQUIRE_TRUE(status == Status::OK, 0, "ssm_scan: delegation to selective_scan failed");
    return status;
}
DECLARE_TYPES(ssm_scan) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_REDUCTION | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(ssm_scan) {
    return SHAPELIST(ConstantShapeHelper::getInstance().bufferForShapeInfo(inputShape->at(0))->primary());
}
#endif

}  // namespace ops
}  // namespace sd
