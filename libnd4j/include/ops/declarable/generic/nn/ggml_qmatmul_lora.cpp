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
// ggml_qmatmul_lora   — Fused quantized base matmul + LoRA delta (forward).
// ggml_qmatmul_lora_bp — Backward pass for the above.
//
// Forward:
//   out = ggml_qmatmul(A, W) + scaling * ((A @ loraAᵀ) @ loraBᵀ)
//
//   where:
//     A      : activations [M,K] or [B,S,K]
//     W      : packed INT8 GGML weight [N,K] (frozen)
//     loraA  : [rank, K]   (trainable)
//     loraB  : [N, rank]   (trainable)
//     scaling: scalar tArg(0)
//
// Backward:
//   dA      = gradOut @ dequant(W)               (base path, no transpose)
//           + scaling * (gradOut @ loraB) @ loraA (lora path)
//   dLoraA  = scaling * (gradOut @ loraB)ᵀ @ A
//   dLoraB  = scaling * gradOutᵀ @ (A @ loraAᵀ)
//   (packed weight is frozen — no dW output)
//
// @author generated for deeplearning4j
//

#include <system/op_boilerplate.h>
#include <array/NDArrayFactory.h>

// ─────────────────────────────────────────────────────────────────────────────
// Forward: ggml_qmatmul_lora
// ─────────────────────────────────────────────────────────────────────────────
#if NOT_EXCLUDED(OP_ggml_qmatmul_lora)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/ggml_qmatmul.h>
#include <ops/declarable/helpers/ggml_dequantize.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(ggml_qmatmul_lora, 4, 1, false, 1, 4) {
    auto activations   = INPUT_VARIABLE(0);  // [M,K] or [B,S,K]
    auto packedWeights = INPUT_VARIABLE(1);  // INT8, rank-1
    auto loraA         = INPUT_VARIABLE(2);  // [rank, K]
    auto loraB         = INPUT_VARIABLE(3);  // [N, rank]
    auto output        = OUTPUT_VARIABLE(0); // [M,N] or [B,S,N]

    double scaling  = T_ARG(0);
    int  quantType  = INT_ARG(0);
    auto N          = static_cast<sd::LongType>(INT_ARG(1));
    auto K          = static_cast<sd::LongType>(INT_ARG(2));
    int  outputDtype = INT_ARG(3);

    if (activations->isEmpty()) {
        return sd::Status::OK;
    }

    // ── Validate ─────────────────────────────────────────────────────────────
    int bpb = helpers::ggmlQMatMulBytesPerBlock(quantType);
    int epb = helpers::ggmlQMatMulElemsPerBlock(quantType);
    REQUIRE_TRUE(bpb > 0, 0,
        "ggml_qmatmul_lora: unsupported quantType %d", quantType);
    REQUIRE_TRUE(K % epb == 0, 0,
        "ggml_qmatmul_lora: K=%lld not divisible by block size %d", (long long)K, epb);

    int actRank = activations->rankOf();
    REQUIRE_TRUE(actRank == 2 || actRank == 3, 0,
        "ggml_qmatmul_lora: activations must be rank 2 or 3, got %d", actRank);

    auto rank = loraA->sizeAt(0);  // LoRA rank

    // ── Step 1: base output via quantized matmul helper ──────────────────────
    helpers::ggmlQMatMul(block.launchContext(),
                         activations, packedWeights, output,
                         quantType, N, K, outputDtype);

    // ── Step 2: LoRA delta ───────────────────────────────────────────────────
    // Flatten [B,S,K] → [M2d,K] for rank-3
    bool rank3 = (actRank == 3);
    sd::LongType B = 1, S = 1;
    if (rank3) {
        B = activations->sizeAt(0);
        S = activations->sizeAt(1);
    }
    sd::LongType M2d = rank3 ? (B * S) : activations->sizeAt(0);

    sd::DataType floatDt = (activations->dataType() == sd::DataType::HALF)
                           ? sd::DataType::HALF : sd::DataType::FLOAT32;

    NDArray* act2d  = nullptr;
    bool reshapedAct = false;
    if (rank3) {
        std::vector<sd::LongType> s = {M2d, K};
        act2d = activations->reshape('c', s);
        reshapedAct = true;
    } else {
        act2d = activations;
    }

    // temp1 = act2d[M2d,K] @ loraAᵀ[K,rank] = [M2d, rank]
    auto temp1 = NDArrayFactory::create('c', std::vector<sd::LongType>{M2d, rank},
                                        floatDt, block.launchContext());
    {
        sd::ops::matmul mmulOp;
        std::vector<sd::LongType> iArgs = {0, 1};  // transposeB=true
        std::vector<NDArray*> inputs  = {act2d, loraA};
        std::vector<NDArray*> outputs = {temp1};
        mmulOp.execute(inputs, outputs, {}, iArgs, {});
    }

    // loraOut2d = temp1[M2d,rank] @ loraBᵀ[rank,N] = [M2d, N]
    // loraB is [N, rank], so transposeB=true gives [M2d,N]
    auto loraOut2d = NDArrayFactory::create('c', std::vector<sd::LongType>{M2d, N},
                                            floatDt, block.launchContext());
    {
        sd::ops::matmul mmulOp;
        std::vector<sd::LongType> iArgs = {0, 1};  // transposeB=true
        std::vector<NDArray*> inputs  = {temp1, loraB};
        std::vector<NDArray*> outputs = {loraOut2d};
        mmulOp.execute(inputs, outputs, {}, iArgs, {});
    }

    // Scale and add: output += scaling * loraOut (with rank-3 reshape if needed)
    loraOut2d->applyScalar(scalar::Multiply, scaling, loraOut2d);

    if (rank3) {
        std::vector<sd::LongType> outFullShape = {B, S, N};
        auto loraOut3d = loraOut2d->reshape('c', outFullShape);
        *output += *loraOut3d;
        delete loraOut3d;
        delete act2d;
    } else {
        *output += *loraOut2d;
    }

    delete temp1;
    delete loraOut2d;

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(ggml_qmatmul_lora) {
    auto actShape = inputShape->at(0);
    int  actRank  = shape::rank(actShape);
    auto N        = static_cast<sd::LongType>(INT_ARG(1));
    int  outputDtype = INT_ARG(3);
    sd::DataType outDt = (outputDtype == 1) ? sd::DataType::HALF : sd::DataType::FLOAT32;

    std::vector<sd::LongType> outShape;
    if (actRank == 2) {
        outShape = {shape::sizeAt(actShape, 0), N};
    } else {
        outShape = {shape::sizeAt(actShape, 0), shape::sizeAt(actShape, 1), N};
    }
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(outDt, 'c', outShape));
}

DECLARE_TYPES(ggml_qmatmul_lora) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {FLOAT32, HALF})    // activations
        ->setAllowedInputTypes(1, {INT8})              // packed weights
        ->setAllowedInputTypes(2, {ALL_FLOATS})        // loraA
        ->setAllowedInputTypes(3, {ALL_FLOATS})        // loraB
        ->setAllowedOutputTypes({FLOAT32, HALF})
        ->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_ggml_qmatmul_lora)

// ─────────────────────────────────────────────────────────────────────────────
// Backward: ggml_qmatmul_lora_bp
// ─────────────────────────────────────────────────────────────────────────────
#if NOT_EXCLUDED(OP_ggml_qmatmul_lora_bp)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/ggml_qmatmul.h>
#include <ops/declarable/helpers/ggml_dequantize.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(ggml_qmatmul_lora_bp, 5, 3, false, 1, 3) {
    auto activations   = INPUT_VARIABLE(0);  // [M,K] or [B,S,K]
    auto packedWeights = INPUT_VARIABLE(1);  // INT8 frozen
    auto loraA         = INPUT_VARIABLE(2);  // [rank, K]
    auto loraB         = INPUT_VARIABLE(3);  // [N, rank]
    auto gradOut       = INPUT_VARIABLE(4);  // [M,N] or [B,S,N]

    auto dA         = OUTPUT_VARIABLE(0);  // same shape as activations
    auto dLoraA     = OUTPUT_VARIABLE(1);  // same shape as loraA [rank,K]
    auto dLoraB     = OUTPUT_VARIABLE(2);  // same shape as loraB [N,rank]

    double scaling  = T_ARG(0);
    int  quantType  = INT_ARG(0);
    auto N          = static_cast<sd::LongType>(INT_ARG(1));
    auto K          = static_cast<sd::LongType>(INT_ARG(2));

    if (activations->isEmpty() || gradOut->isEmpty()) {
        return sd::Status::OK;
    }

    int actRank = activations->rankOf();
    REQUIRE_TRUE(actRank == 2 || actRank == 3, 0,
        "ggml_qmatmul_lora_bp: activations must be rank 2 or 3, got %d", actRank);

    sd::DataType floatDt = (activations->dataType() == sd::DataType::HALF)
                           ? sd::DataType::HALF : sd::DataType::FLOAT32;

    auto rank = loraA->sizeAt(0);

    // ── Flatten rank-3 ────────────────────────────────────────────────────────
    bool rank3 = (actRank == 3);
    sd::LongType B = 1, S = 1;
    if (rank3) {
        B = activations->sizeAt(0);
        S = activations->sizeAt(1);
    }
    sd::LongType M2d = rank3 ? (B * S) : activations->sizeAt(0);

    NDArray* act2d     = nullptr;
    NDArray* gradOut2d = nullptr;
    bool reshapedAct   = false;
    bool reshapedGO    = false;

    if (rank3) {
        std::vector<sd::LongType> s2 = {M2d, K};
        act2d = activations->reshape('c', s2);
        reshapedAct = true;
        std::vector<sd::LongType> sg = {M2d, N};
        gradOut2d = gradOut->reshape('c', sg);
        reshapedGO = true;
    } else {
        act2d     = activations;
        gradOut2d = gradOut;
    }

    // ── Part 1: dA from base path = gradOut @ dequant(W) ─────────────────────
    // Dequantize W: packed → Wf [N, K]
    auto Wf = NDArrayFactory::create('c', std::vector<sd::LongType>{N, K},
                                     floatDt, block.launchContext());
    helpers::ggmlDequantize(block.launchContext(), packedWeights, Wf, quantType);

    // dA_base[M2d,K] = gradOut2d[M2d,N] @ Wf[N,K]  (no transpose)
    auto dA2d = NDArrayFactory::create('c', std::vector<sd::LongType>{M2d, K},
                                       floatDt, block.launchContext());
    {
        sd::ops::matmul mmulOp;
        std::vector<NDArray*> inputs  = {gradOut2d, Wf};
        std::vector<NDArray*> outputs = {dA2d};
        mmulOp.execute(inputs, outputs);
    }
    delete Wf;

    // ── Part 2: LoRA gradients ────────────────────────────────────────────────
    // temp = gradOut2d[M2d,N] @ loraB[N,rank]  (no transpose) → [M2d, rank]
    auto temp = NDArrayFactory::create('c', std::vector<sd::LongType>{M2d, rank},
                                       floatDt, block.launchContext());
    {
        sd::ops::matmul mmulOp;
        std::vector<NDArray*> inputs  = {gradOut2d, loraB};
        std::vector<NDArray*> outputs = {temp};
        mmulOp.execute(inputs, outputs);
    }

    // dA_lora[M2d,K] = temp[M2d,rank] @ loraA[rank,K]  (no transpose) → [M2d,K]
    auto dA_lora = NDArrayFactory::create('c', std::vector<sd::LongType>{M2d, K},
                                           floatDt, block.launchContext());
    {
        sd::ops::matmul mmulOp;
        std::vector<NDArray*> inputs  = {temp, loraA};
        std::vector<NDArray*> outputs = {dA_lora};
        mmulOp.execute(inputs, outputs);
    }
    dA_lora->applyScalar(scalar::Multiply, scaling, dA_lora);

    // dA_total = dA_base + dA_lora
    *dA2d += *dA_lora;
    delete dA_lora;

    // dLoraB[N,rank] = scaling * gradOut2dᵀ[N,M2d] @ act_loraOut[M2d,rank]
    // where act_loraOut = act2d @ loraAᵀ  → [M2d, rank]
    auto actLoraOut = NDArrayFactory::create('c', std::vector<sd::LongType>{M2d, rank},
                                              floatDt, block.launchContext());
    {
        sd::ops::matmul mmulOp;
        std::vector<sd::LongType> iArgs = {0, 1};  // transposeB=true
        std::vector<NDArray*> inputs  = {act2d, loraA};
        std::vector<NDArray*> outputs = {actLoraOut};
        mmulOp.execute(inputs, outputs, {}, iArgs, {});
    }

    {
        // dLoraB[N,rank] = gradOut2dᵀ[N,M2d] @ actLoraOut[M2d,rank]
        auto gradOut2dT = gradOut2d->transpose();
        sd::ops::matmul mmulOp;
        std::vector<NDArray*> inputs  = {gradOut2dT, actLoraOut};
        std::vector<NDArray*> outputs = {dLoraB};
        mmulOp.execute(inputs, outputs);
        delete gradOut2dT;
    }
    dLoraB->applyScalar(scalar::Multiply, scaling, dLoraB);

    delete actLoraOut;

    // dLoraA[rank,K] = scaling * tempᵀ[rank,M2d] @ act2d[M2d,K]
    {
        auto tempT = temp->transpose();
        sd::ops::matmul mmulOp;
        std::vector<NDArray*> inputs  = {tempT, act2d};
        std::vector<NDArray*> outputs = {dLoraA};
        mmulOp.execute(inputs, outputs);
        delete tempT;
    }
    dLoraA->applyScalar(scalar::Multiply, scaling, dLoraA);

    delete temp;

    // ── Assign dA back (reshape if rank-3) ────────────────────────────────────
    if (rank3) {
        std::vector<sd::LongType> fullShape = {B, S, K};
        auto dA3d = dA2d->reshape('c', fullShape);
        dA->assign(dA3d);
        delete dA3d;
        delete act2d;
        delete gradOut2d;
    } else {
        dA->assign(dA2d);
    }

    delete dA2d;

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(ggml_qmatmul_lora_bp) {
    // Outputs: dA (same as activations), dLoraA (same as loraA), dLoraB (same as loraB)
    return SHAPELIST(
        CONSTANT(inputShape->at(0)),  // activations shape
        CONSTANT(inputShape->at(2)),  // loraA shape
        CONSTANT(inputShape->at(3))   // loraB shape
    );
}

DECLARE_TYPES(ggml_qmatmul_lora_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {FLOAT32, HALF})   // activations
        ->setAllowedInputTypes(1, {INT8})              // packed weights
        ->setAllowedInputTypes(2, {ALL_FLOATS})        // loraA
        ->setAllowedInputTypes(3, {ALL_FLOATS})        // loraB
        ->setAllowedInputTypes(4, {ALL_FLOATS})        // gradOut
        ->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_ggml_qmatmul_lora_bp)
