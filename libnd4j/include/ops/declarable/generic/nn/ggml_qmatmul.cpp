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
// ggml_qmatmul — Runtime quantized matmul against GGML-packed weights.
//
// Inputs:
//   0: activations A   — FLOAT32 or HALF, rank 2 [M,K] or rank 3 [B,S,K]
//   1: packedWeights W — INT8 1-D raw GGML block bytes for logical [N,K] weight
//
// Integer args:
//   0: ggmlType   — GgmlQuantType enum (4=Q8_0, 8=Q4_K, 10=Q6_K)
//   1: N          — number of weight rows (output columns)
//   2: K          — inner dimension (must be divisible by block size)
//   3: outputDtype — 0=FLOAT32, 1=HALF
//
// Output:
//   0: [M,N] or [B,S,N] — fp32 accumulation, written in outputDtype
//
// Weight orientation:
//   GGUF stores weight tensors with innermost dimension first (column-major).
//   After reverseShape() in the converter, a linear layer weight originally
//   [hidden=K, out=N] in GGUF becomes [N, K] in ND4J (N rows, K cols).
//   LLaMAArchitecture.java calls weight.permute(1,0) before sd.mmul(), which
//   is equivalent to computing A @ W^T — but since we store W as [N, K] and
//   quantize along K (row direction), we implement:
//     out[m,n] = sum_k A[m,k] * W[n,k]
//   directly without any permutation, matching the semantics of A @ W^T
//   when W is [N, K].
//
// @author generated for deeplearning4j
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_ggml_qmatmul)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/ggml_qmatmul.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(ggml_qmatmul, 2, 1, false, 0, 4) {
    auto activations   = INPUT_VARIABLE(0);
    auto packedWeights = INPUT_VARIABLE(1);
    auto output        = OUTPUT_VARIABLE(0);

    int  quantType  = INT_ARG(0);
    auto N          = static_cast<sd::LongType>(INT_ARG(1));
    auto K          = static_cast<sd::LongType>(INT_ARG(2));
    int  outputDtype = INT_ARG(3);

    // ── Validate ────────────────────────────────────────────────────────────
    int bpb = helpers::ggmlQMatMulBytesPerBlock(quantType);
    int epb = helpers::ggmlQMatMulElemsPerBlock(quantType);
    REQUIRE_TRUE(bpb > 0, 0,
        "ggml_qmatmul: unsupported quantType %d (supported: 4=Q8_0, 8=Q4_K, 10=Q6_K)", quantType);

    REQUIRE_TRUE(K % epb == 0, 0,
        "ggml_qmatmul: K=%lld must be divisible by block size %d for quantType %d",
        (long long)K, epb, quantType);

    sd::LongType expectedBytes = N * (K / epb) * bpb;
    REQUIRE_TRUE(packedWeights->lengthOf() == expectedBytes, 0,
        "ggml_qmatmul: packedWeights length %lld != expected %lld (N=%lld K=%lld bpb=%d epb=%d)",
        (long long)packedWeights->lengthOf(), (long long)expectedBytes,
        (long long)N, (long long)K, bpb, epb);

    int actRank = activations->rankOf();
    REQUIRE_TRUE(actRank == 2 || actRank == 3, 0,
        "ggml_qmatmul: activations must be rank 2 or 3, got rank %d", actRank);

    sd::LongType actK = activations->sizeAt(actRank - 1);
    REQUIRE_TRUE(actK == K, 0,
        "ggml_qmatmul: activation inner dim %lld != K=%lld", (long long)actK, (long long)K);

    REQUIRE_TRUE(outputDtype == 0 || outputDtype == 1, 0,
        "ggml_qmatmul: outputDtype must be 0 (FP32) or 1 (FP16), got %d", outputDtype);

    // ── Execute ─────────────────────────────────────────────────────────────
    helpers::ggmlQMatMul(block.launchContext(),
                         activations, packedWeights, output,
                         quantType, N, K, outputDtype);

    return sd::Status::OK;
}

DECLARE_TYPES(ggml_qmatmul) {
  getOpDescriptor()->addTraits(OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING);
    getOpDescriptor()
        ->setAllowedInputTypes(0, {FLOAT32, HALF})    // activations
        ->setAllowedInputTypes(1, {INT8})              // packed weight bytes
        ->setAllowedOutputTypes({FLOAT32, HALF});
}

DECLARE_SHAPE_FN(ggml_qmatmul) {
    auto actShape = inputShape->at(0);
    int  actRank  = shape::rank(actShape);

    auto N = static_cast<sd::LongType>(INT_ARG(1));
    int  outputDtype = INT_ARG(3);

    sd::DataType outDt = (outputDtype == 1) ? sd::DataType::HALF : sd::DataType::FLOAT32;

    std::vector<sd::LongType> outShape;
    if (actRank == 2) {
        outShape = {shape::sizeAt(actShape, 0), N};
    } else {
        // rank 3: [B, S, K] → [B, S, N]
        outShape = {shape::sizeAt(actShape, 0), shape::sizeAt(actShape, 1), N};
    }

    auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
        outDt, 'c', outShape);

    return SHAPELIST(outputShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_ggml_qmatmul)

// ─────────────────────────────────────────────────────────────────────────────
// ggml_qmatmul_bp — Gradient of ggml_qmatmul w.r.t. activations.
//
// The packed weight is frozen (no gradient returned for it).
//
// Math: forward  out[M,N]  = A[M,K] @ Wf[K,N]   (Wf = dequant(W), W stored [N,K])
//       backward dA[M,K]   = dOut[M,N] @ Wf[N,K]
//
// Strategy: dequantize W→Wf [N,K] via ggml_dequantize helper, then
//           dA = matmul(dOut, Wf)  — no transpose (dOut[M,N] × Wf[N,K] = [M,K])
//
// Inputs:
//   0: activations A   [M,K] or [B,S,K]  — only shape/dtype used
//   1: packedWeights W INT8 rank-1
//   2: gradOut         [M,N] or [B,S,N]  — same rank as activations
//
// iArgs: (0) quantType, (1) N, (2) K
//
// Output:
//   0: gradActivations — same shape & dtype as activations
// ─────────────────────────────────────────────────────────────────────────────
#if NOT_EXCLUDED(OP_ggml_qmatmul_bp)

#include <system/common.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/ggml_dequantize.h>
#include <array/NDArrayFactory.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(ggml_qmatmul_bp, 3, 1, false, 0, 3) {
    auto activations   = INPUT_VARIABLE(0);  // shape reference only
    auto packedWeights = INPUT_VARIABLE(1);  // INT8 packed
    auto gradOut       = INPUT_VARIABLE(2);  // [M,N] or [B,S,N]
    auto gradA         = OUTPUT_VARIABLE(0); // same shape as activations

    int  quantType = INT_ARG(0);
    auto N         = static_cast<sd::LongType>(INT_ARG(1));
    auto K         = static_cast<sd::LongType>(INT_ARG(2));

    if (activations->isEmpty() || gradOut->isEmpty()) {
        return sd::Status::OK;
    }

    int actRank = activations->rankOf();
    REQUIRE_TRUE(actRank == 2 || actRank == 3, 0,
        "ggml_qmatmul_bp: activations must be rank 2 or 3, got %d", actRank);
    REQUIRE_TRUE(gradOut->rankOf() == actRank, 0,
        "ggml_qmatmul_bp: gradOut rank %d must match activations rank %d",
        gradOut->rankOf(), actRank);

    // ── Dequantize W: packed [bytes] → Wf [N, K] float ──────────────────────
    std::vector<sd::LongType> wfShape = {N, K};
    // Determine float dtype matching activations
    sd::DataType floatDt = sd::DataType::FLOAT32;
    // Backward accumulation stays FLOAT32 even for HALF activations.

    // NDArrayFactory::create returns NDArray* (heap allocated)
    auto Wf = NDArrayFactory::create('c', wfShape, floatDt, block.launchContext());
    helpers::ggmlDequantize(block.launchContext(), packedWeights, Wf, quantType);

    // ── Handle rank-3 by flattening batch dims ───────────────────────────────
    bool rank3 = (actRank == 3);
    sd::LongType B = 1, S = 1;
    if (rank3) {
        B = activations->sizeAt(0);
        S = activations->sizeAt(1);
    }
    sd::LongType M2d = rank3 ? (B * S) : activations->sizeAt(0);

    // Reshape gradOut to [M2d, N] for matmul
    NDArray* gradOut2d = nullptr;
    bool reshapedGO = false;
    if (rank3) {
        std::vector<sd::LongType> shape2d = {M2d, N};
        gradOut2d = gradOut->reshape('c', shape2d);
        reshapedGO = true;
    } else {
        gradOut2d = gradOut;
    }

    NDArray* gradOut2dFloat = gradOut2d;
    NDArray* gradOut2dCast = nullptr;
    if (gradOut2d->dataType() != sd::DataType::FLOAT32) {
        gradOut2dCast = gradOut2d->cast(sd::DataType::FLOAT32);
        gradOut2dFloat = gradOut2dCast;
    }

    // dA[M2d,K] = dOut[M2d,N] @ Wf[N,K]  (no transpose needed)
    if (rank3) {
        // Need a 2D intermediate, then reshape back
        std::vector<sd::LongType> gradA2dShape = {M2d, K};
        auto gradA2d = NDArrayFactory::create('c', gradA2dShape, floatDt, block.launchContext());
        {
            sd::ops::matmul mmulOp;
            std::vector<NDArray*> inputs  = {gradOut2dFloat, Wf};
            std::vector<NDArray*> outputs = {gradA2d};
            mmulOp.execute(inputs, outputs);
        }
        // Reshape [M2d,K] → [B,S,K]
        std::vector<sd::LongType> fullShape = {B, S, K};
        auto gradA3d = gradA2d->reshape('c', fullShape);
        gradA->assign(gradA3d);
        delete gradA3d;
        delete gradA2d;
        // gradOut2dCast is released after rank-specific handling.
        delete gradOut2d;
    } else {
        // rank-2: output directly into gradA
        sd::ops::matmul mmulOp;
        std::vector<NDArray*> inputs  = {gradOut2dFloat, Wf};
        std::vector<NDArray*> outputs = {gradA};
        mmulOp.execute(inputs, outputs);
    }

    delete gradOut2dCast;

    delete Wf;

    return sd::Status::OK;
}

DECLARE_SHAPE_FN(ggml_qmatmul_bp) {
    // Output 0: same shape as activations, FLOAT32 dtype for stable backprop.
    auto actShape = inputShape->at(0);
    auto rank = shape::rank(actShape);
    std::vector<sd::LongType> outShape;
    for (int i = 0; i < rank; i++) {
        outShape.push_back(shape::shapeOf(actShape)[i]);
    }
    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(
        sd::DataType::FLOAT32, shape::order(actShape), outShape);
    return SHAPELIST(outputShape);
}

DECLARE_TYPES(ggml_qmatmul_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {FLOAT32, HALF})  // activations (shape ref)
        ->setAllowedInputTypes(1, {INT8})             // packed weights
        ->setAllowedInputTypes(2, {FLOAT32, HALF})   // gradOut
        ->setAllowedOutputTypes({FLOAT32, HALF})
        ->addTraits(OP_TRAIT_MATMUL | OP_TRAIT_FULLY_WRITING | OP_TRAIT_BACKWARD);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_ggml_qmatmul_bp)
