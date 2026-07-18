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
// llama.cpp-compat op names that are pure compositions of existing primitives
// (GLU-family gated activations, Swin window partition/unpartition, and
// sinusoidal/timestep embedding generators). No new kernels.
//

#include <system/op_boilerplate.h>

#include <array/DataTypeUtils.h>
#include <array/NDArrayFactory.h>
#include <helpers/ConstantShapeHelper.h>
#include <ops/BroadcastOpsTuple.h>
#include <ops/declarable/headers/llm.h>

#include <cmath>
#include <vector>

namespace sd {
namespace ops {

// ─── GLU family: out = act(x[..., :half]) * x[..., half:] over the last dim ──
#if NOT_EXCLUDED(OP_swiglu) || NOT_EXCLUDED(OP_geglu) || NOT_EXCLUDED(OP_reglu)
enum GluAct { GLU_SILU, GLU_GELU, GLU_RELU };

static Status gluImpl(graph::Context& block, NDArray* input, NDArray* output, GluAct act) {
    const int rank = input->rankOf();
    REQUIRE_TRUE(rank >= 1, 0, "GLU: input rank must be >= 1");
    REQUIRE_TRUE(input->sizeAt(-1) % 2 == 0, 0,
                 "GLU: last dimension must be even, got %lld", (long long)input->sizeAt(-1));

    if (input->isEmpty()) return Status::OK;

    const LongType half = input->sizeAt(-1) / 2;

    // strided half-views of the last dim (read-only; never written back into)
    std::vector<LongType> gateIdx(2 * rank, 0), upIdx(2 * rank, 0);
    gateIdx[2 * (rank - 1)] = 0;      gateIdx[2 * (rank - 1) + 1] = half;
    upIdx[2 * (rank - 1)] = half;     upIdx[2 * (rank - 1) + 1] = 2 * half;
    NDArray* gateView = (*input)(gateIdx);
    NDArray* upView = (*input)(upIdx);

    // activation of the gate into a contiguous scratch (output-shaped)
    std::vector<LongType> halfShape(rank);
    for (int i = 0; i < rank; i++) halfShape[i] = input->sizeAt(i);
    halfShape[rank - 1] = half;
    NDArray gateAct('c', halfShape, input->dataType(), block.launchContext());
    switch (act) {
        case GLU_SILU: gateView->applyTransform(transform::Swish, &gateAct); break;
        case GLU_GELU: gateView->applyTransform(transform::PreciseGELU, &gateAct); break;
        case GLU_RELU: gateView->applyScalar(scalar::RELU, 0.0, &gateAct); break;
    }

    gateAct.applyPairwiseTransform(pairwise::Multiply, upView, output, nullptr);

    delete gateView;
    delete upView;
    return Status::OK;
}

static sd::ShapeList* gluShape(const ShapeList* inputShape) {
    auto inShape = inputShape->at(0);
    const int rank = shape::rank(inShape);
    std::vector<sd::LongType> outShape(rank);
    for (int i = 0; i < rank; i++) outShape[i] = shape::sizeAt(inShape, static_cast<sd::LongType>(i));
    outShape[rank - 1] /= 2;
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        sd::ArrayOptions::dataType(inShape), 'c', outShape));
}
#endif

#if NOT_EXCLUDED(OP_swiglu)
CUSTOM_OP_IMPL(swiglu, 1, 1, false, 0, 0) {
    return gluImpl(block, INPUT_VARIABLE(0), OUTPUT_VARIABLE(0), GLU_SILU);  // SiLU(gate) * up
}
DECLARE_TYPES(swiglu) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(swiglu) { return gluShape(inputShape); }
#endif

#if NOT_EXCLUDED(OP_geglu)
CUSTOM_OP_IMPL(geglu, 1, 1, false, 0, 0) {
    return gluImpl(block, INPUT_VARIABLE(0), OUTPUT_VARIABLE(0), GLU_GELU);  // tanh-GELU(gate) * up (matches ggml_gelu)
}
DECLARE_TYPES(geglu) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(geglu) { return gluShape(inputShape); }
#endif

#if NOT_EXCLUDED(OP_reglu)
CUSTOM_OP_IMPL(reglu, 1, 1, false, 0, 0) {
    return gluImpl(block, INPUT_VARIABLE(0), OUTPUT_VARIABLE(0), GLU_RELU);  // ReLU(gate) * up
}
DECLARE_TYPES(reglu) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_UNARY_ELEMENTWISE | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(reglu) { return gluShape(inputShape); }
#endif

// ─── Swin window partition / unpartition (reshape + permute) ─────────────────
// Requires H,W divisible by windowSize (the common Swin case; ggml_win_part's
// padding path for indivisible feature maps is intentionally not replicated —
// callers that need padding must pad before this op).
#if NOT_EXCLUDED(OP_win_part)
CUSTOM_OP_IMPL(win_part, 1, 1, false, 0, 1) {
    auto input = INPUT_VARIABLE(0);   // [N, H, W, C]
    auto output = OUTPUT_VARIABLE(0);  // [N*(H/w)*(W/w), w, w, C]

    REQUIRE_TRUE(input->rankOf() == 4, 0, "win_part: input must be rank 4 [N,H,W,C], got %i", input->rankOf());
    const LongType w = INT_ARG(0);
    const LongType N = input->sizeAt(0), H = input->sizeAt(1), W = input->sizeAt(2), C = input->sizeAt(3);
    REQUIRE_TRUE(w > 0 && H % w == 0 && W % w == 0, 0,
                 "win_part: H (%lld) and W (%lld) must be divisible by windowSize (%lld)",
                 (long long)H, (long long)W, (long long)w);
    if (input->isEmpty()) return Status::OK;

    const LongType Hb = H / w, Wb = W / w;
    std::vector<LongType> splitShape = {N, Hb, w, Wb, w, C};
    std::vector<LongType> permAxes = {0, 1, 3, 2, 4, 5};
    NDArray* r1 = input->reshape('c', splitShape);          // [N,Hb,w,Wb,w,C] view
    NDArray* p = r1->permute(permAxes, false, false);       // [N,Hb,Wb,w,w,C] strided

    std::vector<LongType> permShape = {N, Hb, Wb, w, w, C};
    NDArray perm('c', permShape, input->dataType(), block.launchContext());
    perm.assign(p);                                          // contiguous copy of permuted data

    std::vector<LongType> outShape = {N * Hb * Wb, w, w, C};
    NDArray* permFlat = perm.reshape('c', outShape);        // contiguous view
    output->assign(permFlat);

    delete r1; delete p; delete permFlat;
    return Status::OK;
}
DECLARE_TYPES(win_part) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(win_part) {
    auto inShape = inputShape->at(0);
    const LongType w = INT_ARG(0);
    const LongType N = shape::sizeAt(inShape, static_cast<sd::LongType>(0));
    const LongType H = shape::sizeAt(inShape, static_cast<sd::LongType>(1));
    const LongType W = shape::sizeAt(inShape, static_cast<sd::LongType>(2));
    const LongType C = shape::sizeAt(inShape, static_cast<sd::LongType>(3));
    std::vector<sd::LongType> outShape = {N * (H / w) * (W / w), w, w, C};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        sd::ArrayOptions::dataType(inShape), 'c', outShape));
}
#endif

#if NOT_EXCLUDED(OP_win_unpart)
CUSTOM_OP_IMPL(win_unpart, 1, 1, false, 0, 3) {
    auto input = INPUT_VARIABLE(0);   // [numWin, w, w, C]
    auto output = OUTPUT_VARIABLE(0);  // [N, H, W, C]

    REQUIRE_TRUE(input->rankOf() == 4, 0, "win_unpart: input must be rank 4 [numWin,w,w,C], got %i",
                 input->rankOf());
    const LongType w = INT_ARG(0), H = INT_ARG(1), W = INT_ARG(2);
    REQUIRE_TRUE(w > 0 && H % w == 0 && W % w == 0, 0,
                 "win_unpart: H (%lld) and W (%lld) must be divisible by windowSize (%lld)",
                 (long long)H, (long long)W, (long long)w);
    const LongType numWin = input->sizeAt(0), C = input->sizeAt(3);
    const LongType Hb = H / w, Wb = W / w;
    REQUIRE_TRUE(numWin % (Hb * Wb) == 0, 0,
                 "win_unpart: numWindows (%lld) must be divisible by (H/w)*(W/w)=%lld",
                 (long long)numWin, (long long)(Hb * Wb));
    if (input->isEmpty()) return Status::OK;

    const LongType N = numWin / (Hb * Wb);
    std::vector<LongType> splitShape = {N, Hb, Wb, w, w, C};
    std::vector<LongType> permAxes = {0, 1, 3, 2, 4, 5};
    NDArray* r1 = input->reshape('c', splitShape);          // [N,Hb,Wb,w,w,C] view
    NDArray* p = r1->permute(permAxes, false, false);       // [N,Hb,w,Wb,w,C] strided

    std::vector<LongType> permShape = {N, Hb, w, Wb, w, C};
    NDArray perm('c', permShape, input->dataType(), block.launchContext());
    perm.assign(p);

    std::vector<LongType> outShape = {N, H, W, C};
    NDArray* permFlat = perm.reshape('c', outShape);
    output->assign(permFlat);

    delete r1; delete p; delete permFlat;
    return Status::OK;
}
DECLARE_TYPES(win_unpart) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(win_unpart) {
    auto inShape = inputShape->at(0);
    const LongType w = INT_ARG(0), H = INT_ARG(1), W = INT_ARG(2);
    const LongType numWin = shape::sizeAt(inShape, static_cast<sd::LongType>(0));
    const LongType C = shape::sizeAt(inShape, static_cast<sd::LongType>(3));
    const LongType N = numWin / ((H / w) * (W / w));
    std::vector<sd::LongType> outShape = {N, H, W, C};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
        sd::ArrayOptions::dataType(inShape), 'c', outShape));
}
#endif

// ─── Embedding generators (build freqs host-side, broadcast, sin/cos) ────────
#if NOT_EXCLUDED(OP_timestep_embedding) || NOT_EXCLUDED(OP_sinusoidal_position_encoding)
// Writes cos(args)||sin(args) (cosFirst=true, ggml_timestep_embedding order) or
// sin(args)||cos(args) (cosFirst=false) into [T, dim]; zero-pads a final column
// when dim is odd. args[t,j] = pos[t] * freq[j], freq[j] = base^(-2j/scaleDim).
static Status buildEmbedding(graph::Context& block, NDArray* positions, NDArray* output,
                             LongType dim, double period, bool cosFirst) {
    auto ctx = block.launchContext();
    const auto outType = output->dataType();
    const LongType T = positions->lengthOf();
    const LongType half = dim / 2;
    if (positions->isEmpty() || half == 0) { if (!output->isEmpty()) output->nullify(); return Status::OK; }

    std::vector<float> freqsData(half);
    const double logPeriod = std::log(period);
    for (LongType j = 0; j < half; j++)
        freqsData[j] = static_cast<float>(std::exp(-logPeriod * static_cast<double>(j) / static_cast<double>(half)));
    std::vector<LongType> freqShape = {1, half};
    NDArray* freqsF = NDArrayFactory::create('c', freqShape, freqsData, ctx);
    NDArray* freqs = freqsF->cast(outType);

    std::vector<LongType> colShape = {T, 1};
    NDArray* posCast = positions->cast(outType);
    NDArray* posCol = posCast->reshape('c', colShape);

    std::vector<LongType> argsShape = {T, half};
    NDArray args('c', argsShape, outType, ctx);
    posCol->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), freqs, &args, true);

    NDArray first('c', argsShape, outType, ctx), second('c', argsShape, outType, ctx);
    args.applyTransform(cosFirst ? transform::Cosine : transform::Sin, &first);
    args.applyTransform(cosFirst ? transform::Sin : transform::Cosine, &second);

    // scatter the two halves into the output's contiguous column ranges
    std::vector<LongType> firstIdx = {0, 0, 0, half};
    std::vector<LongType> secondIdx = {0, 0, half, 2 * half};
    NDArray* outFirst = (*output)(firstIdx);
    NDArray* outSecond = (*output)(secondIdx);
    outFirst->assign(&first);
    outSecond->assign(&second);
    delete outFirst; delete outSecond;

    if (dim > 2 * half) {  // odd dim → trailing zero column
        std::vector<LongType> tailIdx = {0, 0, 2 * half, dim};
        NDArray* tail = (*output)(tailIdx);
        tail->nullify();
        delete tail;
    }

    delete freqsF; delete freqs; delete posCast; delete posCol;
    return Status::OK;
}

static sd::ShapeList* embeddingShape(const ShapeList* inputShape, LongType dim, DataType forceType,
                                     bool useForce) {
    auto inShape = inputShape->at(0);
    const LongType T = shape::length(inShape);
    DataType t = useForce ? forceType : sd::ArrayOptions::dataType(inShape);
    if (!DataTypeUtils::isR(t)) t = DataType::FLOAT32;  // integer positions → float embedding
    std::vector<sd::LongType> outShape = {T, dim};
    return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(t, 'c', outShape));
}
#endif

#if NOT_EXCLUDED(OP_timestep_embedding)
CUSTOM_OP_IMPL(timestep_embedding, 1, 1, false, 0, -2) {
    auto timesteps = INPUT_VARIABLE(0);
    const LongType dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : 128;
    const double maxPeriod = block.getIArguments()->size() > 1 ? static_cast<double>(INT_ARG(1)) : 10000.0;
    return buildEmbedding(block, timesteps, OUTPUT_VARIABLE(0), dim, maxPeriod, /*cosFirst=*/true);
}
DECLARE_TYPES(timestep_embedding) {
    getOpDescriptor()->setAllowedInputTypes({ALL_INTS, ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(timestep_embedding) {
    const LongType dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : 128;
    return embeddingShape(inputShape, dim, DataType::FLOAT32, false);
}
#endif

#if NOT_EXCLUDED(OP_sinusoidal_position_encoding)
CUSTOM_OP_IMPL(sinusoidal_position_encoding, 1, 1, false, 0, -2) {
    auto positions = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);
    const LongType dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : output->sizeAt(-1);
    // sin||cos half-split layout (common transformer variant); base 10000
    return buildEmbedding(block, positions, output, dim, 10000.0, /*cosFirst=*/false);
}
DECLARE_TYPES(sinusoidal_position_encoding) {
    getOpDescriptor()->setAllowedInputTypes({ALL_INTS, ALL_FLOATS})->setAllowedOutputTypes({ALL_FLOATS})
        ->addTraits(OP_TRAIT_FULLY_WRITING);
}
DECLARE_SHAPE_FN(sinusoidal_position_encoding) {
    REQUIRE_TRUE(block.getIArguments()->size() > 0, 0,
                 "sinusoidal_position_encoding: I arg 0 (embedDim) is required for shape inference");
    const LongType dim = INT_ARG(0);
    return embeddingShape(inputShape, dim, DataType::FLOAT32, false);
}
#endif

}  // namespace ops
}  // namespace sd
