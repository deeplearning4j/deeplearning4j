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
// @author Adam Gibson
//
// loftq_init — LoftQ LoRA initialization (arXiv:2310.08659)
//
// Computes LoRA matrices B [outDim, rank] and A [rank, inDim] such that
//   W ≈ Q + B @ A
// where Q is the quantized weight, using SVD of the residual.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_loftq_init)

#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/svd.h>
#include <math/templatemath.h>
#include <cmath>

namespace sd {
namespace ops {

// ---------------------------------------------------------------------------
// Internal helpers (anonymous namespace — translation-unit private)
// ---------------------------------------------------------------------------
namespace {

// NF4 codebook: 16 values derived from the normal distribution quantile function
// (standard QLoRA codebook, arXiv:2305.14314 Table 1)
static const float NF4_CODEBOOK[16] = {
    -1.0f,                -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f,-0.18477343022823334f, -0.09105003625154495f,  0.0f,
     0.07958029955625534f,  0.16093020141124725f,  0.24611230194568634f,  0.33791524171829224f,
     0.44070982933044434f,  0.5626170039176941f,   0.7229568362236023f,   1.0f
};

// FP4 codebook (1 sign, 2-bit exponent, 1-bit mantissa)
static const float FP4_CODEBOOK[16] = {
    0.0f, 0.0625f, 0.125f, 0.1875f, 0.25f, 0.3125f, 0.375f, 0.4375f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 1.0f
};

// Find nearest value in a sorted 16-entry codebook
SD_INLINE float nearestCode(float value, const float* codebook) {
    float best     = codebook[0];
    float bestDist = sd::math::sd_abs<float, float>(value - best);
    for (int i = 1; i < 16; i++) {
        float dist = sd::math::sd_abs<float, float>(value - codebook[i]);
        if (dist < bestDist) {
            bestDist = dist;
            best     = codebook[i];
        }
    }
    return best;
}

// Block-wise NF4 quantization and immediate dequantization
// result has same shape as weight; each block of blockSize elements
// is independently scaled to [-1,1] before nearest-code lookup.
void quantizeNF4(NDArray* weight, int blockSize, NDArray* result) {
    auto* src = weight->bufferAsT<float>();
    auto* dst = result->bufferAsT<float>();
    LongType total = weight->lengthOf();

    int numBlocks = (int)((total + blockSize - 1) / blockSize);
    for (int b = 0; b < numBlocks; b++) {
        LongType start = (LongType)b * blockSize;
        LongType end   = sd::math::sd_min<LongType>(start + blockSize, total);

        // Find absmax in block
        float absMax = 0.0f;
        for (LongType i = start; i < end; i++) {
            float v = sd::math::sd_abs<float, float>(src[i]);
            if (v > absMax) absMax = v;
        }
        if (absMax == 0.0f) {
            for (LongType i = start; i < end; i++) dst[i] = 0.0f;
            continue;
        }
        float scale = absMax;
        for (LongType i = start; i < end; i++) {
            float normalized = src[i] / scale;
            dst[i] = nearestCode(normalized, NF4_CODEBOOK) * scale;
        }
    }
}

// Block-wise FP4 quantization and immediate dequantization
void quantizeFP4(NDArray* weight, int blockSize, NDArray* result) {
    auto* src = weight->bufferAsT<float>();
    auto* dst = result->bufferAsT<float>();
    LongType total = weight->lengthOf();

    int numBlocks = (int)((total + blockSize - 1) / blockSize);
    for (int b = 0; b < numBlocks; b++) {
        LongType start = (LongType)b * blockSize;
        LongType end   = sd::math::sd_min<LongType>(start + blockSize, total);

        float absMax = 0.0f;
        for (LongType i = start; i < end; i++) {
            float v = sd::math::sd_abs<float, float>(src[i]);
            if (v > absMax) absMax = v;
        }
        if (absMax == 0.0f) {
            for (LongType i = start; i < end; i++) dst[i] = 0.0f;
            continue;
        }
        float scale = absMax;
        for (LongType i = start; i < end; i++) {
            float sign       = src[i] >= 0.0f ? 1.0f : -1.0f;
            float normalized = sd::math::sd_abs<float, float>(src[i]) / scale;
            dst[i] = sign * nearestCode(normalized, FP4_CODEBOOK) * scale;
        }
    }
}

}  // anonymous namespace


// ---------------------------------------------------------------------------
// CUSTOM_OP_IMPL
// ---------------------------------------------------------------------------
CUSTOM_OP_IMPL(loftq_init, 1, 2, false, 0, 5) {
    auto weight = INPUT_VARIABLE(0);   // [outDim, inDim]
    auto loraB  = OUTPUT_VARIABLE(0);  // [outDim, rank]
    auto loraA  = OUTPUT_VARIABLE(1);  // [rank, inDim]

    REQUIRE_TRUE(weight->rankOf() == 2, 0,
        "loftq_init: weight must be a 2-D matrix, got rank %d", weight->rankOf());

    const LongType outDim   = weight->sizeAt(0);
    const LongType inDim    = weight->sizeAt(1);
    const LongType diagSize = sd::math::sd_min<LongType>(outDim, inDim);

    const int rank          = INT_ARG(0);
    const int numIterations = block.numI() > 1 ? INT_ARG(1) : 1;
    const int quantBits     = block.numI() > 2 ? INT_ARG(2) : 4;
    const int blockSize     = block.numI() > 3 ? INT_ARG(3) : 64;
    const int quantTypeFlag = block.numI() > 4 ? INT_ARG(4) : 0;  // 0=NF4, 1=FP4

    REQUIRE_TRUE(rank > 0 && rank <= diagSize, 0,
        "loftq_init: rank must be in [1, min(outDim=%lld, inDim=%lld)], got %d",
        (long long)outDim, (long long)inDim, rank);
    REQUIRE_TRUE(numIterations >= 1, 0,
        "loftq_init: numIterations must be >= 1, got %d", numIterations);
    REQUIRE_TRUE(quantBits == 4 || quantBits == 8, 0,
        "loftq_init: quantBits must be 4 or 8, got %d", quantBits);
    REQUIRE_TRUE(blockSize > 0, 0,
        "loftq_init: blockSize must be > 0, got %d", blockSize);

    // Work in FLOAT32
    auto W = weight->dataType() == DataType::FLOAT32
             ? weight->dup('c')
             : weight->cast(DataType::FLOAT32);

    // Initialize B and A to zero for first iteration
    double zero = 0.0;
    loraB->assign(zero);
    loraA->assign(zero);

    // Temporaries for SVD outputs: S [diagSize], U [outDim, diagSize], Vt [diagSize, inDim]
    auto S  = NDArrayFactory::create<float>('c', {diagSize},          weight->getContext());
    auto U  = NDArrayFactory::create<float>('f', {outDim, diagSize},  weight->getContext());
    auto Vt = NDArrayFactory::create<float>('f', {diagSize, inDim},   weight->getContext());

    // Scratch arrays
    auto Q      = NDArrayFactory::create<float>('c', {outDim, inDim}, weight->getContext());
    auto target = NDArrayFactory::create<float>('c', {outDim, inDim}, weight->getContext());
    auto R      = NDArrayFactory::create<float>('c', {outDim, inDim}, weight->getContext());

    for (int iter = 0; iter < numIterations; iter++) {
        // target = W - B@A  (first iter: target = W since B=A=0)
        if (iter == 0) {
            target->assign(W);
        } else {
            // Recompute B@A: create a temporary matrix on the heap so we
            // can pass a pointer to MmulHelper and then delete it explicitly.
            std::vector<sd::LongType> baShape = {outDim, inDim};
            NDArray* BA = new NDArray('c', baShape, DataType::FLOAT32, weight->getContext());
            MmulHelper::mmul(loraB, loraA, BA, 1.0f, 0.0f);
            target->assign(W);
            *target -= *BA;
            delete BA;
        }

        // Quantize target → Q
        if (quantTypeFlag == 0 || quantBits == 4) {
            quantizeNF4(target, blockSize, Q);
        } else {
            quantizeFP4(target, blockSize, Q);
        }

        // R = target - Q
        R->assign(target);
        *R -= *Q;

        // Economy SVD: R = U @ diag(S) @ Vt
        // calcUV=true, fullUV=false (thin SVD), switchNum=0
        helpers::svd(block.launchContext(), R, {S, U, Vt},
                     /*fullUV=*/false, /*calcUV=*/true, /*switchNum=*/0);

        // Extract top-r components and form B, A
        // sqrtS_r[i] = sqrt(S[i])  for i in [0, rank)
        // B[:, i]   = U[:, i] * sqrtS_r[i]
        // A[i, :]   = sqrtS_r[i] * Vt[i, :]
        const float* sPtr  = S->bufferAsT<float>();
        const float* uPtr  = U->bufferAsT<float>();    // F-order: U(row, col) = uPtr[col*outDim + row]
        const float* vtPtr = Vt->bufferAsT<float>();   // F-order: Vt(row, col) = vtPtr[col*diagSize + row]
        float* bPtr  = loraB->bufferAsT<float>(); // C-order: B(row, col) = bPtr[row*rank + col]
        float* aPtr  = loraA->bufferAsT<float>(); // C-order: A(row, col) = aPtr[row*inDim + col]

        for (int r = 0; r < rank; r++) {
            float sqrtS_val = (sPtr[r] > 0.0f) ? sd::math::sd_sqrt<float, float>(sPtr[r]) : 0.0f;

            // B[:, r] = U[:, r] * sqrtS_val
            for (LongType row = 0; row < outDim; row++) {
                // U is F-order [outDim, diagSize]: element (row, r) = uPtr[r*outDim + row]
                bPtr[row * rank + r] = uPtr[(LongType)r * outDim + row] * sqrtS_val;
            }

            // A[r, :] = sqrtS_val * Vt[r, :]
            for (LongType col = 0; col < inDim; col++) {
                // Vt is F-order [diagSize, inDim]: element (r, col) = vtPtr[col*diagSize + r]
                aPtr[(LongType)r * inDim + col] = sqrtS_val * vtPtr[col * diagSize + r];
            }
        }
    }

    // Clean up heap-allocated temporaries
    delete S;
    delete U;
    delete Vt;
    delete Q;
    delete target;
    delete R;
    if (W != weight) delete W;

    return Status::OK;
}


// ---------------------------------------------------------------------------
// DECLARE_TYPES
// ---------------------------------------------------------------------------
DECLARE_TYPES(loftq_init) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, {DataType::FLOAT32, DataType::DOUBLE, DataType::HALF})
        ->setAllowedOutputTypes(0, {DataType::FLOAT32})
        ->setAllowedOutputTypes(1, {DataType::FLOAT32});
}


// ---------------------------------------------------------------------------
// DECLARE_SHAPE_FN
// ---------------------------------------------------------------------------
DECLARE_SHAPE_FN(loftq_init) {
    auto weightShapeInfo = inputShape->at(0);
    REQUIRE_TRUE(shape::rank(weightShapeInfo) == 2, 0,
        "loftq_init: weight must be 2-D, got rank %d", shape::rank(weightShapeInfo));

    const LongType outDim = shape::sizeAt(weightShapeInfo, static_cast<LongType>(0));
    const LongType inDim  = shape::sizeAt(weightShapeInfo, static_cast<LongType>(1));
    const int rank        = INT_ARG(0);

    // loraB: [outDim, rank]
    auto bShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
        DataType::FLOAT32, 'c', {outDim, (LongType)rank});

    // loraA: [rank, inDim]
    auto aShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(
        DataType::FLOAT32, 'c', {(LongType)rank, inDim});

    return SHAPELIST(bShapeInfo, aShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_loftq_init)
