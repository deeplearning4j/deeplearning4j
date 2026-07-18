/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// MLIR-accelerated attention operations
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>
#include <ops/declarable/platform/mlir/mlirUtils.h>

#if defined(HAVE_MLIR)

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// dot_product_attention MLIR implementation
// Computes scaled dot-product attention: softmax(Q * K^T / sqrt(dk)) * V
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(dot_product_attention, ENGINE_CPU)

PLATFORM_IMPL(dot_product_attention, ENGINE_CPU) {
    auto* queries = INPUT_VARIABLE(0);   // [batch, seqLenQ, depth] or [batch, heads, seqLenQ, depth]
    auto* keys = INPUT_VARIABLE(1);      // [batch, seqLenK, depth] or [batch, heads, seqLenK, depth]
    auto* values = INPUT_VARIABLE(2);    // [batch, seqLenK, depthV] or [batch, heads, seqLenK, depthV]
    auto* mask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;  // optional attention mask

    auto* output = OUTPUT_VARIABLE(0);   // attention output
    auto* weights = block.outputWidth() > 1 ? OUTPUT_VARIABLE(1) : nullptr; // optional attention weights

    // Scale factor (usually 1/sqrt(dk))
    double scale = block.numT() > 0 ? T_ARG(0) : 0.0;  // 0 means auto-compute
    bool useCausalMask = block.numB() > 0 ? B_ARG(0) : false;
    double dropout = block.numT() > 1 ? T_ARG(1) : 0.0;

    std::vector<NDArray*> inputs = {queries, keys, values};
    if (mask != nullptr) inputs.push_back(mask);

    std::vector<NDArray*> outputs = {output};
    if (weights != nullptr) outputs.push_back(weights);

    auto status = executeMlirEx("dot_product_attention", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(dot_product_attention, ENGINE_CPU) {
    auto* queries = INPUT_VARIABLE(0);
    auto* keys = INPUT_VARIABLE(1);
    auto* values = INPUT_VARIABLE(2);

    Requirements req("MLIR DOT_PRODUCT_ATTENTION");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(queries->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(queries->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// dot_product_attention_bp (backprop) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(dot_product_attention_bp, ENGINE_CPU)

PLATFORM_IMPL(dot_product_attention_bp, ENGINE_CPU) {
    auto* queries = INPUT_VARIABLE(0);
    auto* keys = INPUT_VARIABLE(1);
    auto* values = INPUT_VARIABLE(2);
    auto* gradOut = INPUT_VARIABLE(3);  // gradient from output
    auto* mask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

    auto* gradQ = OUTPUT_VARIABLE(0);  // gradient w.r.t. queries
    auto* gradK = OUTPUT_VARIABLE(1);  // gradient w.r.t. keys
    auto* gradV = OUTPUT_VARIABLE(2);  // gradient w.r.t. values

    double scale = block.numT() > 0 ? T_ARG(0) : 0.0;
    bool useCausalMask = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {queries, keys, values, gradOut};
    if (mask != nullptr) inputs.push_back(mask);

    std::vector<NDArray*> outputs = {gradQ, gradK, gradV};

    auto status = executeMlir("dot_product_attention_bp", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(dot_product_attention_bp, ENGINE_CPU) {
    auto* queries = INPUT_VARIABLE(0);

    Requirements req("MLIR DOT_PRODUCT_ATTENTION_BP");

    // Backward ops fall through to native C++ — no MLIR backward support
    req.expectTrue(false, "Backward ops use native C++ implementation");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// multi_head_attention MLIR implementation
// Full multi-head attention with Q/K/V projections
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(multi_head_attention, ENGINE_CPU)

PLATFORM_IMPL(multi_head_attention, ENGINE_CPU) {
    auto* queries = INPUT_VARIABLE(0);    // [batch, seqLen, embedDim]
    auto* keys = INPUT_VARIABLE(1);       // [batch, seqLen, embedDim]
    auto* values = INPUT_VARIABLE(2);     // [batch, seqLen, embedDim]
    auto* Wq = INPUT_VARIABLE(3);         // query projection weights
    auto* Wk = INPUT_VARIABLE(4);         // key projection weights
    auto* Wv = INPUT_VARIABLE(5);         // value projection weights
    auto* Wo = INPUT_VARIABLE(6);         // output projection weights
    auto* mask = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;
    auto* bq = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;  // query bias
    auto* bk = block.width() > 9 ? INPUT_VARIABLE(9) : nullptr;  // key bias
    auto* bv = block.width() > 10 ? INPUT_VARIABLE(10) : nullptr; // value bias
    auto* bo = block.width() > 11 ? INPUT_VARIABLE(11) : nullptr; // output bias

    auto* output = OUTPUT_VARIABLE(0);
    auto* weights = block.outputWidth() > 1 ? OUTPUT_VARIABLE(1) : nullptr;

    int numHeads = INT_ARG(0);
    double dropout = block.numT() > 0 ? T_ARG(0) : 0.0;
    bool useCausalMask = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {queries, keys, values, Wq, Wk, Wv, Wo};
    if (mask != nullptr) inputs.push_back(mask);
    if (bq != nullptr) inputs.push_back(bq);
    if (bk != nullptr) inputs.push_back(bk);
    if (bv != nullptr) inputs.push_back(bv);
    if (bo != nullptr) inputs.push_back(bo);

    std::vector<NDArray*> outputs = {output};
    if (weights != nullptr) outputs.push_back(weights);

    auto status = executeMlirEx("multi_head_attention", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(multi_head_attention, ENGINE_CPU) {
    auto* queries = INPUT_VARIABLE(0);
    auto* Wq = INPUT_VARIABLE(3);

    Requirements req("MLIR MULTI_HEAD_ATTENTION");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(queries->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(queries->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// multi_head_attention_bp (backprop) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(multi_head_attention_bp, ENGINE_CPU)

PLATFORM_IMPL(multi_head_attention_bp, ENGINE_CPU) {
    auto* queries = INPUT_VARIABLE(0);
    auto* keys = INPUT_VARIABLE(1);
    auto* values = INPUT_VARIABLE(2);
    auto* Wq = INPUT_VARIABLE(3);
    auto* Wk = INPUT_VARIABLE(4);
    auto* Wv = INPUT_VARIABLE(5);
    auto* Wo = INPUT_VARIABLE(6);
    auto* gradOut = INPUT_VARIABLE(7);
    auto* mask = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;

    auto* gradQ = OUTPUT_VARIABLE(0);
    auto* gradK = OUTPUT_VARIABLE(1);
    auto* gradV = OUTPUT_VARIABLE(2);
    auto* gradWq = OUTPUT_VARIABLE(3);
    auto* gradWk = OUTPUT_VARIABLE(4);
    auto* gradWv = OUTPUT_VARIABLE(5);
    auto* gradWo = OUTPUT_VARIABLE(6);

    int numHeads = INT_ARG(0);

    std::vector<NDArray*> inputs = {queries, keys, values, Wq, Wk, Wv, Wo, gradOut};
    if (mask != nullptr) inputs.push_back(mask);

    std::vector<NDArray*> outputs = {gradQ, gradK, gradV, gradWq, gradWk, gradWv, gradWo};

    auto status = executeMlir("multi_head_attention_bp", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(multi_head_attention_bp, ENGINE_CPU) {
    auto* queries = INPUT_VARIABLE(0);

    Requirements req("MLIR MULTI_HEAD_ATTENTION_BP");

    // Backward ops fall through to native C++ — no MLIR backward support
    req.expectTrue(false, "Backward ops use native C++ implementation");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// self_attention MLIR implementation
// Simplified self-attention where Q=K=V
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(self_attention, ENGINE_CPU)

PLATFORM_IMPL(self_attention, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);      // [batch, seqLen, embedDim]
    auto* Wqkv = INPUT_VARIABLE(1);       // combined Q/K/V projection
    auto* Wo = INPUT_VARIABLE(2);         // output projection
    auto* mask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;

    auto* output = OUTPUT_VARIABLE(0);

    int numHeads = INT_ARG(0);
    double dropout = block.numT() > 0 ? T_ARG(0) : 0.0;
    bool useCausalMask = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {input, Wqkv, Wo};
    if (mask != nullptr) inputs.push_back(mask);

    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("self_attention", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(self_attention, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SELF_ATTENTION");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
