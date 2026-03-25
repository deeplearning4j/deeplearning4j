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
// Quantization and dequantization operations using GGML kernels
// Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, and other GGML quantization formats
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "llamacppUtils.h"

#if HAVE_LLAMACPP

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// QUANTIZE_Q4_0 - Quantize to 4-bit format (type 0)
PLATFORM_IMPL(quantize_q4_0, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    // Create quantized tensor
    int64_t ne[4] = {input->sizeAt(-1), 1, 1, 1};
    for (int i = 1; i < input->rankOf(); i++) {
        ne[i] = input->sizeAt(i - 1);
    }

    struct ggml_tensor* ggml_output = ggml_new_tensor(ctx, GGML_TYPE_Q4_0, input->rankOf(), ne);
    ggml_set_name(ggml_output, "output");

    // Copy and quantize
    struct ggml_tensor* ggml_quantized = ggml_cpy(ctx, ggml_input, ggml_output);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_quantized);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_quantized, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(quantize_q4_0, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP QUANTIZE_Q4_0 OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input] {
        return llamacppUtils::isSupportedType(input->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasQuantizedSupport(), "quantization support"), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// QUANTIZE_Q8_0 - Quantize to 8-bit format
PLATFORM_IMPL(quantize_q8_0, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    int64_t ne[4] = {input->sizeAt(-1), 1, 1, 1};
    for (int i = 1; i < input->rankOf(); i++) {
        ne[i] = input->sizeAt(i - 1);
    }

    struct ggml_tensor* ggml_output = ggml_new_tensor(ctx, GGML_TYPE_Q8_0, input->rankOf(), ne);
    ggml_set_name(ggml_output, "output");

    struct ggml_tensor* ggml_quantized = ggml_cpy(ctx, ggml_input, ggml_output);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_quantized);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_quantized, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(quantize_q8_0, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP QUANTIZE_Q8_0 OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input] {
        return llamacppUtils::isSupportedType(input->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasQuantizedSupport(), "quantization support"), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// DEQUANTIZE - Dequantize from any quantized format to float
PLATFORM_IMPL(dequantize, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    // Create float output tensor
    int64_t ne[4] = {input->sizeAt(-1), 1, 1, 1};
    for (int i = 1; i < input->rankOf(); i++) {
        ne[i] = input->sizeAt(i - 1);
    }

    struct ggml_tensor* ggml_output = ggml_new_tensor(ctx, GGML_TYPE_F32, input->rankOf(), ne);
    ggml_set_name(ggml_output, "output");

    struct ggml_tensor* ggml_dequantized = ggml_cpy(ctx, ggml_input, ggml_output);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_dequantized);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_dequantized, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(dequantize, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP DEQUANTIZE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([output] {
        return llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasQuantizedSupport(), "quantization support"), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// QUANTIZED_MUL_MAT - Matrix multiplication with quantized weights
static void quantizedMulMatLlamaCpp(NDArray* input, NDArray* quantizedWeights, NDArray* output) {
    llamacppUtils::GgmlContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");
    struct ggml_tensor* ggml_weights = llamacppUtils::createGgmlTensor(ctx, quantizedWeights, "weights");

    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_weights, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(quantized_mul_mat, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty() || weights->isEmpty()) return sd::Status::OK;

    quantizedMulMatLlamaCpp(input, weights, output);
    return sd::Status::OK;
}

PLATFORM_CHECK(quantized_mul_mat, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP QUANTIZED_MUL_MAT OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasQuantizedSupport(), "quantization support"), NO_MSG);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SCALE - Scale tensor by scalar value
static void scaleLlamaCpp(NDArray* input, NDArray* output, float scale) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_scale(ctx, ggml_input, scale);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, output);
}

PLATFORM_IMPL(scale, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0f;

    scaleLlamaCpp(input, output, scale);
    return sd::Status::OK;
}

PLATFORM_CHECK(scale, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP SCALE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ADD_INPLACE - In-place addition (accumulator pattern)
static void addInplaceLlamaCpp(NDArray* accumulator, NDArray* input) {
    llamacppUtils::GgmlContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_acc = llamacppUtils::createGgmlTensor(ctx, accumulator, "accumulator");
    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensor(ctx, input, "input");

    struct ggml_tensor* ggml_output = ggml_add_inplace(ctx, ggml_acc, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    llamacppUtils::copyGgmlToNDArray(ggml_output, accumulator);
}

PLATFORM_IMPL(add_inplace, ENGINE_CPU) {
    auto accumulator = INPUT_VARIABLE(0);
    auto input = INPUT_VARIABLE(1);

    if (accumulator->isEmpty() || input->isEmpty()) return sd::Status::OK;

    addInplaceLlamaCpp(accumulator, input);
    return sd::Status::OK;
}

PLATFORM_CHECK(add_inplace, ENGINE_CPU) {
    auto accumulator = INPUT_VARIABLE(0);
    auto input = INPUT_VARIABLE(1);

    Requirements req("LLAMACPP ADD_INPLACE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable([accumulator, input] {
        return llamacppUtils::isSupportedType(accumulator->dataType()) &&
               llamacppUtils::isSupportedType(input->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(accumulator->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// GGML_DEQUANTIZE - Dequantize raw GGML quantized bytes using llamacpp backend
// Platform accelerated path - used when HAVE_LLAMACPP is available

static ggml_type dequantTypeToGgml(int quantType) {
    switch (quantType) {
        case 0: return GGML_TYPE_Q4_0;
        case 1: return GGML_TYPE_Q4_1;
        case 2: return GGML_TYPE_Q5_0;
        case 3: return GGML_TYPE_Q5_1;
        case 4: return GGML_TYPE_Q8_0;
        case 5: return GGML_TYPE_Q8_1;
        case 6: return GGML_TYPE_Q2_K;
        case 7: return GGML_TYPE_Q3_K;
        case 8: return GGML_TYPE_Q4_K;
        case 9: return GGML_TYPE_Q5_K;
        case 10: return GGML_TYPE_Q6_K;
        default: return GGML_TYPE_Q4_0;
    }
}

static ggml_type outputDtypeArgToGgml(int dtypeArg) {
    switch (dtypeArg) {
        case 0: return GGML_TYPE_F32;
        case 1: return GGML_TYPE_F16;
        case 2: return GGML_TYPE_BF16;
        default: return GGML_TYPE_F32;
    }
}

PLATFORM_IMPL(ggml_dequantize, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int quantType = INT_ARG(0);
    int outputDtypeArg = INT_ARG(1);

    ggml_type qtype = dequantTypeToGgml(quantType);
    ggml_type outGgmlType = outputDtypeArgToGgml(outputDtypeArg);
    sd::LongType numElements = output->lengthOf();

    size_t rawBytes = input->lengthOf();
    size_t outputBytes = numElements * ggml_type_size(outGgmlType);
    size_t ctxSize = rawBytes + outputBytes + 256 * 1024 * 1024;
    llamacppUtils::GgmlContextGuard ctx(ctxSize);

    struct ggml_tensor* ggml_src = ggml_new_tensor_1d(ctx, qtype, numElements);
    ggml_set_name(ggml_src, "quantized_src");

    size_t expectedBytes = ggml_nbytes(ggml_src);
    size_t availableBytes = input->lengthOf();
    size_t copyBytes = std::min(expectedBytes, availableBytes);
    memcpy(ggml_src->data, input->buffer(), copyBytes);

    struct ggml_tensor* ggml_dst = ggml_new_tensor_1d(ctx, outGgmlType, numElements);
    ggml_set_name(ggml_dst, "dequantized_dst");

    struct ggml_tensor* ggml_result = ggml_cpy(ctx, ggml_src, ggml_dst);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraph(ctx, graph);

    memcpy(output->buffer(), ggml_dst->data, output->lengthOf() * output->sizeOfT());

    return sd::Status::OK;
}

PLATFORM_CHECK(ggml_dequantize, ENGINE_CPU) {
    Requirements req("LLAMACPP GGML_DEQUANTIZE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasQuantizedSupport(), "quantization support"), NO_MSG);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP
