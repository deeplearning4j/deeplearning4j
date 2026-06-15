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
// CUDA implementations of quantization operations using GGML kernels
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "../llamacppUtils.h"

#if HAVE_LLAMACPP && defined(GGML_USE_CUDA)

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// QUANTIZE_Q4_0 - CUDA
PLATFORM_IMPL(quantize_q4_0, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    int64_t ne[4] = {input->sizeAt(-1), 1, 1, 1};
    for (int i = 1; i < input->rankOf(); i++) {
        ne[i] = input->sizeAt(i - 1);
    }

    struct ggml_tensor* ggml_output = ggml_new_tensor(ctx, GGML_TYPE_Q4_0, input->rankOf(), ne);
    struct ggml_tensor* ggml_quantized = ggml_cpy(ctx, ggml_input, ggml_output);
    ggml_set_name(ggml_quantized, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_quantized);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_quantized, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(quantize_q4_0, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA QUANTIZE_Q4_0 OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input] {
        return llamacppUtils::isSupportedType(input->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasQuantizedSupport(), "quantization support"), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// QUANTIZE_Q8_0 - CUDA
PLATFORM_IMPL(quantize_q8_0, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    int64_t ne[4] = {input->sizeAt(-1), 1, 1, 1};
    for (int i = 1; i < input->rankOf(); i++) {
        ne[i] = input->sizeAt(i - 1);
    }

    struct ggml_tensor* ggml_output = ggml_new_tensor(ctx, GGML_TYPE_Q8_0, input->rankOf(), ne);
    struct ggml_tensor* ggml_quantized = ggml_cpy(ctx, ggml_input, ggml_output);
    ggml_set_name(ggml_quantized, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_quantized);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_quantized, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(quantize_q8_0, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA QUANTIZE_Q8_0 OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input] {
        return llamacppUtils::isSupportedType(input->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasQuantizedSupport(), "quantization support"), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// DEQUANTIZE - CUDA
PLATFORM_IMPL(dequantize, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(64 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    int64_t ne[4] = {input->sizeAt(-1), 1, 1, 1};
    for (int i = 1; i < input->rankOf(); i++) {
        ne[i] = input->sizeAt(i - 1);
    }

    struct ggml_tensor* ggml_output = ggml_new_tensor(ctx, GGML_TYPE_F32, input->rankOf(), ne);
    struct ggml_tensor* ggml_dequantized = ggml_cpy(ctx, ggml_input, ggml_output);
    ggml_set_name(ggml_dequantized, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_dequantized);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_dequantized, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(dequantize, ENGINE_CUDA) {
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA DEQUANTIZE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([output] {
        return llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasQuantizedSupport(), "quantization support"), NO_MSG);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// QUANTIZED_MUL_MAT - CUDA
PLATFORM_IMPL(quantized_mul_mat, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty() || weights->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(128 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_weights = llamacppUtils::createGgmlTensorCuda(ctx, weights, ctx.getBackend(), "weights");

    struct ggml_tensor* ggml_output = ggml_mul_mat(ctx, ggml_weights, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(quantized_mul_mat, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA QUANTIZED_MUL_MAT OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasQuantizedSupport(), "quantization support"), NO_MSG);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// SCALE - CUDA
PLATFORM_IMPL(scale, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float scale = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0f;

    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");
    struct ggml_tensor* ggml_output = ggml_scale(ctx, ggml_input, scale);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, output, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(scale, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    Requirements req("LLAMACPP CUDA SCALE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([input, output] {
        return llamacppUtils::isSupportedType(input->dataType()) &&
               llamacppUtils::isSupportedType(output->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// ADD_INPLACE - CUDA
PLATFORM_IMPL(add_inplace, ENGINE_CUDA) {
    auto accumulator = INPUT_VARIABLE(0);
    auto input = INPUT_VARIABLE(1);

    if (accumulator->isEmpty() || input->isEmpty()) return sd::Status::OK;

    llamacppUtils::GgmlCudaContextGuard ctx(32 * 1024 * 1024);

    struct ggml_tensor* ggml_acc = llamacppUtils::createGgmlTensorCuda(ctx, accumulator, ctx.getBackend(), "accumulator");
    struct ggml_tensor* ggml_input = llamacppUtils::createGgmlTensorCuda(ctx, input, ctx.getBackend(), "input");

    struct ggml_tensor* ggml_output = ggml_add_inplace(ctx, ggml_acc, ggml_input);
    ggml_set_name(ggml_output, "output");

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_output);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    llamacppUtils::copyGgmlCudaToNDArray(ggml_output, accumulator, ctx.getBackend());
    return sd::Status::OK;
}

PLATFORM_CHECK(add_inplace, ENGINE_CUDA) {
    auto accumulator = INPUT_VARIABLE(0);
    auto input = INPUT_VARIABLE(1);

    Requirements req("LLAMACPP CUDA ADD_INPLACE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable([accumulator, input] {
        return llamacppUtils::isSupportedType(accumulator->dataType()) &&
               llamacppUtils::isSupportedType(input->dataType());
    }, TYPECHECK_MSG), NO_MSG);
    req.expectLessEq(makeInfoVariable(accumulator->rankOf(), RANK_MSG_INPUT0), 4);
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// GGML_DEQUANTIZE - CUDA (llamacpp accelerated path)

static ggml_type dequantTypeToGgmlCuda(int quantType) {
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
        case 11: return GGML_TYPE_Q8_K;
        case 12: return GGML_TYPE_IQ2_XXS;
        case 13: return GGML_TYPE_IQ2_XS;
        case 14: return GGML_TYPE_IQ3_XXS;
        case 15: return GGML_TYPE_IQ1_S;
        case 16: return GGML_TYPE_IQ4_NL;
        case 17: return GGML_TYPE_IQ3_S;
        case 18: return GGML_TYPE_IQ2_S;
        case 19: return GGML_TYPE_IQ4_XS;
        case 20: return GGML_TYPE_IQ1_M;
        case 21: return GGML_TYPE_TQ1_0;
        case 22: return GGML_TYPE_TQ2_0;
        default: return GGML_TYPE_Q4_0;
    }
}

static ggml_type outputDtypeArgToGgmlCuda(int dtypeArg) {
    switch (dtypeArg) {
        case 0: return GGML_TYPE_F32;
        case 1: return GGML_TYPE_F16;
        case 2: return GGML_TYPE_BF16;
        default: return GGML_TYPE_F32;
    }
}

PLATFORM_IMPL(ggml_dequantize, ENGINE_CUDA) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int quantType = INT_ARG(0);
    int outputDtypeArg = INT_ARG(1);

    ggml_type qtype = dequantTypeToGgmlCuda(quantType);
    ggml_type outGgmlType = outputDtypeArgToGgmlCuda(outputDtypeArg);
    sd::LongType numElements = output->lengthOf();

    size_t rawBytes = input->lengthOf();
    size_t outputBytes = numElements * ggml_type_size(outGgmlType);
    size_t ctxSize = rawBytes + outputBytes + 256 * 1024 * 1024;
    llamacppUtils::GgmlCudaContextGuard ctx(ctxSize);

    struct ggml_tensor* ggml_src = ggml_new_tensor_1d(ctx, qtype, numElements);
    ggml_set_name(ggml_src, "quantized_src");

    // Copy raw bytes from device to GGML tensor via host staging
    size_t expectedBytes = ggml_nbytes(ggml_src);
    size_t availableBytes = input->lengthOf();
    size_t copyBytes = std::min(expectedBytes, availableBytes);

    // Need to get raw bytes to host first, then into GGML buffer
    std::vector<uint8_t> hostBuf(copyBytes);
    cudaMemcpy(hostBuf.data(), input->specialBuffer(), copyBytes, cudaMemcpyDeviceToHost);
    memcpy(ggml_src->data, hostBuf.data(), copyBytes);

    // Upload to CUDA backend
    ggml_backend_tensor_set(ggml_src, hostBuf.data(), 0, copyBytes);

    struct ggml_tensor* ggml_dst = ggml_new_tensor_1d(ctx, outGgmlType, numElements);
    ggml_set_name(ggml_dst, "dequantized_dst");

    struct ggml_tensor* ggml_result = ggml_cpy(ctx, ggml_src, ggml_dst);

    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, ggml_result);
    llamacppUtils::executeGgmlGraphCuda(ctx, graph, ctx.getBackend());

    // Copy result back: GGML CUDA -> host -> NDArray special buffer
    size_t outBytes = numElements * ggml_type_size(outGgmlType);
    std::vector<uint8_t> hostOut(outBytes);
    ggml_backend_tensor_get(ggml_dst, hostOut.data(), 0, outBytes);
    cudaMemcpy(output->specialBuffer(), hostOut.data(), outBytes, cudaMemcpyHostToDevice);

    return sd::Status::OK;
}

PLATFORM_CHECK(ggml_dequantize, ENGINE_CUDA) {
    Requirements req("LLAMACPP CUDA GGML_DEQUANTIZE OP");
    req.expectTrue(block.isUseLLAMACPP(), IS_USE_LLAMACPP_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasCudaBackend(), "CUDA backend available"), NO_MSG);
    req.expectTrue(makeInfoVariable(llamacppUtils::hasQuantizedSupport(), "quantization support"), NO_MSG);
    req.logTheSuccess();
    return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_LLAMACPP && GGML_USE_CUDA
