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
// MLIR-accelerated activation functions
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
// relu MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(relu, ENGINE_CPU)

PLATFORM_IMPL(relu, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("relu", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(relu, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR RELU");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(shape::strideDescendingCAscendingF(input->shapeInfo()), "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// leaky_relu MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(lrelu, ENGINE_CPU)

PLATFORM_IMPL(lrelu, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    double alpha = block.numT() > 0 ? T_ARG(0) : 0.01;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("lrelu", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(lrelu, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR LRELU");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// sigmoid MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(sigmoid, ENGINE_CPU)

PLATFORM_IMPL(sigmoid, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("sigmoid", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(sigmoid, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SIGMOID");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(shape::strideDescendingCAscendingF(input->shapeInfo()), "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// tanh MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(tanh, ENGINE_CPU)

PLATFORM_IMPL(tanh, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("tanh", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(tanh, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR TANH");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// softmax MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(softmax, ENGINE_CPU)

PLATFORM_IMPL(softmax, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    int axis = block.numI() > 0 ? INT_ARG(0) : -1;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("softmax", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(softmax, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SOFTMAX");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(shape::strideDescendingCAscendingF(input->shapeInfo()), "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// gelu MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(gelu, ENGINE_CPU)

PLATFORM_IMPL(gelu, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    bool precise = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("gelu", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(gelu, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR GELU");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// silu (swish) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(silu, ENGINE_CPU)

PLATFORM_IMPL(silu, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("silu", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(silu, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SILU");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// elu MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(elu, ENGINE_CPU)

PLATFORM_IMPL(elu, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    double alpha = block.numT() > 0 ? T_ARG(0) : 1.0;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("elu", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(elu, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR ELU");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
