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
// MLIR-accelerated unary/elementwise operations
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
// exp MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(exp, ENGINE_CPU)

PLATFORM_IMPL(exp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("exp", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR exp failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(exp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR EXP");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// log MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(log, ENGINE_CPU)

PLATFORM_IMPL(log, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("log", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR log failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(log, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR LOG");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// sqrt MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(sqrt, ENGINE_CPU)

PLATFORM_IMPL(sqrt, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("sqrt", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR sqrt failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(sqrt, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SQRT");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// rsqrt (reciprocal sqrt) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(rsqrt, ENGINE_CPU)

PLATFORM_IMPL(rsqrt, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("rsqrt", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR rsqrt failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(rsqrt, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR RSQRT");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// abs MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(abs, ENGINE_CPU)

PLATFORM_IMPL(abs, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("abs", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR abs failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(abs, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR ABS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// neg MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(neg, ENGINE_CPU)

PLATFORM_IMPL(neg, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("neg", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR neg failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(neg, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR NEG");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// reciprocal MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reciprocal, ENGINE_CPU)

PLATFORM_IMPL(reciprocal, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reciprocal", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reciprocal failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reciprocal, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR RECIPROCAL");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// square MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(square, ENGINE_CPU)

PLATFORM_IMPL(square, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("square", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR square failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(square, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SQUARE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// sin MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(sin, ENGINE_CPU)

PLATFORM_IMPL(sin, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("sin", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR sin failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(sin, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SIN");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// cos MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(cos, ENGINE_CPU)

PLATFORM_IMPL(cos, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("cos", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR cos failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(cos, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR COS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// floor MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(Floor, ENGINE_CPU)

PLATFORM_IMPL(Floor, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("floor", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR floor failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(Floor, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR FLOOR");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// ceil MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(ceil, ENGINE_CPU)

PLATFORM_IMPL(ceil, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("ceil", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR ceil failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(ceil, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR CEIL");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// round MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(Round, ENGINE_CPU)

PLATFORM_IMPL(Round, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("round", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR round failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(Round, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR ROUND");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// erf MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(erf, ENGINE_CPU)

PLATFORM_IMPL(erf, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("erf", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR erf failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(erf, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR ERF");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
