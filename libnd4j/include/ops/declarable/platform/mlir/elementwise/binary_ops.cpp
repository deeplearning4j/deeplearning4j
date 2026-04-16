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
// MLIR-accelerated binary/elementwise operations
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
// add MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(add, ENGINE_CPU)

PLATFORM_IMPL(add, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("add", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR add failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(add, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);

    Requirements req("MLIR ADD");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(x->dataType() == y->dataType(), "Same dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(shape::strideDescendingCAscendingF(x->shapeInfo()), "X contiguous") &&
    req.expectTrue(shape::strideDescendingCAscendingF(y->shapeInfo()), "Y contiguous");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// subtract MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(subtract, ENGINE_CPU)

PLATFORM_IMPL(subtract, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("subtract", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR subtract failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(subtract, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);

    Requirements req("MLIR SUBTRACT");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(x->dataType() == y->dataType(), "Same dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// multiply MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(multiply, ENGINE_CPU)

PLATFORM_IMPL(multiply, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("multiply", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR multiply failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(multiply, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);

    Requirements req("MLIR MULTIPLY");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(x->dataType() == y->dataType(), "Same dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(shape::strideDescendingCAscendingF(x->shapeInfo()), "X contiguous") &&
    req.expectTrue(shape::strideDescendingCAscendingF(y->shapeInfo()), "Y contiguous");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// divide MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(divide, ENGINE_CPU)

PLATFORM_IMPL(divide, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("divide", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR divide failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(divide, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);

    Requirements req("MLIR DIVIDE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(x->dataType() == y->dataType(), "Same dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// maximum MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(maximum, ENGINE_CPU)

PLATFORM_IMPL(maximum, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("maximum", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR maximum failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(maximum, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);

    Requirements req("MLIR MAXIMUM");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(x->dataType() == y->dataType(), "Same dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// minimum MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(minimum, ENGINE_CPU)

PLATFORM_IMPL(minimum, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("minimum", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR minimum failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(minimum, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);

    Requirements req("MLIR MINIMUM");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(x->dataType() == y->dataType(), "Same dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// pow MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(Pow, ENGINE_CPU)

PLATFORM_IMPL(Pow, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("pow", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR pow failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(Pow, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR POW");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// squared_difference MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(squaredsubtract, ENGINE_CPU)

PLATFORM_IMPL(squaredsubtract, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("squared_subtract", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR squared_subtract failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(squaredsubtract, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR SQUARED_SUBTRACT");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
