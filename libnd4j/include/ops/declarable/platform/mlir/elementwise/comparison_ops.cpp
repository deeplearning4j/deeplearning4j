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
// MLIR-accelerated comparison and logical operations
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
// equals MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(equals, ENGINE_CPU)

PLATFORM_IMPL(equals, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("equals", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR equals failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(equals, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR EQUALS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(shape::strideDescendingCAscendingF(x->shapeInfo()), "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// not_equals MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(not_equals, ENGINE_CPU)

PLATFORM_IMPL(not_equals, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("not_equals", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR not_equals failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(not_equals, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR NOT_EQUALS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// greater MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(greater, ENGINE_CPU)

PLATFORM_IMPL(greater, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("greater", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR greater failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(greater, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR GREATER");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(shape::strideDescendingCAscendingF(x->shapeInfo()), "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// greater_equal MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(greater_equal, ENGINE_CPU)

PLATFORM_IMPL(greater_equal, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("greater_equal", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR greater_equal failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(greater_equal, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR GREATER_EQUAL");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// less MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(less, ENGINE_CPU)

PLATFORM_IMPL(less, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("less", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR less failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(less, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR LESS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(shape::strideDescendingCAscendingF(x->shapeInfo()), "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// less_equal MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(less_equal, ENGINE_CPU)

PLATFORM_IMPL(less_equal, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("less_equal", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR less_equal failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(less_equal, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR LESS_EQUAL");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// where (select) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(Where, ENGINE_CPU)

PLATFORM_IMPL(Where, ENGINE_CPU) {
    auto* condition = INPUT_VARIABLE(0);  // bool tensor
    auto* x = INPUT_VARIABLE(1);          // values where condition is true
    auto* y = INPUT_VARIABLE(2);          // values where condition is false
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {condition, x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("where", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR where failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(Where, ENGINE_CPU) {
    auto* condition = INPUT_VARIABLE(0);
    auto* x = INPUT_VARIABLE(1);

    Requirements req("MLIR WHERE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// logical_and MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(boolean_and, ENGINE_CPU)

PLATFORM_IMPL(boolean_and, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("logical_and", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR logical_and failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(boolean_and, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR LOGICAL_AND");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// logical_or MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(boolean_or, ENGINE_CPU)

PLATFORM_IMPL(boolean_or, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("logical_or", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR logical_or failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(boolean_or, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR LOGICAL_OR");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// logical_xor MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(boolean_xor, ENGINE_CPU)

PLATFORM_IMPL(boolean_xor, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* y = INPUT_VARIABLE(1);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x, y};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("logical_xor", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR logical_xor failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(boolean_xor, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR LOGICAL_XOR");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// logical_not MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(boolean_not, ENGINE_CPU)

PLATFORM_IMPL(boolean_not, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("logical_not", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR logical_not failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(boolean_not, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR LOGICAL_NOT");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// isnan MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(isnan, ENGINE_CPU)

PLATFORM_IMPL(isnan, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("isnan", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR isnan failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(isnan, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR ISNAN");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Floating point dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// isinf MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(isinf, ENGINE_CPU)

PLATFORM_IMPL(isinf, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("isinf", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR isinf failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(isinf, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR ISINF");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Floating point dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// isfinite MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(isfinite, ENGINE_CPU)

PLATFORM_IMPL(isfinite, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);
    auto* z = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {x};
    std::vector<NDArray*> outputs = {z};

    auto status = executeMlir("isfinite", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR isfinite failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(isfinite, ENGINE_CPU) {
    auto* x = INPUT_VARIABLE(0);

    Requirements req("MLIR ISFINITE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(x->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Floating point dtype") &&
    req.expectTrue(x->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
