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
// MLIR-accelerated shape manipulation operations
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
// reshape MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reshape, ENGINE_CPU)

PLATFORM_IMPL(reshape, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    // New shape can come from input variable or int args
    std::vector<LongType> newShape;
    if (block.width() > 1) {
        auto* shapeArr = INPUT_VARIABLE(1);
        for (LongType i = 0; i < shapeArr->lengthOf(); i++) {
            newShape.push_back(shapeArr->e<LongType>(i));
        }
    } else {
        for (int i = 0; i < block.numI(); i++) {
            newShape.push_back(INT_ARG(i));
        }
    }

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reshape", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reshape failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reshape, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR RESHAPE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// transpose / permute MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(transpose, ENGINE_CPU)

PLATFORM_IMPL(transpose, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    // Permutation order
    std::vector<LongType> permutation;
    if (block.width() > 1) {
        auto* permArr = INPUT_VARIABLE(1);
        for (LongType i = 0; i < permArr->lengthOf(); i++) {
            permutation.push_back(permArr->e<LongType>(i));
        }
    } else if (block.numI() > 0) {
        for (int i = 0; i < block.numI(); i++) {
            permutation.push_back(INT_ARG(i));
        }
    }

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("transpose", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR transpose failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(transpose, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR TRANSPOSE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// concat MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(concat, ENGINE_CPU)

PLATFORM_IMPL(concat, ENGINE_CPU) {
    auto* output = OUTPUT_VARIABLE(0);

    // Collect all input arrays
    std::vector<NDArray*> inputs;
    for (int i = 0; i < block.width() - 1; i++) {  // Last input may be axis
        inputs.push_back(INPUT_VARIABLE(i));
    }

    // Axis along which to concatenate
    int axis = 0;
    if (block.numI() > 0) {
        axis = INT_ARG(0);
    } else if (block.width() > 1) {
        // Check if last input is scalar (axis)
        auto* lastInput = INPUT_VARIABLE(block.width() - 1);
        if (lastInput->isScalar()) {
            axis = lastInput->e<int>(0);
            inputs.pop_back();
        }
    }

    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("concat", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR concat failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(concat, ENGINE_CPU) {
    auto* input0 = INPUT_VARIABLE(0);

    Requirements req("MLIR CONCAT");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input0->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(input0->lengthOf() >= MLIR_MIN_TENSOR_SIZE / 4, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// slice / strided_slice MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(slice, ENGINE_CPU)

PLATFORM_IMPL(slice, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    // Begin indices and sizes
    std::vector<LongType> begin, size;

    if (block.width() > 2) {
        auto* beginArr = INPUT_VARIABLE(1);
        auto* sizeArr = INPUT_VARIABLE(2);
        for (LongType i = 0; i < beginArr->lengthOf(); i++) {
            begin.push_back(beginArr->e<LongType>(i));
        }
        for (LongType i = 0; i < sizeArr->lengthOf(); i++) {
            size.push_back(sizeArr->e<LongType>(i));
        }
    }

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("slice", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR slice failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(slice, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SLICE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// strided_slice MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(strided_slice, ENGINE_CPU)

PLATFORM_IMPL(strided_slice, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("strided_slice", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR strided_slice failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(strided_slice, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR STRIDED_SLICE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// gather MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(gather, ENGINE_CPU)

PLATFORM_IMPL(gather, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* indices = INPUT_VARIABLE(1);
    auto* output = OUTPUT_VARIABLE(0);

    int axis = block.numI() > 0 ? INT_ARG(0) : 0;

    std::vector<NDArray*> inputs = {input, indices};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("gather", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR gather failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(gather, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* indices = INPUT_VARIABLE(1);

    Requirements req("MLIR GATHER");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectIn(indices->dataType(), {INT32, INT64}, "Index dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// scatter_update MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(scatter_update, ENGINE_CPU)

PLATFORM_IMPL(scatter_update, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* indices = INPUT_VARIABLE(1);
    auto* updates = INPUT_VARIABLE(2);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {input, indices, updates};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("scatter_update", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR scatter_update failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(scatter_update, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SCATTER_UPDATE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// tile MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(tile, ENGINE_CPU)

PLATFORM_IMPL(tile, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    // Repetitions per dimension
    std::vector<LongType> reps;
    if (block.width() > 1) {
        auto* repsArr = INPUT_VARIABLE(1);
        for (LongType i = 0; i < repsArr->lengthOf(); i++) {
            reps.push_back(repsArr->e<LongType>(i));
        }
    } else {
        for (int i = 0; i < block.numI(); i++) {
            reps.push_back(INT_ARG(i));
        }
    }

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("tile", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR tile failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(tile, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR TILE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE / 4, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// split MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(split, ENGINE_CPU)

PLATFORM_IMPL(split, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    int numSplits = block.numOutputs();
    int axis = block.numI() > 0 ? INT_ARG(0) : 0;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs;
    for (int i = 0; i < numSplits; i++) {
        outputs.push_back(OUTPUT_VARIABLE(i));
    }

    auto status = executeMlir("split", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR split failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(split, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR SPLIT");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// stack MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(stack, ENGINE_CPU)

PLATFORM_IMPL(stack, ENGINE_CPU) {
    auto* output = OUTPUT_VARIABLE(0);

    int axis = block.numI() > 0 ? INT_ARG(0) : 0;

    std::vector<NDArray*> inputs;
    for (int i = 0; i < block.width(); i++) {
        inputs.push_back(INPUT_VARIABLE(i));
    }

    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("stack", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR stack failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(stack, ENGINE_CPU) {
    auto* input0 = INPUT_VARIABLE(0);

    Requirements req("MLIR STACK");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(input0->lengthOf() >= MLIR_MIN_TENSOR_SIZE / 4, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// unstack MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(unstack, ENGINE_CPU)

PLATFORM_IMPL(unstack, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    int axis = block.numI() > 0 ? INT_ARG(0) : 0;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs;
    for (int i = 0; i < block.numOutputs(); i++) {
        outputs.push_back(OUTPUT_VARIABLE(i));
    }

    auto status = executeMlir("unstack", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR unstack failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(unstack, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR UNSTACK");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
