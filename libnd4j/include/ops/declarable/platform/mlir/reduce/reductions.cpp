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
// MLIR-accelerated reduction operations
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
// reduce_sum MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reduce_sum, ENGINE_CPU)

PLATFORM_IMPL(reduce_sum, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    // Get dimensions to reduce along
    std::vector<LongType> dimensions;
    if (block.width() > 1) {
        auto* dims = INPUT_VARIABLE(1);
        for (LongType i = 0; i < dims->lengthOf(); i++) {
            dimensions.push_back(dims->e<LongType>(i));
        }
    } else if (block.numI() > 0) {
        for (int i = 0; i < block.numI(); i++) {
            dimensions.push_back(INT_ARG(i));
        }
    }

    bool keepDims = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reduce_sum", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reduce_sum failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reduce_sum, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR REDUCE_SUM");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_mean MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reduce_mean, ENGINE_CPU)

PLATFORM_IMPL(reduce_mean, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<LongType> dimensions;
    if (block.width() > 1) {
        auto* dims = INPUT_VARIABLE(1);
        for (LongType i = 0; i < dims->lengthOf(); i++) {
            dimensions.push_back(dims->e<LongType>(i));
        }
    } else if (block.numI() > 0) {
        for (int i = 0; i < block.numI(); i++) {
            dimensions.push_back(INT_ARG(i));
        }
    }

    bool keepDims = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reduce_mean", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reduce_mean failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reduce_mean, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR REDUCE_MEAN");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_max MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reduce_max, ENGINE_CPU)

PLATFORM_IMPL(reduce_max, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<LongType> dimensions;
    if (block.width() > 1) {
        auto* dims = INPUT_VARIABLE(1);
        for (LongType i = 0; i < dims->lengthOf(); i++) {
            dimensions.push_back(dims->e<LongType>(i));
        }
    } else if (block.numI() > 0) {
        for (int i = 0; i < block.numI(); i++) {
            dimensions.push_back(INT_ARG(i));
        }
    }

    bool keepDims = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reduce_max", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reduce_max failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reduce_max, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR REDUCE_MAX");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_min MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reduce_min, ENGINE_CPU)

PLATFORM_IMPL(reduce_min, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<LongType> dimensions;
    if (block.width() > 1) {
        auto* dims = INPUT_VARIABLE(1);
        for (LongType i = 0; i < dims->lengthOf(); i++) {
            dimensions.push_back(dims->e<LongType>(i));
        }
    } else if (block.numI() > 0) {
        for (int i = 0; i < block.numI(); i++) {
            dimensions.push_back(INT_ARG(i));
        }
    }

    bool keepDims = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reduce_min", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reduce_min failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reduce_min, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR REDUCE_MIN");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16, INT32, INT64}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_prod MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reduce_prod, ENGINE_CPU)

PLATFORM_IMPL(reduce_prod, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<LongType> dimensions;
    if (block.width() > 1) {
        auto* dims = INPUT_VARIABLE(1);
        for (LongType i = 0; i < dims->lengthOf(); i++) {
            dimensions.push_back(dims->e<LongType>(i));
        }
    } else if (block.numI() > 0) {
        for (int i = 0; i < block.numI(); i++) {
            dimensions.push_back(INT_ARG(i));
        }
    }

    bool keepDims = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reduce_prod", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reduce_prod failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reduce_prod, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR REDUCE_PROD");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_variance (var) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reduce_variance, ENGINE_CPU)

PLATFORM_IMPL(reduce_variance, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<LongType> dimensions;
    if (block.width() > 1) {
        auto* dims = INPUT_VARIABLE(1);
        for (LongType i = 0; i < dims->lengthOf(); i++) {
            dimensions.push_back(dims->e<LongType>(i));
        }
    } else if (block.numI() > 0) {
        for (int i = 0; i < block.numI(); i++) {
            dimensions.push_back(INT_ARG(i));
        }
    }

    bool keepDims = block.numB() > 0 ? B_ARG(0) : false;
    bool biasCorrected = block.numB() > 1 ? B_ARG(1) : false;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reduce_variance", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reduce_variance failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reduce_variance, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR REDUCE_VARIANCE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_stdev (std) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reduce_stdev, ENGINE_CPU)

PLATFORM_IMPL(reduce_stdev, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<LongType> dimensions;
    if (block.width() > 1) {
        auto* dims = INPUT_VARIABLE(1);
        for (LongType i = 0; i < dims->lengthOf(); i++) {
            dimensions.push_back(dims->e<LongType>(i));
        }
    } else if (block.numI() > 0) {
        for (int i = 0; i < block.numI(); i++) {
            dimensions.push_back(INT_ARG(i));
        }
    }

    bool keepDims = block.numB() > 0 ? B_ARG(0) : false;
    bool biasCorrected = block.numB() > 1 ? B_ARG(1) : false;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reduce_stdev", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reduce_stdev failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reduce_stdev, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR REDUCE_STDEV");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_norm2 (L2 norm) MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reduce_norm2, ENGINE_CPU)

PLATFORM_IMPL(reduce_norm2, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<LongType> dimensions;
    if (block.width() > 1) {
        auto* dims = INPUT_VARIABLE(1);
        for (LongType i = 0; i < dims->lengthOf(); i++) {
            dimensions.push_back(dims->e<LongType>(i));
        }
    } else if (block.numI() > 0) {
        for (int i = 0; i < block.numI(); i++) {
            dimensions.push_back(INT_ARG(i));
        }
    }

    bool keepDims = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reduce_norm2", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reduce_norm2 failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reduce_norm2, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR REDUCE_NORM2");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_logsumexp MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(reduce_logsumexp, ENGINE_CPU)

PLATFORM_IMPL(reduce_logsumexp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<LongType> dimensions;
    if (block.width() > 1) {
        auto* dims = INPUT_VARIABLE(1);
        for (LongType i = 0; i < dims->lengthOf(); i++) {
            dimensions.push_back(dims->e<LongType>(i));
        }
    } else if (block.numI() > 0) {
        for (int i = 0; i < block.numI(); i++) {
            dimensions.push_back(INT_ARG(i));
        }
    }

    bool keepDims = block.numB() > 0 ? B_ARG(0) : false;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlir("reduce_logsumexp", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR reduce_logsumexp failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(reduce_logsumexp, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR REDUCE_LOGSUMEXP");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE}, "Supported dtype") &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold") &&
    req.expectTrue(input->ews() == 1 || input->ews() == 0, "Contiguous memory");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
