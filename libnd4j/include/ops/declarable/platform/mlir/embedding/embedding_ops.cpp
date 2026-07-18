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
// MLIR-accelerated embedding operations
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
// embedding_lookup MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(embedding_lookup, ENGINE_CPU)

PLATFORM_IMPL(embedding_lookup, ENGINE_CPU) {
    auto* params = INPUT_VARIABLE(0);    // embedding table [vocab_size, embed_dim]
    auto* indices = INPUT_VARIABLE(1);   // indices [batch, ...]
    auto* output = OUTPUT_VARIABLE(0);   // [batch, ..., embed_dim]

    int partitionMode = block.numI() > 0 ? INT_ARG(0) : 0;

    std::vector<NDArray*> inputs = {params, indices};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("embedding_lookup", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(embedding_lookup, ENGINE_CPU) {
    auto* params = INPUT_VARIABLE(0);
    auto* indices = INPUT_VARIABLE(1);

    Requirements req("MLIR EMBEDDING_LOOKUP");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(params->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(params->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// one_hot MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(onehot, ENGINE_CPU)

PLATFORM_IMPL(onehot, ENGINE_CPU) {
    auto* indices = INPUT_VARIABLE(0);   // integer indices
    auto* output = OUTPUT_VARIABLE(0);

    int depth = INT_ARG(0);
    int axis = block.numI() > 1 ? INT_ARG(1) : -1;
    double onValue = block.numT() > 0 ? T_ARG(0) : 1.0;
    double offValue = block.numT() > 1 ? T_ARG(1) : 0.0;

    std::vector<NDArray*> inputs = {indices};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("one_hot", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(onehot, ENGINE_CPU) {
    auto* indices = INPUT_VARIABLE(0);
    auto* output = OUTPUT_VARIABLE(0);

    Requirements req("MLIR ONE_HOT");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(indices->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(indices->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// segment_sum MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(segment_sum, ENGINE_CPU)

PLATFORM_IMPL(segment_sum, ENGINE_CPU) {
    auto* data = INPUT_VARIABLE(0);
    auto* segmentIds = INPUT_VARIABLE(1);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {data, segmentIds};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("segment_sum", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(segment_sum, ENGINE_CPU) {
    auto* data = INPUT_VARIABLE(0);

    Requirements req("MLIR SEGMENT_SUM");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(data->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(data->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// segment_mean MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(segment_mean, ENGINE_CPU)

PLATFORM_IMPL(segment_mean, ENGINE_CPU) {
    auto* data = INPUT_VARIABLE(0);
    auto* segmentIds = INPUT_VARIABLE(1);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {data, segmentIds};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("segment_mean", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(segment_mean, ENGINE_CPU) {
    auto* data = INPUT_VARIABLE(0);

    Requirements req("MLIR SEGMENT_MEAN");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(data->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(data->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// segment_max MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(segment_max, ENGINE_CPU)

PLATFORM_IMPL(segment_max, ENGINE_CPU) {
    auto* data = INPUT_VARIABLE(0);
    auto* segmentIds = INPUT_VARIABLE(1);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {data, segmentIds};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("segment_max", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(segment_max, ENGINE_CPU) {
    auto* data = INPUT_VARIABLE(0);

    Requirements req("MLIR SEGMENT_MAX");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(data->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(data->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// segment_min MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(segment_min, ENGINE_CPU)

PLATFORM_IMPL(segment_min, ENGINE_CPU) {
    auto* data = INPUT_VARIABLE(0);
    auto* segmentIds = INPUT_VARIABLE(1);
    auto* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {data, segmentIds};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("segment_min", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(segment_min, ENGINE_CPU) {
    auto* data = INPUT_VARIABLE(0);

    Requirements req("MLIR SEGMENT_MIN");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(data->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(data->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// unsorted_segment_sum MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(unsorted_segment_sum, ENGINE_CPU)

PLATFORM_IMPL(unsorted_segment_sum, ENGINE_CPU) {
    auto* data = INPUT_VARIABLE(0);
    auto* segmentIds = INPUT_VARIABLE(1);
    auto* output = OUTPUT_VARIABLE(0);

    int numSegments = INT_ARG(0);

    std::vector<NDArray*> inputs = {data, segmentIds};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("unsorted_segment_sum", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(unsorted_segment_sum, ENGINE_CPU) {
    auto* data = INPUT_VARIABLE(0);

    Requirements req("MLIR UNSORTED_SEGMENT_SUM");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(data->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(data->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// unique MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(unique, ENGINE_CPU)

PLATFORM_IMPL(unique, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* values = OUTPUT_VARIABLE(0);   // unique values
    auto* indices = OUTPUT_VARIABLE(1);  // indices into unique values

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {values, indices};

    auto status = executeMlirEx("unique", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(unique, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR UNIQUE");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// top_k MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(top_k, ENGINE_CPU)

PLATFORM_IMPL(top_k, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);
    auto* values = OUTPUT_VARIABLE(0);
    auto* indices = OUTPUT_VARIABLE(1);

    int k = INT_ARG(0);
    bool sorted = block.numB() > 0 ? B_ARG(0) : true;

    std::vector<NDArray*> inputs = {input};
    std::vector<NDArray*> outputs = {values, indices};

    auto status = executeMlirEx("top_k", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(top_k, ENGINE_CPU) {
    auto* input = INPUT_VARIABLE(0);

    Requirements req("MLIR TOP_K");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(input->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(input->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// in_top_k MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(in_top_k, ENGINE_CPU)

PLATFORM_IMPL(in_top_k, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);  // [batch, classes]
    auto* targets = INPUT_VARIABLE(1);      // [batch] - integer class indices
    auto* output = OUTPUT_VARIABLE(0);      // [batch] - bool

    int k = INT_ARG(0);

    std::vector<NDArray*> inputs = {predictions, targets};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("in_top_k", block, inputs, outputs);

    if (status != Status::OK) {
        return Status::BAD_ARGUMENTS;
    }

    return Status::OK;
}

PLATFORM_CHECK(in_top_k, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);

    Requirements req("MLIR IN_TOP_K");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(predictions->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}) &&
    req.expectTrue(predictions->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
