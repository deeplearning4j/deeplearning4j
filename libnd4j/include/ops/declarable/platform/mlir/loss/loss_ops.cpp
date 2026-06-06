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
// MLIR-accelerated loss function operations
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
// softmax_cross_entropy_with_logits MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(softmax_cross_entropy_loss_with_logits, ENGINE_CPU)

PLATFORM_IMPL(softmax_cross_entropy_loss_with_logits, ENGINE_CPU) {
    auto* logits = INPUT_VARIABLE(0);   // [batch, classes]
    auto* labels = INPUT_VARIABLE(1);   // [batch, classes] - one-hot or soft labels
    auto* output = OUTPUT_VARIABLE(0);  // [batch] or scalar

    int axis = block.numI() > 0 ? INT_ARG(0) : -1;

    std::vector<NDArray*> inputs = {logits, labels};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("softmax_cross_entropy_with_logits", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR softmax_cross_entropy_with_logits failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(softmax_cross_entropy_loss_with_logits, ENGINE_CPU) {
    auto* logits = INPUT_VARIABLE(0);

    Requirements req("MLIR SOFTMAX_CROSS_ENTROPY_WITH_LOGITS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(logits->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(logits->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// softmax_cross_entropy_with_logits_bp MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(softmax_cross_entropy_loss_with_logits_grad, ENGINE_CPU)

PLATFORM_IMPL(softmax_cross_entropy_loss_with_logits_grad, ENGINE_CPU) {
    auto* logits = INPUT_VARIABLE(0);
    auto* labels = INPUT_VARIABLE(1);
    auto* gradO = INPUT_VARIABLE(2);

    auto* gradLogits = OUTPUT_VARIABLE(0);

    std::vector<NDArray*> inputs = {logits, labels, gradO};
    std::vector<NDArray*> outputs = {gradLogits};

    auto status = executeMlir("softmax_cross_entropy_with_logits_bp", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR softmax_cross_entropy_with_logits_bp failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(softmax_cross_entropy_loss_with_logits_grad, ENGINE_CPU) {
    auto* logits = INPUT_VARIABLE(0);

    Requirements req("MLIR SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_BP");

    // Backward ops fall through to native C++ — no MLIR backward support
    req.expectTrue(false, "Backward ops use native C++ implementation");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// sparse_softmax_cross_entropy_with_logits MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(sparse_softmax_cross_entropy_loss_with_logits, ENGINE_CPU)

PLATFORM_IMPL(sparse_softmax_cross_entropy_loss_with_logits, ENGINE_CPU) {
    auto* labels = INPUT_VARIABLE(0);   // [batch] - integer class indices
    auto* logits = INPUT_VARIABLE(1);   // [batch, classes]
    auto* output = OUTPUT_VARIABLE(0);  // [batch]

    std::vector<NDArray*> inputs = {logits, labels};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("sparse_softmax_cross_entropy_with_logits", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR sparse_softmax_cross_entropy_with_logits failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(sparse_softmax_cross_entropy_loss_with_logits, ENGINE_CPU) {
    auto* labels = INPUT_VARIABLE(0);
    auto* logits = INPUT_VARIABLE(1);

    Requirements req("MLIR SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(logits->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(logits->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// sigmoid_cross_entropy_with_logits MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(sigm_cross_entropy_loss, ENGINE_CPU)

PLATFORM_IMPL(sigm_cross_entropy_loss, ENGINE_CPU) {
    auto* logits = INPUT_VARIABLE(0);   // [batch, ...]
    auto* labels = INPUT_VARIABLE(1);   // [batch, ...] - binary labels
    auto* weights = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    int reductionMode = block.numI() > 0 ? INT_ARG(0) : 0;  // 0=none, 1=mean, 2=sum

    std::vector<NDArray*> inputs = {logits, labels};
    if (weights != nullptr) inputs.push_back(weights);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("sigmoid_cross_entropy_with_logits", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR sigmoid_cross_entropy_with_logits failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(sigm_cross_entropy_loss, ENGINE_CPU) {
    auto* logits = INPUT_VARIABLE(0);

    Requirements req("MLIR SIGMOID_CROSS_ENTROPY_WITH_LOGITS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(logits->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(logits->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// mean_squared_error MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(mean_sqerr_loss, ENGINE_CPU)

PLATFORM_IMPL(mean_sqerr_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);
    auto* labels = INPUT_VARIABLE(1);
    auto* weights = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    int reductionMode = block.numI() > 0 ? INT_ARG(0) : 1;  // default: mean

    std::vector<NDArray*> inputs = {predictions, labels};
    if (weights != nullptr) inputs.push_back(weights);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("mean_squared_error", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR mean_squared_error failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(mean_sqerr_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);

    Requirements req("MLIR MEAN_SQUARED_ERROR");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(predictions->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(predictions->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// mean_absolute_error MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(mean_pairwssqerr_loss, ENGINE_CPU)

PLATFORM_IMPL(mean_pairwssqerr_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);
    auto* labels = INPUT_VARIABLE(1);
    auto* weights = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    int reductionMode = block.numI() > 0 ? INT_ARG(0) : 1;

    std::vector<NDArray*> inputs = {predictions, labels};
    if (weights != nullptr) inputs.push_back(weights);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("mean_pairwise_squared_error", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR mean_pairwise_squared_error failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(mean_pairwssqerr_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);

    Requirements req("MLIR MEAN_PAIRWISE_SQUARED_ERROR");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(predictions->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(predictions->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// huber_loss MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(huber_loss, ENGINE_CPU)

PLATFORM_IMPL(huber_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);
    auto* labels = INPUT_VARIABLE(1);
    auto* weights = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    double delta = block.numT() > 0 ? T_ARG(0) : 1.0;
    int reductionMode = block.numI() > 0 ? INT_ARG(0) : 1;

    std::vector<NDArray*> inputs = {predictions, labels};
    if (weights != nullptr) inputs.push_back(weights);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("huber_loss", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR huber_loss failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(huber_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);

    Requirements req("MLIR HUBER_LOSS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(predictions->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(predictions->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// hinge_loss MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(hinge_loss, ENGINE_CPU)

PLATFORM_IMPL(hinge_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);
    auto* labels = INPUT_VARIABLE(1);  // labels in {-1, 1}
    auto* weights = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    int reductionMode = block.numI() > 0 ? INT_ARG(0) : 1;

    std::vector<NDArray*> inputs = {predictions, labels};
    if (weights != nullptr) inputs.push_back(weights);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("hinge_loss", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR hinge_loss failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(hinge_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);

    Requirements req("MLIR HINGE_LOSS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(predictions->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(predictions->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// log_loss MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(log_loss, ENGINE_CPU)

PLATFORM_IMPL(log_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);  // probabilities
    auto* labels = INPUT_VARIABLE(1);
    auto* weights = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    double epsilon = block.numT() > 0 ? T_ARG(0) : 1e-7;
    int reductionMode = block.numI() > 0 ? INT_ARG(0) : 1;

    std::vector<NDArray*> inputs = {predictions, labels};
    if (weights != nullptr) inputs.push_back(weights);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("log_loss", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR log_loss failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(log_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);

    Requirements req("MLIR LOG_LOSS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(predictions->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(predictions->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// cosine_distance_loss MLIR implementation
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(cosine_distance_loss, ENGINE_CPU)

PLATFORM_IMPL(cosine_distance_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);
    auto* labels = INPUT_VARIABLE(1);
    auto* weights = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto* output = OUTPUT_VARIABLE(0);

    int axis = block.numI() > 0 ? INT_ARG(0) : -1;
    int reductionMode = block.numI() > 1 ? INT_ARG(1) : 1;

    std::vector<NDArray*> inputs = {predictions, labels};
    if (weights != nullptr) inputs.push_back(weights);
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("cosine_distance_loss", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR cosine_distance_loss failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(cosine_distance_loss, ENGINE_CPU) {
    auto* predictions = INPUT_VARIABLE(0);

    Requirements req("MLIR COSINE_DISTANCE_LOSS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(predictions->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(predictions->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

//////////////////////////////////////////////////////////////////////////
// ctc_loss MLIR implementation (Connectionist Temporal Classification)
//////////////////////////////////////////////////////////////////////////

DECLARE_PLATFORM(ctc_loss, ENGINE_CPU)

PLATFORM_IMPL(ctc_loss, ENGINE_CPU) {
    auto* logits = INPUT_VARIABLE(0);           // [time, batch, classes]
    auto* labels = INPUT_VARIABLE(1);           // [batch, maxLabelLen]
    auto* logitLengths = INPUT_VARIABLE(2);     // [batch]
    auto* labelLengths = INPUT_VARIABLE(3);     // [batch]
    auto* output = OUTPUT_VARIABLE(0);          // [batch]

    int blankIndex = block.numI() > 0 ? INT_ARG(0) : 0;

    std::vector<NDArray*> inputs = {logits, labels, logitLengths, labelLengths};
    std::vector<NDArray*> outputs = {output};

    auto status = executeMlirEx("ctc_loss", block, inputs, outputs);

    if (status != Status::OK()) {
        return Status::CODE(ND4J_STATUS_BAD_ARGUMENTS, "MLIR ctc_loss failed");
    }

    return Status::OK();
}

PLATFORM_CHECK(ctc_loss, ENGINE_CPU) {
    auto* logits = INPUT_VARIABLE(0);

    Requirements req("MLIR CTC_LOSS");

    req.expectTrue(mlirEnabled(), "MLIR enabled") &&
    req.expectIn(logits->dataType(), {FLOAT32, DOUBLE, HALF, BFLOAT16}, "Supported dtype") &&
    req.expectTrue(logits->lengthOf() >= MLIR_MIN_TENSOR_SIZE, "Size threshold");

    return req;
}

} // namespace platforms
} // namespace ops
} // namespace sd

#endif // HAVE_MLIR
