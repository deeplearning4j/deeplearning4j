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
// Metal Performance Shaders - Loss function operations
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"
#include <cmath>
#include <cfloat>

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Mean Squared Error Loss
// MSE = mean((predictions - labels)^2)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(mean_sqerr_loss, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (predictions->isEmpty()) return sd::Status::OK;

    const float* predPtr = predictions->bufferAsT<float>();
    const float* labelPtr = labels->bufferAsT<float>();
    float* outPtr = output->bufferAsT<float>();

    LongType length = predictions->lengthOf();
    float sum = 0.0f;

    for (LongType i = 0; i < length; i++) {
        float diff = predPtr[i] - labelPtr[i];
        sum += diff * diff;
    }

    outPtr[0] = sum / length;

    return sd::Status::OK;
}

PLATFORM_CHECK(mean_sqerr_loss, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);

    Requirements req("MPS MEAN_SQERR_LOSS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(predictions->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*predictions), "Predictions must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Mean Absolute Error Loss
// MAE = mean(|predictions - labels|)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(mean_absolute_error, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (predictions->isEmpty()) return sd::Status::OK;

    const float* predPtr = predictions->bufferAsT<float>();
    const float* labelPtr = labels->bufferAsT<float>();
    float* outPtr = output->bufferAsT<float>();

    LongType length = predictions->lengthOf();
    float sum = 0.0f;

    for (LongType i = 0; i < length; i++) {
        sum += fabsf(predPtr[i] - labelPtr[i]);
    }

    outPtr[0] = sum / length;

    return sd::Status::OK;
}

PLATFORM_CHECK(mean_absolute_error, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);

    Requirements req("MPS MEAN_ABSOLUTE_ERROR OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(predictions->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*predictions), "Predictions must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Huber Loss
// L = 0.5 * x^2 if |x| <= delta, else delta * (|x| - 0.5 * delta)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(huber_loss, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (predictions->isEmpty()) return sd::Status::OK;

    float delta = 1.0f;
    if (block.getTArguments()->size() > 0) {
        delta = T_ARG(0);
    }

    const float* predPtr = predictions->bufferAsT<float>();
    const float* labelPtr = labels->bufferAsT<float>();
    float* outPtr = output->bufferAsT<float>();

    LongType length = predictions->lengthOf();
    float sum = 0.0f;

    for (LongType i = 0; i < length; i++) {
        float diff = fabsf(predPtr[i] - labelPtr[i]);
        if (diff <= delta) {
            sum += 0.5f * diff * diff;
        } else {
            sum += delta * (diff - 0.5f * delta);
        }
    }

    outPtr[0] = sum / length;

    return sd::Status::OK;
}

PLATFORM_CHECK(huber_loss, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);

    Requirements req("MPS HUBER_LOSS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(predictions->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*predictions), "Predictions must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Binary Cross Entropy Loss
// BCE = -mean(labels * log(predictions) + (1 - labels) * log(1 - predictions))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(sigm_cross_entropy_loss, ENGINE_CPU) {
    auto logits = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (logits->isEmpty()) return sd::Status::OK;

    const float* logitPtr = logits->bufferAsT<float>();
    const float* labelPtr = labels->bufferAsT<float>();
    float* outPtr = output->bufferAsT<float>();

    LongType length = logits->lengthOf();
    float sum = 0.0f;
    const float epsilon = 1e-7f;

    for (LongType i = 0; i < length; i++) {
        float logit = logitPtr[i];
        float label = labelPtr[i];

        // Numerically stable sigmoid cross entropy:
        // max(logit, 0) - logit * label + log(1 + exp(-|logit|))
        float maxVal = std::max(logit, 0.0f);
        sum += maxVal - logit * label + log1pf(expf(-fabsf(logit)));
    }

    outPtr[0] = sum / length;

    return sd::Status::OK;
}

PLATFORM_CHECK(sigm_cross_entropy_loss, ENGINE_CPU) {
    auto logits = INPUT_VARIABLE(0);

    Requirements req("MPS SIGM_CROSS_ENTROPY_LOSS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(logits->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*logits), "Logits must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Softmax Cross Entropy Loss
// CE = -sum(labels * log(softmax(logits)))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(softmax_cross_entropy_loss, ENGINE_CPU) {
    auto logits = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (logits->isEmpty()) return sd::Status::OK;

    const float* logitPtr = logits->bufferAsT<float>();
    const float* labelPtr = labels->bufferAsT<float>();
    float* outPtr = output->bufferAsT<float>();

    // Assuming last dimension is classes
    LongType batchSize = 1;
    LongType numClasses = logits->sizeAt(-1);

    for (int i = 0; i < logits->rankOf() - 1; i++) {
        batchSize *= logits->sizeAt(i);
    }

    float totalLoss = 0.0f;
    const float epsilon = 1e-7f;

    for (LongType b = 0; b < batchSize; b++) {
        LongType offset = b * numClasses;

        // Find max for numerical stability
        float maxLogit = logitPtr[offset];
        for (LongType c = 1; c < numClasses; c++) {
            if (logitPtr[offset + c] > maxLogit) {
                maxLogit = logitPtr[offset + c];
            }
        }

        // Compute log(sum(exp(logit - max)))
        float sumExp = 0.0f;
        for (LongType c = 0; c < numClasses; c++) {
            sumExp += expf(logitPtr[offset + c] - maxLogit);
        }
        float logSumExp = logf(sumExp) + maxLogit;

        // Compute cross entropy: -sum(label * (logit - logSumExp))
        float batchLoss = 0.0f;
        for (LongType c = 0; c < numClasses; c++) {
            if (labelPtr[offset + c] > 0) {
                batchLoss -= labelPtr[offset + c] * (logitPtr[offset + c] - logSumExp);
            }
        }
        totalLoss += batchLoss;
    }

    outPtr[0] = totalLoss / batchSize;

    return sd::Status::OK;
}

PLATFORM_CHECK(softmax_cross_entropy_loss, ENGINE_CPU) {
    auto logits = INPUT_VARIABLE(0);

    Requirements req("MPS SOFTMAX_CROSS_ENTROPY_LOSS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(logits->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*logits), "Logits must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Sparse Softmax Cross Entropy Loss
// Labels are class indices, not one-hot encoded
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(sparse_softmax_cross_entropy_loss_with_logits, ENGINE_CPU) {
    auto logits = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);  // Integer class indices
    auto output = OUTPUT_VARIABLE(0);

    if (logits->isEmpty()) return sd::Status::OK;

    const float* logitPtr = logits->bufferAsT<float>();
    const int* labelPtr = labels->bufferAsT<int>();
    float* outPtr = output->bufferAsT<float>();

    LongType batchSize = logits->sizeAt(0);
    LongType numClasses = logits->sizeAt(-1);

    float totalLoss = 0.0f;

    for (LongType b = 0; b < batchSize; b++) {
        LongType offset = b * numClasses;
        int trueClass = labelPtr[b];

        // Find max for numerical stability
        float maxLogit = logitPtr[offset];
        for (LongType c = 1; c < numClasses; c++) {
            if (logitPtr[offset + c] > maxLogit) {
                maxLogit = logitPtr[offset + c];
            }
        }

        // Compute log(sum(exp(logit - max)))
        float sumExp = 0.0f;
        for (LongType c = 0; c < numClasses; c++) {
            sumExp += expf(logitPtr[offset + c] - maxLogit);
        }
        float logSumExp = logf(sumExp) + maxLogit;

        // Cross entropy for true class: -(logit[trueClass] - logSumExp)
        totalLoss -= (logitPtr[offset + trueClass] - logSumExp);
    }

    outPtr[0] = totalLoss / batchSize;

    return sd::Status::OK;
}

PLATFORM_CHECK(sparse_softmax_cross_entropy_loss_with_logits, ENGINE_CPU) {
    auto logits = INPUT_VARIABLE(0);

    Requirements req("MPS SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(logits->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*logits), "Logits must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Cosine Distance Loss
// 1 - cosine_similarity(predictions, labels)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(cosine_distance_loss, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (predictions->isEmpty()) return sd::Status::OK;

    const float* predPtr = predictions->bufferAsT<float>();
    const float* labelPtr = labels->bufferAsT<float>();
    float* outPtr = output->bufferAsT<float>();

    LongType length = predictions->lengthOf();

    float dotProduct = 0.0f;
    float normPred = 0.0f;
    float normLabel = 0.0f;

    for (LongType i = 0; i < length; i++) {
        dotProduct += predPtr[i] * labelPtr[i];
        normPred += predPtr[i] * predPtr[i];
        normLabel += labelPtr[i] * labelPtr[i];
    }

    float cosineSim = dotProduct / (sqrtf(normPred) * sqrtf(normLabel) + 1e-8f);
    outPtr[0] = 1.0f - cosineSim;

    return sd::Status::OK;
}

PLATFORM_CHECK(cosine_distance_loss, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);

    Requirements req("MPS COSINE_DISTANCE_LOSS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(predictions->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*predictions), "Predictions must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Hinge Loss
// max(0, 1 - labels * predictions)
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(hinge_loss, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (predictions->isEmpty()) return sd::Status::OK;

    const float* predPtr = predictions->bufferAsT<float>();
    const float* labelPtr = labels->bufferAsT<float>();
    float* outPtr = output->bufferAsT<float>();

    LongType length = predictions->lengthOf();
    float sum = 0.0f;

    for (LongType i = 0; i < length; i++) {
        float margin = 1.0f - labelPtr[i] * predPtr[i];
        sum += std::max(0.0f, margin);
    }

    outPtr[0] = sum / length;

    return sd::Status::OK;
}

PLATFORM_CHECK(hinge_loss, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);

    Requirements req("MPS HINGE_LOSS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(predictions->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*predictions), "Predictions must be contiguous");
    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Log Loss (Binary Cross Entropy with raw probabilities)
// -mean(labels * log(predictions + eps) + (1 - labels) * log(1 - predictions + eps))
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(log_loss, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);
    auto labels = INPUT_VARIABLE(1);
    auto output = OUTPUT_VARIABLE(0);

    if (predictions->isEmpty()) return sd::Status::OK;

    float epsilon = 1e-7f;
    if (block.getTArguments()->size() > 0) {
        epsilon = T_ARG(0);
    }

    const float* predPtr = predictions->bufferAsT<float>();
    const float* labelPtr = labels->bufferAsT<float>();
    float* outPtr = output->bufferAsT<float>();

    LongType length = predictions->lengthOf();
    float sum = 0.0f;

    for (LongType i = 0; i < length; i++) {
        float pred = std::max(epsilon, std::min(1.0f - epsilon, predPtr[i]));
        float label = labelPtr[i];
        sum -= label * logf(pred) + (1.0f - label) * logf(1.0f - pred);
    }

    outPtr[0] = sum / length;

    return sd::Status::OK;
}

PLATFORM_CHECK(log_loss, ENGINE_CPU) {
    auto predictions = INPUT_VARIABLE(0);

    Requirements req("MPS LOG_LOSS OP");
    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");
    req.expectTrue(predictions->dataType() == DataType::FLOAT32, "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*predictions), "Predictions must be contiguous");
    req.logTheSuccess();
    return req;
}

#else  // !HAVE_MPS

PLATFORM_IMPL(mean_sqerr_loss, ENGINE_CPU)                           { return sd::Status::OK; }
PLATFORM_CHECK(mean_sqerr_loss, ENGINE_CPU)                          { return Requirements("MPS MEAN_SQERR_LOSS STUB"); }

PLATFORM_IMPL(mean_absolute_error, ENGINE_CPU)                       { return sd::Status::OK; }
PLATFORM_CHECK(mean_absolute_error, ENGINE_CPU)                      { return Requirements("MPS MEAN_ABS_ERROR STUB"); }

PLATFORM_IMPL(huber_loss, ENGINE_CPU)                                { return sd::Status::OK; }
PLATFORM_CHECK(huber_loss, ENGINE_CPU)                               { return Requirements("MPS HUBER_LOSS STUB"); }

PLATFORM_IMPL(sigm_cross_entropy_loss, ENGINE_CPU)                   { return sd::Status::OK; }
PLATFORM_CHECK(sigm_cross_entropy_loss, ENGINE_CPU)                  { return Requirements("MPS SIGM_XENT_LOSS STUB"); }

PLATFORM_IMPL(softmax_cross_entropy_loss, ENGINE_CPU)                { return sd::Status::OK; }
PLATFORM_CHECK(softmax_cross_entropy_loss, ENGINE_CPU)               { return Requirements("MPS SOFTMAX_XENT_LOSS STUB"); }

PLATFORM_IMPL(sparse_softmax_cross_entropy_loss_with_logits, ENGINE_CPU)  { return sd::Status::OK; }
PLATFORM_CHECK(sparse_softmax_cross_entropy_loss_with_logits, ENGINE_CPU) { return Requirements("MPS SPARSE_SOFTMAX_XENT STUB"); }

PLATFORM_IMPL(cosine_distance_loss, ENGINE_CPU)                      { return sd::Status::OK; }
PLATFORM_CHECK(cosine_distance_loss, ENGINE_CPU)                     { return Requirements("MPS COSINE_DIST_LOSS STUB"); }

PLATFORM_IMPL(hinge_loss, ENGINE_CPU)                                { return sd::Status::OK; }
PLATFORM_CHECK(hinge_loss, ENGINE_CPU)                               { return Requirements("MPS HINGE_LOSS STUB"); }

PLATFORM_IMPL(log_loss, ENGINE_CPU)                                  { return sd::Status::OK; }
PLATFORM_CHECK(log_loss, ENGINE_CPU)                                 { return Requirements("MPS LOG_LOSS STUB"); }

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
