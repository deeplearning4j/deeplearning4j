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

#include <array/NDArrayFactory.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "pjrtUtils.h"

#if defined(HAVE_PJRT) && HAVE_PJRT

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
// Helper to build batch normalization forward HLO (training mode)
static std::string buildBatchNormTrainingHLO(const std::vector<sd::LongType>& inputShape,
                                              sd::LongType featureDim,
                                              double epsilon,
                                              DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  sd::LongType batch = inputShape[0];
  sd::LongType channels = inputShape[featureDim];

  hlo << "HloModule batchnorm_training_module\n\n";

  // Add computation for variance calculation
  hlo << "add_computation {\n";
  hlo << "  x = " << typeStr << "[] parameter(0)\n";
  hlo << "  y = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[] add(x, y)\n";
  hlo << "}\n\n";

  hlo << "ENTRY main {\n";

  // Parameters: input, scale (gamma), offset (beta)
  if (inputShape.size() == 4) {
    // NHWC format
    hlo << "  input = " << typeStr << "[" << inputShape[0] << "," << inputShape[1] << ","
        << inputShape[2] << "," << inputShape[3] << "] parameter(0)\n";
  } else if (inputShape.size() == 2) {
    // NC format (dense layer)
    hlo << "  input = " << typeStr << "[" << inputShape[0] << "," << inputShape[1] << "] parameter(0)\n";
  }

  hlo << "  scale = " << typeStr << "[" << channels << "] parameter(1)\n";
  hlo << "  offset = " << typeStr << "[" << channels << "] parameter(2)\n";

  // Use batch-norm-training HLO instruction
  // This returns a tuple: (output, batch_mean, batch_var)
  if (inputShape.size() == 4) {
    hlo << "  bn_result = (" << typeStr << "[" << inputShape[0] << "," << inputShape[1] << ","
        << inputShape[2] << "," << inputShape[3] << "], " << typeStr << "[" << channels << "], "
        << typeStr << "[" << channels << "]) ";
    hlo << "batch-norm-training(input, scale, offset), epsilon=" << epsilon << ", feature_index=" << featureDim << "\n";
  } else {
    hlo << "  bn_result = (" << typeStr << "[" << inputShape[0] << "," << inputShape[1] << "], "
        << typeStr << "[" << channels << "], " << typeStr << "[" << channels << "]) ";
    hlo << "batch-norm-training(input, scale, offset), epsilon=" << epsilon << ", feature_index=1\n";
  }

  hlo << "  ROOT output = " << typeStr << "[";
  if (inputShape.size() == 4) {
    hlo << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3];
  } else {
    hlo << inputShape[0] << "," << inputShape[1];
  }
  hlo << "] get-tuple-element(bn_result), index=0\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Helper to build batch normalization forward HLO (inference mode)
static std::string buildBatchNormInferenceHLO(const std::vector<sd::LongType>& inputShape,
                                               sd::LongType featureDim,
                                               double epsilon,
                                               DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  sd::LongType channels = inputShape[featureDim];

  hlo << "HloModule batchnorm_inference_module\n\n";

  hlo << "ENTRY main {\n";

  // Parameters: input, scale (gamma), offset (beta), mean, variance
  if (inputShape.size() == 4) {
    hlo << "  input = " << typeStr << "[" << inputShape[0] << "," << inputShape[1] << ","
        << inputShape[2] << "," << inputShape[3] << "] parameter(0)\n";
  } else {
    hlo << "  input = " << typeStr << "[" << inputShape[0] << "," << inputShape[1] << "] parameter(0)\n";
  }

  hlo << "  scale = " << typeStr << "[" << channels << "] parameter(1)\n";
  hlo << "  offset = " << typeStr << "[" << channels << "] parameter(2)\n";
  hlo << "  mean = " << typeStr << "[" << channels << "] parameter(3)\n";
  hlo << "  variance = " << typeStr << "[" << channels << "] parameter(4)\n";

  // Use batch-norm-inference HLO instruction
  hlo << "  ROOT output = " << typeStr << "[";
  if (inputShape.size() == 4) {
    hlo << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3];
  } else {
    hlo << inputShape[0] << "," << inputShape[1];
  }
  hlo << "] batch-norm-inference(input, scale, offset, mean, variance), ";
  hlo << "epsilon=" << epsilon << ", feature_index=" << featureDim << "\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Helper to build batch normalization backward HLO
static std::string buildBatchNormGradHLO(const std::vector<sd::LongType>& inputShape,
                                          sd::LongType featureDim,
                                          double epsilon,
                                          DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  sd::LongType channels = inputShape[featureDim];

  hlo << "HloModule batchnorm_grad_module\n\n";

  hlo << "ENTRY main {\n";

  // Parameters: input, scale, mean, variance, grad_output
  if (inputShape.size() == 4) {
    hlo << "  input = " << typeStr << "[" << inputShape[0] << "," << inputShape[1] << ","
        << inputShape[2] << "," << inputShape[3] << "] parameter(0)\n";
  } else {
    hlo << "  input = " << typeStr << "[" << inputShape[0] << "," << inputShape[1] << "] parameter(0)\n";
  }

  hlo << "  scale = " << typeStr << "[" << channels << "] parameter(1)\n";
  hlo << "  mean = " << typeStr << "[" << channels << "] parameter(2)\n";
  hlo << "  variance = " << typeStr << "[" << channels << "] parameter(3)\n";

  if (inputShape.size() == 4) {
    hlo << "  grad_output = " << typeStr << "[" << inputShape[0] << "," << inputShape[1] << ","
        << inputShape[2] << "," << inputShape[3] << "] parameter(4)\n";
  } else {
    hlo << "  grad_output = " << typeStr << "[" << inputShape[0] << "," << inputShape[1] << "] parameter(4)\n";
  }

  // Use batch-norm-grad HLO instruction
  // Returns tuple: (grad_input, grad_scale, grad_offset)
  hlo << "  bn_grad = (";
  if (inputShape.size() == 4) {
    hlo << typeStr << "[" << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3] << "]";
  } else {
    hlo << typeStr << "[" << inputShape[0] << "," << inputShape[1] << "]";
  }
  hlo << ", " << typeStr << "[" << channels << "], " << typeStr << "[" << channels << "]) ";
  hlo << "batch-norm-grad(input, scale, mean, variance, grad_output), ";
  hlo << "epsilon=" << epsilon << ", feature_index=" << featureDim << "\n";

  // Return grad_input as primary output
  hlo << "  ROOT grad_input = " << typeStr << "[";
  if (inputShape.size() == 4) {
    hlo << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3];
  } else {
    hlo << inputShape[0] << "," << inputShape[1];
  }
  hlo << "] get-tuple-element(bn_grad), index=0\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Batch Normalization forward implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(batchnorm, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto mean = INPUT_VARIABLE(1);
  auto variance = INPUT_VARIABLE(2);
  NDArray* gamma = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  NDArray* beta = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;

  auto output = OUTPUT_VARIABLE(0);

  // Parameters
  bool applyScale = block.getBArguments()->size() > 0 ? B_ARG(0) : true;
  bool applyOffset = block.getBArguments()->size() > 1 ? B_ARG(1) : true;
  double epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5;

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for batchnorm operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  // Determine feature dimension (last dim for NHWC, dim 1 for NCHW)
  sd::LongType featureDim = inputShape.size() - 1;  // Assume NHWC

  // Create scale and offset if not provided
  std::unique_ptr<NDArray> defaultGamma, defaultBeta;
  if (!gamma) {
    defaultGamma = std::make_unique<NDArray>(input->ordering(), std::vector<sd::LongType>{inputShape[featureDim]}, input->dataType(), input->getContext());
    double one = 1.0;
    defaultGamma->assign(one);
    gamma = defaultGamma.get();
  }
  if (!beta) {
    defaultBeta = std::make_unique<NDArray>(input->ordering(), std::vector<sd::LongType>{inputShape[featureDim]}, input->dataType(), input->getContext());
    double zero = 0.0;
    defaultBeta->assign(zero);
    beta = defaultBeta.get();
  }

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "batchnorm_inf_";
  for (auto s : inputShape) keyStream << s << "_";
  keyStream << epsilon << "_" << input->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module for inference
  std::string hloModule = buildBatchNormInferenceHLO(inputShape, featureDim, epsilon, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU batchnorm program");
  }

  // Create device buffers and execute
  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufScale(client, gamma);
  PjrtBuffer bufOffset(client, beta);
  PjrtBuffer bufMean(client, mean);
  PjrtBuffer bufVar(client, variance);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn, &bufScale, &bufOffset, &bufMean, &bufVar};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(batchnorm, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT BATCHNORM OP");

  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.expectTrue(input->rankOf() >= 2 && input->rankOf() <= 4,
                 REQUIRE_TRUE_MSG("Input must be 2D, 3D, or 4D tensor"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Batch Normalization backward implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(batchnorm_bp, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto mean = INPUT_VARIABLE(1);
  auto variance = INPUT_VARIABLE(2);
  NDArray* gamma = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  NDArray* gradOut = block.width() > 4 ? INPUT_VARIABLE(4) : INPUT_VARIABLE(3);

  auto gradI = OUTPUT_VARIABLE(0);
  NDArray* gradG = block.width() > 1 ? OUTPUT_VARIABLE(1) : nullptr;
  NDArray* gradB = block.width() > 2 ? OUTPUT_VARIABLE(2) : nullptr;

  // Parameters
  bool applyScale = block.getBArguments()->size() > 0 ? B_ARG(0) : true;
  bool applyOffset = block.getBArguments()->size() > 1 ? B_ARG(1) : true;
  double epsilon = block.getTArguments()->size() > 0 ? T_ARG(0) : 1e-5;

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for batchnorm_bp operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());
  sd::LongType featureDim = inputShape.size() - 1;

  // Create default gamma if not provided
  std::unique_ptr<NDArray> defaultGamma;
  if (!gamma) {
    defaultGamma = std::make_unique<NDArray>(input->ordering(), std::vector<sd::LongType>{inputShape[featureDim]}, input->dataType(), input->getContext());
    double one = 1.0;
    defaultGamma->assign(one);
    gamma = defaultGamma.get();
  }

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "batchnorm_bp_";
  for (auto s : inputShape) keyStream << s << "_";
  keyStream << epsilon << "_" << input->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module
  std::string hloModule = buildBatchNormGradHLO(inputShape, featureDim, epsilon, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU batchnorm_bp program");
  }

  // Create device buffers and execute
  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufScale(client, gamma);
  PjrtBuffer bufMean(client, mean);
  PjrtBuffer bufVar(client, variance);
  PjrtBuffer bufGradOut(client, gradOut);
  PjrtBuffer bufGradI(client, gradI);

  std::vector<PjrtBuffer*> inputs = {&bufIn, &bufScale, &bufMean, &bufVar, &bufGradOut};
  std::vector<PjrtBuffer*> outputs = {&bufGradI};
  executable->execute(inputs, outputs);

  bufGradI.toHost(gradI);

  // For gradient of gamma and beta, use reduce operations
  if (gradG != nullptr) {
    // Gradient w.r.t. gamma: sum of gradOut * normalized_input
    // Simplified: compute on host for now
    // TODO: Add HLO for gamma/beta gradients
  }

  if (gradB != nullptr) {
    // Gradient w.r.t. beta: sum of gradOut over all dims except feature dim
    if (inputShape.size() == 4) {
      gradOut->reduceAlongDimension(reduce::Sum, *gradB, {0, 1, 2});
    } else {
      gradOut->reduceAlongDimension(reduce::Sum, *gradB, {0});
    }
  }

  return Status::OK;
}

PLATFORM_CHECK(batchnorm_bp, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT BATCHNORM_BP OP");

  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.expectTrue(input->rankOf() >= 2 && input->rankOf() <= 4,
                 REQUIRE_TRUE_MSG("Input must be 2D, 3D, or 4D tensor"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Layer Normalization
//////////////////////////////////////////////////////////////////////////

static std::string buildLayerNormHLO(const std::vector<sd::LongType>& inputShape,
                                      const std::vector<int>& normAxes,
                                      double epsilon,
                                      DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule layernorm_module\n\n";

  hlo << "add_computation {\n";
  hlo << "  x = " << typeStr << "[] parameter(0)\n";
  hlo << "  y = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[] add(x, y)\n";
  hlo << "}\n\n";

  hlo << "ENTRY main {\n";

  // Input
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] parameter(0)\n";

  // Compute mean across normalization axes
  // For layer norm on last axis of [batch, seq, hidden]:
  // mean shape would be [batch, seq]

  // Calculate number of elements for averaging
  sd::LongType numElements = 1;
  for (int axis : normAxes) {
    numElements *= inputShape[axis];
  }

  // Reduce to compute mean
  hlo << "  init_zero = " << typeStr << "[] constant(0)\n";
  hlo << "  sum = " << typeStr << "[";
  bool first = true;
  for (size_t i = 0; i < inputShape.size(); i++) {
    bool isNormAxis = false;
    for (int axis : normAxes) {
      if ((size_t)axis == i) isNormAxis = true;
    }
    if (!isNormAxis) {
      if (!first) hlo << ",";
      hlo << inputShape[i];
      first = false;
    }
  }
  hlo << "] reduce(input, init_zero), dimensions={";
  for (size_t i = 0; i < normAxes.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << normAxes[i];
  }
  hlo << "}, to_apply=add_computation\n";

  // Mean = sum / count
  hlo << "  divisor = " << typeStr << "[] constant(" << numElements << ")\n";
  hlo << "  mean_reduced = " << typeStr << "[";
  first = true;
  for (size_t i = 0; i < inputShape.size(); i++) {
    bool isNormAxis = false;
    for (int axis : normAxes) {
      if ((size_t)axis == i) isNormAxis = true;
    }
    if (!isNormAxis) {
      if (!first) hlo << ",";
      hlo << inputShape[i];
      first = false;
    }
  }
  hlo << "] divide(sum, broadcast(divisor))\n";

  // For simplicity, this is a skeleton - full implementation would:
  // 1. Broadcast mean back to input shape
  // 2. Compute variance
  // 3. Normalize: (x - mean) / sqrt(var + eps)
  // 4. Apply scale and offset

  // Placeholder output (needs full implementation)
  hlo << "  ROOT output = " << typeStr << "[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] parameter(0)\n";  // Placeholder
  hlo << "}\n";

  return hlo.str();
}

PLATFORM_IMPL(layer_norm, ENGINE_TPU) {
  // Layer normalization - normalize across specified axes
  auto input = INPUT_VARIABLE(0);
  NDArray* gain = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;
  NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;

  auto output = OUTPUT_VARIABLE(0);
  NDArray* mean = block.width() > 1 ? OUTPUT_VARIABLE(1) : nullptr;
  NDArray* std = block.width() > 2 ? OUTPUT_VARIABLE(2) : nullptr;

  // For now, fall back to CPU implementation
  // Full TPU implementation requires complex HLO for arbitrary axis normalization
  THROW_EXCEPTION("TPU layer_norm not yet fully implemented - use CPU fallback");

  return Status::OK;
}

PLATFORM_CHECK(layer_norm, ENGINE_TPU) {
  Requirements req("PJRT LAYER_NORM OP");
  // Currently not fully implemented
  req.expectFalse(true, REQUIRE_TRUE_MSG("TPU layer_norm not yet implemented"));
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_PJRT
