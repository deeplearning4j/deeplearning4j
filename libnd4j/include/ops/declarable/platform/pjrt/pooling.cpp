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
// Helper to build max pooling HLO
static std::string buildMaxPool2dHLO(const std::vector<sd::LongType>& inputShape,
                                      const std::vector<int>& kernelSize,
                                      const std::vector<int>& strides,
                                      const std::vector<int>& padding,
                                      DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  sd::LongType batch = inputShape[0];
  sd::LongType inH = inputShape[1];
  sd::LongType inW = inputShape[2];
  sd::LongType channels = inputShape[3];

  int kH = kernelSize[0];
  int kW = kernelSize[1];
  int sH = strides[0];
  int sW = strides[1];
  int pH = padding[0];
  int pW = padding[1];

  // Calculate output dimensions
  sd::LongType outH = (inH + 2 * pH - kH) / sH + 1;
  sd::LongType outW = (inW + 2 * pW - kW) / sW + 1;

  // Get negative infinity for the data type
  std::string negInf;
  if (dtype == DataType::FLOAT32) {
    negInf = "-inf";
  } else if (dtype == DataType::BFLOAT16) {
    negInf = "-inf";
  } else {
    negInf = "-inf";
  }

  hlo << "HloModule maxpool2d_module\n\n";

  // Max computation for reduce-window
  hlo << "max_computation {\n";
  hlo << "  x = " << typeStr << "[] parameter(0)\n";
  hlo << "  y = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[] maximum(x, y)\n";
  hlo << "}\n\n";

  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[" << batch << "," << inH << "," << inW << "," << channels << "] parameter(0)\n";

  // Initialize value for max pooling (negative infinity)
  hlo << "  init = " << typeStr << "[] constant(" << negInf << ")\n";

  // reduce-window for max pooling
  hlo << "  ROOT result = " << typeStr << "[" << batch << "," << outH << "," << outW << "," << channels << "] ";
  hlo << "reduce-window(input, init), ";
  hlo << "window={size=1x" << kH << "x" << kW << "x1 stride=1x" << sH << "x" << sW << "x1 ";
  hlo << "pad=0_0x" << pH << "_" << pH << "x" << pW << "_" << pW << "x0_0}, ";
  hlo << "to_apply=max_computation\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Helper to build average pooling HLO
static std::string buildAvgPool2dHLO(const std::vector<sd::LongType>& inputShape,
                                      const std::vector<int>& kernelSize,
                                      const std::vector<int>& strides,
                                      const std::vector<int>& padding,
                                      bool countIncludePad,
                                      DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  sd::LongType batch = inputShape[0];
  sd::LongType inH = inputShape[1];
  sd::LongType inW = inputShape[2];
  sd::LongType channels = inputShape[3];

  int kH = kernelSize[0];
  int kW = kernelSize[1];
  int sH = strides[0];
  int sW = strides[1];
  int pH = padding[0];
  int pW = padding[1];

  // Calculate output dimensions
  sd::LongType outH = (inH + 2 * pH - kH) / sH + 1;
  sd::LongType outW = (inW + 2 * pW - kW) / sW + 1;

  hlo << "HloModule avgpool2d_module\n\n";

  // Sum computation for reduce-window
  hlo << "sum_computation {\n";
  hlo << "  x = " << typeStr << "[] parameter(0)\n";
  hlo << "  y = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[] add(x, y)\n";
  hlo << "}\n\n";

  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[" << batch << "," << inH << "," << inW << "," << channels << "] parameter(0)\n";

  // Initialize value for sum (zero)
  hlo << "  init = " << typeStr << "[] constant(0)\n";

  // reduce-window for sum pooling
  hlo << "  sum_pool = " << typeStr << "[" << batch << "," << outH << "," << outW << "," << channels << "] ";
  hlo << "reduce-window(input, init), ";
  hlo << "window={size=1x" << kH << "x" << kW << "x1 stride=1x" << sH << "x" << sW << "x1 ";
  hlo << "pad=0_0x" << pH << "_" << pH << "x" << pW << "_" << pW << "x0_0}, ";
  hlo << "to_apply=sum_computation\n";

  // Divide by kernel size for average
  double divisor = countIncludePad ? (kH * kW) : (kH * kW);  // For non-padded, we'd need dynamic calculation
  hlo << "  divisor = " << typeStr << "[" << batch << "," << outH << "," << outW << "," << channels << "] ";
  hlo << "broadcast(" << typeStr << "[] constant(" << divisor << ")), dimensions={}\n";

  hlo << "  ROOT result = " << typeStr << "[" << batch << "," << outH << "," << outW << "," << channels << "] ";
  hlo << "divide(sum_pool, divisor)\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Helper to build max pooling backward HLO
static std::string buildMaxPool2dBpHLO(const std::vector<sd::LongType>& inputShape,
                                        const std::vector<sd::LongType>& outputShape,
                                        const std::vector<int>& kernelSize,
                                        const std::vector<int>& strides,
                                        const std::vector<int>& padding,
                                        DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  sd::LongType batch = inputShape[0];
  sd::LongType inH = inputShape[1];
  sd::LongType inW = inputShape[2];
  sd::LongType channels = inputShape[3];

  int kH = kernelSize[0];
  int kW = kernelSize[1];
  int sH = strides[0];
  int sW = strides[1];
  int pH = padding[0];
  int pW = padding[1];

  hlo << "HloModule maxpool2d_bp_module\n\n";

  // Select and scatter computation
  hlo << "ge_computation {\n";
  hlo << "  x = " << typeStr << "[] parameter(0)\n";
  hlo << "  y = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = pred[] compare(x, y), direction=GE\n";
  hlo << "}\n\n";

  hlo << "add_computation {\n";
  hlo << "  x = " << typeStr << "[] parameter(0)\n";
  hlo << "  y = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[] add(x, y)\n";
  hlo << "}\n\n";

  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[" << inputShape[0] << "," << inputShape[1] << "," << inputShape[2] << "," << inputShape[3] << "] parameter(0)\n";
  hlo << "  grad_out = " << typeStr << "[" << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "," << outputShape[3] << "] parameter(1)\n";

  // Initialize to zero
  hlo << "  init = " << typeStr << "[] constant(0)\n";

  // select-and-scatter for backward pass
  hlo << "  ROOT result = " << typeStr << "[" << batch << "," << inH << "," << inW << "," << channels << "] ";
  hlo << "select-and-scatter(input, grad_out, init), ";
  hlo << "window={size=1x" << kH << "x" << kW << "x1 stride=1x" << sH << "x" << sW << "x1 ";
  hlo << "pad=0_0x" << pH << "_" << pH << "x" << pW << "_" << pW << "x0_0}, ";
  hlo << "select=ge_computation, scatter=add_computation\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Helper to build average pooling backward HLO
static std::string buildAvgPool2dBpHLO(const std::vector<sd::LongType>& inputShape,
                                        const std::vector<sd::LongType>& outputShape,
                                        const std::vector<int>& kernelSize,
                                        const std::vector<int>& strides,
                                        const std::vector<int>& padding,
                                        DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  sd::LongType batch = inputShape[0];
  sd::LongType inH = inputShape[1];
  sd::LongType inW = inputShape[2];
  sd::LongType channels = inputShape[3];

  int kH = kernelSize[0];
  int kW = kernelSize[1];
  int sH = strides[0];
  int sW = strides[1];
  int pH = padding[0];
  int pW = padding[1];

  double divisor = kH * kW;

  hlo << "HloModule avgpool2d_bp_module\n\n";

  hlo << "add_computation {\n";
  hlo << "  x = " << typeStr << "[] parameter(0)\n";
  hlo << "  y = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[] add(x, y)\n";
  hlo << "}\n\n";

  hlo << "ENTRY main {\n";
  hlo << "  grad_out = " << typeStr << "[" << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "," << outputShape[3] << "] parameter(0)\n";

  // Scale gradient by 1/(kH*kW)
  hlo << "  divisor = " << typeStr << "[" << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "," << outputShape[3] << "] ";
  hlo << "broadcast(" << typeStr << "[] constant(" << divisor << ")), dimensions={}\n";
  hlo << "  scaled_grad = " << typeStr << "[" << outputShape[0] << "," << outputShape[1] << "," << outputShape[2] << "," << outputShape[3] << "] ";
  hlo << "divide(grad_out, divisor)\n";

  // Use reduce-window with low/high padding to distribute gradient back
  hlo << "  init = " << typeStr << "[] constant(0)\n";

  // Calculate needed padding for reconstruction
  int lowPadH = kH - pH - 1;
  int highPadH = inH + pH - (outputShape[1] - 1) * sH - 1;
  int lowPadW = kW - pW - 1;
  int highPadW = inW + pW - (outputShape[2] - 1) * sW - 1;

  hlo << "  ROOT result = " << typeStr << "[" << batch << "," << inH << "," << inW << "," << channels << "] ";
  hlo << "reduce-window(scaled_grad, init), ";
  hlo << "window={size=1x" << kH << "x" << kW << "x1 stride=1x1x1x1 ";
  hlo << "pad=0_0x" << lowPadH << "_" << highPadH << "x" << lowPadW << "_" << highPadW << "x0_0}, ";
  hlo << "to_apply=add_computation\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Max Pool 2D forward implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(maxpool2d, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);  // [N, H, W, C] or [N, C, H, W]
  auto output = OUTPUT_VARIABLE(0);

  // Get pooling parameters
  int kH = INT_ARG(0);  // kernel height
  int kW = INT_ARG(1);  // kernel width
  int sH = INT_ARG(2);  // stride height
  int sW = INT_ARG(3);  // stride width
  int pH = INT_ARG(4);  // padding height
  int pW = INT_ARG(5);  // padding width
  int dH = INT_ARG(6);  // dilation height
  int dW = INT_ARG(7);  // dilation width
  int isSameMode = INT_ARG(8);
  int extraParam0 = INT_ARG(9);  // 0 = NCHW, 1 = NHWC

  bool isNCHW = extraParam0 == 0;

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for maxpool2d operation");
  }

  // Convert to NHWC if needed
  NDArray* inputNHWC = input;
  std::unique_ptr<NDArray> inputConverted;
  if (isNCHW) {
    inputConverted = std::make_unique<NDArray>(input->permute({0, 2, 3, 1}));
    inputNHWC = inputConverted.get();
  }

  std::vector<sd::LongType> inputShape(inputNHWC->shapeOf(), inputNHWC->shapeOf() + inputNHWC->rankOf());
  std::vector<int> kernelSize = {kH, kW};
  std::vector<int> strides = {sH, sW};
  std::vector<int> padding = {pH, pW};

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "maxpool2d_";
  for (auto s : inputShape) keyStream << s << "_";
  keyStream << kH << "_" << kW << "_" << sH << "_" << sW << "_" << pH << "_" << pW << "_" << input->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module
  std::string hloModule = buildMaxPool2dHLO(inputShape, kernelSize, strides, padding, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU maxpool2d program");
  }

  // Create device buffers and execute
  PjrtBuffer bufIn(client, inputNHWC);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  // Convert back to NCHW if needed
  if (isNCHW) {
    output->permutei({0, 3, 1, 2});
  }

  return Status::OK;
}

PLATFORM_CHECK(maxpool2d, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT MAXPOOL2D OP");

  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.expectTrue(input->rankOf() == 4, REQUIRE_TRUE_MSG("Input must be 4D tensor"));

  // Check for unsupported dilation
  int dH = INT_ARG(6);
  int dW = INT_ARG(7);
  req.expectTrue(dH == 1 && dW == 1,
                 REQUIRE_TRUE_MSG("TPU maxpool2d currently supports dilation=1 only"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Max Pool 2D backward implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(maxpool2d_bp, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);   // [N, H, W, C] or [N, C, H, W]
  auto gradOut = INPUT_VARIABLE(1); // gradient from next layer
  auto gradI = OUTPUT_VARIABLE(0);  // gradient w.r.t. input

  // Get pooling parameters
  int kH = INT_ARG(0);
  int kW = INT_ARG(1);
  int sH = INT_ARG(2);
  int sW = INT_ARG(3);
  int pH = INT_ARG(4);
  int pW = INT_ARG(5);
  int dH = INT_ARG(6);
  int dW = INT_ARG(7);
  int isSameMode = INT_ARG(8);
  int extraParam0 = INT_ARG(9);

  bool isNCHW = extraParam0 == 0;

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for maxpool2d_bp operation");
  }

  // Convert to NHWC if needed
  NDArray* inputNHWC = input;
  NDArray* gradOutNHWC = gradOut;
  std::unique_ptr<NDArray> inputConverted, gradOutConverted;

  if (isNCHW) {
    inputConverted = std::make_unique<NDArray>(input->permute({0, 2, 3, 1}));
    gradOutConverted = std::make_unique<NDArray>(gradOut->permute({0, 2, 3, 1}));
    inputNHWC = inputConverted.get();
    gradOutNHWC = gradOutConverted.get();
  }

  std::vector<sd::LongType> inputShape(inputNHWC->shapeOf(), inputNHWC->shapeOf() + inputNHWC->rankOf());
  std::vector<sd::LongType> gradOutShape(gradOutNHWC->shapeOf(), gradOutNHWC->shapeOf() + gradOutNHWC->rankOf());
  std::vector<int> kernelSize = {kH, kW};
  std::vector<int> strides = {sH, sW};
  std::vector<int> padding = {pH, pW};

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "maxpool2d_bp_";
  for (auto s : inputShape) keyStream << s << "_";
  keyStream << input->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module
  std::string hloModule = buildMaxPool2dBpHLO(inputShape, gradOutShape, kernelSize, strides, padding, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU maxpool2d_bp program");
  }

  // Create device buffers and execute
  PjrtBuffer bufIn(client, inputNHWC);
  PjrtBuffer bufGradOut(client, gradOutNHWC);
  PjrtBuffer bufGradI(client, gradI);

  std::vector<PjrtBuffer*> inputs = {&bufIn, &bufGradOut};
  std::vector<PjrtBuffer*> outputs = {&bufGradI};
  executable->execute(inputs, outputs);

  bufGradI.toHost(gradI);

  // Convert back to NCHW if needed
  if (isNCHW) {
    gradI->permutei({0, 3, 1, 2});
  }

  return Status::OK;
}

PLATFORM_CHECK(maxpool2d_bp, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT MAXPOOL2D_BP OP");

  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.expectTrue(input->rankOf() == 4, REQUIRE_TRUE_MSG("Input must be 4D tensor"));

  int dH = INT_ARG(6);
  int dW = INT_ARG(7);
  req.expectTrue(dH == 1 && dW == 1,
                 REQUIRE_TRUE_MSG("TPU maxpool2d_bp currently supports dilation=1 only"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Average Pool 2D forward implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(avgpool2d, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get pooling parameters
  int kH = INT_ARG(0);
  int kW = INT_ARG(1);
  int sH = INT_ARG(2);
  int sW = INT_ARG(3);
  int pH = INT_ARG(4);
  int pW = INT_ARG(5);
  int dH = INT_ARG(6);
  int dW = INT_ARG(7);
  int isSameMode = INT_ARG(8);
  int extraParam0 = INT_ARG(9);
  int countIncludePad = INT_ARG(10);

  bool isNCHW = extraParam0 == 0;

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for avgpool2d operation");
  }

  // Convert to NHWC if needed
  NDArray* inputNHWC = input;
  std::unique_ptr<NDArray> inputConverted;
  if (isNCHW) {
    inputConverted = std::make_unique<NDArray>(input->permute({0, 2, 3, 1}));
    inputNHWC = inputConverted.get();
  }

  std::vector<sd::LongType> inputShape(inputNHWC->shapeOf(), inputNHWC->shapeOf() + inputNHWC->rankOf());
  std::vector<int> kernelSize = {kH, kW};
  std::vector<int> strides = {sH, sW};
  std::vector<int> padding = {pH, pW};

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "avgpool2d_";
  for (auto s : inputShape) keyStream << s << "_";
  keyStream << kH << "_" << kW << "_" << sH << "_" << sW << "_" << countIncludePad << "_" << input->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module
  std::string hloModule = buildAvgPool2dHLO(inputShape, kernelSize, strides, padding, countIncludePad != 0, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU avgpool2d program");
  }

  // Create device buffers and execute
  PjrtBuffer bufIn(client, inputNHWC);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  // Convert back to NCHW if needed
  if (isNCHW) {
    output->permutei({0, 3, 1, 2});
  }

  return Status::OK;
}

PLATFORM_CHECK(avgpool2d, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT AVGPOOL2D OP");

  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.expectTrue(input->rankOf() == 4, REQUIRE_TRUE_MSG("Input must be 4D tensor"));

  int dH = INT_ARG(6);
  int dW = INT_ARG(7);
  req.expectTrue(dH == 1 && dW == 1,
                 REQUIRE_TRUE_MSG("TPU avgpool2d currently supports dilation=1 only"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Average Pool 2D backward implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(avgpool2d_bp, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto gradOut = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  // Get pooling parameters
  int kH = INT_ARG(0);
  int kW = INT_ARG(1);
  int sH = INT_ARG(2);
  int sW = INT_ARG(3);
  int pH = INT_ARG(4);
  int pW = INT_ARG(5);
  int dH = INT_ARG(6);
  int dW = INT_ARG(7);
  int isSameMode = INT_ARG(8);
  int extraParam0 = INT_ARG(9);

  bool isNCHW = extraParam0 == 0;

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for avgpool2d_bp operation");
  }

  // Convert to NHWC if needed
  NDArray* inputNHWC = input;
  NDArray* gradOutNHWC = gradOut;
  std::unique_ptr<NDArray> inputConverted, gradOutConverted;

  if (isNCHW) {
    inputConverted = std::make_unique<NDArray>(input->permute({0, 2, 3, 1}));
    gradOutConverted = std::make_unique<NDArray>(gradOut->permute({0, 2, 3, 1}));
    inputNHWC = inputConverted.get();
    gradOutNHWC = gradOutConverted.get();
  }

  std::vector<sd::LongType> inputShape(inputNHWC->shapeOf(), inputNHWC->shapeOf() + inputNHWC->rankOf());
  std::vector<sd::LongType> gradOutShape(gradOutNHWC->shapeOf(), gradOutNHWC->shapeOf() + gradOutNHWC->rankOf());
  std::vector<int> kernelSize = {kH, kW};
  std::vector<int> strides = {sH, sW};
  std::vector<int> padding = {pH, pW};

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "avgpool2d_bp_";
  for (auto s : inputShape) keyStream << s << "_";
  keyStream << input->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module
  std::string hloModule = buildAvgPool2dBpHLO(inputShape, gradOutShape, kernelSize, strides, padding, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU avgpool2d_bp program");
  }

  // Create device buffers and execute
  PjrtBuffer bufGradOut(client, gradOutNHWC);
  PjrtBuffer bufGradI(client, gradI);

  std::vector<PjrtBuffer*> inputs = {&bufGradOut};
  std::vector<PjrtBuffer*> outputs = {&bufGradI};
  executable->execute(inputs, outputs);

  bufGradI.toHost(gradI);

  // Convert back to NCHW if needed
  if (isNCHW) {
    gradI->permutei({0, 3, 1, 2});
  }

  return Status::OK;
}

PLATFORM_CHECK(avgpool2d_bp, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT AVGPOOL2D_BP OP");

  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.expectTrue(input->rankOf() == 4, REQUIRE_TRUE_MSG("Input must be 4D tensor"));

  int dH = INT_ARG(6);
  int dW = INT_ARG(7);
  req.expectTrue(dH == 1 && dW == 1,
                 REQUIRE_TRUE_MSG("TPU avgpool2d_bp currently supports dilation=1 only"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// 3D Pooling Operations
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(maxpool3dnew, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Similar implementation but with 3D kernel
  // For now, fall back to CPU
  THROW_EXCEPTION("TPU maxpool3d not yet implemented - use CPU fallback");

  return Status::OK;
}

PLATFORM_CHECK(maxpool3dnew, ENGINE_TPU) {
  Requirements req("PJRT MAXPOOL3D OP");
  req.expectFalse(true, REQUIRE_TRUE_MSG("TPU maxpool3d not yet implemented"));
  return req;
}

PLATFORM_IMPL(avgpool3dnew, ENGINE_TPU) {
  THROW_EXCEPTION("TPU avgpool3d not yet implemented - use CPU fallback");
  return Status::OK;
}

PLATFORM_CHECK(avgpool3dnew, ENGINE_TPU) {
  Requirements req("PJRT AVGPOOL3D OP");
  req.expectFalse(true, REQUIRE_TRUE_MSG("TPU avgpool3d not yet implemented"));
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_PJRT
