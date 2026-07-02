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
// Helper to build conv2d backward HLO for input gradient
static std::string buildConv2dInputGradHLO(const std::vector<sd::LongType>& inputShape,
                                            const std::vector<sd::LongType>& filterShape,
                                            const std::vector<sd::LongType>& gradOutShape,
                                            const std::vector<int>& strides,
                                            const std::vector<int>& padding,
                                            DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  sd::LongType batch = inputShape[0];
  sd::LongType inH = inputShape[1];
  sd::LongType inW = inputShape[2];
  sd::LongType inC = inputShape[3];
  sd::LongType kH = filterShape[0];
  sd::LongType kW = filterShape[1];
  sd::LongType outF = filterShape[3];

  hlo << "HloModule conv2d_input_grad_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  grad_out = " << typeStr << "[" << gradOutShape[0] << "," << gradOutShape[1] << ","
      << gradOutShape[2] << "," << gradOutShape[3] << "] parameter(0)\n";
  hlo << "  filter = " << typeStr << "[" << kH << "," << kW << "," << inC << "," << outF << "] parameter(1)\n";

  // Transpose filter for backward convolution
  hlo << "  filter_t = " << typeStr << "[" << kH << "," << kW << "," << outF << "," << inC << "] transpose(filter), dimensions={0,1,3,2}\n";

  // Reverse filter spatially
  hlo << "  filter_rev = " << typeStr << "[" << kH << "," << kW << "," << outF << "," << inC << "] reverse(filter_t), dimensions={0,1}\n";

  // Backward convolution (transposed convolution)
  hlo << "  ROOT result = " << typeStr << "[" << batch << "," << inH << "," << inW << "," << inC << "] ";
  hlo << "convolution(grad_out, filter_rev), ";
  hlo << "window={size=" << kH << "x" << kW << " stride=1x1";
  hlo << " pad=" << (kH - 1) << "_" << (kH - 1) << "x" << (kW - 1) << "_" << (kW - 1) << "}, ";
  hlo << "dim_labels=b01f_01oi->b01f\n";
  hlo << "}\n";

  return hlo.str();
}

// Helper to build conv2d backward HLO for filter gradient
static std::string buildConv2dFilterGradHLO(const std::vector<sd::LongType>& inputShape,
                                             const std::vector<sd::LongType>& filterShape,
                                             const std::vector<sd::LongType>& gradOutShape,
                                             const std::vector<int>& strides,
                                             const std::vector<int>& padding,
                                             DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  sd::LongType batch = inputShape[0];
  sd::LongType inH = inputShape[1];
  sd::LongType inW = inputShape[2];
  sd::LongType inC = inputShape[3];
  sd::LongType kH = filterShape[0];
  sd::LongType kW = filterShape[1];
  sd::LongType outF = filterShape[3];

  hlo << "HloModule conv2d_filter_grad_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[" << batch << "," << inH << "," << inW << "," << inC << "] parameter(0)\n";
  hlo << "  grad_out = " << typeStr << "[" << gradOutShape[0] << "," << gradOutShape[1] << ","
      << gradOutShape[2] << "," << gradOutShape[3] << "] parameter(1)\n";

  // Compute filter gradient using convolution
  hlo << "  ROOT result = " << typeStr << "[" << kH << "," << kW << "," << inC << "," << outF << "] ";
  hlo << "convolution(input, grad_out), ";
  hlo << "window={size=" << gradOutShape[1] << "x" << gradOutShape[2] << " stride=1x1";
  hlo << " pad=" << padding[0] << "_" << padding[0] << "x" << padding[1] << "_" << padding[1] << "}, ";
  hlo << "dim_labels=f01b_01bf->01bf\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Conv2D forward implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(conv2d, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);   // [N, H, W, C] or [N, C, H, W]
  auto weights = INPUT_VARIABLE(1); // [kH, kW, iC, oC]
  NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto output = OUTPUT_VARIABLE(0);

  // Get convolution parameters
  int kH = INT_ARG(0);  // kernel height
  int kW = INT_ARG(1);  // kernel width
  int sH = INT_ARG(2);  // stride height
  int sW = INT_ARG(3);  // stride width
  int pH = INT_ARG(4);  // padding height
  int pW = INT_ARG(5);  // padding width
  int dH = INT_ARG(6);  // dilation height
  int dW = INT_ARG(7);  // dilation width
  int isSameMode = INT_ARG(8);  // same mode
  int isNCHW = block.getIArguments()->size() > 9 ? INT_ARG(9) : 0;

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for conv2d operation");
  }

  // Convert to NHWC if needed (TPU prefers NHWC)
  NDArray* inputNHWC = input;
  NDArray* inputConverted = nullptr;
  if (isNCHW) {
    { std::vector<sd::LongType> _perm_inputConverted = {0, 2, 3, 1}; inputConverted = input->permute(_perm_inputConverted, false, false); };
    inputNHWC = inputConverted;
  }

  std::vector<sd::LongType> inputShape(inputNHWC->shapeOf(), inputNHWC->shapeOf() + inputNHWC->rankOf());
  std::vector<sd::LongType> filterShape(weights->shapeOf(), weights->shapeOf() + weights->rankOf());
  std::vector<int> strides = {sH, sW};
  std::vector<int> padding = {pH, pW};

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "conv2d_";
  for (auto s : inputShape) keyStream << s << "_";
  for (auto s : filterShape) keyStream << s << "_";
  keyStream << sH << "_" << sW << "_" << pH << "_" << pW << "_" << input->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module
  std::string hloModule = buildConv2dHLO(inputShape, filterShape, strides, padding, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU conv2d program");
  }

  // Create device buffers and execute
  PjrtBuffer bufIn(client, inputNHWC);
  PjrtBuffer bufWeights(client, weights);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn, &bufWeights};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  // Handle bias if present
  if (bias != nullptr) {
    // Add bias to output
    (*output) += (*bias);
  }

  bufOut.toHost(output);

  // Convert back to NCHW if needed
  if (isNCHW) {
    output->permutei({0, 3, 1, 2}, false, false);
  }

  delete inputConverted;
  return Status::OK;
}

PLATFORM_CHECK(conv2d, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);

  Requirements req("PJRT CONV2D OP");

  req.expectTrue(isTpuSupportedType(input->dataType()),
                 "Input data type must be TPU-compatible");
  req.expectTrue(isTpuSupportedType(weights->dataType()),
                 "Weights data type must be TPU-compatible");
  req.expectTrue(input->dataType() == weights->dataType(),
                 "Input and weights must have matching data types");
  req.expectTrue(input->rankOf() == 4, "Input must be 4D tensor");
  req.expectTrue(weights->rankOf() == 4, "Weights must be 4D tensor");

  // Check for unsupported dilation (TPU has limited dilation support)
  int dH = INT_ARG(6);
  int dW = INT_ARG(7);
  req.expectTrue(dH == 1 && dW == 1,
                 "TPU conv2d currently supports dilation=1 only");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Conv2D backward implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(conv2d_bp, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);    // [N, H, W, C] or [N, C, H, W]
  auto weights = INPUT_VARIABLE(1);  // [kH, kW, iC, oC]
  auto gradOut = INPUT_VARIABLE(2);  // gradient from next layer

  auto gradI = OUTPUT_VARIABLE(0);   // gradient w.r.t. input
  auto gradW = OUTPUT_VARIABLE(1);   // gradient w.r.t. weights
  NDArray* gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;

  // Get convolution parameters
  int kH = INT_ARG(0);
  int kW = INT_ARG(1);
  int sH = INT_ARG(2);
  int sW = INT_ARG(3);
  int pH = INT_ARG(4);
  int pW = INT_ARG(5);
  int dH = INT_ARG(6);
  int dW = INT_ARG(7);
  int isSameMode = INT_ARG(8);
  int isNCHW = block.getIArguments()->size() > 9 ? INT_ARG(9) : 0;

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for conv2d_bp operation");
  }

  // Convert to NHWC if needed
  NDArray* inputNHWC = input;
  NDArray* gradOutNHWC = gradOut;
  NDArray *inputConverted = nullptr, *gradOutConverted = nullptr;

  if (isNCHW) {
    { std::vector<sd::LongType> _perm_inputConverted = {0, 2, 3, 1}; inputConverted = input->permute(_perm_inputConverted, false, false); };
    { std::vector<sd::LongType> _perm_gradOutConverted = {0, 2, 3, 1}; gradOutConverted = gradOut->permute(_perm_gradOutConverted, false, false); };
    inputNHWC = inputConverted;
    gradOutNHWC = gradOutConverted;
  }

  std::vector<sd::LongType> inputShape(inputNHWC->shapeOf(), inputNHWC->shapeOf() + inputNHWC->rankOf());
  std::vector<sd::LongType> filterShape(weights->shapeOf(), weights->shapeOf() + weights->rankOf());
  std::vector<sd::LongType> gradOutShape(gradOutNHWC->shapeOf(), gradOutNHWC->shapeOf() + gradOutNHWC->rankOf());
  std::vector<int> strides = {sH, sW};
  std::vector<int> padding = {pH, pW};

  // Compute input gradient
  std::string inputGradHLO = buildConv2dInputGradHLO(inputShape, filterShape, gradOutShape, strides, padding, input->dataType());
  std::ostringstream keyStream1;
  keyStream1 << "conv2d_bp_input_";
  for (auto s : inputShape) keyStream1 << s << "_";
  keyStream1 << input->dataType();

  auto& cache = HLOExecutableCache::getInstance();
  auto execInputGrad = cache.getOrCompile(client, keyStream1.str(), inputGradHLO);

  if (execInputGrad && execInputGrad->isValid()) {
    PjrtBuffer bufGradOut(client, gradOutNHWC);
    PjrtBuffer bufWeights(client, weights);
    PjrtBuffer bufGradI(client, gradI);

    std::vector<PjrtBuffer*> inputs = {&bufGradOut, &bufWeights};
    std::vector<PjrtBuffer*> outputs = {&bufGradI};
    execInputGrad->execute(inputs, outputs);
    bufGradI.toHost(gradI);
  }

  // Compute filter gradient
  std::string filterGradHLO = buildConv2dFilterGradHLO(inputShape, filterShape, gradOutShape, strides, padding, input->dataType());
  std::ostringstream keyStream2;
  keyStream2 << "conv2d_bp_filter_";
  for (auto s : filterShape) keyStream2 << s << "_";
  keyStream2 << input->dataType();

  auto execFilterGrad = cache.getOrCompile(client, keyStream2.str(), filterGradHLO);

  if (execFilterGrad && execFilterGrad->isValid()) {
    PjrtBuffer bufInput(client, inputNHWC);
    PjrtBuffer bufGradOut(client, gradOutNHWC);
    PjrtBuffer bufGradW(client, gradW);

    std::vector<PjrtBuffer*> inputs = {&bufInput, &bufGradOut};
    std::vector<PjrtBuffer*> outputs = {&bufGradW};
    execFilterGrad->execute(inputs, outputs);
    bufGradW.toHost(gradW);
  }

  // Compute bias gradient if needed
  if (gradB != nullptr) {
    // Bias gradient is sum over batch and spatial dimensions
    std::vector<sd::LongType> biasAxes = {0, 1, 2};
    gradOutNHWC->reduceAlongDimension(reduce::Sum, gradB, &biasAxes);
  }

  // Convert gradients back to NCHW if needed
  if (isNCHW) {
    gradI->permutei({0, 3, 1, 2}, false, false);
  }

  delete inputConverted;
  delete gradOutConverted;
  return Status::OK;
}

PLATFORM_CHECK(conv2d_bp, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto gradOut = INPUT_VARIABLE(2);

  Requirements req("PJRT CONV2D_BP OP");

  req.expectTrue(isTpuSupportedType(input->dataType()),
                 "Input data type must be TPU-compatible");
  req.expectTrue(isTpuSupportedType(weights->dataType()),
                 "Weights data type must be TPU-compatible");
  req.expectTrue(input->rankOf() == 4, "Input must be 4D tensor");
  req.expectTrue(weights->rankOf() == 4, "Weights must be 4D tensor");

  int dH = INT_ARG(6);
  int dW = INT_ARG(7);
  req.expectTrue(dH == 1 && dW == 1,
                 "TPU conv2d_bp currently supports dilation=1 only");

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_PJRT
