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
// Activation function HLO builders
//////////////////////////////////////////////////////////////////////////

static std::string buildReluHLO(const std::vector<sd::LongType>& shape, DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);
  std::string shapeStr = shapeToString(shape);

  hlo << "HloModule relu_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  x = " << typeStr << "[" << shapeStr << "] parameter(0)\n";
  hlo << "  zero = " << typeStr << "[] constant(0)\n";
  hlo << "  zero_broadcast = " << typeStr << "[" << shapeStr << "] broadcast(zero), dimensions={}\n";
  hlo << "  ROOT result = " << typeStr << "[" << shapeStr << "] maximum(x, zero_broadcast)\n";
  hlo << "}\n";

  return hlo.str();
}

static std::string buildSigmoidHLO(const std::vector<sd::LongType>& shape, DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);
  std::string shapeStr = shapeToString(shape);

  hlo << "HloModule sigmoid_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  x = " << typeStr << "[" << shapeStr << "] parameter(0)\n";
  hlo << "  ROOT result = " << typeStr << "[" << shapeStr << "] logistic(x)\n";
  hlo << "}\n";

  return hlo.str();
}

static std::string buildTanhHLO(const std::vector<sd::LongType>& shape, DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);
  std::string shapeStr = shapeToString(shape);

  hlo << "HloModule tanh_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  x = " << typeStr << "[" << shapeStr << "] parameter(0)\n";
  hlo << "  ROOT result = " << typeStr << "[" << shapeStr << "] tanh(x)\n";
  hlo << "}\n";

  return hlo.str();
}

static std::string buildSoftmaxHLO(const std::vector<sd::LongType>& shape, DataType dtype, int axis) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);
  std::string shapeStr = shapeToString(shape);

  // For softmax, we need: exp(x - max(x)) / sum(exp(x - max(x)))
  hlo << "HloModule softmax_module\n\n";
  hlo << "add_computation {\n";
  hlo << "  lhs = " << typeStr << "[] parameter(0)\n";
  hlo << "  rhs = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[] add(lhs, rhs)\n";
  hlo << "}\n\n";
  hlo << "max_computation {\n";
  hlo << "  lhs = " << typeStr << "[] parameter(0)\n";
  hlo << "  rhs = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[] maximum(lhs, rhs)\n";
  hlo << "}\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  x = " << typeStr << "[" << shapeStr << "] parameter(0)\n";

  // Build reduction shape (all dims except the softmax axis)
  std::vector<sd::LongType> reduceShape;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int>(i) != axis) {
      reduceShape.push_back(shape[i]);
    }
  }
  std::string reduceShapeStr = reduceShape.empty() ? "" : shapeToString(reduceShape);

  // Get max along axis
  hlo << "  neg_inf = " << typeStr << "[] constant(-inf)\n";
  if (reduceShape.empty()) {
    hlo << "  max_x = " << typeStr << "[] reduce(x, neg_inf), dimensions={" << axis << "}, to_apply=max_computation\n";
  } else {
    hlo << "  max_x = " << typeStr << "[" << reduceShapeStr << "] reduce(x, neg_inf), dimensions={" << axis << "}, to_apply=max_computation\n";
  }
  hlo << "  max_broadcast = " << typeStr << "[" << shapeStr << "] broadcast(max_x), dimensions={";
  bool first = true;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int>(i) != axis) {
      if (!first) hlo << ",";
      hlo << i;
      first = false;
    }
  }
  hlo << "}\n";

  // Subtract max and exponentiate
  hlo << "  x_shifted = " << typeStr << "[" << shapeStr << "] subtract(x, max_broadcast)\n";
  hlo << "  exp_x = " << typeStr << "[" << shapeStr << "] exponential(x_shifted)\n";

  // Sum along axis
  hlo << "  zero = " << typeStr << "[] constant(0)\n";
  if (reduceShape.empty()) {
    hlo << "  sum_exp = " << typeStr << "[] reduce(exp_x, zero), dimensions={" << axis << "}, to_apply=add_computation\n";
  } else {
    hlo << "  sum_exp = " << typeStr << "[" << reduceShapeStr << "] reduce(exp_x, zero), dimensions={" << axis << "}, to_apply=add_computation\n";
  }
  hlo << "  sum_broadcast = " << typeStr << "[" << shapeStr << "] broadcast(sum_exp), dimensions={";
  first = true;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int>(i) != axis) {
      if (!first) hlo << ",";
      hlo << i;
      first = false;
    }
  }
  hlo << "}\n";

  // Divide to get softmax
  hlo << "  ROOT result = " << typeStr << "[" << shapeStr << "] divide(exp_x, sum_broadcast)\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// ReLU implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(relu, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for relu operation");
  }

  std::vector<sd::LongType> shape(input->shapeOf(), input->shapeOf() + input->rankOf());

  // Build cache key and HLO
  std::ostringstream keyStream;
  keyStream << "relu_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << input->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildReluHLO(shape, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU relu program");
  }

  // Execute
  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);
  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);
  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(relu, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  Requirements req("PJRT RELU OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Sigmoid implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(sigmoid, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for sigmoid operation");
  }

  std::vector<sd::LongType> shape(input->shapeOf(), input->shapeOf() + input->rankOf());

  std::ostringstream keyStream;
  keyStream << "sigmoid_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << input->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildSigmoidHLO(shape, input->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU sigmoid program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);
  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);
  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(sigmoid, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  Requirements req("PJRT SIGMOID OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Tanh implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(tanh, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for tanh operation");
  }

  std::vector<sd::LongType> shape(input->shapeOf(), input->shapeOf() + input->rankOf());

  std::ostringstream keyStream;
  keyStream << "tanh_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << input->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildTanhHLO(shape, input->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU tanh program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);
  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);
  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(tanh, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  Requirements req("PJRT TANH OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Softmax implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(softmax, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Get the axis for softmax (default to last dimension)
  int axis = block.getIArguments()->size() > 0 ? INT_ARG(0) : -1;
  if (axis < 0) {
    axis = input->rankOf() + axis;
  }

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for softmax operation");
  }

  std::vector<sd::LongType> shape(input->shapeOf(), input->shapeOf() + input->rankOf());

  std::ostringstream keyStream;
  keyStream << "softmax_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << input->dataType() << "_axis" << axis;
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildSoftmaxHLO(shape, input->dataType(), axis);

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU softmax program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);
  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);
  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(softmax, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  Requirements req("PJRT SOFTMAX OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Log Softmax implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(log_softmax, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  int axis = block.getIArguments()->size() > 0 ? INT_ARG(0) : -1;
  if (axis < 0) {
    axis = input->rankOf() + axis;
  }

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for log_softmax operation");
  }

  // For log_softmax, we use: x - log(sum(exp(x)))
  // This is more numerically stable as: x - max(x) - log(sum(exp(x - max(x))))
  std::vector<sd::LongType> shape(input->shapeOf(), input->shapeOf() + input->rankOf());

  std::ostringstream keyStream;
  keyStream << "log_softmax_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << input->dataType() << "_axis" << axis;
  std::string cacheKey = keyStream.str();

  // Build log_softmax HLO (similar to softmax but with log at the end)
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(input->dataType());
  std::string shapeStr = shapeToString(shape);

  hlo << "HloModule log_softmax_module\n\n";
  hlo << "add_computation {\n";
  hlo << "  lhs = " << typeStr << "[] parameter(0)\n";
  hlo << "  rhs = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[] add(lhs, rhs)\n";
  hlo << "}\n\n";
  hlo << "max_computation {\n";
  hlo << "  lhs = " << typeStr << "[] parameter(0)\n";
  hlo << "  rhs = " << typeStr << "[] parameter(1)\n";
  hlo << "  ROOT result = " << typeStr << "[] maximum(lhs, rhs)\n";
  hlo << "}\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  x = " << typeStr << "[" << shapeStr << "] parameter(0)\n";
  hlo << "  neg_inf = " << typeStr << "[] constant(-inf)\n";
  hlo << "  max_x = " << typeStr << "[] reduce(x, neg_inf), dimensions={" << axis << "}, to_apply=max_computation\n";
  hlo << "  max_broadcast = " << typeStr << "[" << shapeStr << "] broadcast(max_x), dimensions={}\n";
  hlo << "  x_shifted = " << typeStr << "[" << shapeStr << "] subtract(x, max_broadcast)\n";
  hlo << "  exp_x = " << typeStr << "[" << shapeStr << "] exponential(x_shifted)\n";
  hlo << "  zero = " << typeStr << "[] constant(0)\n";
  hlo << "  sum_exp = " << typeStr << "[] reduce(exp_x, zero), dimensions={" << axis << "}, to_apply=add_computation\n";
  hlo << "  log_sum = " << typeStr << "[] log(sum_exp)\n";
  hlo << "  log_broadcast = " << typeStr << "[" << shapeStr << "] broadcast(log_sum), dimensions={}\n";
  hlo << "  ROOT result = " << typeStr << "[" << shapeStr << "] subtract(x_shifted, log_broadcast)\n";
  hlo << "}\n";

  std::string hloModule = hlo.str();

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU log_softmax program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);
  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);
  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(log_softmax, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  Requirements req("PJRT LOG_SOFTMAX OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_PJRT
