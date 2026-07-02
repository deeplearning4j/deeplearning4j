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
// Helper to build reduction HLO for various operations
static std::string buildReductionHLO(const std::vector<sd::LongType>& inputShape,
                                      const std::vector<sd::LongType>& outputShape,
                                      const std::vector<int>& dimensions,
                                      const std::string& reductionType,
                                      DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule reduction_" << reductionType << "_module\n\n";

  // Build the reduction computation
  if (reductionType == "sum" || reductionType == "mean") {
    hlo << "add_computation {\n";
    hlo << "  x = " << typeStr << "[] parameter(0)\n";
    hlo << "  y = " << typeStr << "[] parameter(1)\n";
    hlo << "  ROOT result = " << typeStr << "[] add(x, y)\n";
    hlo << "}\n\n";
  } else if (reductionType == "max") {
    hlo << "max_computation {\n";
    hlo << "  x = " << typeStr << "[] parameter(0)\n";
    hlo << "  y = " << typeStr << "[] parameter(1)\n";
    hlo << "  ROOT result = " << typeStr << "[] maximum(x, y)\n";
    hlo << "}\n\n";
  } else if (reductionType == "min") {
    hlo << "min_computation {\n";
    hlo << "  x = " << typeStr << "[] parameter(0)\n";
    hlo << "  y = " << typeStr << "[] parameter(1)\n";
    hlo << "  ROOT result = " << typeStr << "[] minimum(x, y)\n";
    hlo << "}\n\n";
  } else if (reductionType == "prod") {
    hlo << "mul_computation {\n";
    hlo << "  x = " << typeStr << "[] parameter(0)\n";
    hlo << "  y = " << typeStr << "[] parameter(1)\n";
    hlo << "  ROOT result = " << typeStr << "[] multiply(x, y)\n";
    hlo << "}\n\n";
  }

  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] parameter(0)\n";

  // Initial value for reduction
  std::string initVal;
  std::string compName;
  if (reductionType == "sum" || reductionType == "mean") {
    initVal = "0";
    compName = "add_computation";
  } else if (reductionType == "max") {
    initVal = "-inf";
    compName = "max_computation";
  } else if (reductionType == "min") {
    initVal = "inf";
    compName = "min_computation";
  } else if (reductionType == "prod") {
    initVal = "1";
    compName = "mul_computation";
  }

  hlo << "  init = " << typeStr << "[] constant(" << initVal << ")\n";

  // Perform reduction
  hlo << "  reduced = " << typeStr << "[";
  for (size_t i = 0; i < outputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << outputShape[i];
  }
  hlo << "] reduce(input, init), dimensions={";
  for (size_t i = 0; i < dimensions.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << dimensions[i];
  }
  hlo << "}, to_apply=" << compName << "\n";

  // For mean, divide by count
  if (reductionType == "mean") {
    sd::LongType count = 1;
    for (int dim : dimensions) {
      count *= inputShape[dim];
    }
    hlo << "  divisor = " << typeStr << "[";
    for (size_t i = 0; i < outputShape.size(); i++) {
      if (i > 0) hlo << ",";
      hlo << outputShape[i];
    }
    hlo << "] broadcast(" << typeStr << "[] constant(" << count << ")), dimensions={}\n";
    hlo << "  ROOT result = " << typeStr << "[";
    for (size_t i = 0; i < outputShape.size(); i++) {
      if (i > 0) hlo << ",";
      hlo << outputShape[i];
    }
    hlo << "] divide(reduced, divisor)\n";
  } else {
    hlo << "  ROOT result = " << typeStr << "[";
    for (size_t i = 0; i < outputShape.size(); i++) {
      if (i > 0) hlo << ",";
      hlo << outputShape[i];
    }
    hlo << "] copy(reduced)\n";
  }

  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Helper to build full reduction HLO (reduce to scalar)
static std::string buildFullReductionHLO(const std::vector<sd::LongType>& inputShape,
                                          const std::string& reductionType,
                                          DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule full_reduction_" << reductionType << "_module\n\n";

  // Build the reduction computation
  std::string compName;
  std::string initVal;

  if (reductionType == "sum" || reductionType == "mean") {
    hlo << "add_computation {\n";
    hlo << "  x = " << typeStr << "[] parameter(0)\n";
    hlo << "  y = " << typeStr << "[] parameter(1)\n";
    hlo << "  ROOT result = " << typeStr << "[] add(x, y)\n";
    hlo << "}\n\n";
    compName = "add_computation";
    initVal = "0";
  } else if (reductionType == "max") {
    hlo << "max_computation {\n";
    hlo << "  x = " << typeStr << "[] parameter(0)\n";
    hlo << "  y = " << typeStr << "[] parameter(1)\n";
    hlo << "  ROOT result = " << typeStr << "[] maximum(x, y)\n";
    hlo << "}\n\n";
    compName = "max_computation";
    initVal = "-inf";
  } else if (reductionType == "min") {
    hlo << "min_computation {\n";
    hlo << "  x = " << typeStr << "[] parameter(0)\n";
    hlo << "  y = " << typeStr << "[] parameter(1)\n";
    hlo << "  ROOT result = " << typeStr << "[] minimum(x, y)\n";
    hlo << "}\n\n";
    compName = "min_computation";
    initVal = "inf";
  } else if (reductionType == "prod") {
    hlo << "mul_computation {\n";
    hlo << "  x = " << typeStr << "[] parameter(0)\n";
    hlo << "  y = " << typeStr << "[] parameter(1)\n";
    hlo << "  ROOT result = " << typeStr << "[] multiply(x, y)\n";
    hlo << "}\n\n";
    compName = "mul_computation";
    initVal = "1";
  }

  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] parameter(0)\n";

  hlo << "  init = " << typeStr << "[] constant(" << initVal << ")\n";

  // Reduce all dimensions
  hlo << "  reduced = " << typeStr << "[] reduce(input, init), dimensions={";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << i;
  }
  hlo << "}, to_apply=" << compName << "\n";

  if (reductionType == "mean") {
    sd::LongType count = 1;
    for (auto s : inputShape) count *= s;
    hlo << "  divisor = " << typeStr << "[] constant(" << count << ")\n";
    hlo << "  ROOT result = " << typeStr << "[] divide(reduced, divisor)\n";
  } else {
    hlo << "  ROOT result = " << typeStr << "[] copy(reduced)\n";
  }

  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// reduce_sum implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(reduce_sum, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for reduce_sum operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  // Get dimensions to reduce
  std::vector<int> dimensions;
  if (block.getIArguments()->size() > 0) {
    for (size_t i = 0; i < block.getIArguments()->size(); i++) {
      int dim = INT_ARG(i);
      if (dim < 0) dim += input->rankOf();
      dimensions.push_back(dim);
    }
  }

  bool keepDims = block.getBArguments()->size() > 0 ? B_ARG(0) : false;

  std::string cacheKey;
  std::string hloModule;

  if (dimensions.empty()) {
    // Full reduction to scalar
    std::ostringstream keyStream;
    keyStream << "reduce_sum_full_";
    for (auto s : inputShape) keyStream << s << "_";
    keyStream << input->dataType();
    cacheKey = keyStream.str();

    hloModule = buildFullReductionHLO(inputShape, "sum", input->dataType());
  } else {
    // Partial reduction
    std::vector<sd::LongType> outputShape(output->shapeOf(), output->shapeOf() + output->rankOf());

    std::ostringstream keyStream;
    keyStream << "reduce_sum_";
    for (auto s : inputShape) keyStream << s << "_";
    for (auto d : dimensions) keyStream << d << "_";
    keyStream << input->dataType();
    cacheKey = keyStream.str();

    hloModule = buildReductionHLO(inputShape, outputShape, dimensions, "sum", input->dataType());
  }

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU reduce_sum program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(reduce_sum, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT REDUCE_SUM OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_mean implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(reduce_mean, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for reduce_mean operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  std::vector<int> dimensions;
  if (block.getIArguments()->size() > 0) {
    for (size_t i = 0; i < block.getIArguments()->size(); i++) {
      int dim = INT_ARG(i);
      if (dim < 0) dim += input->rankOf();
      dimensions.push_back(dim);
    }
  }

  std::string cacheKey;
  std::string hloModule;

  if (dimensions.empty()) {
    std::ostringstream keyStream;
    keyStream << "reduce_mean_full_";
    for (auto s : inputShape) keyStream << s << "_";
    keyStream << input->dataType();
    cacheKey = keyStream.str();

    hloModule = buildFullReductionHLO(inputShape, "mean", input->dataType());
  } else {
    std::vector<sd::LongType> outputShape(output->shapeOf(), output->shapeOf() + output->rankOf());

    std::ostringstream keyStream;
    keyStream << "reduce_mean_";
    for (auto s : inputShape) keyStream << s << "_";
    for (auto d : dimensions) keyStream << d << "_";
    keyStream << input->dataType();
    cacheKey = keyStream.str();

    hloModule = buildReductionHLO(inputShape, outputShape, dimensions, "mean", input->dataType());
  }

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU reduce_mean program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(reduce_mean, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT REDUCE_MEAN OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_max implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(reduce_max, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for reduce_max operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  std::vector<int> dimensions;
  if (block.getIArguments()->size() > 0) {
    for (size_t i = 0; i < block.getIArguments()->size(); i++) {
      int dim = INT_ARG(i);
      if (dim < 0) dim += input->rankOf();
      dimensions.push_back(dim);
    }
  }

  std::string cacheKey;
  std::string hloModule;

  if (dimensions.empty()) {
    std::ostringstream keyStream;
    keyStream << "reduce_max_full_";
    for (auto s : inputShape) keyStream << s << "_";
    keyStream << input->dataType();
    cacheKey = keyStream.str();

    hloModule = buildFullReductionHLO(inputShape, "max", input->dataType());
  } else {
    std::vector<sd::LongType> outputShape(output->shapeOf(), output->shapeOf() + output->rankOf());

    std::ostringstream keyStream;
    keyStream << "reduce_max_";
    for (auto s : inputShape) keyStream << s << "_";
    for (auto d : dimensions) keyStream << d << "_";
    keyStream << input->dataType();
    cacheKey = keyStream.str();

    hloModule = buildReductionHLO(inputShape, outputShape, dimensions, "max", input->dataType());
  }

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU reduce_max program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(reduce_max, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT REDUCE_MAX OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_min implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(reduce_min, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for reduce_min operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  std::vector<int> dimensions;
  if (block.getIArguments()->size() > 0) {
    for (size_t i = 0; i < block.getIArguments()->size(); i++) {
      int dim = INT_ARG(i);
      if (dim < 0) dim += input->rankOf();
      dimensions.push_back(dim);
    }
  }

  std::string cacheKey;
  std::string hloModule;

  if (dimensions.empty()) {
    std::ostringstream keyStream;
    keyStream << "reduce_min_full_";
    for (auto s : inputShape) keyStream << s << "_";
    keyStream << input->dataType();
    cacheKey = keyStream.str();

    hloModule = buildFullReductionHLO(inputShape, "min", input->dataType());
  } else {
    std::vector<sd::LongType> outputShape(output->shapeOf(), output->shapeOf() + output->rankOf());

    std::ostringstream keyStream;
    keyStream << "reduce_min_";
    for (auto s : inputShape) keyStream << s << "_";
    for (auto d : dimensions) keyStream << d << "_";
    keyStream << input->dataType();
    cacheKey = keyStream.str();

    hloModule = buildReductionHLO(inputShape, outputShape, dimensions, "min", input->dataType());
  }

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU reduce_min program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(reduce_min, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT REDUCE_MIN OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// reduce_prod implementation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(reduce_prod, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for reduce_prod operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  std::vector<int> dimensions;
  if (block.getIArguments()->size() > 0) {
    for (size_t i = 0; i < block.getIArguments()->size(); i++) {
      int dim = INT_ARG(i);
      if (dim < 0) dim += input->rankOf();
      dimensions.push_back(dim);
    }
  }

  std::string cacheKey;
  std::string hloModule;

  if (dimensions.empty()) {
    std::ostringstream keyStream;
    keyStream << "reduce_prod_full_";
    for (auto s : inputShape) keyStream << s << "_";
    keyStream << input->dataType();
    cacheKey = keyStream.str();

    hloModule = buildFullReductionHLO(inputShape, "prod", input->dataType());
  } else {
    std::vector<sd::LongType> outputShape(output->shapeOf(), output->shapeOf() + output->rankOf());

    std::ostringstream keyStream;
    keyStream << "reduce_prod_";
    for (auto s : inputShape) keyStream << s << "_";
    for (auto d : dimensions) keyStream << d << "_";
    keyStream << input->dataType();
    cacheKey = keyStream.str();

    hloModule = buildReductionHLO(inputShape, outputShape, dimensions, "prod", input->dataType());
  }

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU reduce_prod program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(reduce_prod, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT REDUCE_PROD OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// argmax implementation
//////////////////////////////////////////////////////////////////////////

static std::string buildArgmaxHLO(const std::vector<sd::LongType>& inputShape,
                                   int dimension,
                                   DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule argmax_module\n\n";

  // Comparison computation for argmax
  hlo << "argmax_computation {\n";
  hlo << "  val_a = " << typeStr << "[] parameter(0)\n";
  hlo << "  idx_a = s32[] parameter(1)\n";
  hlo << "  val_b = " << typeStr << "[] parameter(2)\n";
  hlo << "  idx_b = s32[] parameter(3)\n";
  hlo << "  cmp = pred[] compare(val_a, val_b), direction=GT\n";
  hlo << "  max_val = " << typeStr << "[] select(cmp, val_a, val_b)\n";
  hlo << "  max_idx = s32[] select(cmp, idx_a, idx_b)\n";
  hlo << "  ROOT result = (" << typeStr << "[], s32[]) tuple(max_val, max_idx)\n";
  hlo << "}\n\n";

  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] parameter(0)\n";

  // Create iota for indices
  hlo << "  iota = s32[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] iota(), iota_dimension=" << dimension << "\n";

  // Initial values
  hlo << "  init_val = " << typeStr << "[] constant(-inf)\n";
  hlo << "  init_idx = s32[] constant(0)\n";

  // Reduce with argmax computation
  hlo << "  result = (" << typeStr << "[";
  bool first = true;
  for (size_t i = 0; i < inputShape.size(); i++) {
    if ((int)i != dimension) {
      if (!first) hlo << ",";
      hlo << inputShape[i];
      first = false;
    }
  }
  hlo << "], s32[";
  first = true;
  for (size_t i = 0; i < inputShape.size(); i++) {
    if ((int)i != dimension) {
      if (!first) hlo << ",";
      hlo << inputShape[i];
      first = false;
    }
  }
  hlo << "]) reduce(input, iota, init_val, init_idx), ";
  hlo << "dimensions={" << dimension << "}, to_apply=argmax_computation\n";

  hlo << "  ROOT indices = s32[";
  first = true;
  for (size_t i = 0; i < inputShape.size(); i++) {
    if ((int)i != dimension) {
      if (!first) hlo << ",";
      hlo << inputShape[i];
      first = false;
    }
  }
  hlo << "] get-tuple-element(result), index=1\n";
  hlo << "}\n";

  return hlo.str();
}

PLATFORM_IMPL(argmax, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  int dimension = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
  if (dimension < 0) dimension += input->rankOf();

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for argmax operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  std::ostringstream keyStream;
  keyStream << "argmax_";
  for (auto s : inputShape) keyStream << s << "_";
  keyStream << dimension << "_" << input->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildArgmaxHLO(inputShape, dimension, input->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU argmax program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(argmax, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT ARGMAX OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// argmin implementation
//////////////////////////////////////////////////////////////////////////

static std::string buildArgminHLO(const std::vector<sd::LongType>& inputShape,
                                   int dimension,
                                   DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule argmin_module\n\n";

  hlo << "argmin_computation {\n";
  hlo << "  val_a = " << typeStr << "[] parameter(0)\n";
  hlo << "  idx_a = s32[] parameter(1)\n";
  hlo << "  val_b = " << typeStr << "[] parameter(2)\n";
  hlo << "  idx_b = s32[] parameter(3)\n";
  hlo << "  cmp = pred[] compare(val_a, val_b), direction=LT\n";
  hlo << "  min_val = " << typeStr << "[] select(cmp, val_a, val_b)\n";
  hlo << "  min_idx = s32[] select(cmp, idx_a, idx_b)\n";
  hlo << "  ROOT result = (" << typeStr << "[], s32[]) tuple(min_val, min_idx)\n";
  hlo << "}\n\n";

  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] parameter(0)\n";

  hlo << "  iota = s32[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] iota(), iota_dimension=" << dimension << "\n";

  hlo << "  init_val = " << typeStr << "[] constant(inf)\n";
  hlo << "  init_idx = s32[] constant(0)\n";

  hlo << "  result = (" << typeStr << "[";
  bool first = true;
  for (size_t i = 0; i < inputShape.size(); i++) {
    if ((int)i != dimension) {
      if (!first) hlo << ",";
      hlo << inputShape[i];
      first = false;
    }
  }
  hlo << "], s32[";
  first = true;
  for (size_t i = 0; i < inputShape.size(); i++) {
    if ((int)i != dimension) {
      if (!first) hlo << ",";
      hlo << inputShape[i];
      first = false;
    }
  }
  hlo << "]) reduce(input, iota, init_val, init_idx), ";
  hlo << "dimensions={" << dimension << "}, to_apply=argmin_computation\n";

  hlo << "  ROOT indices = s32[";
  first = true;
  for (size_t i = 0; i < inputShape.size(); i++) {
    if ((int)i != dimension) {
      if (!first) hlo << ",";
      hlo << inputShape[i];
      first = false;
    }
  }
  hlo << "] get-tuple-element(result), index=1\n";
  hlo << "}\n";

  return hlo.str();
}

PLATFORM_IMPL(argmin, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  int dimension = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
  if (dimension < 0) dimension += input->rankOf();

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for argmin operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  std::ostringstream keyStream;
  keyStream << "argmin_";
  for (auto s : inputShape) keyStream << s << "_";
  keyStream << dimension << "_" << input->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildArgminHLO(inputShape, dimension, input->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU argmin program");
  }

  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(argmin, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT ARGMIN OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_PJRT
