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
// Helper to build transpose HLO
static std::string buildTransposeHLO(const std::vector<sd::LongType>& inputShape,
                                      const std::vector<int>& permutation,
                                      DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule transpose_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] parameter(0)\n";

  // Calculate output shape based on permutation
  std::vector<sd::LongType> outputShape(inputShape.size());
  for (size_t i = 0; i < permutation.size(); i++) {
    outputShape[i] = inputShape[permutation[i]];
  }

  hlo << "  ROOT result = " << typeStr << "[";
  for (size_t i = 0; i < outputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << outputShape[i];
  }
  hlo << "] transpose(input), dimensions={";
  for (size_t i = 0; i < permutation.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << permutation[i];
  }
  hlo << "}\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Helper to build reshape HLO
static std::string buildReshapeHLO(const std::vector<sd::LongType>& inputShape,
                                    const std::vector<sd::LongType>& outputShape,
                                    DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule reshape_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] parameter(0)\n";

  hlo << "  ROOT result = " << typeStr << "[";
  for (size_t i = 0; i < outputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << outputShape[i];
  }
  hlo << "] reshape(input)\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Helper to build concat HLO
static std::string buildConcatHLO(const std::vector<std::vector<sd::LongType>>& inputShapes,
                                   const std::vector<sd::LongType>& outputShape,
                                   int axis,
                                   DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule concat_module\n\n";
  hlo << "ENTRY main {\n";

  // Define all input parameters
  for (size_t i = 0; i < inputShapes.size(); i++) {
    hlo << "  input" << i << " = " << typeStr << "[";
    for (size_t j = 0; j < inputShapes[i].size(); j++) {
      if (j > 0) hlo << ",";
      hlo << inputShapes[i][j];
    }
    hlo << "] parameter(" << i << ")\n";
  }

  // Concatenate
  hlo << "  ROOT result = " << typeStr << "[";
  for (size_t i = 0; i < outputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << outputShape[i];
  }
  hlo << "] concatenate(";
  for (size_t i = 0; i < inputShapes.size(); i++) {
    if (i > 0) hlo << ", ";
    hlo << "input" << i;
  }
  hlo << "), dimensions={" << axis << "}\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Transpose operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(transpose, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for transpose operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  // Get permutation from arguments or default (reverse)
  std::vector<int> permutation;
  if (block.getIArguments()->size() > 0) {
    for (size_t i = 0; i < block.getIArguments()->size(); i++) {
      permutation.push_back(INT_ARG(i));
    }
  } else {
    // Default: reverse all dimensions
    for (int i = input->rankOf() - 1; i >= 0; i--) {
      permutation.push_back(i);
    }
  }

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "transpose_";
  for (auto s : inputShape) keyStream << s << "_";
  for (auto p : permutation) keyStream << p << "_";
  keyStream << input->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module
  std::string hloModule = buildTransposeHLO(inputShape, permutation, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU transpose program");
  }

  // Create device buffers and execute
  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(transpose, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT TRANSPOSE OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Reshape operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(reshape, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for reshape operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());
  std::vector<sd::LongType> outputShape(output->shapeOf(), output->shapeOf() + output->rankOf());

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "reshape_";
  for (auto s : inputShape) keyStream << s << "_";
  for (auto s : outputShape) keyStream << s << "_";
  keyStream << input->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module
  std::string hloModule = buildReshapeHLO(inputShape, outputShape, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU reshape program");
  }

  // Create device buffers and execute
  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(reshape, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT RESHAPE OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Concat operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(concat, ENGINE_TPU) {
  auto output = OUTPUT_VARIABLE(0);

  int numInputs = block.width() - 1;  // Last input might be axis
  int axis = INT_ARG(0);
  if (axis < 0) axis += output->rankOf();

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for concat operation");
  }

  // Collect input shapes
  std::vector<std::vector<sd::LongType>> inputShapes;
  std::vector<NDArray*> inputArrays;
  for (int i = 0; i < numInputs; i++) {
    auto inp = INPUT_VARIABLE(i);
    inputArrays.push_back(inp);
    inputShapes.push_back(std::vector<sd::LongType>(inp->shapeOf(), inp->shapeOf() + inp->rankOf()));
  }

  std::vector<sd::LongType> outputShape(output->shapeOf(), output->shapeOf() + output->rankOf());

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "concat_" << axis << "_";
  for (const auto& shape : inputShapes) {
    for (auto s : shape) keyStream << s << "_";
  }
  keyStream << inputArrays[0]->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module
  std::string hloModule = buildConcatHLO(inputShapes, outputShape, axis, inputArrays[0]->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU concat program");
  }

  // Create device buffers
  std::vector<std::unique_ptr<PjrtBuffer>> inputBuffers;
  std::vector<PjrtBuffer*> inputs;
  for (auto arr : inputArrays) {
    inputBuffers.push_back(std::make_unique<PjrtBuffer>(client, arr));
    inputs.push_back(inputBuffers.back().get());
  }

  PjrtBuffer bufOut(client, output);
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(concat, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT CONCAT OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Slice operation
//////////////////////////////////////////////////////////////////////////

static std::string buildSliceHLO(const std::vector<sd::LongType>& inputShape,
                                  const std::vector<sd::LongType>& begin,
                                  const std::vector<sd::LongType>& end,
                                  const std::vector<sd::LongType>& strides,
                                  DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule slice_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] parameter(0)\n";

  // Calculate output shape
  std::vector<sd::LongType> outputShape;
  for (size_t i = 0; i < inputShape.size(); i++) {
    sd::LongType size = (end[i] - begin[i] + strides[i] - 1) / strides[i];
    outputShape.push_back(size);
  }

  hlo << "  ROOT result = " << typeStr << "[";
  for (size_t i = 0; i < outputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << outputShape[i];
  }
  hlo << "] slice(input), slice={";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ", ";
    hlo << "[" << begin[i] << ":" << end[i] << ":" << strides[i] << "]";
  }
  hlo << "}\n";
  hlo << "}\n";

  return hlo.str();
}

PLATFORM_IMPL(slice, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for slice operation");
  }

  std::vector<sd::LongType> inputShape(input->shapeOf(), input->shapeOf() + input->rankOf());

  // Get begin, size from arguments
  std::vector<sd::LongType> begin, end, strides;
  int rank = input->rankOf();

  // Arguments are: begin0, begin1, ..., size0, size1, ...
  for (int i = 0; i < rank; i++) {
    begin.push_back(INT_ARG(i));
  }
  for (int i = 0; i < rank; i++) {
    sd::LongType size = INT_ARG(rank + i);
    end.push_back(begin[i] + size);
  }
  for (int i = 0; i < rank; i++) {
    strides.push_back(1);  // Default stride
  }

  // Build cache key
  std::ostringstream keyStream;
  keyStream << "slice_";
  for (auto s : inputShape) keyStream << s << "_";
  for (auto b : begin) keyStream << b << "_";
  for (auto e : end) keyStream << e << "_";
  keyStream << input->dataType();
  std::string cacheKey = keyStream.str();

  // Build HLO module
  std::string hloModule = buildSliceHLO(inputShape, begin, end, strides, input->dataType());

  // Get or compile executable
  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU slice program");
  }

  // Create device buffers and execute
  PjrtBuffer bufIn(client, input);
  PjrtBuffer bufOut(client, output);

  std::vector<PjrtBuffer*> inputs = {&bufIn};
  std::vector<PjrtBuffer*> outputs = {&bufOut};
  executable->execute(inputs, outputs);

  bufOut.toHost(output);

  return Status::OK;
}

PLATFORM_CHECK(slice, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);

  Requirements req("PJRT SLICE OP");
  req.expectTrue(isTpuSupportedType(input->dataType()),
                 REQUIRE_TRUE_MSG("Input data type must be TPU-compatible"));

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Tile operation
//////////////////////////////////////////////////////////////////////////

static std::string buildTileHLO(const std::vector<sd::LongType>& inputShape,
                                 const std::vector<sd::LongType>& multiples,
                                 DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  // Calculate output shape
  std::vector<sd::LongType> outputShape;
  for (size_t i = 0; i < inputShape.size(); i++) {
    outputShape.push_back(inputShape[i] * multiples[i]);
  }

  hlo << "HloModule tile_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << inputShape[i];
  }
  hlo << "] parameter(0)\n";

  // Broadcast to create tiling effect
  // This is a simplified implementation - full implementation would use
  // reshape + broadcast + reshape pattern
  hlo << "  ROOT result = " << typeStr << "[";
  for (size_t i = 0; i < outputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << outputShape[i];
  }
  hlo << "] broadcast(input), dimensions={";
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << i;
  }
  hlo << "}\n";
  hlo << "}\n";

  return hlo.str();
}

PLATFORM_IMPL(tile, ENGINE_TPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // For complex tile operations, fall back to CPU
  // TPU tile requires specific HLO pattern
  THROW_EXCEPTION("TPU tile not yet fully implemented - use CPU fallback");

  return Status::OK;
}

PLATFORM_CHECK(tile, ENGINE_TPU) {
  Requirements req("PJRT TILE OP");
  req.expectFalse(true, REQUIRE_TRUE_MSG("TPU tile not yet implemented"));
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_PJRT
