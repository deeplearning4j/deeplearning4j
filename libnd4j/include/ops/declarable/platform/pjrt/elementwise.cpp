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
// Helper to build binary elementwise HLO
static std::string buildBinaryElementwiseHLO(const std::vector<sd::LongType>& shapeA,
                                              const std::vector<sd::LongType>& shapeB,
                                              const std::vector<sd::LongType>& outputShape,
                                              const std::string& operation,
                                              DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule binary_" << operation << "_module\n\n";
  hlo << "ENTRY main {\n";

  // Input A
  hlo << "  a = " << typeStr << "[";
  for (size_t i = 0; i < shapeA.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << shapeA[i];
  }
  hlo << "] parameter(0)\n";

  // Input B
  hlo << "  b = " << typeStr << "[";
  for (size_t i = 0; i < shapeB.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << shapeB[i];
  }
  hlo << "] parameter(1)\n";

  // Handle broadcasting if shapes differ
  bool needsBroadcast = (shapeA != shapeB);

  if (needsBroadcast) {
    // Broadcast smaller tensor to match larger one
    // This is simplified - full implementation would handle all broadcast cases
    if (shapeA.size() < shapeB.size() || (shapeA.size() == shapeB.size() && shapeA < shapeB)) {
      hlo << "  a_broadcast = " << typeStr << "[";
      for (size_t i = 0; i < outputShape.size(); i++) {
        if (i > 0) hlo << ",";
        hlo << outputShape[i];
      }
      hlo << "] broadcast(a), dimensions={";
      // Calculate broadcast dimensions
      size_t offset = outputShape.size() - shapeA.size();
      for (size_t i = 0; i < shapeA.size(); i++) {
        if (i > 0) hlo << ",";
        hlo << (offset + i);
      }
      hlo << "}\n";

      hlo << "  ROOT result = " << typeStr << "[";
      for (size_t i = 0; i < outputShape.size(); i++) {
        if (i > 0) hlo << ",";
        hlo << outputShape[i];
      }
      hlo << "] " << operation << "(a_broadcast, b)\n";
    } else {
      hlo << "  b_broadcast = " << typeStr << "[";
      for (size_t i = 0; i < outputShape.size(); i++) {
        if (i > 0) hlo << ",";
        hlo << outputShape[i];
      }
      hlo << "] broadcast(b), dimensions={";
      size_t offset = outputShape.size() - shapeB.size();
      for (size_t i = 0; i < shapeB.size(); i++) {
        if (i > 0) hlo << ",";
        hlo << (offset + i);
      }
      hlo << "}\n";

      hlo << "  ROOT result = " << typeStr << "[";
      for (size_t i = 0; i < outputShape.size(); i++) {
        if (i > 0) hlo << ",";
        hlo << outputShape[i];
      }
      hlo << "] " << operation << "(a, b_broadcast)\n";
    }
  } else {
    hlo << "  ROOT result = " << typeStr << "[";
    for (size_t i = 0; i < outputShape.size(); i++) {
      if (i > 0) hlo << ",";
      hlo << outputShape[i];
    }
    hlo << "] " << operation << "(a, b)\n";
  }

  hlo << "}\n";
  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Helper to build unary elementwise HLO
static std::string buildUnaryElementwiseHLO(const std::vector<sd::LongType>& shape,
                                             const std::string& operation,
                                             DataType dtype) {
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(dtype);

  hlo << "HloModule unary_" << operation << "_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << shape[i];
  }
  hlo << "] parameter(0)\n";

  hlo << "  ROOT result = " << typeStr << "[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << shape[i];
  }
  hlo << "] " << operation << "(input)\n";
  hlo << "}\n";

  return hlo.str();
}

//////////////////////////////////////////////////////////////////////////
// Add operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(add, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for add operation");
  }

  std::vector<sd::LongType> shapeA(x->shapeOf(), x->shapeOf() + x->rankOf());
  std::vector<sd::LongType> shapeB(y->shapeOf(), y->shapeOf() + y->rankOf());
  std::vector<sd::LongType> outputShape(z->shapeOf(), z->shapeOf() + z->rankOf());

  std::ostringstream keyStream;
  keyStream << "add_";
  for (auto s : shapeA) keyStream << s << "_";
  for (auto s : shapeB) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildBinaryElementwiseHLO(shapeA, shapeB, outputShape, "add", x->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU add program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufY(client, y);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX, &bufY};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(add, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);

  Requirements req("PJRT ADD OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "X data type must be TPU-compatible");
  req.expectTrue(isTpuSupportedType(y->dataType()),
                 "Y data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Subtract operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(subtract, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for subtract operation");
  }

  std::vector<sd::LongType> shapeA(x->shapeOf(), x->shapeOf() + x->rankOf());
  std::vector<sd::LongType> shapeB(y->shapeOf(), y->shapeOf() + y->rankOf());
  std::vector<sd::LongType> outputShape(z->shapeOf(), z->shapeOf() + z->rankOf());

  std::ostringstream keyStream;
  keyStream << "subtract_";
  for (auto s : shapeA) keyStream << s << "_";
  for (auto s : shapeB) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildBinaryElementwiseHLO(shapeA, shapeB, outputShape, "subtract", x->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU subtract program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufY(client, y);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX, &bufY};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(subtract, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);

  Requirements req("PJRT SUBTRACT OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "X data type must be TPU-compatible");
  req.expectTrue(isTpuSupportedType(y->dataType()),
                 "Y data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Multiply operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(multiply, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for multiply operation");
  }

  std::vector<sd::LongType> shapeA(x->shapeOf(), x->shapeOf() + x->rankOf());
  std::vector<sd::LongType> shapeB(y->shapeOf(), y->shapeOf() + y->rankOf());
  std::vector<sd::LongType> outputShape(z->shapeOf(), z->shapeOf() + z->rankOf());

  std::ostringstream keyStream;
  keyStream << "multiply_";
  for (auto s : shapeA) keyStream << s << "_";
  for (auto s : shapeB) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildBinaryElementwiseHLO(shapeA, shapeB, outputShape, "multiply", x->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU multiply program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufY(client, y);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX, &bufY};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(multiply, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);

  Requirements req("PJRT MULTIPLY OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "X data type must be TPU-compatible");
  req.expectTrue(isTpuSupportedType(y->dataType()),
                 "Y data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Divide operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(divide, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for divide operation");
  }

  std::vector<sd::LongType> shapeA(x->shapeOf(), x->shapeOf() + x->rankOf());
  std::vector<sd::LongType> shapeB(y->shapeOf(), y->shapeOf() + y->rankOf());
  std::vector<sd::LongType> outputShape(z->shapeOf(), z->shapeOf() + z->rankOf());

  std::ostringstream keyStream;
  keyStream << "divide_";
  for (auto s : shapeA) keyStream << s << "_";
  for (auto s : shapeB) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildBinaryElementwiseHLO(shapeA, shapeB, outputShape, "divide", x->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU divide program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufY(client, y);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX, &bufY};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(divide, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);

  Requirements req("PJRT DIVIDE OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "X data type must be TPU-compatible");
  req.expectTrue(isTpuSupportedType(y->dataType()),
                 "Y data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Negate operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(negate, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for negate operation");
  }

  std::vector<sd::LongType> shape(x->shapeOf(), x->shapeOf() + x->rankOf());

  std::ostringstream keyStream;
  keyStream << "negate_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildUnaryElementwiseHLO(shape, "negate", x->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU negate program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(negate, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);

  Requirements req("PJRT NEGATE OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Abs operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(abs, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for abs operation");
  }

  std::vector<sd::LongType> shape(x->shapeOf(), x->shapeOf() + x->rankOf());

  std::ostringstream keyStream;
  keyStream << "abs_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildUnaryElementwiseHLO(shape, "abs", x->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU abs program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(abs, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);

  Requirements req("PJRT ABS OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Square operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(square, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for square operation");
  }

  std::vector<sd::LongType> shape(x->shapeOf(), x->shapeOf() + x->rankOf());

  // Square is x * x
  std::ostringstream hlo;
  std::string typeStr = getHLOTypeString(x->dataType());

  hlo << "HloModule square_module\n\n";
  hlo << "ENTRY main {\n";
  hlo << "  input = " << typeStr << "[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << shape[i];
  }
  hlo << "] parameter(0)\n";
  hlo << "  ROOT result = " << typeStr << "[";
  for (size_t i = 0; i < shape.size(); i++) {
    if (i > 0) hlo << ",";
    hlo << shape[i];
  }
  hlo << "] multiply(input, input)\n";
  hlo << "}\n";

  std::ostringstream keyStream;
  keyStream << "square_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hlo.str());

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU square program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(square, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);

  Requirements req("PJRT SQUARE OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Sqrt operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(sqrt, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for sqrt operation");
  }

  std::vector<sd::LongType> shape(x->shapeOf(), x->shapeOf() + x->rankOf());

  std::ostringstream keyStream;
  keyStream << "sqrt_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildUnaryElementwiseHLO(shape, "sqrt", x->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU sqrt program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(sqrt, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);

  Requirements req("PJRT SQRT OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Exp operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(exp, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for exp operation");
  }

  std::vector<sd::LongType> shape(x->shapeOf(), x->shapeOf() + x->rankOf());

  std::ostringstream keyStream;
  keyStream << "exp_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildUnaryElementwiseHLO(shape, "exponential", x->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU exp program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(exp, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);

  Requirements req("PJRT EXP OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Log operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(log, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for log operation");
  }

  std::vector<sd::LongType> shape(x->shapeOf(), x->shapeOf() + x->rankOf());

  std::ostringstream keyStream;
  keyStream << "log_";
  for (auto s : shape) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildUnaryElementwiseHLO(shape, "log", x->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU log program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(log, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);

  Requirements req("PJRT LOG OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "Input data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
// Power operation
//////////////////////////////////////////////////////////////////////////

PLATFORM_IMPL(Pow, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  auto& client = getGlobalPjrtClient();
  if (!client.isValid()) {
    THROW_EXCEPTION("TPU/PJRT client not available for Pow operation");
  }

  std::vector<sd::LongType> shapeA(x->shapeOf(), x->shapeOf() + x->rankOf());
  std::vector<sd::LongType> shapeB(y->shapeOf(), y->shapeOf() + y->rankOf());
  std::vector<sd::LongType> outputShape(z->shapeOf(), z->shapeOf() + z->rankOf());

  std::ostringstream keyStream;
  keyStream << "pow_";
  for (auto s : shapeA) keyStream << s << "_";
  for (auto s : shapeB) keyStream << s << "_";
  keyStream << x->dataType();
  std::string cacheKey = keyStream.str();

  std::string hloModule = buildBinaryElementwiseHLO(shapeA, shapeB, outputShape, "power", x->dataType());

  auto& cache = HLOExecutableCache::getInstance();
  auto executable = cache.getOrCompile(client, cacheKey, hloModule);

  if (!executable || !executable->isValid()) {
    THROW_EXCEPTION("Failed to compile TPU Pow program");
  }

  PjrtBuffer bufX(client, x);
  PjrtBuffer bufY(client, y);
  PjrtBuffer bufZ(client, z);

  std::vector<PjrtBuffer*> inputs = {&bufX, &bufY};
  std::vector<PjrtBuffer*> outputs = {&bufZ};
  executable->execute(inputs, outputs);

  bufZ.toHost(z);

  return Status::OK;
}

PLATFORM_CHECK(Pow, ENGINE_TPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);

  Requirements req("PJRT POW OP");
  req.expectTrue(isTpuSupportedType(x->dataType()),
                 "X data type must be TPU-compatible");
  req.expectTrue(isTpuSupportedType(y->dataType()),
                 "Y data type must be TPU-compatible");

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_PJRT
