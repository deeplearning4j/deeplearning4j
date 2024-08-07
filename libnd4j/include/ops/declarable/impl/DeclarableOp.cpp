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
// @author raver119@gmail.com
//
#include <array/NDArrayFactory.h>
#include <exceptions/datatype_exception.h>
#include <exceptions/graph_exception.h>
#include <graph/exceptions/unresolved_input_exception.h>
#include <helpers/ShapeUtils.h>
#include <helpers/StringUtils.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/OpRegistrator.h>

#include <cstdarg>

namespace sd {
namespace ops {

ErrorResult conditionHelper(const char *file, int line, int condition, int argNumber, const char *format, ...) {
  std::string message;
  if (!condition) {
    va_list args;
    char buffer[512]; // Assuming the message won't exceed 512 characters. Adjust if needed.

    int written = snprintf(buffer, sizeof(buffer), "Error at [%s:%i:%i]:\n", file, line, argNumber);
    if (written > 0 && written < sizeof(buffer)) {
      message += buffer;
    }

    va_start(args, format);
    written = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    if (written > 0 && written < sizeof(buffer)) {
      message += buffer;
    }

    message += "\n";

    return {Status::BAD_PARAMS, message };
  }
  return {Status::OK, "" };
}

DeclarableOp::DeclarableOp() {
  // no-op
}

DeclarableOp::DeclarableOp(const char *name, bool isLogical) {
  _descriptor = new OpDescriptor(name, isLogical);
  _name = name;
}

DeclarableOp::DeclarableOp(const char *name, int numInputs, bool scalar) {
  _descriptor = new OpDescriptor(numInputs, name, scalar);
  _name = name;
}

DeclarableOp::DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace) {
  _descriptor = new OpDescriptor(numInputs, numOutputs, opName, allowsInplace);
  _name = opName;
}

DeclarableOp::DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent) {
  _descriptor = new OpDescriptor(numInputs, numOutputs, opName, allowsInplace, divergent);
  _name = opName;
}

DeclarableOp::DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs,
                           int iArgs) {
  _descriptor = new OpDescriptor(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs);
  _name = opName;
}

DeclarableOp::~DeclarableOp() {
  if (_descriptor != nullptr) delete _descriptor;

  if (_scalar != nullptr) delete _scalar;
}

OpDescriptor *DeclarableOp::getOpDescriptor() { return _descriptor; }

std::string *DeclarableOp::getOpName() { return _descriptor->getOpName(); }

LongType DeclarableOp::getOpHash() { return _descriptor->getHash(); }

NDArray *DeclarableOp::getNullifiedZ(Context &block, int inputId) {
  auto result = getZ(block, inputId);
  if (result != nullptr && !block.isInplace()) result->nullify();

  return result;
}

NDArray *DeclarableOp::getZ(Context &ctx, int inputId) {
  NDArray *z = nullptr;

  if (ctx.isFastPath()) {
    if (ctx.fastpath_out().size() <= inputId) {
      if (ctx.isInplace()) {
        z = ctx.fastpath_in()[inputId];
      } else
        THROW_EXCEPTION("fastpath_out: unresolved output array");
    } else {
      z = ctx.fastpath_out()[inputId];
    }
  } else {
    std::pair<int, int> pair(ctx.nodeId(), inputId);

    if (ctx.isInplace()) {
      z = ctx.variable(inputId)->getNDArray();

      // hypothetically it's possible to have no variable. chances are low, but who knows. let's just create it for now
      if (!ctx.getVariableSpace()->hasVariable(pair)) {
        auto var = new Variable();
        ctx.getVariableSpace()->putVariable(pair, var);
      }

      // now we're saving input array as output array
      auto var = ctx.getVariableSpace()->getVariable(pair);
      var->markRemovable(false);
      var->setNDArray(z);
    } else if (!ctx.isInplace()) {
      auto var = ctx.variable(pair);
      if (var->getNDArray() != nullptr && var->getNDArray()->nonNull()) {
        z = var->getNDArray();
      } else {
        sd_printf("Can't get Z variable for node_%i!\n", ctx.nodeId());
      }
    } else {
      THROW_EXCEPTION("getZ: Unable to return z variable!");
    }
  }

  return z;
}

int DeclarableOp::prepareOutputs(Context &ctx) {
  auto workspace = ctx.getWorkspace();
  GraphProfile *prof = nullptr;
  NodeProfile *node = nullptr;
  std::chrono::time_point<std::chrono::system_clock> inputEnd, inputStart, shapeStart, shapeEnd, arrayStart, arrayEnd;
  bool canUseFastPath = true;

  auto fp = ctx.isFastPath();

  if (Environment::getInstance().isProfiling()) {
    if (ctx.getVariableSpace() != nullptr && ctx.getVariableSpace()->flowPath() != nullptr) {
      prof = ctx.getVariableSpace()->flowPath()->profile();
      node = prof->nodeById(ctx.nodeId());
    }
  }

  if (ctx.isInplace()) {
    if (Environment::getInstance().isProfiling() && node != nullptr) {
      if (fp) {
        //
      } else {
        for (auto p : *ctx.inputs()) {
          auto var = ctx.variable(p);
          if (var->variableType() == NDARRAY) {
            NDArray *array = var->getNDArray();

            node->addInputShape(array->shapeInfo());
            node->addOutputShape(array->shapeInfo());
          }
        }
      }
    }

    // if that's not fp, we can still propagate inputs and outputs
    if (!fp) {
      int cnt = 0;
      auto id = ctx.nodeId();
      auto vs = ctx.getVariableSpace();
      for (auto p : *ctx.inputs()) {
        auto var = ctx.variable(p);
        if (var->variableType() == NDARRAY) {
          NDArray *array = var->getNDArray();
          ctx.setInputArray(cnt, array);
          ctx.setOutputArray(cnt, array);

          // in case of this override we might need to update outputs in the Graph VariableSpace as well
          if (vs != nullptr) {
            if (vs->hasVariable(id, cnt)) {
              auto v2 = vs->getVariable(id, cnt);
              if (!v2->hasNDArray()) {
                v2->setNDArray(array);
                v2->markRemovable(false);
              }
            } else {
              auto v2 = vs->putVariable(id, cnt, array);
              v2->markRemovable(false);
            }
          }

          cnt++;
        } else {
          canUseFastPath = false;
        }
      }
    }

    if (!canUseFastPath) ctx.forbidFastPath(true);

    // do nothing, getZ result will do the trick
    return static_cast<int>(ctx.width());
  } else {
    // if op is not inplace - we should pre-allocate arrays
    ShapeList inSha;
    int results = 0;

    if (Environment::getInstance().isProfiling() && node != nullptr) inputStart = std::chrono::system_clock::now();

    int cntIn = 0;
    // we build list of input shapes
    if (fp) {
      for (const auto p : ctx.fastpath_in()) {
        inSha.push_back(p == nullptr ? nullptr : p->shapeInfo());
      }
    } else {
      int arrCnt = 0;
      for (auto p : *ctx.inputs()) {
        auto var = ctx.variable(p);
        if (var->variableType() == NDARRAY) {
          NDArray *array = var->getNDArray();
          var->markRemovable(false);
          if (array == nullptr)
            throw unresolved_input_exception::build("OP PREPARE OUTPUTS: Variable wasn't resolved prior shape calculation", p);

          inSha.push_back(array->shapeInfo());

          // we're also filling ctx with arrays
          if (canUseFastPath) ctx.setInputArray(arrCnt++, array);
        } else {
          canUseFastPath = false;
        }
        cntIn++;
      }
    }

    // if we override shape function, we'll return size of fastPath
    if (fp && ctx.shapeFunctionOverride()) {
      return (int)ctx.fastpath_out().size();
    }

    // optionally saving input time
    if (Environment::getInstance().isProfiling() && node != nullptr) {
      inputEnd = std::chrono::system_clock::now();
      auto inputTime = std::chrono::duration_cast<std::chrono::nanoseconds>(inputEnd - inputStart).count();
      node->setInputTime(inputTime);

      // saving output shapes in profile
      for (int e = 0; e < inSha.size(); e++) node->addInputShape(inSha.at(e));

      shapeStart = std::chrono::system_clock::now();
    }

    auto outSha = this->calculateOutputShape(&inSha, ctx);
    if (Environment::getInstance().isDebugAndVerbose()) {
      sd_printf("Node_%i: %s\n", ctx.nodeId(), this->getOpDescriptor()->getOpName()->c_str());
      sd_printf("Input shapes:\n",0);
      for (int e = 0; e < inSha.size(); e++) {
        if (inSha.at(e) != nullptr) {
          sd_printf("Shape_%i: ", e);
          shape::printShapeInfoLinear(inSha.at(e));
        } else {
          sd_printf("Shape_%i: nullptr\n", e);
        }
      }
      sd_printf("Output shapes:\n",0);
      for (int e = 0; e < outSha->size(); e++) {
        if (outSha->at(e) != nullptr) {
          sd_printf("Shape_%i: ", e);
          shape::printShapeInfoLinear(outSha->at(e));
        } else {
          sd_printf("Shape_%i: nullptr\n", e);
        }
      }
    }


    results = outSha->size();

    // optionally saving shapeTime
    if (Environment::getInstance().isProfiling() && node != nullptr) {
      shapeEnd = std::chrono::system_clock::now();
      auto prepTime = std::chrono::duration_cast<std::chrono::nanoseconds>(shapeEnd - shapeStart).count();
      node->setShapeFunctionTime(prepTime);

      // saving output shapes in profile
      for (int e = 0; e < outSha->size(); e++) node->addOutputShape(outSha->at(e));

      arrayStart = std::chrono::system_clock::now();
    }

    int cnt = 0;
    for (int jj = 0; jj < outSha->size(); jj++) {
      auto out = outSha->at(jj);
      if (!fp) {
        // we need to check, if Z is really needed
        std::pair<int, int> pair(ctx.nodeId(), cnt++);

        if (!ctx.isValueAvailable(pair.second)) {
          if (Environment::getInstance().isDebugAndVerbose())
            shape::printShapeInfoLinear("OP PREPARE OUTPUTS: Going to create variable with shape", out);

          // we're creating non-initialized array here
          auto outArr = new NDArray(out, true, ctx.launchContext(), false);

          ctx.pushNDArrayToVariableSpace(pair, outArr);

          if (canUseFastPath) ctx.setOutputArray(pair.second, outArr);
        } else {
          // validate/compare shapes here. existent vs provided in outSha
          auto var = ctx.variable(pair);
          auto shape = var->getNDArray()->shapeInfo();

          if (canUseFastPath) ctx.setOutputArray(pair.second, var->getNDArray());

          // note we only compare the shapes here not the shape info which may
          // have extra information attached to it. We compare data types and empty status down below.
          // sometimes empty strides (that don't actually matter) can cause errors, we omit this on purpose
          if (!shape::equalsSoft(out, shape)) {
            auto eShape = ShapeUtils::shapeAsString(out);
            auto aShape = ShapeUtils::shapeAsString(shape);
            auto eShapeInfoString = ShapeUtils::shapeInfoAsString(out);
            auto aShapeInfoString = ShapeUtils::shapeInfoAsString(shape);
            delete outSha;

            sd_printf(
                "OP PREPARE OUTPUTS: Op name: %s Failed to set output for op context. Expected vs provided shapes mismatch %s vs %s at index %i with expected shape info %s and output "
                "shape info %s\n",
                getOpName()->c_str(),eShape.c_str(), aShape.c_str(), pair.second, eShapeInfoString.c_str(), aShapeInfoString.c_str());

            THROW_EXCEPTION("OP PREPARE OUTPUTS: Expected vs provided shapes mismatch first case");
          }

          if (shape::isEmptyConst(out) != shape::isEmptyConst(shape)) {
            sd_printf("OP PREPARE OUTPUTS: First array empty: %d Second shape empty: %d\n", shape::isEmptyConst(out), shape::isEmptyConst(shape));

            THROW_EXCEPTION("OP PREPARE OUTPUTS: Expected vs provided shapes mismatch");
          }

          // checking out data type equality
          if (ArrayOptions::dataType(out) != ArrayOptions::dataType(shape)) {
            std::string msg =
                "Provided array [" + StringUtils::valueToString<int>(pair.second) + "] has unexpected data type";
            throw datatype_exception::build(msg, ArrayOptions::dataType(out), ArrayOptions::dataType(shape));
          }
        }
      } else {
        auto fout = ctx.fastpath_out();
        auto idx = cnt++;
        if (fout.size() <= idx) {
          // array doesnt exist
          auto outArr = new NDArray(out, true, ctx.launchContext());
          ctx.setOutputArray(idx, outArr, true);
        } else {
          auto array = fout[idx];
          int shapeEquals = shape::equalsSoft(out, array->shapeInfo());
          int arrayEmpty = array->isEmpty();
          // checking out shape equality
          if (!shapeEquals) {
            auto eShape = ShapeUtils::shapeAsString(out);
            auto aShape = ShapeUtils::shapeAsString(array->shapeInfo());
            auto eShapeInfoString = ShapeUtils::shapeInfoAsString(out);
            auto aShapeInfoString = ShapeUtils::shapeInfoAsString(array->shapeInfo());
            if (eShapeInfoString != aShapeInfoString) {
              delete outSha;
              std::string errorMessage;
              errorMessage += "OP PREPARE OUTPUTS: Op name: ";
              errorMessage += getOpName()->c_str();
              errorMessage += " Failed to set output for op context. Expected vs provided shapes mismatch ";
              errorMessage += eShape;
              errorMessage += " vs ";
              errorMessage += aShape;
              errorMessage += " at index ";
              errorMessage += std::to_string(idx);
              errorMessage += " with expected shape info ";
              errorMessage += eShapeInfoString;
              errorMessage += " and output shape info ";
              errorMessage += aShapeInfoString;
              errorMessage += ". Conditions, shapeEquals: ";
              errorMessage += std::to_string(shapeEquals);
              errorMessage += ", array empty: ";
              errorMessage += std::to_string(arrayEmpty);
              errorMessage += "\n";
              errorMessage += "Expected shape info: ";
              errorMessage += eShapeInfoString;
              errorMessage += "\n";
              errorMessage += "Provided shape info: ";
              errorMessage += aShapeInfoString;
              errorMessage += "\n";
              errorMessage += "Expected shape: ";
              errorMessage += eShape;
              errorMessage += "\n";
              errorMessage += "Provided shape: ";
              errorMessage += aShape;
              errorMessage += "\n";
              THROW_EXCEPTION(errorMessage.c_str());

            }
          }
        }
      }
    }

    if (!canUseFastPath) ctx.forbidFastPath(true);

    delete outSha;

    // saving arrayTime
    if (Environment::getInstance().isProfiling() && node != nullptr) {
      arrayEnd = std::chrono::system_clock::now();
      auto arrayTime = std::chrono::duration_cast<std::chrono::nanoseconds>(arrayEnd - arrayStart).count();
      node->setArrayTime(arrayTime);
    }

    return results;
  }
}

void DeclarableOp::storeResult(Context &block, int outputNumber, NDArray *array) {
  this->storeResult(block, outputNumber, *array);
}

void DeclarableOp::storeResult(Context &ctx, int outputNumber, NDArray &array) {
  ctx.pushNDArrayToVariableSpace(ctx.nodeId(), outputNumber, &array, !ctx.isInplace());
}

bool DeclarableOp::allocateResult(Context &block, LongType *shape) {
  auto var = block.variable(block.getNodeId(), 0);

  auto workspace = block.getWorkspace();

  LongType len = shape::length(shape);
  LongType *__shape;
  ALLOCATE(__shape, workspace, shape::shapeInfoLength(shape), sd::LongType);  // new int[shape[0] * 2 + 4];

  memcpy(__shape, shape, shape::shapeInfoByteLength(shape));

  // if that's first run - we probably have nothing here
  if (var->getNDArray() == nullptr) {
    auto desc = new ShapeDescriptor(__shape);
    DataBuffer *buffer =
        new DataBuffer(len * sizeof(int8_t),desc->dataType(), workspace);
    var->setNDArray(new NDArray(buffer, desc, block.launchContext()));
    if (Environment::getInstance().isDeleteShapeInfo()) delete desc;
  } else if (var->getNDArray()->lengthOf() != len) {
    // if length not match - lets reallocate array
    delete var->getNDArray();
    auto desc = new ShapeDescriptor(__shape);
    DataBuffer *buffer =
        new DataBuffer(len * sizeof(int8_t), desc->dataType(), workspace);
    var->setNDArray(new NDArray(buffer, desc, block.launchContext()));
    if (Environment::getInstance().isDeleteShapeInfo()) delete desc;
  }

  return true;
}


void DeclarableOp::traceExecIfNeeded(Context &block) {
  if(OpRegistrator::getInstance().traceOps()) {
    std::vector<const LongType *> *inputShapeBuffers = new std::vector<const LongType *>();
    for(int i = 0; i < block.width(); i++) {
      inputShapeBuffers->push_back(block.variable(i)->getNDArray()->shapeInfo());
    }
    std::vector<const LongType *> *outputShapeBuffers = new std::vector<const LongType *>();
    for(int i = 0; i < block.outputWidth(); i++) {
      outputShapeBuffers->push_back(block.fastpath_out()[i]->shapeInfo());
    }

    OpExecTrace *opExecTrace = new OpExecTrace(inputShapeBuffers,outputShapeBuffers, getOpName());
    OpRegistrator::getInstance().registerOpExec(opExecTrace);
  }
}

bool DeclarableOp::allocateResult(Context &block, std::initializer_list<LongType> &shape, char order) {
  auto var = block.variable(block.getNodeId(), 0);
  auto workspace = block.getWorkspace();

  LongType len = shape::length(shape);
  // if that's first run - we probably have nothing here
  if (var->getNDArray() == nullptr) {
    var->setNDArray(new NDArray(order, shape, block.dataType(), block.launchContext()));
  } else if (var->getNDArray()->lengthOf() != len) {
    // if length not match - lets reallocate array
    delete var->getNDArray();
    var->setNDArray(new NDArray(order, shape, block.dataType(), block.launchContext()));
  }

  return true;
}

Status DeclarableOp::validateDataTypes(Context &block) {
  _registrator.lock();
  if (!_registered) {
    _registered = true;
    this->registerTypes();
  }
  _registrator.unlock();

  // rolling over inputs first
  int cnt = 0, inT = 0;
#if defined(__NEC__)
  sd::DataType inputTypes[SD_MAX_INPUT_SIZE];
  if (block.width() > SD_MAX_INPUT_SIZE) {
    sd_printf("%s:%d Exceeded allowed input size (%d) \n", __FILE__, __LINE__, SD_MAX_INPUT_SIZE);
    THROW_EXCEPTION("Provided inputs are more than allowed");
  }
#else
  std::vector<DataType> inputTypes(block.width());
#endif
  if (block.isFastPath()) {
    for (auto array : block.fastpath_in()) {
      if (array == nullptr) {
        continue;
      }

      auto dtype = array->dataType();

      inputTypes[inT++] = dtype;
      if (!_descriptor->checkInputMatch(cnt, dtype)) {
        auto ctype = DataTypeUtils::asString(dtype);

        auto inputTypes2 = _descriptor->getInputTypesForInput(cnt);
        if (inputTypes2.size() > 1) {
          std::string allTypes;
          for (int i = 0; i < inputTypes2.size(); i++) {
            allTypes += DataTypeUtils::asString(inputTypes2[i]);
            if (i < inputTypes2.size() - 1) {
              allTypes += ",";
            }
          }
          sd_printf("Op [%s] failed check for input [%i], DataType: [%s] Expected data types[%s]\n",
                    _descriptor->getOpName()->data(), cnt, ctype.c_str(), allTypes.c_str());
        } else if (!inputTypes2.size() < 1) {
          auto typeAsString = DataTypeUtils::asString(inputTypes2[0]);
          sd_printf("Op [%s] failed check for input [%i], DataType: [%s] Expected data type[%s]\n",
                    _descriptor->getOpName()->data(), cnt, ctype.c_str(), typeAsString.c_str());
        } else {
          sd_printf("Op [%s] data types empty \n", _descriptor->getOpName()->data());
        }

        return Status::BAD_ARGUMENTS;
      }
      cnt++;
    }

  } else {
    for (auto &p : *(block.inputs())) {
      auto var = block.variable(p);

      // only validating non-null variables
      if (var != nullptr && var->hasNDArray()) {
        auto array = var->getNDArray();
        inputTypes[inT++] = array->dataType();
        if (!_descriptor->checkInputMatch(cnt, array->dataType())) {
          auto ctype = DataTypeUtils::asString(array->dataType());
          std::string errorMessage = "Op [" + std::string(_descriptor->getOpName()->data()) +
                                     "] failed check for input [" + std::to_string(cnt) + "], DataType: [" + ctype +
                                     "]\n";
          THROW_EXCEPTION(errorMessage.c_str());
          return Status::BAD_ARGUMENTS;
        }
      }

      cnt++;
    }
  }

  if (block.isFastPath()) {
    int index = 0;
    for (auto array : block.fastpath_out()) {
      if (array == nullptr) continue;

      auto cType = array->dataType();

      if (_descriptor->isSameMode()) {
        if (index >= block.width()) {
          if (block.fastpath_in().size() == 0) continue;

          auto ia = block.fastpath_in()[0];

          if (ia->dataType() != cType) {
            auto t = DataTypeUtils::asString(cType);
            std::string errorMessage = "Op [" + std::string(_descriptor->getOpName()->data()) +
                                       "] failed check for output [" + std::to_string(index) + "], DataType: [" + t +
                                       "]\n";
            THROW_EXCEPTION(errorMessage.c_str());
            return Status::BAD_ARGUMENTS;
          }
        } else {
          // for same mode, output type must be the same as input type
          auto ia = block.fastpath_in()[index];

          if (ia->dataType() != cType) {
            auto t = DataTypeUtils::asString(cType);
            std::string errorMessage = "Op [" + std::string(_descriptor->getOpName()->data()) +
                                       "] failed check for output [" + std::to_string(index) + "], DataType: [" + t +
                                       "]\n";
            THROW_EXCEPTION(errorMessage.c_str());
            return Status::BAD_ARGUMENTS;
          }
        }
      } else if (_descriptor->isInherit(index)) {
        // in inherit mode, output type must be the same as one of input types
        if (std::find(std::begin(inputTypes), std::end(inputTypes), cType) == std::end(inputTypes)) {
          auto t = DataTypeUtils::asString(cType);
          std::string errorMessage = "Op [" + std::string(_descriptor->getOpName()->data()) +
                                     "] failed check for output [" + std::to_string(index) + "], DataType: [" + t +
                                     "].\n";
          THROW_EXCEPTION(errorMessage.c_str());
          return Status::BAD_ARGUMENTS;
        }

      } else if (!_descriptor->checkOutputMatch(index, cType)) {
        auto t = DataTypeUtils::asString(cType);
        std::string errorMessage = "Op [" + std::string(_descriptor->getOpName()->data()) +
                                   "] failed check for output [" + std::to_string(index) + "], DataType: [" + t +
                                   "];\n";
        THROW_EXCEPTION(errorMessage.c_str());
        return Status::BAD_ARGUMENTS;
      }
      index++;
    }
  } else {
    // checking optionally available outputs
    auto varSpace = block.getVariableSpace();
    for (int index = 0; index < DataTypeUtils::max<int>(); index++) {
      if (varSpace != nullptr && varSpace->hasVariable(block.nodeId(), index)) {
        auto var = block.variable(block.nodeId(), index);

        // only validating non-null variables
        if (var != nullptr && var->hasNDArray()) {
          auto array = var->getNDArray();
          auto cType = array->dataType();

          if (_descriptor->isSameMode()) {
            if (index >= block.width()) {
              if (block.width() == 0) continue;
              auto iv = block.variable(0);

              if (iv->getNDArray()->dataType() != cType) {
                auto t = DataTypeUtils::asString(cType);
                std::string errorMessage = "Op [" + std::string(_descriptor->getOpName()->data()) +
                                           "] failed check for output [" + std::to_string(index) + "], DataType: [" +
                                           t + "]\n";
                THROW_EXCEPTION(errorMessage.c_str());
                return Status::BAD_ARGUMENTS;
              }
            } else {
              // for same mode, output type must be the same as input type
              auto iv = block.variable(index);

              if (iv->getNDArray()->dataType() != cType) {
                auto t = DataTypeUtils::asString(cType);
                std::string errorMessage = "Op [" + std::string(_descriptor->getOpName()->data()) +
                                           "] failed check for output [" + std::to_string(index) + "], DataType: [" +
                                           t + "]\n";
                THROW_EXCEPTION(errorMessage.c_str());
                return Status::BAD_ARGUMENTS;
              }
            }
          } else if (_descriptor->isInherit(index)) {
            // in inherit mode, output type must be the same as one of input types
            if (std::find(std::begin(inputTypes), std::end(inputTypes), cType) == std::end(inputTypes)) {
              auto t = DataTypeUtils::asString(cType);
              std::string errorMessage = "Op [" + std::string(_descriptor->getOpName()->data()) +
                                         "] failed check for output [" + std::to_string(index) + "], DataType: [" + t +
                                         "].\n";
              THROW_EXCEPTION(errorMessage.c_str());
              return Status::BAD_ARGUMENTS;
            }

          } else if (!_descriptor->checkOutputMatch(index, cType)) {
            auto t = DataTypeUtils::asString(cType);
            std::string errorMessage = "Op [" + std::string(_descriptor->getOpName()->data()) +
                                       "] failed check for output [" + std::to_string(index) + "], DataType: [" + t +
                                       "];\n";
            THROW_EXCEPTION(errorMessage.c_str());
            return Status::BAD_ARGUMENTS;
          }
        }
      } else
        break;
    }
  }

  return Status::OK;
}

Status DeclarableOp::execute(Context *block) {
  sd_debug("Executing op: [%s]\n", this->getOpName()->c_str());

  std::chrono::time_point<std::chrono::system_clock> timeEnter, timeStart, timeEnd;
  LongType prepTime, outerTime;

  LongType memoryBefore =
      block->workspace() == nullptr ? 0L : block->workspace()->getSpilledSize() + block->workspace()->getUsedSize();
  if (Environment::getInstance().isProfiling()) timeEnter = std::chrono::system_clock::now();
  // basic validation: ensure inputs are set
  REQUIRE_OK(this->validateNonEmptyInput(*block));

  // ensure number of IArgs, TArgs match our expectations
  REQUIRE_OK(this->validateArguments(*block));

  // validating data types for inputs and (optionally) outputs
  REQUIRE_OK(this->validateDataTypes(*block));

  // this method will allocate output NDArrays for this op
  auto numOutputs = this->prepareOutputs(*block);

  if (Environment::getInstance().isProfiling()) {
    timeStart = std::chrono::system_clock::now();
    prepTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeStart - timeEnter).count();
  }

  Status status;
  bool hasHelper = false;

  // platform helpers use might be forbidden for various reasons, so we'll check it out first
  if (block->helpersAllowed() && Environment::getInstance().helpersAllowed()) {
    // if we have platform-specific helper for this op - invoke it
    if (OpRegistrator::getInstance().hasHelper(this->getOpHash(), block->engine())) {
      auto helper = OpRegistrator::getInstance().getPlatformHelper(this->getOpHash(), block->engine());
      if (helper->isUsable(*block)) {
        status = helper->invokeHelper(*block);
        hasHelper = true;
      }
    }
  }

  std::vector<NDArray> inputsToCheck;
  if(Environment::getInstance().isCheckInputChange()) {
    for(int i = 0; i < block->width(); i++) {
      auto array = block->array(i);
      inputsToCheck.push_back(array->dup(false));

    }
  }

  std::vector<NDArray> outputsToCheck;
  if(Environment::getInstance().isCheckOutputChange()) {
    for(int i = 0; i < numOutputs; i++) {
      auto array = block->fastpath_out()[i];
      outputsToCheck.push_back(array->dup(false));
    }

    printf("outputs to check %d\n", outputsToCheck.size());
  }


  if (!hasHelper) status = this->validateAndExecute(*block);
  //validate when inputs are changed when they shouldn't be
  if(Environment::getInstance().isCheckInputChange() && !this->getOpDescriptor()->allowsInplace()) {
    for(int i = 0; i < block->width(); i++) {
      auto array = block->array(i);
      bool arrayInOutputs = false;
      for(int j = 0 ; j < numOutputs; j++) {
        //only test for underlying buffer, note there are
        //a  limited number of ways to figure this out.
        //this is a best effort way to determine if we're looking at the same underlying input
        //the reason we have to test this way is when an array is passed down from java
        //we usually create a new ndarray and wrap the existing buffer.
        //due to this wrapping we can't directly just compare ndarray objects.
        if(array->buffer() == block->outputArray(j)->buffer()) {
          arrayInOutputs = true;
          break;
        }
      }

      if(arrayInOutputs) {
        continue;
      }

      if(!array->equalsTo(&inputsToCheck[i])) {
        std::string errorMessage;
        errorMessage += "Input array ";
        errorMessage += std::to_string(i);
        errorMessage += " has been changed after execution of op ";
        errorMessage += this->getOpName()->c_str();
        errorMessage += "\n";
        THROW_EXCEPTION(errorMessage.c_str());
      }

    }
  }

  if(Environment::getInstance().isCheckOutputChange()) {
    printf("Checking output change on num output arrays: %d\n", outputsToCheck.size());
    for (int i = 0; i < outputsToCheck.size(); i++) {
      auto array = block->outputArray(i);
      if(array == nullptr || array->isEmpty()) {
        continue;
      }

      if (array->equalsTo(&outputsToCheck[i])) {
        std::string errorMessage;
        errorMessage += "Output array ";
        errorMessage += std::to_string(i);
        errorMessage += " has not been changed after execution of op ";
        errorMessage += this->getOpName()->c_str();
        errorMessage += "\n";
        THROW_EXCEPTION(errorMessage.c_str());
      } else {
        printf("Array at %d is not equal\n", i);
      }
    }
  }

  // optionally saving execution time
  if (Environment::getInstance().isProfiling()) {
    timeEnd = std::chrono::system_clock::now();
    outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
    block->setInnerTime(outerTime);
    sd_debug("%s [%s] prepTime %lld time %lld \n", hasHelper ? "helper" : "ordinary", this->getOpName()->c_str(),
             static_cast<sd::LongType>(prepTime), static_cast<sd::LongType>(outerTime));
  }

  if (Environment::getInstance().isProfiling() && block->getVariableSpace() != nullptr) {
    auto fp = block->getVariableSpace()->flowPath();
    if (fp != nullptr) {
      auto p = fp->profile();
      if (p != nullptr) {
        LongType memoryAfter = block->workspace() == nullptr
                               ? 0L
                               : block->workspace()->getSpilledSize() + block->workspace()->getUsedSize();
        LongType memoryUsed = memoryAfter - memoryBefore;
        p->nodeById(block->nodeId())->setPreparationTime(prepTime);
        p->nodeById(block->nodeId())->setExecutionTime(outerTime);
        p->nodeById(block->nodeId())->setTotalSize(memoryUsed);
      }
    }
  }

  // now we print out all outputs for this node
  if (Environment::getInstance().isDebugAndVerbose()) {
    sd_printf("Op with name %s and num inputs %i \n", this->getOpName()->c_str(), block->width());
    auto vs = block->getVariableSpace();
    int numInputs = block->width();
    for (int e = 0; e < numInputs; e++) {
      auto array = block->isFastPath() ?  block->fastpath_in()[e]
                                       : vs->getVariable(block->nodeId(), e)->getNDArray();
      sd_printf("Checking input %d  block fast path %d op name %s with array shape information %s\n",e,block->isFastPath(),this->getOpName()->c_str(),
                ShapeUtils::shapeInfoAsString(array->shapeInfo()).c_str());
      auto shape = ShapeUtils::shapeAsString(array);
      //limit size preview for string arrays due to allocation size when debugging
      int sizePreview = array->isS() ? 2 : 32;
      auto first = array->isEmpty() ? new std::string("Empty NDArray") : array->asString(sizePreview);
      auto type = DataTypeUtils::asString(array->dataType());

      sd_printf("node_%i:%i input  shape: %s; dtype: %s; first values %s\n", block->nodeId(), e, shape.c_str(),
                type.c_str(), first->c_str());
    }

    for (int e = 0; e < numOutputs; e++) {
      // if given output index doesn't exist - we're done
      sd_printf("Declarable op execute: processing output %d for op %s\n",e,this->getOpName()->c_str());

      if (!block->isFastPath()) {
        if (!vs->hasVariable(block->nodeId(), e)) break;
      } else {
        // we have to check either in or out stack, depending on isInplace()
        if (block->isInplace()) {
          if (block->fastpath_out().size() <= e) break;
        } else {
          if (block->fastpath_out().size() <= e) break;
        }
      }

      auto array = block->isFastPath() ?  block->fastpath_out()[e]
                                       : vs->getVariable(block->nodeId(), e)->getNDArray();

      if(array == nullptr) {
        THROW_EXCEPTION("DeclarableOp::execute: array is nullptr");
      }

      auto shape = ShapeUtils::shapeAsString(array);
      LongType len = sd::math::sd_min<LongType>(32, array->isEmpty() || array->isScalar() ? 1 : array->lengthOf());
      sd_printf("array to string: Len of array is %lld real len is %lld data buffer length %lld array offset %lld array is attached %d array is view %d\n",
                len,array->lengthOf(),array->dataBuffer()->getNumElements(), array->offset(),array->isAttached(),array->isView());
      fflush(stdout);
      auto first = array->isEmpty() ? new std::string("Empty NDArray") : array->asString(len);
      auto type = DataTypeUtils::asString(array->dataType());

      sd_printf("node_%i:%i result shape: %s; dtype: %s; first values %s\n", block->nodeId(), e, shape.c_str(),
                type.c_str(), first->c_str());
    }
  }

  traceExecIfNeeded(*block);


  return status;
}

void DeclarableOp::overwriteResult(Context &block, int outputIdx, NDArray *array, bool remove) {
  if (block.isFastPath()) {
    if (remove && block.fastpath_out()[outputIdx] != nullptr) {
      // delete reference/call destructor if remove is true
      sd_debug("Deleting extra reference in fast path at idx %d\n",outputIdx);
      delete block.fastpath_out()[outputIdx];
    }
    sd_debug("In fast path, setting variable\n", 0);
    block.fastpath_out()[outputIdx] = array;
  } else if (block.getVariableSpace() == nullptr) {
    THROW_EXCEPTION("Var space should not be null before pushing variable!");
  } else {
    block.pushNDArrayToVariableSpace(block.nodeId(), outputIdx, array, remove);
    sd_debug("After pushing variable\n", 0);
    auto varSpace = block.getVariableSpace();
    if (varSpace == nullptr) {
      THROW_EXCEPTION("Var space should not be null!");
    }
    sd_debug("After getting var space\n", 0);
    if (varSpace->hasVariable(block.getNodeId(), outputIdx)) {
      sd_debug("calling get variable\n", 0);
      auto var = varSpace->getVariable(block.getNodeId(), outputIdx);
      sd_debug("after calling get variable", 0);
      if (var->getNDArray() != nullptr && var->isRemovable()) delete var->getNDArray();

      var->setNDArray(array);
      var->markRemovable(true);
    } else {
      sd_debug("Creating new variable\n", 0);
      auto var = new Variable(array, nullptr, block.getNodeId(), outputIdx);
      varSpace->putVariable(block.getNodeId(), outputIdx, var);
      sd_debug("Putting variable\n", 0);
    }
  }
}

void DeclarableOp::overwriteResult(Context &block, int outputIdx, NDArray *array) {
  block.pushNDArrayToVariableSpace(block.nodeId(), outputIdx, array);
  auto varSpace = block.getVariableSpace();
  if (varSpace != nullptr && varSpace->hasVariable(block.getNodeId(), outputIdx)) {
    auto var = varSpace->getVariable(block.getNodeId(), outputIdx);
    if (var->getNDArray() != nullptr && var->isRemovable()) delete var->getNDArray();

    var->setNDArray(array);
    var->markRemovable(true);
  } else if(varSpace != nullptr) {
    auto var = new Variable(array, nullptr, block.getNodeId(), outputIdx);
    varSpace->putVariable(block.getNodeId(), outputIdx, var);
  }
}

void DeclarableOp::overwriteResult(Context &block, int outputIdx, NDArrayList *list) {
  block.pushNDArrayListToVariableSpace(block.nodeId(), outputIdx, list);
  auto varSpace = block.getVariableSpace();
  if (varSpace->hasVariable(block.getNodeId(), outputIdx)) {
    auto var = varSpace->getVariable(block.getNodeId(), outputIdx);
    var->setNDArrayList(list);
  } else {
    auto var = new Variable(nullptr, nullptr, block.getNodeId(), outputIdx);
    var->setNDArrayList(list);
    varSpace->putVariable(block.getNodeId(), outputIdx, var);
  }
}

Status DeclarableOp::validateArguments(Context &block) {
  /*
   * We're checking number of T and I arguments. If number of args is finite number - we check strict equality
   * If number of args is variable (-1), but variables MUST be present - we check for non-zero number of arguments
   */
  if (_descriptor->getNumberOfTArgs() > 0) {
    if ((int)block.getTArguments()->size() < _descriptor->getNumberOfTArgs()) {
      sd_printf("%s: %i T args expected, but %i received\n", this->getOpName()->c_str(),
                _descriptor->getNumberOfTArgs(), block.getTArguments()->size());
      return Status::BAD_PARAMS;
    }
  } else if (_descriptor->getNumberOfTArgs() == -1)
    if (block.getTArguments()->size() == 0) {
      sd_printf("%s: Number of T arguments should be positive number, but got 0 arguments\n",
                this->getOpName()->c_str());
      return Status::BAD_PARAMS;
    }

  if (_descriptor->getNumberOfIArgs() > 0) {
    if ((int)block.getIArguments()->size() < _descriptor->getNumberOfIArgs()) {
      sd_printf("%s: %i int args expected, but %i received\n", this->getOpName()->c_str(),
                _descriptor->getNumberOfIArgs(), block.getIArguments()->size());
      return Status::BAD_PARAMS;
    }
  } else if (_descriptor->getNumberOfIArgs() == -1)
    if (block.getIArguments()->size() == 0) {
      sd_printf("%s: Number of Integer arguments should be positive number, but got 0 arguments\n",
                this->getOpName()->c_str());
      return Status::BAD_PARAMS;
    }

  return Status::OK;
}

Status DeclarableOp::validateInputDimensions(Context &block, int rank) {
  if (block.width() == 0) return Status::OK;

  for (auto p : *block.inputs()) {
    auto v = block.variable(p);
    NDArray *aV = v->getNDArray();

    if (aV == nullptr) return Status::BAD_INPUT;

    if (aV->rankOf() != rank) return Status::BAD_DIMENSIONS;
  }

  return Status::OK;
}

Status DeclarableOp::validateInput2D(Context &block) { return validateInputDimensions(block, 2); }

Status DeclarableOp::validateInput3D(Context &block) { return validateInputDimensions(block, 3); }

Status DeclarableOp::validateInput4D(Context &block) { return validateInputDimensions(block, 4); }

Status DeclarableOp::validateNonEmptyInput(Context &block) {
  if (this->getOpDescriptor()->getNumberOfInputs() == -2 || this->getOpDescriptor()->getNumberOfInputs() == 0)
    return Status::OK;

  if (block.width() < 1 && !block.isFastPath() && block.fastpath_in().size() < 1) {
    sd_printf("%s: no operands provided for the op", this->getOpName()->c_str());
    return Status::BAD_INPUT;
  }

  int cnt = 0;
  for (auto p : *block.inputs()) {
    auto v = block.variable(p);
    if (v == nullptr) {
      if (this->getOpName() != nullptr) {
        sd_printf("Node [%i:<%s>]: Variable [%i] (%i:%i) is NULL\n", block.getNodeId(), this->getOpName()->c_str(), cnt,
                  p.first, p.second);
      } else {
        sd_printf("Node [%i:<noname>]: Variable [%i] (%i:%i) is NULL\n", block.getNodeId(), cnt, p.first, p.second);
      }
      return Status::BAD_INPUT;
    }

    if (v->variableType() == NDARRAY) {
      NDArray *aV = v->getNDArray();

      // if array is empty intentionally - we're ok with that
      if (v->hasNDArray() && v->isEmpty()) continue;

      if (aV == nullptr || !aV->nonNull()) {
        if (this->getOpName() != nullptr) {
          sd_printf("Node [%i:<%s>]: NDArray [%i] (%i:%i) is NULL\n", block.getNodeId(), this->getOpName()->c_str(),
                    cnt, p.first, p.second);
        } else {
          sd_printf("Node [%i:<noname>]: NDArray [%i] (%i:%i) is NULL\n", block.getNodeId(), cnt, p.first, p.second);
        }
        return Status::BAD_INPUT;
      }
    }

    cnt++;
  }

  return Status::OK;
}

Status DeclarableOp::validateOrdersMatch(Context &block) {
  if (block.width() == 0) return Status::OK;

  NDArray *a0 = block.variable(0)->getNDArray();
  for (auto p : *block.inputs()) {
    auto v = block.variable(p);
    NDArray *aV = v->getNDArray();
    if (a0->ordering() != aV->ordering()) return Status::BAD_ORDER;
  }

  return Status::OK;
}

Status DeclarableOp::execute(RandomGenerator &rng, const std::vector<NDArray *> &inputs,
                             const std::vector<NDArray *> &outputs, const std::vector<double> &tArgs,
                             const std::vector<LongType> &iArgs, const std::vector<bool> &bArgs,
                             const std::vector<DataType> &dArgs, bool isInplace, DataType type) {
  VariableSpace variableSpace;
  FlowPath fp;
  variableSpace.setFlowPath(&fp);

  int cnt = -1;
  std::vector<int> in;
  for (auto v : inputs) {
    if (v == nullptr) continue;

    auto var = new Variable(v);
    var->markRemovable(false);
    in.push_back(cnt);
    variableSpace.putVariable(cnt--, var);
  }

  int et = 0;
  for (auto v : outputs) {
    auto var = new Variable(v);
    var->markRemovable(false);
    std::pair<int, int> pair(1, et++);
    variableSpace.putVariable(pair, var);
  }

  Context block(1, &variableSpace, false);
  block.fillInputs(in);
  block.markInplace(isInplace);
  block.setDataType(0, type);

  // we need this line for tests basically
  // if (rng != nullptr)
  block.setRng(rng);

  for (int e = 0; e < tArgs.size(); e++) block.getTArguments()->emplace_back(tArgs.at(e));

  // FIXME: iargs should be sd::LongType
  for (int e = 0; e < iArgs.size(); e++) block.getIArguments()->emplace_back(static_cast<int>(iArgs.at(e)));

  for (int e = 0; e < bArgs.size(); e++) block.getBArguments()->push_back(static_cast<int>(bArgs.at(e)));

  for (int e = 0; e < dArgs.size(); e++) block.getDArguments()->push_back(dArgs.at(e));

  Status result = this->execute(&block);

  return result;
}

Status DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs) {
  return execute(inputs, outputs, std::vector<double>(), std::vector<LongType>(), std::vector<bool>(),
                 std::vector<DataType>());
}

template <>
Status DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs,
                             std::initializer_list<double> tArgs) {
  return execute(inputs, outputs, tArgs, std::vector<LongType>(), std::vector<bool>(), std::vector<DataType>());
}

template <>
Status DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs,
                             std::initializer_list<DataType> dArgs) {
  return execute(inputs, outputs, std::vector<double>(), std::vector<LongType>(), std::vector<bool>(), dArgs);
}

template <>
Status DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs,
                             std::initializer_list<float> tArgs) {
  std::vector<double> realArgs;
  for (auto v : tArgs) realArgs.emplace_back(v);

  return execute(inputs, outputs, realArgs, std::vector<LongType>(), std::vector<bool>(),
                 std::vector<DataType>());
}

template <>
Status DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs,
                             std::initializer_list<LongType> iArgs) {
  return execute(inputs, outputs, std::vector<double>(), iArgs, std::vector<bool>(), std::vector<DataType>());
}

template <>
Status DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs,
                             std::initializer_list<int> iArgs) {
  std::vector<LongType> realArgs;
  for (auto v : iArgs) realArgs.emplace_back(v);

  return execute(inputs, outputs, std::vector<double>(), realArgs, std::vector<bool>(), std::vector<DataType>());
}

template <>
Status DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs,
                             std::initializer_list<bool> bArgs) {
  return execute(inputs, outputs, std::vector<double>(), std::vector<LongType>(), bArgs, std::vector<DataType>());
}

Status DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs,
                             const std::vector<double> &tArgs, const std::vector<LongType> &iArgs,
                             const std::vector<bool> &bArgs, const std::vector<DataType> &dArgs, bool isInplace) {
  Context ctx(1);

  for (int e = 0; e < inputs.size(); e++) {
    ctx.setInputArray(e, inputs[e]);
  }

  for (int e = 0; e < outputs.size(); e++) {
    ctx.setOutputArray(e, outputs[e]);
  }

  if (isInplace) ctx.markInplace(isInplace);

  ctx.setIArguments(iArgs);
  ctx.setTArguments(tArgs);
  ctx.setBArguments(bArgs);
  ctx.setDArguments(dArgs);

  return execute(&ctx);
}

ResultSet DeclarableOp::evaluate(const std::vector<NDArray *> &inputs) {
  return evaluate(inputs, std::vector<double>(), std::vector<LongType>(), std::vector<bool>(),
                  std::vector<DataType>());
}

template <>
ResultSet DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<int> iArgs) {
  std::vector<LongType> realArgs;
  for (auto v : iArgs) realArgs.emplace_back(v);

  return evaluate(inputs, std::vector<double>(), realArgs, std::vector<bool>(), std::vector<DataType>());
}

template <>
ResultSet DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<LongType> iArgs) {
  return evaluate(inputs, std::vector<double>(), iArgs, std::vector<bool>(), std::vector<DataType>());
}

template <>
ResultSet DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<float> tArgs) {
  std::vector<double> realArgs;
  for (auto v : tArgs) realArgs.emplace_back(v);

  return evaluate(inputs, realArgs, std::vector<LongType>(), std::vector<bool>(), std::vector<DataType>());
}

template <>
ResultSet DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<double> tArgs) {
  return evaluate(inputs, tArgs, std::vector<LongType>(), std::vector<bool>(), std::vector<DataType>());
}

template <>
ResultSet DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<bool> bArgs) {
  return evaluate(inputs, std::vector<double>(), std::vector<LongType>(), bArgs, std::vector<DataType>());
}

template <>
ResultSet DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<DataType> bArgs) {
  return evaluate(inputs, std::vector<double>(), std::vector<LongType>(), std::vector<bool>(), bArgs);
}

ResultSet DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, const std::vector<double> &tArgs,
                                 const std::vector<LongType> &iArgs, const std::vector<bool> &bArgs,
                                 const std::vector<DataType> &dArgs, bool isInplace) {
  VariableSpace variableSpace;
  // ResultSet arrayList;
  FlowPath fp;
  variableSpace.setFlowPath(&fp);

  int cnt = -1;
  std::vector<int> in;
  for (auto v : inputs) {
    if (v == nullptr) continue;

    auto var = new Variable(v);
    var->markRemovable(false);
    in.push_back(cnt);
    variableSpace.putVariable(cnt--, var);
  }

  Context block(1, &variableSpace, false);
  block.setDataType(0, FLOAT32);
  block.fillInputs(in);
  block.markInplace(isInplace);

  for (int e = 0; e < tArgs.size(); e++) block.getTArguments()->emplace_back(tArgs.at(e));

  for (int e = 0; e < iArgs.size(); e++) block.getIArguments()->emplace_back(iArgs.at(e));

  for (int e = 0; e < bArgs.size(); e++) block.getBArguments()->push_back(bArgs.at(e));

  for (int e = 0; e < dArgs.size(); e++) block.getDArguments()->push_back(dArgs.at(e));

  Status status = this->execute(&block);
  ResultSet arrayList;
  if (isInplace) arrayList.setNonRemovable();

  arrayList.setStatus(status);
  if (status != Status::OK) return arrayList;

  if (!isInplace) {
    if (block.isFastPath()) {
      // note this *is* similar to the code below but we use fast paths instead
      // we need to ensure variables don't get freed allowing reuse of outputs
      // as views
      for (int e = 0; e < DataTypeUtils::max<int>(); e++) {
        std::pair<int, int> pair(1, e);
        if (variableSpace.hasVariable(pair)) {
          auto var = variableSpace.getVariable(pair);
          auto arr = var->getNDArray();
          if (!arr->isAttached()) {
            var->markRemovable(false);
            arr->setContext(LaunchContext::defaultContext());
          }
        } else
          break;
      }
      for (int e = 0; e < block.fastpath_out().size(); e++) {
        auto arr = block.fastpath_out()[e];
        if (!arr->isAttached()) {
          arr->setContext(LaunchContext::defaultContext());
          arrayList.push_back(arr);
        } else {
          arrayList.push_back(arr->detach());
        }
      }

      arrayList.setNonRemovable();

    } else {
      for (int e = 0; e < DataTypeUtils::max<int>(); e++) {
        std::pair<int, int> pair(1, e);
        if (variableSpace.hasVariable(pair)) {
          auto var = variableSpace.getVariable(pair);
          auto arr = var->getNDArray();
          if (!arr->isAttached()) {
            var->markRemovable(false);
            arr->setContext(LaunchContext::defaultContext());
            arrayList.push_back(arr);
          } else {
            arrayList.push_back(arr->detach());
          }
        } else
          break;
      }
    }

  } else {
    for (auto v : inputs) {
      arrayList.push_back(v);
    }
  }

  return arrayList;
}

ResultSet DeclarableOp::execute(const OpArgsHolder &holder, bool isInplace) {
  // FIXME: add DArgs to OpArgsHolder
  return evaluate(holder.getInArrs(), holder.getTArgs(), holder.getIArgs(), holder.getBArgs(), std::vector<DataType>(),
                  isInplace);
}

Status DeclarableOp::validateInputDimensionsMatch(Context &block) {
  if (block.width() == 0) return Status::OK;

  NDArray *a0 = block.array(0);
  for (int e = 1; e < block.width(); e++) {
    auto aV = block.array(e);
    if (!shape::equalsSoft(a0->shapeInfo(), aV->shapeInfo())) return Status::BAD_DIMENSIONS;
  }

  return Status::OK;
}

Status DeclarableOp::validateInputLengthMatch(Context &block) {
  if (block.width() == 0) return Status::OK;

  LongType l0 = block.array(0)->lengthOf();
  for (uint32_t e = 0; e < block.width(); e++) {
    if (l0 != block.array(e)->lengthOf()) return Status::BAD_LENGTH;
  }

  return Status::OK;
}

samediff::EmptyHandling DeclarableOp::emptyHandling() { return samediff::EmptyHandling::EMPTY_SKIP; }

void DeclarableOp::registerTypes() { this->getOpDescriptor()->setSameMode(true); }


}  // namespace ops
}  // namespace sd
