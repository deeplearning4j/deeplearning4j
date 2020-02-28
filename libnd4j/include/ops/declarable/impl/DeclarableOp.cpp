/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <ops/declarable/DeclarableOp.h>
#include <Status.h>
#include <helpers/ShapeUtils.h>
#include <NDArrayFactory.h>
#include <exceptions/graph_exception.h>
#include <exceptions/unresolved_input_exception.h>
#include <ops/declarable/OpRegistrator.h>
#include <exceptions/datatype_exception.h>
#include <helpers/StringUtils.h>
#include <cstdarg>

namespace nd4j {
    namespace ops {
        Nd4jStatus conditionHelper(const char *file, int line, int condition, int argNumber, const char *format, ...) {
            if (!condition) {
                va_list args;

                printf("Error at [%s:%i:%i]:\n", file, line, argNumber);
                va_start(args, format);
                vprintf(format, args);
                va_end(args);
                printf("\n");
                fflush(stdout);

                return ND4J_STATUS_BAD_PARAMS;
            }
            return ND4J_STATUS_OK;
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

        DeclarableOp::DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs) {
            _descriptor = new OpDescriptor(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs);
            _name = opName;
        }

        DeclarableOp::~DeclarableOp() {
            if (_descriptor != nullptr)
                delete _descriptor;

            if (_scalar != nullptr)
                delete _scalar;
        }

        OpDescriptor* DeclarableOp::getOpDescriptor() {
            return _descriptor;
        }

        std::string *DeclarableOp::getOpName() {
            return _descriptor->getOpName();
        }

        Nd4jLong DeclarableOp::getOpHash() {
            return _descriptor->getHash();
        }


        nd4j::NDArray* nd4j::ops::DeclarableOp::getZ(Context& ctx, int inputId) {
            NDArray* z = nullptr;

            if (ctx.isFastPath()) {
                if (ctx.fastpath_out().size() <= inputId) {
                    if (ctx.isInplace()) {
                        z = ctx.fastpath_in()[inputId];
                    } else
                        throw std::runtime_error("fastpath_out: unresolved output array");
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
                        nd4j_printf("Can't get Z variable for node_%i!\n", ctx.nodeId());
                    }
                } else {
                    nd4j_printf("BOOM!\n", "");
                    throw std::runtime_error("Boom!");
                }
            }

            return z;
        }

        int nd4j::ops::DeclarableOp::prepareOutputs(Context &ctx) {
            auto workspace = ctx.getWorkspace();
            GraphProfile *prof = nullptr;
            NodeProfile *node = nullptr;
            std::chrono::time_point<std::chrono::system_clock> inputEnd, inputStart, shapeStart, shapeEnd, arrayStart, arrayEnd;
            bool canUseFastPath = true;

            auto fp = ctx.isFastPath();

            if (Environment::getInstance()->isProfiling()) {
                if (ctx.getVariableSpace() != nullptr && ctx.getVariableSpace()->flowPath() != nullptr) {
                    prof = ctx.getVariableSpace()->flowPath()->profile();
                    node = prof->nodeById(ctx.nodeId());
                }
            }

            if (ctx.isInplace()) {
                if (Environment::getInstance()->isProfiling() && node != nullptr) {
                    if (fp) {
                        //
                    } else {
                        for (auto p: *ctx.inputs()) {
                            auto var = ctx.variable(p);
                            if (var->variableType() == VariableType::NDARRAY) {
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
                    for (auto p: *ctx.inputs()) {
                        auto var = ctx.variable(p);
                        if (var->variableType() == VariableType::NDARRAY) {
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

                if (!canUseFastPath)
                    ctx.forbidFastPath(true);

                // do nothing, getZ result will do the trick
                return static_cast<int>(ctx.width());
            } else {
                // if op is not inplace - we should pre-allocate arrays
                ShapeList inSha;
                int results = 0;

                if (Environment::getInstance()->isProfiling() && node != nullptr)
                    inputStart = std::chrono::system_clock::now();

                int cntIn = 0;
                // we build list of input shapes
                if (fp) {
                    for (const auto p:ctx.fastpath_in()) {
                        inSha.push_back(p == nullptr ? nullptr : p->getShapeInfo());
                    }
                } else {
                    int arrCnt = 0;
                    for (auto p: *ctx.inputs()) {
                        auto var = ctx.variable(p);
                        if (var->variableType() == VariableType::NDARRAY) {
                            NDArray *array = var->getNDArray();
                            if (array == nullptr)
                                throw unresolved_input_exception::build("Variable wasn't resolved prior shape calculation", p);

                            inSha.push_back(array->getShapeInfo());

                            // we're also filling ctx with arrays
                            if (canUseFastPath)
                                ctx.setInputArray(arrCnt++, array);
                        } else {
                            canUseFastPath = false;
                        }
                        cntIn++;
                    }
                }

                // if we override shape function, we'll return size of fastPath
                if (fp && ctx.shapeFunctionOverride()) {
                    return (int) ctx.fastpath_out().size();
                }

                // optionally saving input time
                if (Environment::getInstance()->isProfiling() && node != nullptr) {
                    inputEnd = std::chrono::system_clock::now();
                    auto inputTime = std::chrono::duration_cast<std::chrono::nanoseconds>(inputEnd - inputStart).count();
                    node->setInputTime(inputTime);

                    // saving output shapes in profile
                    for (int e = 0; e < inSha.size(); e++)
                        node->addInputShape(inSha.at(e));

                    shapeStart = std::chrono::system_clock::now();
                }

                auto outSha = this->calculateOutputShape(&inSha, ctx);
                results = outSha->size();

                // optionally saving shapeTime
                if (Environment::getInstance()->isProfiling() && node != nullptr) {
                    shapeEnd = std::chrono::system_clock::now();
                    auto prepTime = std::chrono::duration_cast<std::chrono::nanoseconds>(shapeEnd - shapeStart).count();
                    node->setShapeFunctionTime(prepTime);

                    // saving output shapes in profile
                    for (int e = 0; e < outSha->size(); e++)
                        node->addOutputShape(outSha->at(e));

                    arrayStart = std::chrono::system_clock::now();
                }

                int cnt = 0;

                for (auto out: *outSha->asVector()) {
                    if (!fp) {
                        // we need to check, if Z is really needed
                        std::pair<int, int> pair(ctx.nodeId(), cnt++);

                        if (!ctx.isValueAvailable(pair.second)) {
                            if (Environment::getInstance()->isDebugAndVerbose())
                                shape::printShapeInfoLinear("Going to create variable with shape", out);

                            auto outArr = new NDArray(out, true, ctx.launchContext());

                            ctx.pushNDArrayToVariableSpace(pair, outArr);

                            if (canUseFastPath)
                                ctx.setOutputArray(pair.second, outArr);
                        } else {
                            // validate/compare shapes here. existent vs provided in outSha
                            auto var = ctx.variable(pair);
                            auto shape = var->getNDArray()->shapeInfo();

                            if (canUseFastPath)
                                ctx.setOutputArray(pair.second, var->getNDArray());

                            if (!shape::equalsSoft(out, shape) || shape::isEmpty(out) != shape::isEmpty(shape)) {
                                auto eShape = ShapeUtils::shapeAsString(out);
                                auto aShape = ShapeUtils::shapeAsString(shape);

                                //outSha->destroy();
                                delete outSha;

                                nd4j_printf("Expected vs provided shapes mismatch %s vs %s at index %i\n", eShape.c_str(), aShape.c_str(), pair.second);
                                throw std::runtime_error("Expected vs provided shapes mismatch");
                            }

                            /*
                             * FIXME: we want to uncomment this eventually, and check data types equality
                            //checking out data type equality
                            if (ArrayOptions::dataType(out) != ArrayOptions::dataType(shape)) {
                                std::string msg = "Provided array [" + StringUtils::valueToString<int>(pair.second) + "] has unexpected data type";
                                throw nd4j::datatype_exception::build(msg, ArrayOptions::dataType(out), ArrayOptions::dataType(shape));
                            }
                             */
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
                            // checking out shape equality
                            if (!shape::equalsSoft(out, array->shapeInfo()) || shape::isEmpty(out) != array->isEmpty()) {
                                auto eShape = ShapeUtils::shapeAsString(out);
                                auto aShape = ShapeUtils::shapeAsString(array->shapeInfo());

                                //outSha->destroy();
                                delete outSha;

                                nd4j_printf("Expected vs provided shape mismatch %s vs %s at index %i\n", eShape.c_str(), aShape.c_str(), idx);
                                throw std::runtime_error("Expected vs provided shape mismatch");
                            }
                        }
                    }
                }

                if (!canUseFastPath)
                    ctx.forbidFastPath(true);

                delete outSha;

                // saving arrayTime
                if (Environment::getInstance()->isProfiling() && node != nullptr) {
                    arrayEnd = std::chrono::system_clock::now();
                    auto arrayTime = std::chrono::duration_cast<std::chrono::nanoseconds>(arrayEnd - arrayStart).count();
                    node->setArrayTime(arrayTime);
                }

                return results;
            }
        }

        void nd4j::ops::DeclarableOp::storeResult(Context &block, int outputNumber, NDArray* array) {
            this->storeResult(block, outputNumber, *array);
        }

        void nd4j::ops::DeclarableOp::storeResult(nd4j::graph::Context &ctx, int outputNumber, NDArray& array) {
            ctx.pushNDArrayToVariableSpace(ctx.nodeId(), outputNumber, &array, !ctx.isInplace());
        }

        bool nd4j::ops::DeclarableOp::allocateResult(Context& block, Nd4jLong* shape) {
            auto var = block.variable(block.getNodeId(), 0);

            auto workspace = block.getWorkspace();

            Nd4jLong len = shape::length(shape);
            Nd4jLong* __shape;
            ALLOCATE(__shape, workspace, shape::shapeInfoLength(shape), Nd4jLong); //new int[shape[0] * 2 + 4];

            memcpy(__shape, shape, shape::shapeInfoByteLength(shape));

            // if that's first run - we probably have nothing here
            if (var->getNDArray() == nullptr) {

                std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(len * sizeof(int8_t), ArrayOptions::dataType(__shape), workspace);
                var->setNDArray(new NDArray(buffer, ShapeDescriptor(__shape), block.launchContext()));
            }
            else if(var->getNDArray()->lengthOf() != len) {
                // if length not match - lets reallocate array
                delete var->getNDArray();
                std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(len * sizeof(int8_t), ArrayOptions::dataType(__shape), workspace);
                var->setNDArray(new NDArray(buffer, ShapeDescriptor(__shape), block.launchContext()));
            }

            return true;
        }


        bool nd4j::ops::DeclarableOp::allocateResult(Context& block, std::initializer_list<Nd4jLong>& shape, char order) {
            auto var = block.variable(block.getNodeId(), 0);
            auto workspace = block.getWorkspace();

            Nd4jLong len = shape::length(shape);
            // if that's first run - we probably have nothing here
            if (var->getNDArray() == nullptr) {
                var->setNDArray(new NDArray(order, shape, block.dataType(), block.launchContext()));
            } else if(var->getNDArray()->lengthOf() != len) {
                // if length not match - lets reallocate array
                delete var->getNDArray();
                var->setNDArray(new NDArray(order, shape, block.dataType(), block.launchContext()));
            }

            return true;
        }

        Nd4jStatus nd4j::ops::DeclarableOp::validateDataTypes(Context& block) {
            _registrator.lock();
            if (!_registered) {
                _registered = true;
                this->registerTypes();
            }
            _registrator.unlock();

            // rolling over inputs first
            int cnt = 0, inT = 0;
            std::vector<nd4j::DataType> inputTypes(block.width());
            if (block.isFastPath()) {
                for (auto array: block.fastpath_in()) {
                    if (array == nullptr)
                        continue;

                    inputTypes[inT++] = array->dataType();
                    if (!_descriptor->checkInputMatch(cnt, array->dataType())) {
                        auto ctype = DataTypeUtils::asString(array->dataType());
                        nd4j_printf("Op [%s] failed check for input [%i], DataType: [%s]\n",
                                    _descriptor->getOpName()->data(), cnt, ctype.c_str());
                        return ND4J_STATUS_BAD_ARGUMENTS;
                    }
                    cnt++;
                }
            } else {
                for (auto &p: *(block.inputs())) {
                    auto var = block.variable(p);

                    // we're not checking validity, if ANY types were explicitly allowed
                    //if (block.dataType(cnt) == nd4j::DataType::ANY)
                    //    continue;

                    // only validating non-null variables
                    if (var != nullptr && var->hasNDArray()) {
                        auto array = var->getNDArray();

                        inputTypes[inT++] = array->dataType();
                        if (!_descriptor->checkInputMatch(cnt, array->dataType())) {
                            auto ctype = DataTypeUtils::asString(array->dataType());
                            nd4j_printf("Op [%s] failed check for input [%i], DataType: [%s]\n",
                                        _descriptor->getOpName()->data(), cnt, ctype.c_str());
                            return ND4J_STATUS_BAD_ARGUMENTS;
                        }
                    }

                    cnt++;
                }
            }

            if (block.isFastPath()) {
                int index = 0;
                for (auto array: block.fastpath_out()) {
                    if (array == nullptr)
                        continue;

                    auto cType = array->dataType();

                    if (_descriptor->isSameMode()) {

                        if (index >= block.width()) {
                            if (block.fastpath_in().size() == 0)
                                continue;

                            auto ia = block.fastpath_in()[0];

                            if (ia->dataType() != cType) {
                                auto t = DataTypeUtils::asString(cType);
                                nd4j_printf("Op [%s] failed check for output [%i], DataType: [%s]\n",
                                            _descriptor->getOpName()->data(), index, t.c_str());
                                return ND4J_STATUS_BAD_ARGUMENTS;
                            }
                        } else {
                            // for same mode, output type must be the same as input type
                            auto ia = block.fastpath_in()[index];

                            if (ia->dataType() != cType) {
                                auto t = DataTypeUtils::asString(cType);
                                nd4j_printf("Op [%s] failed check for output [%i], DataType: [%s]\n",
                                            _descriptor->getOpName()->data(), index, t.c_str());
                                return ND4J_STATUS_BAD_ARGUMENTS;
                            }
                        }
                    } else if (_descriptor->isInherit(index)) {
                        // in inherit mode, output type must be the same as one of input types
                        if (std::find(inputTypes.begin(), inputTypes.end(), cType) == inputTypes.end()) {
                            auto t = DataTypeUtils::asString(cType);
                            nd4j_printf("Op [%s] failed check for output [%i], DataType: [%s].\n",
                                        _descriptor->getOpName()->data(), index, t.c_str());
                            return ND4J_STATUS_BAD_ARGUMENTS;
                        }

                    } else if (!_descriptor->checkOutputMatch(index, cType)) {
                        auto t = DataTypeUtils::asString(cType);
                        nd4j_printf("Op [%s] failed check for output [%i], DataType: [%s];\n",
                                    _descriptor->getOpName()->data(), index, t.c_str());
                        return ND4J_STATUS_BAD_ARGUMENTS;
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
                                    if (block.width() == 0)
                                        continue;

                                    auto iv = block.variable(0);

                                    if (iv->getNDArray()->dataType() != cType) {
                                        auto t = DataTypeUtils::asString(cType);
                                        nd4j_printf("Op [%s] failed check for output [%i], DataType: [%s]\n",
                                                    _descriptor->getOpName()->data(), index, t.c_str());
                                        return ND4J_STATUS_BAD_ARGUMENTS;
                                    }
                                } else {
                                    // for same mode, output type must be the same as input type
                                    auto iv = block.variable(index);

                                    if (iv->getNDArray()->dataType() != cType) {
                                        auto t = DataTypeUtils::asString(cType);
                                        nd4j_printf("Op [%s] failed check for output [%i], DataType: [%s]\n",
                                                    _descriptor->getOpName()->data(), index, t.c_str());
                                        return ND4J_STATUS_BAD_ARGUMENTS;
                                    }
                                }
                            } else if (_descriptor->isInherit(index)) {
                                // in inherit mode, output type must be the same as one of input types
                                if (std::find(inputTypes.begin(), inputTypes.end(), cType) == inputTypes.end()) {
                                    auto t = DataTypeUtils::asString(cType);
                                    nd4j_printf("Op [%s] failed check for output [%i], DataType: [%s].\n",
                                                _descriptor->getOpName()->data(), index, t.c_str());
                                    return ND4J_STATUS_BAD_ARGUMENTS;
                                }

                            } else if (!_descriptor->checkOutputMatch(index, cType)) {
                                auto t = DataTypeUtils::asString(cType);
                                nd4j_printf("Op [%s] failed check for output [%i], DataType: [%s];\n",
                                            _descriptor->getOpName()->data(), index, t.c_str());
                                return ND4J_STATUS_BAD_ARGUMENTS;
                            }
                        }
                    } else
                        break;
                }
            }


            return ND4J_STATUS_OK;
        }

        Nd4jStatus nd4j::ops::DeclarableOp::execute(Context* block) {
            nd4j_debug("Executing op: [%s]\n", this->getOpName()->c_str());

            std::chrono::time_point<std::chrono::system_clock> timeEnter, timeStart, timeEnd;
            Nd4jLong prepTime, outerTime;

            Nd4jLong memoryBefore = block->workspace() == nullptr ? 0L : block->workspace()->getSpilledSize() + block->workspace()->getUsedSize();
            if (Environment::getInstance()->isProfiling())
                timeEnter = std::chrono::system_clock::now();

            // basic validation: ensure inputs are set
            REQUIRE_OK(this->validateNonEmptyInput(*block));

            // ensure number of IArgs, TArgs match our expectations
            REQUIRE_OK(this->validateArguments(*block));

            // validating data types for inputs and (optionally) outputs
            REQUIRE_OK(this->validateDataTypes(*block));


            // this method will allocate output NDArrays for this op
            auto numOutputs = this->prepareOutputs(*block);

            if (Environment::getInstance()->isProfiling()) {
                timeStart = std::chrono::system_clock::now();
                prepTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeStart - timeEnter).count();
            }


            Nd4jStatus status;
            bool hasHelper = false;

            // platform helpers use might be forbidden for various reasons, so we'll check it out first
            if (block->helpersAllowed() && nd4j::Environment::getInstance()->helpersAllowed()) {
                // if we have platform-specific helper for this op - invoke it
                if (OpRegistrator::getInstance()->hasHelper(this->getOpHash(), block->engine())) {
                    auto helper = OpRegistrator::getInstance()->getPlatformHelper(this->getOpHash(), block->engine());
                    if (helper->isUsable(*block)) {
                        status = helper->invokeHelper(*block);
                        hasHelper = true;
                    }
                }
            }

            // if we don't have platform-specific helper - invoke generic implementation
            if (!hasHelper)
                status = this->validateAndExecute(*block);

            // optionally saving execution time
            if (Environment::getInstance()->isProfiling()) {
                timeEnd = std::chrono::system_clock::now();
                outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
                block->setInnerTime(outerTime);
            }

            if (Environment::getInstance()->isProfiling() && block->getVariableSpace() != nullptr) {
                auto fp = block->getVariableSpace()->flowPath();
                if (fp != nullptr) {
                    auto p = fp->profile();
                    if (p != nullptr) {
                        Nd4jLong memoryAfter = block->workspace() == nullptr ? 0L : block->workspace()->getSpilledSize() + block->workspace()->getUsedSize();
                        Nd4jLong memoryUsed = memoryAfter - memoryBefore;
                        p->nodeById(block->nodeId())->setPreparationTime(prepTime);
                        p->nodeById(block->nodeId())->setExecutionTime(outerTime);
                        p->nodeById(block->nodeId())->setTotalSize(memoryUsed);
                    }
                }
            }


            // now we print out all outputs for this node
            if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                auto vs = block->getVariableSpace();

                for (int e = 0; e < numOutputs; e++) {
                    // if given output index doesn't exist - we're done

                    if (!block->isFastPath()) {
                        if (!vs->hasVariable(block->nodeId(), e))
                            break;
                    } else {
                        // we have to check either in or out stack, depending on isInplace()
                        if (block->isInplace()) {
                            if (block->fastpath_in().size() <= e)
                                break;
                        } else {
                            if (block->fastpath_out().size() <= e)
                                break;
                        }
                    }

                    auto array = block->isFastPath() ? block->isInplace() ? block->fastpath_in()[e] : block->fastpath_out()[e] : vs->getVariable(block->nodeId(), e)->getNDArray();

                    auto shape = ShapeUtils::shapeAsString(array);
                    auto first = array->isEmpty() ? std::string("Empty NDArray") : array->asString(32);
                    auto type = DataTypeUtils::asString(array->dataType());

                    nd4j_printf("node_%i:%i result shape: %s; dtype: %s; first values %s\n", block->nodeId(), e, shape.c_str(), type.c_str(), first.c_str());
                }
            }

            return status;
        }

        void DeclarableOp::overwriteResult(Context &block, int outputIdx, NDArray *array) {
            throw std::runtime_error("Overwrite result used!");
            //block.pushNDArrayToVariableSpace(block.nodeId(), outputIdx, array);
            /*
            auto varSpace = block.getVariableSpace();
            if (varSpace->hasVariable(block.getNodeId(), outputIdx)) {
                auto var = varSpace->getVariable(block.getNodeId(), outputIdx);
                if (var->getNDArray() != nullptr && var->isRemovable())
                    delete var->getNDArray();

                var->setNDArray(array);
                var->markRemovable(true);
            } else {
                auto var = new Variable(array, nullptr, block.getNodeId(), outputIdx);
                varSpace->putVariable(block.getNodeId(), outputIdx, var);
            }
            */
        }

        void DeclarableOp::overwriteResult(Context &block, int outputIdx, NDArrayList *list) {
            throw std::runtime_error("Overwrite result used!");
            //block.pushNDArrayListToVariableSpace(block.nodeId(), outputIdx, list);
            /*
            auto varSpace = block.getVariableSpace();
            if (varSpace->hasVariable(block.getNodeId(), outputIdx)) {
                auto var = varSpace->getVariable(block.getNodeId(), outputIdx);
                var->setNDArrayList(list);
            } else {
                auto var = new Variable(nullptr, nullptr, block.getNodeId(), outputIdx);
                var->setNDArrayList(list);
                varSpace->putVariable(block.getNodeId(), outputIdx, var);
            }
            */
        }

        Nd4jStatus nd4j::ops::DeclarableOp::validateArguments(Context& block) {
            /*
             * We're checking number of T and I arguments. If number of args is finite number - we check strict equality
             * If number of args is variable (-1), but variables MUST be present - we check for non-zero number of arguments
             */
            if (_descriptor->getNumberOfTArgs() > 0) {
                if ((int) block.getTArguments()->size() < _descriptor->getNumberOfTArgs()) {
                    nd4j_printf("%s: %i T args expected, but %i received\n", this->getOpName()->c_str(), _descriptor->getNumberOfTArgs(), block.getTArguments()->size());
                    return ND4J_STATUS_BAD_PARAMS;
                }
            } else
            if (_descriptor->getNumberOfTArgs() == -1)
                if (block.getTArguments()->size() == 0) {
                    nd4j_printf("%s: Number of T arguments should be positive number, but got 0 arguments\n", this->getOpName()->c_str());
                    return ND4J_STATUS_BAD_PARAMS;
                }

            if (_descriptor->getNumberOfIArgs() > 0) {
                if ((int) block.getIArguments()->size() < _descriptor->getNumberOfIArgs()) {
                    nd4j_printf("%s: %i int args expected, but %i received\n", this->getOpName()->c_str(), _descriptor->getNumberOfIArgs(), block.getIArguments()->size());
                    return ND4J_STATUS_BAD_PARAMS;
                }
            } else
            if (_descriptor->getNumberOfIArgs() == -1)
                if (block.getIArguments()->size() == 0) {
                    nd4j_printf("%s: Number of Integer arguments should be positive number, but got 0 arguments\n", this->getOpName()->c_str());
                    return ND4J_STATUS_BAD_PARAMS;
                }


            return ND4J_STATUS_OK;
        }

        Nd4jStatus nd4j::ops::DeclarableOp::validateInputDimensions(Context& block, int rank) {
            if (block.width() == 0)
                return ND4J_STATUS_OK;

            for (auto p: *block.inputs()) {
                auto v = block.variable(p);
                NDArray *aV = v->getNDArray();

                if (aV == nullptr)
                    return ND4J_STATUS_BAD_INPUT;

                if (aV->rankOf() != rank)
                    return ND4J_STATUS_BAD_DIMENSIONS;
            }

            return ND4J_STATUS_OK;
        }

        Nd4jStatus nd4j::ops::DeclarableOp::validateInput2D(Context& block) {
            return validateInputDimensions(block, 2);
        }

        Nd4jStatus nd4j::ops::DeclarableOp::validateInput3D(Context& block) {
            return validateInputDimensions(block, 3);
        }

        Nd4jStatus nd4j::ops::DeclarableOp::validateInput4D(Context& block) {
            return validateInputDimensions(block, 4);
        }

        Nd4jStatus nd4j::ops::DeclarableOp::validateNonEmptyInput(Context& block) {
            if (this->getOpDescriptor()->getNumberOfInputs() == -2 || this->getOpDescriptor()->getNumberOfInputs() == 0)
                return Status::OK();

            if (block.width() < 1) {
                nd4j_printf("%s: no operands provided for the op", this->getOpName()->c_str());
                return ND4J_STATUS_BAD_INPUT;
            }


            int cnt = 0;
            for (auto p: *block.inputs()) {
                auto v = block.variable(p);
                if (v == nullptr) {
                    if (this->getOpName() != nullptr) {
                        nd4j_printf("Node [%i:<%s>]: Variable [%i] (%i:%i) is NULL\n", block.getNodeId(), this->getOpName()->c_str(), cnt, p.first, p.second);
                    } else {
                        nd4j_printf("Node [%i:<noname>]: Variable [%i] (%i:%i) is NULL\n", block.getNodeId(), cnt, p.first, p.second);
                    }
                    return ND4J_STATUS_BAD_INPUT;
                }

                if (v->variableType() == VariableType::NDARRAY) {
                    NDArray *aV = v->getNDArray();

                    // if array is empty intentionally - we're ok with that
                    if (v->hasNDArray() && v->isEmpty())
                        continue;

                    if (aV == nullptr || !aV->nonNull()) {
                        if (this->getOpName() != nullptr) {
                            nd4j_printf("Node [%i:<%s>]: NDArray [%i] (%i:%i) is NULL\n", block.getNodeId(), this->getOpName()->c_str(), cnt, p.first, p.second);
                        } else {
                            nd4j_printf("Node [%i:<noname>]: NDArray [%i] (%i:%i) is NULL\n", block.getNodeId(), cnt, p.first, p.second);
                        }
                        return ND4J_STATUS_BAD_INPUT;
                    }
                }

                cnt++;
            }

            return ND4J_STATUS_OK;
        }

        Nd4jStatus nd4j::ops::DeclarableOp::validateOrdersMatch(Context& block) {
            if (block.width() == 0)
                return ND4J_STATUS_OK;

            NDArray *a0 = block.variable(0)->getNDArray();
            for (auto p: *block.inputs()) {
                auto v = block.variable(p);
                NDArray *aV = v->getNDArray();
                if (a0->ordering() != aV->ordering())
                    return ND4J_STATUS_BAD_ORDER;
            }

            return ND4J_STATUS_OK;
        }

        Nd4jStatus nd4j::ops::DeclarableOp::execute(nd4j::graph::RandomGenerator& rng, const std::vector<NDArray*>& inputs, const std::vector<NDArray*>& outputs, const std::vector<double>& tArgs, const std::vector<Nd4jLong>& iArgs, const std::vector<bool>& bArgs, const std::vector<nd4j::DataType>& dArgs, bool isInplace, nd4j::DataType type) {
            VariableSpace variableSpace;
            FlowPath fp;
            variableSpace.setFlowPath(&fp);

            int cnt = -1;
            std::vector<int> in;
            for (auto v: inputs) {
                if (v == nullptr)
                    continue;

                auto var = new Variable(v);
                var->markRemovable(false);
                in.push_back(cnt);
                variableSpace.putVariable(cnt--, var);
            }

            int et = 0;
            for (auto v: outputs) {
                auto var = new Variable(v);
                var->markRemovable(false);
                std::pair<int,int> pair(1, et++);
                variableSpace.putVariable(pair, var);
            }

            Context block(1, &variableSpace, false);
            block.fillInputs(in);
            block.markInplace(isInplace);
            block.setDataType(0, type);

            // we need this line for tests basically
            //if (rng != nullptr)
            block.setRng(rng);

            for (int e = 0; e < tArgs.size(); e++)
                block.getTArguments()->emplace_back(tArgs.at(e));

            // FIXME: iargs should be Nd4jLong
            for (int e = 0; e < iArgs.size(); e++)
                block.getIArguments()->emplace_back(static_cast<int>(iArgs.at(e)));

            for (int e = 0; e < bArgs.size(); e++)
                block.getBArguments()->push_back(static_cast<int>(bArgs.at(e)));

            for (int e = 0; e < dArgs.size(); e++)
                block.getDArguments()->push_back(dArgs.at(e));

            Nd4jStatus result = this->execute(&block);

            return result;
        }

        Nd4jStatus DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs) {
            return execute(inputs, outputs, std::vector<double>(), std::vector<Nd4jLong>(), std::vector<bool>(), std::vector<nd4j::DataType>());
        }

        template <>
        Nd4jStatus DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs, std::initializer_list<double> tArgs) {
            return execute(inputs, outputs, tArgs, std::vector<Nd4jLong>(), std::vector<bool>(), std::vector<nd4j::DataType>());
        }

        template <>
        Nd4jStatus DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs, std::initializer_list<nd4j::DataType> dArgs) {
            return execute(inputs, outputs, std::vector<double>(), std::vector<Nd4jLong>(), std::vector<bool>(), dArgs);
        }

        template <>
        Nd4jStatus DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs, std::initializer_list<float> tArgs) {
            std::vector<double> realArgs;
            for (auto v:tArgs)
                realArgs.emplace_back(v);

            return execute(inputs, outputs, realArgs, std::vector<Nd4jLong>(), std::vector<bool>(), std::vector<nd4j::DataType>());
        }

        template <>
        Nd4jStatus DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs, std::initializer_list<Nd4jLong> iArgs) {
            return execute(inputs, outputs, std::vector<double>(), iArgs, std::vector<bool>(), std::vector<nd4j::DataType>());
        }

        template <>
        Nd4jStatus DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs, std::initializer_list<int> iArgs) {
            std::vector<Nd4jLong> realArgs;
            for (auto v:iArgs)
                realArgs.emplace_back(v);

            return execute(inputs, outputs, std::vector<double>(), realArgs, std::vector<bool>(), std::vector<nd4j::DataType>());
        }

        template <>
        Nd4jStatus DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs, std::initializer_list<bool> bArgs) {
            return execute(inputs, outputs, std::vector<double>(), std::vector<Nd4jLong>(), bArgs, std::vector<nd4j::DataType>());
        }

        Nd4jStatus DeclarableOp::execute(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs, const std::vector<double> &tArgs, const std::vector<Nd4jLong> &iArgs, const std::vector<bool> &bArgs, const std::vector<nd4j::DataType> &dArgs, bool isInplace) {
            Context ctx(1);

            for (int e = 0; e < inputs.size(); e++) {
                ctx.setInputArray(e, inputs[e]);
            }

            for (int e = 0; e < outputs.size(); e++) {
                ctx.setOutputArray(e, outputs[e]);
            }


            if (isInplace)
                ctx.markInplace(isInplace);

            ctx.setIArguments(iArgs);
            ctx.setTArguments(tArgs);
            ctx.setBArguments(bArgs);
            ctx.setDArguments(dArgs);

            return execute(&ctx);
        }

        nd4j::ResultSet *DeclarableOp::evaluate(const std::vector<NDArray *> &inputs) {
            return evaluate(inputs, std::vector<double>(), std::vector<Nd4jLong>(), std::vector<bool>(), std::vector<nd4j::DataType>());
        }

        template <>
        nd4j::ResultSet *DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<int> iArgs) {
            std::vector<Nd4jLong> realArgs;
            for (auto v:iArgs)
                realArgs.emplace_back(v);

            return evaluate(inputs, std::vector<double>(), realArgs, std::vector<bool>(), std::vector<nd4j::DataType>());
        }

        template <>
        nd4j::ResultSet *DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<Nd4jLong> iArgs) {
            return evaluate(inputs, std::vector<double>(), iArgs, std::vector<bool>(), std::vector<nd4j::DataType>());
        }

        template <>
        nd4j::ResultSet *DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<float> tArgs) {
            std::vector<double> realArgs;
            for (auto v:tArgs)
                realArgs.emplace_back(v);

            return evaluate(inputs, realArgs, std::vector<Nd4jLong>(), std::vector<bool>(), std::vector<nd4j::DataType>());
        }

        template <>
        nd4j::ResultSet *DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<double> tArgs) {
            return evaluate(inputs, tArgs, std::vector<Nd4jLong>(), std::vector<bool>(), std::vector<nd4j::DataType>());
        }

        template <>
        nd4j::ResultSet *DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<bool> bArgs) {
            return evaluate(inputs, std::vector<double>(), std::vector<Nd4jLong>(), bArgs, std::vector<nd4j::DataType>());
        }

        template <>
        nd4j::ResultSet *DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, std::initializer_list<nd4j::DataType> bArgs) {
            return evaluate(inputs, std::vector<double>(), std::vector<Nd4jLong>(), std::vector<bool>(), bArgs);
        }

        nd4j::ResultSet *DeclarableOp::evaluate(const std::vector<NDArray *> &inputs, const std::vector<double> &tArgs, const std::vector<Nd4jLong> &iArgs, const std::vector<bool> &bArgs, const std::vector<nd4j::DataType> &dArgs, bool isInplace) {
            VariableSpace variableSpace;
            //ResultSet arrayList;
            FlowPath fp;
            variableSpace.setFlowPath(&fp);

            int cnt = -1;
            std::vector<int> in;
            for (auto v: inputs) {
                if (v == nullptr)
                    continue;

                auto var = new Variable(v);
                var->markRemovable(false);
                in.push_back(cnt);
                variableSpace.putVariable(cnt--, var);
            }

            Context block(1, &variableSpace, false);
            block.setDataType(0, nd4j::DataType::FLOAT32);
            block.fillInputs(in);
            block.markInplace(isInplace);
            // block.setRNG(ProviderRNG::getInstance().getRNG());

            for (int e = 0; e < tArgs.size(); e++)
                block.getTArguments()->emplace_back(tArgs.at(e));

            for (int e = 0; e < iArgs.size(); e++)
                block.getIArguments()->emplace_back(iArgs.at(e));

            for (int e = 0; e < bArgs.size(); e++)
                block.getBArguments()->push_back(bArgs.at(e));

            for (int e = 0; e < dArgs.size(); e++)
                block.getDArguments()->push_back(dArgs.at(e));

            Nd4jStatus status = this->execute(&block);
            auto arrayList = new ResultSet();
            if (isInplace)
                arrayList->setNonRemovable();

            arrayList->setStatus(status);
            if (status != ND4J_STATUS_OK)
                return arrayList;

            if (!isInplace) {
                for (int e = 0; e < DataTypeUtils::max<int>(); e++) {
                    std::pair<int, int> pair(1, e);
                    if (variableSpace.hasVariable(pair)) {
                        auto var = variableSpace.getVariable(pair);
                        auto arr = var->getNDArray();
                        if (!arr->isAttached()) {
                            var->markRemovable(false);
                            arr->setContext(nd4j::LaunchContext::defaultContext());
                            arrayList->push_back(arr);
                        } else {
                            arrayList->push_back(arr->detach());
                        }
                    } else
                        break;
                }
            } else {
                for (auto v:inputs) {
                    arrayList->push_back(v);
                }
            }

            return arrayList;
        }

        nd4j::ResultSet* nd4j::ops::DeclarableOp::execute(const nd4j::OpArgsHolder& holder, bool isInplace) {
            // FIXME: add DArgs to OpArgsHolder
            return evaluate(holder.getInArrs(), holder.getTArgs(), holder.getIArgs(), holder.getBArgs(), std::vector<nd4j::DataType>(), isInplace);
        }

        Nd4jStatus nd4j::ops::DeclarableOp::validateInputDimensionsMatch(Context& block) {
            if (block.width() == 0)
                return ND4J_STATUS_OK;

            NDArray *a0 = block.array(0);
            for (int e = 0; e < block.width(); e++) {
                auto aV = block.array(e);
                if (!shape::equalsSoft(a0->getShapeInfo(), aV->getShapeInfo()))
                    return ND4J_STATUS_BAD_DIMENSIONS;
            }

            return ND4J_STATUS_OK;
        }

        Nd4jStatus nd4j::ops::DeclarableOp::validateInputLengthMatch(Context& block) {
            if (block.width() == 0)
                return ND4J_STATUS_OK;


            Nd4jLong l0 = block.array(0)->lengthOf();
            for (uint32_t e = 0; e < block.width(); e++) {
                if (l0 != block.array(e)->lengthOf())
                    return ND4J_STATUS_BAD_LENGTH;
            }

            return ND4J_STATUS_OK;
        }

        samediff::EmptyHandling DeclarableOp::emptyHandling() {
            return samediff::EmptyHandling::EMPTY_SKIP;
        }

        void DeclarableOp::registerTypes() {
            this->getOpDescriptor()->setSameMode(true);
        }

        /*
        template <typename T>
        int* nd4j::ops::DeclarableOp::calculateOutputShape(int* inputShape, nd4j::graph::Block& block) {
            // default implementation suits transform, so just returns the same shape

            int* newshape;
            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(inputShape), int);
            memcpy(newshape, inputShape, shape::shapeInfoByteLength(inputShape));

            return newshape;
        }
        */
    }
}