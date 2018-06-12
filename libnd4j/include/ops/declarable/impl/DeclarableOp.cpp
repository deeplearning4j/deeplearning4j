//
// Created by raver119 on 07.10.2017.
//

#include <ops/declarable/DeclarableOp.h>
#include <helpers/ProviderRNG.h>
#include <Status.h>
#include <helpers/ShapeUtils.h>

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

        template<typename T>
        DeclarableOp<T>::DeclarableOp() {
            // no-op
        }

        template<typename T>
        DeclarableOp<T>::DeclarableOp(const char *name, bool isLogical) {
            _descriptor = new OpDescriptor(name, isLogical);
        }

        template <typename T>
        DeclarableOp<T>::DeclarableOp(const char *name, int numInputs, bool scalar) {
            _descriptor = new OpDescriptor(numInputs, name, scalar);
        }

        template<typename T>
        DeclarableOp<T>::DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace) {
            _descriptor = new OpDescriptor(numInputs, numOutputs, opName, allowsInplace);
        }

        template<typename T>
        DeclarableOp<T>::DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent) {
            _descriptor = new OpDescriptor(numInputs, numOutputs, opName, allowsInplace, divergent);
        }

        template<typename T>
        DeclarableOp<T>::DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs) {
            _descriptor = new OpDescriptor(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs);
        }

        template<typename T>
        DeclarableOp<T>::~DeclarableOp() {
            if (_descriptor != nullptr)
                delete _descriptor;
        }

        template<typename T>
        OpDescriptor* DeclarableOp<T>::getOpDescriptor() {
            return _descriptor;
        }

        template<typename T>
        std::string *DeclarableOp<T>::getOpName() {
            return _descriptor->getOpName();
        }

        template<typename T>
        Nd4jLong DeclarableOp<T>::getOpHash() {
            return _descriptor->getHash();
        }


        template <typename T>
        nd4j::NDArray<T>* nd4j::ops::DeclarableOp<T>::getZ(Context<T>& ctx, int inputId) {
            NDArray<T>* z = nullptr;

            std::pair<int, int> pair(ctx.nodeId(), inputId);

            if (ctx.isInplace()) {
                z = ctx.variable(inputId)->getNDArray();

                // hypothetically it's possible to have no variable. chances are low, but who knows. let's just create it for now
                if (!ctx.getVariableSpace()->hasVariable(pair)) {
                    auto var = new Variable<T>();
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
                nd4j_printf("BOOM!\n","");
                throw std::runtime_error("Boom!");
            }

            return z;
        }



        template <typename T>
        bool nd4j::ops::DeclarableOp<T>::prepareOutputs(Context<T> &ctx) {
            auto workspace = ctx.getWorkspace();
            GraphProfile *prof = nullptr;
            NodeProfile *node = nullptr;
            std::chrono::time_point<std::chrono::system_clock> inputEnd, inputStart, shapeStart, shapeEnd, arrayStart, arrayEnd;

            if (Environment::getInstance()->isProfiling()) {
                if (ctx.getVariableSpace() != nullptr && ctx.getVariableSpace()->flowPath() != nullptr) {
                    prof = ctx.getVariableSpace()->flowPath()->profile();
                    node = prof->nodeById(ctx.nodeId());
                }
            }

            if (ctx.isInplace()) {
                // do nothing, getZ result will do the trick
            } else {
                // if op is not inplace - we should pre-allocate arrays

                ShapeList inSha;

                if (Environment::getInstance()->isProfiling() && node != nullptr)
                    inputStart = std::chrono::system_clock::now();

                int cntIn = 0;
                // we build list of input shapes
                for (auto p: *ctx.inputs()) {
                    auto var = ctx.variable(p);
                    if (var->variableType() == VariableType::NDARRAY) {
                        NDArray<T> *array = var->getNDArray();
                        inSha.push_back(array->getShapeInfo());

                    }
                    cntIn++;
                }

                // optionally saving input time
                if (Environment::getInstance()->isProfiling() && node != nullptr) {
                    inputEnd = std::chrono::system_clock::now();
                    auto inputTime = std::chrono::duration_cast<std::chrono::nanoseconds>(inputEnd - inputStart).count();
                    node->setInputTime(inputTime);

                    shapeStart = std::chrono::system_clock::now();
                }

                auto outSha = this->calculateOutputShape(&inSha, ctx);

                // optionally saving shapeTime
                if (Environment::getInstance()->isProfiling() && node != nullptr) {
                    shapeEnd = std::chrono::system_clock::now();
                    auto prepTime = std::chrono::duration_cast<std::chrono::nanoseconds>(shapeEnd - shapeStart).count();
                    node->setShapeFunctionTime(prepTime);

                    arrayStart = std::chrono::system_clock::now();
                }

                int cnt = 0;
                for (auto out: *outSha->asVector()) {
                    // we need to check, if Z is really needed
                    std::pair<int, int> pair(ctx.nodeId(), cnt++);

                    if (!ctx.isValueAvailable(pair.second)) {
                        auto outArr = new NDArray<T>(out, true, workspace);

                        ctx.pushNDArrayToVariableSpace(pair, outArr);
                    } else {
                        // validate/compare shapes here. existent vs provided in outSha
                        auto var = ctx.variable(pair);
                        auto shape = var->getNDArray()->shapeInfo();

                        if (!shape::equalsSoft(out, shape)) {
                            auto eShape = ShapeUtils<T>::shapeAsString(out);
                            auto aShape = ShapeUtils<T>::shapeAsString(shape);

                            outSha->destroy();
                            delete outSha;

                            nd4j_printf("Expected vs provided shapes mismatch: %s vs %s\n", eShape.c_str(), aShape.c_str());
                            throw std::runtime_error("Expected vs provided shapes mismatch");
                        }
                    }
                }

                outSha->destroy();
                delete outSha;

                // saving arrayTime
                if (Environment::getInstance()->isProfiling() && node != nullptr) {
                    arrayEnd = std::chrono::system_clock::now();
                    auto arrayTime = std::chrono::duration_cast<std::chrono::nanoseconds>(arrayEnd - arrayStart).count();
                    node->setArrayTime(arrayTime);
                }
            }

            return true;
        }

        template <typename T>
        void nd4j::ops::DeclarableOp<T>::storeResult(Context<T> &block, int outputNumber, NDArray<T>* array) {
            this->storeResult(block, outputNumber, *array);
        }

        template <typename T>
        void nd4j::ops::DeclarableOp<T>::storeResult(nd4j::graph::Context<T> &ctx, int outputNumber, NDArray<T>& array) {

            if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                T mean = array.meanNumber();
                //if (mean == (T) 0.0f || (mean < (T) 1e-5f && mean > (T) -1e-5f))
                //    nd4j_debug("node_%i:%i result has 0.0 as mean\n", block.getNodeId(), outputNumber);
                nd4j_debug("node_%i:%i result length: [%i]; mean [%f]\n", ctx.nodeId(), outputNumber, (int) array.lengthOf(), (float) mean);
            }

            ctx.pushNDArrayToVariableSpace(ctx.nodeId(), outputNumber, &array, !ctx.isInplace());
        }


        template <typename T>
        bool nd4j::ops::DeclarableOp<T>::allocateResult(Context<T>& block, Nd4jLong* shape) {
            auto var = block.variable(block.getNodeId(), 0);

            auto workspace = block.getWorkspace();

            Nd4jLong len = shape::length(shape);
            Nd4jLong* __shape;
            ALLOCATE(__shape, workspace, shape::shapeInfoLength(shape[0]), Nd4jLong); //new int[shape[0] * 2 + 4];

            memcpy(__shape, shape, shape::shapeInfoByteLength(shape[0]));

            // if that's first run - we probably have nothing here
            if (var->getNDArray() == nullptr) {
                T* buffer;
                ALLOCATE(buffer, workspace, len, T);

                var->setNDArray(new NDArray<T>(buffer, __shape, workspace));
                var->getNDArray()->triggerAllocationFlag(true, true);
            } else if(var->getNDArray()->lengthOf() != len) {
                // if length not match - lets reallocate array
                delete var->getNDArray();
                T* buffer;
                ALLOCATE(buffer, workspace, len, T);

                var->setNDArray(new NDArray<T>(buffer, __shape, workspace));
                var->getNDArray()->triggerAllocationFlag(true, true);
            }

            return true;
        }



        template <typename T>
        bool nd4j::ops::DeclarableOp<T>::allocateResult(Context<T>& block, std::initializer_list<Nd4jLong>& shape, char order) {
            auto var = block.variable(block.getNodeId(), 0);
            auto workspace = block.getWorkspace();

            Nd4jLong len = shape::length(shape);
            // if that's first run - we probably have nothing here
            if (var->getNDArray() == nullptr) {
                var->setNDArray(new NDArray<T>(order, shape, workspace));
                var->getNDArray()->triggerAllocationFlag(true, true);
            } else if(var->getNDArray()->lengthOf() != len) {
                // if length not match - lets reallocate array
                delete var->getNDArray();
                var->setNDArray(new NDArray<T>(order, shape, workspace));
                var->getNDArray()->triggerAllocationFlag(true, true);
            }

            return true;
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::execute(Context<T>* block) {
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

            // this method will allocate output NDArrays for this op
            this->prepareOutputs(*block);

            if (Environment::getInstance()->isProfiling()) {
                timeStart = std::chrono::system_clock::now();
                prepTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeStart - timeEnter).count();
            }

            Nd4jStatus status = this->validateAndExecute(*block);

            // optionally saving execution time
            if (Environment::getInstance()->isProfiling()) {
                timeEnd = std::chrono::system_clock::now();
                outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
                block->setInnerTime(outerTime);
            }

            if (Environment::getInstance()->isProfiling()) {
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

            return status;
        }

        template <typename T>
        void DeclarableOp<T>::overwriteResult(Context<T> &block, int outputIdx, NDArray<T> *array) {
            block.pushNDArrayToVariableSpace(block.nodeId(), outputIdx, array);
            /*
            auto varSpace = block.getVariableSpace();
            if (varSpace->hasVariable(block.getNodeId(), outputIdx)) {
                auto var = varSpace->getVariable(block.getNodeId(), outputIdx);
                if (var->getNDArray() != nullptr && var->isRemovable())
                    delete var->getNDArray();

                var->setNDArray(array);
                var->markRemovable(true);
            } else {
                auto var = new Variable<T>(array, nullptr, block.getNodeId(), outputIdx);
                varSpace->putVariable(block.getNodeId(), outputIdx, var);
            }
            */
        }

        template <typename T>
        void DeclarableOp<T>::overwriteResult(Context<T> &block, int outputIdx, NDArrayList<T> *list) {
            block.pushNDArrayListToVariableSpace(block.nodeId(), outputIdx, list);
            /*
            auto varSpace = block.getVariableSpace();
            if (varSpace->hasVariable(block.getNodeId(), outputIdx)) {
                auto var = varSpace->getVariable(block.getNodeId(), outputIdx);
                var->setNDArrayList(list);
            } else {
                auto var = new Variable<T>(nullptr, nullptr, block.getNodeId(), outputIdx);
                var->setNDArrayList(list);
                varSpace->putVariable(block.getNodeId(), outputIdx, var);
            }
            */
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateArguments(Context<T>& block) {
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

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInputDimensions(Context<T>& block, int rank) {
            if (block.width() == 0)
                return ND4J_STATUS_OK;

            for (auto p: *block.inputs()) {
                auto v = block.variable(p);
                NDArray<T> *aV = v->getNDArray();

                if (aV == nullptr)
                    return ND4J_STATUS_BAD_INPUT;

                if (aV->rankOf() != rank)
                    return ND4J_STATUS_BAD_DIMENSIONS;
            }

            return ND4J_STATUS_OK;
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInput2D(Context<T>& block) {
            return validateInputDimensions(block, 2);
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInput3D(Context<T>& block) {
            return validateInputDimensions(block, 3);
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInput4D(Context<T>& block) {
            return validateInputDimensions(block, 4);
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateNonEmptyInput(Context<T>& block) {
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
                    NDArray<T> *aV = v->getNDArray();

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

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateOrdersMatch(Context<T>& block) {
            if (block.width() == 0)
                return ND4J_STATUS_OK;

            NDArray<T> *a0 = block.variable(0)->getNDArray();
            for (auto p: *block.inputs()) {
                auto v = block.variable(p);
                NDArray<T> *aV = v->getNDArray();
                if (a0->ordering() != aV->ordering())
                    return ND4J_STATUS_BAD_ORDER;
            }

            return ND4J_STATUS_OK;
        }

        template<typename T>
        nd4j::ResultSet<T>*  nd4j::ops::DeclarableOp<T>::execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<T> tArgs, std::initializer_list<Nd4jLong> iArgs, bool isInplace) {
            std::vector<NDArray<T>*> ins(inputs);
            std::vector<T> tas(tArgs);
            std::vector<Nd4jLong> ias(iArgs);
            return this->execute(ins, tas, ias, isInplace);
        }

        template<typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs , std::initializer_list<T> tArgs, std::initializer_list<Nd4jLong> iArgs, bool isInplace) {
            std::vector<NDArray<T>*> ins(inputs);
            std::vector<NDArray<T>*> ous(outputs);
            std::vector<T> tas(tArgs);
            std::vector<Nd4jLong> ias(iArgs);
            return this->execute(ins, ous, tas, ias, isInplace);
        }

        template<typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::execute(nd4j::random::RandomBuffer *rng, std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs , std::initializer_list<T> tArgs, std::initializer_list<Nd4jLong> iArgs, bool isInplace) {
            std::vector<NDArray<T>*> ins(inputs);
            std::vector<NDArray<T>*> ous(outputs);
            std::vector<T> tas(tArgs);
            std::vector<Nd4jLong> ias(iArgs);
            return this->execute(rng, ins, ous, tas, ias, isInplace);
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::execute(std::vector<NDArray<T>*>& inputs, std::vector<NDArray<T>*>& outputs, std::vector<T>& tArgs, std::vector<Nd4jLong>& iArgs, bool isInplace) {
            // TODO: nullptr here might be replaced
            return execute(ProviderRNG::getInstance().getRNG(), inputs, outputs, tArgs, iArgs, isInplace);
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::execute(nd4j::random::RandomBuffer *rng, std::vector<NDArray<T>*>& inputs, std::vector<NDArray<T>*>& outputs, std::vector<T>& tArgs, std::vector<Nd4jLong>& iArgs, bool isInplace) {
            VariableSpace<T> variableSpace;
            FlowPath fp;
            variableSpace.setFlowPath(&fp);

            int cnt = -1;
            std::vector<int> in;
            for (auto v: inputs) {
                if (v == nullptr)
                    continue;

                auto var = new Variable<T>(v);
                var->markRemovable(false);
                in.push_back(cnt);
                variableSpace.putVariable(cnt--, var);
            }

            int et = 0;
            for (auto v: outputs) {
                auto var = new Variable<T>(v);
                var->markRemovable(false);
                std::pair<int,int> pair(1, et++);
                variableSpace.putVariable(pair, var);
            }

            Context<T> block(1, &variableSpace, false);
            block.fillInputs(in);
            block.markInplace(isInplace);

            // we need this line for tests basically
            if (rng != nullptr)
                block.setRNG(rng);

            for (int e = 0; e < tArgs.size(); e++)
                block.getTArguments()->emplace_back(tArgs.at(e));

            // FIXME: iargs should be Nd4jLong
            for (int e = 0; e < iArgs.size(); e++)
                block.getIArguments()->emplace_back(static_cast<int>(iArgs.at(e)));

            Nd4jStatus result = this->execute(&block);

            return result;
        }

        template <typename T>
        nd4j::ResultSet<T>* nd4j::ops::DeclarableOp<T>::execute(std::vector<NDArray<T>*>& inputs, std::vector<T>& tArgs, std::vector<Nd4jLong>& iArgs, bool isInplace) {
            VariableSpace<T> variableSpace;
            auto arrayList = new ResultSet<T>();
            //ResultSet<T> arrayList;
            FlowPath fp;
            variableSpace.setFlowPath(&fp);

            if (isInplace)
                arrayList->setNonRemovable();

            int cnt = -1;
            std::vector<int> in;
            for (auto v: inputs) {
                if (v == nullptr)
                    continue;

                auto var = new Variable<T>(v);
                var->markRemovable(false);
                in.push_back(cnt);
                variableSpace.putVariable(cnt--, var);
            }
            
            Context<T> block(1, &variableSpace, false);
            block.fillInputs(in);
            block.markInplace(isInplace);
            block.setRNG(ProviderRNG::getInstance().getRNG());

            for (int e = 0; e < tArgs.size(); e++)
                block.getTArguments()->emplace_back(tArgs.at(e));


            for (int e = 0; e < iArgs.size(); e++)
                block.getIArguments()->emplace_back(iArgs.at(e));

            Nd4jStatus status = this->execute(&block);
            arrayList->setStatus(status);
            if (status != ND4J_STATUS_OK)
                return arrayList;


            for (int e = 0; e < 65536; e++) {
                std::pair<int,int> pair(1, e);
                if (variableSpace.hasVariable(pair)) {
                    auto var = variableSpace.getVariable(pair);
                    auto arr = var->getNDArray();
                    if (!arr->isAttached()) {
                        var->markRemovable(false);
                        arrayList->push_back(arr);
                    } else {
                        arrayList->push_back(arr->detach());
                    }
                } else
                    break;
            }

            return arrayList;
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInputDimensionsMatch(Context<T>& block) {
            if (block.width() == 0)
                return ND4J_STATUS_OK;


            NDArray<T> *a0 = block.variable(0)->getNDArray();
            for (auto p: *block.inputs()) {
                auto v = block.variable(p);
                NDArray<T> *aV = v->getNDArray();
                if (!shape::equalsSoft(a0->getShapeInfo(), aV->getShapeInfo()))
                    return ND4J_STATUS_BAD_DIMENSIONS;
            }

            return ND4J_STATUS_OK;
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInputLengthMatch(Context<T>& block) {
            if (block.width() == 0)
                return ND4J_STATUS_OK;


            Nd4jLong l0 = block.variable(0)->getNDArray()->lengthOf();
            for (uint32_t e = 0; e < block.width(); e++) {
                if (l0 != block.variable(e)->getNDArray()->lengthOf())
                    return ND4J_STATUS_BAD_LENGTH;
            }

            return ND4J_STATUS_OK;
        }


        /*
        template <typename T>
        int* nd4j::ops::DeclarableOp<T>::calculateOutputShape(int* inputShape, nd4j::graph::Block<T>& block) {
            // default implementation suits transform, so just returns the same shape

            int* newshape;
            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(inputShape), int);
            memcpy(newshape, inputShape, shape::shapeInfoByteLength(inputShape));

            return newshape;
        }
        */


        template class ND4J_EXPORT DeclarableOp<float>;
        template class ND4J_EXPORT DeclarableOp<float16>;
        template class ND4J_EXPORT DeclarableOp<double>;
    }
}