//
// Created by raver119 on 07.10.2017.
//

#include <ops/declarable/DeclarableOp.h>

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
        Nd4jIndex DeclarableOp<T>::getOpHash() {
            return _descriptor->getHash();
        }


        template <typename T>
        nd4j::NDArray<T>* nd4j::ops::DeclarableOp<T>::getZ(Block<T>& block, int inputId) {
            NDArray<T>* z = nullptr;

            if (block.isInplace()) {
                z = block.variable(inputId)->getNDArray();
            } else if (!block.isInplace() && block.getVariableSpace()->hasVariable(block.getNodeId())) {
                std::pair<int, int> pair(block.getNodeId(), inputId);

                auto var = block.getVariableSpace()->getVariable(pair);
                if (var->getNDArray() != nullptr && var->getNDArray()->nonNull()) {
                    z = var->getNDArray();
                } else {
/*
            auto shapeList = new ShapeList();
            for (auto v: block.getVariables()) {
                shapeList->push_back(v->getNDArray()->getShapeInfo());
            }

            auto shapes = this->calculateOutputShape(shapeList, block);
            int *shape = shapes->at(inputId);
            z = new NDArray<T>();
*/
                    nd4j_printf("Can't get Z variable!\n","");
                }
            }

            return z;
        }



        template <typename T>
        bool nd4j::ops::DeclarableOp<T>::prepareOutputs(Block<T> &block) {
            auto workspace = block.getWorkspace();

            if (block.isInplace()) {
                // do nothing, getZ result will do the trick
            } else {
                // if op is not inplace - we should pre-allocate arrays

                ShapeList inSha;

                int cntIn = 0;
                for (auto p: *block.inputs()) {
                    auto var = block.getVariableSpace()->getVariable(p);
                    NDArray<T> *array = var->getNDArray();
                    inSha.push_back(array->getShapeInfo());

                    //array->printShapeInfo("prepOutput");
                    cntIn++;
                }
                //nd4j_printf("Input shapes: %i\n", cntIn);

                auto outSha = this->calculateOutputShape(&inSha, block);
                int cnt = 0;
                //nd4j_printf("Output shapes: %i; Rank_0: %i\n", outSha->size(), outSha->at(0)[0]);
                for (auto out: *outSha->asVector()) {
                    // we need to check, if Z is really needed
                    std::pair<int, int> pair(block.getNodeId(), cnt++);
                    if (block.getVariableSpace()->hasVariable(pair)) {
                        auto var = block.getVariableSpace()->getVariable(pair);
                        if (var->getNDArray() != nullptr && var->getNDArray()->nonNull())
                            continue;
                    }

                    auto outArr = new NDArray<T>(out, workspace);

                    auto var = block.getVariableSpace()->getVariable(pair);
                    if (var == nullptr) {
                        var = new Variable<T>(outArr);
                        block.getVariableSpace()->putVariable(pair, var);
                    } else {
                        //block.getVariableSpace()->putVariable(pair, outArr);
                        var->setNDArray(outArr);
                    }
                }

                outSha->destroy();
                delete outSha;
            }

            return true;
        }


        template <typename T>
        void nd4j::ops::DeclarableOp<T>::storeResult(Block<T> &block, int outputNumber, NDArray<T>& array) {

            if (nd4j::Environment::getInstance()->isDebug()) {
                T mean = array.meanNumber();
                //if (mean == (T) 0.0f || (mean < (T) 1e-5f && mean > (T) -1e-5f))
                //    nd4j_debug("node_%i:%i result has 0.0 as mean\n", block.getNodeId(), outputNumber);
                nd4j_debug("node_%i:%i result mean [%f]\n", block.getNodeId(), outputNumber, (float) mean);
            }

            // if that's the only output - treat it as singular variable
            if (outputNumber == 0 && this->getOpDescriptor()->getNumberOfOutputs() == 1) {
                // we're adding this check, to avoid saving in legacy execution mechanism
                if (!block.getVariableSpace()->hasVariable(block.getNodeId())) {
                    nd4j_debug("Skipping storeResult for node_%i:%i\n", block.getNodeId(), outputNumber);
                    return;
                }

                auto variable = block.getVariableSpace()->getVariable(block.getNodeId());
                variable->setNDArray(&array);
            } else {
                // otherwise - reference it as pair key
                std::pair<int, int> pair((int) block.getNodeId(), outputNumber);
                if (block.getVariableSpace()->hasVariable(pair)) {
                    auto variable = block.getVariableSpace()->getVariable(pair);
                    variable->setNDArray(&array);
                } else {
                    block.getVariableSpace()->putVariable(pair, &array);
                }
            }
        }


        template <typename T>
        bool nd4j::ops::DeclarableOp<T>::allocateResult(Block<T>& block, int* shape) {
            auto var = block.getVariableSpace()->getVariable(block.getNodeId());

            auto workspace = block.getWorkspace();

            Nd4jIndex len = shape::length(shape);
            int* __shape;
            ALLOCATE(__shape, workspace, shape::shapeInfoLength(shape[0]), int); //new int[shape[0] * 2 + 4];

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

                var ->setNDArray(new NDArray<T>(buffer, __shape, workspace));
                var->getNDArray()->triggerAllocationFlag(true, true);
            }

            return true;
        }



        template <typename T>
        bool nd4j::ops::DeclarableOp<T>::allocateResult(Block<T>& block, std::initializer_list<int>& shape, char order) {
            auto var = block.getVariableSpace()->getVariable(block.getNodeId());
            auto workspace = block.getWorkspace();

            Nd4jIndex len = shape::length(shape);
            // if that's first run - we probably have nothing here
            if (var->getNDArray() == nullptr) {
                var ->setNDArray(new NDArray<T>(order, shape, workspace));
                var->getNDArray()->triggerAllocationFlag(true, true);
            } else if(var->getNDArray()->lengthOf() != len) {
                // if length not match - lets reallocate array
                delete var->getNDArray();
                var ->setNDArray(new NDArray<T>(order, shape, workspace));
                var->getNDArray()->triggerAllocationFlag(true, true);
            }

            return true;
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::execute(Block<T>* block) {
            if (block != nullptr)
                _block = block;
            else
                throw std::invalid_argument("Block is NULL");

            nd4j_debug("Executing op: [%s]\n", this->getOpName()->c_str());

            // basic validation: ensure inputs are set
            REQUIRE_OK(this->validateNonEmptyInput(*block));

            // ensure number of IArgs, TArgs match our expectations
            REQUIRE_OK(this->validateArguments(*block));

            // this method will allocate output NDArrays for this op
            this->prepareOutputs(*block);

            auto timeStart = std::chrono::system_clock::now();

            Nd4jStatus status = this->validateAndExecute(*block);

            auto timeEnd = std::chrono::system_clock::now();
            auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
            block->setInnerTime(outerTime);

            return status;
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateArguments(Block<T>& block) {
            /*
             * We're checking number of T and I arguments. If number of args is finite number - we check strict equality
             * If number of args is variable (-1), but variables MUST be present - we check for non-zero number of arguments
             */
            if (_descriptor->getNumberOfTArgs() > 0) {
                if ((int) block.getTArguments()->size() < _descriptor->getNumberOfTArgs()) {
                    nd4j_debug("% T args expected, but %i received", _descriptor->getNumberOfTArgs(), block.getTArguments()->size());
                    return ND4J_STATUS_BAD_PARAMS;
                }
            } else
            if (_descriptor->getNumberOfTArgs() == -1)
                if (block.getTArguments()->size() == 0)
                    return ND4J_STATUS_BAD_PARAMS;

            if (_descriptor->getNumberOfIArgs() > 0) {
                if ((int) block.getIArguments()->size() < _descriptor->getNumberOfIArgs()) {
                    nd4j_debug("% int args expected, but %i received", _descriptor->getNumberOfIArgs(), block.getIArguments()->size());
                    return ND4J_STATUS_BAD_PARAMS;
                }
            } else
            if (_descriptor->getNumberOfIArgs() == -1)
                if (block.getIArguments()->size() == 0)
                    return ND4J_STATUS_BAD_PARAMS;


            return ND4J_STATUS_OK;
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInputDimensions(Block<T>& block, int rank) {
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
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInput2D(Block<T>& block) {
            return validateInputDimensions(block, 2);
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInput3D(Block<T>& block) {
            return validateInputDimensions(block, 3);
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInput4D(Block<T>& block) {
            return validateInputDimensions(block, 4);
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateNonEmptyInput(Block<T>& block) {
            if (block.width() < 1)
                return ND4J_STATUS_BAD_INPUT;


            int cnt = 0;
            for (auto p: *block.inputs()) {
                auto v = block.getVariableSpace()->getVariable(p);
                if (v == nullptr) {
                    if (this->getOpName() != nullptr) {
                        nd4j_printf("Node [%i:<%s>]: Variable [%i] (%i:%i) is NULL\n", block.getNodeId(), this->getOpName()->c_str(), cnt, 0, 0);
                    } else {
                        nd4j_printf("Node [%i:<%s>]: Variable [%i] (%i:%i) is NULL\n", block.getNodeId(), cnt, 0, 0);
                    }
                    throw "Bad input";
                }

                NDArray<T> *aV = v->getNDArray();

                if (aV == nullptr || !aV->nonNull())
                    return ND4J_STATUS_BAD_INPUT;

                cnt++;
            }

            return ND4J_STATUS_OK;
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateOrdersMatch(Block<T>& block) {
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
        nd4j::ArrayList<T>*  nd4j::ops::DeclarableOp<T>::execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace) {
            std::vector<NDArray<T>*> ins(inputs);
            std::vector<T> tas(tArgs);
            std::vector<int> ias(iArgs);
            return this->execute(ins, tas, ias, isInplace);
        }

        template<typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs , std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace) {
            std::vector<NDArray<T>*> ins(inputs);
            std::vector<NDArray<T>*> ous(outputs);
            std::vector<T> tas(tArgs);
            std::vector<int> ias(iArgs);
            return this->execute(ins, ous, tas, ias, isInplace);
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::execute(std::vector<NDArray<T>*>& inputs, std::vector<NDArray<T>*>& outputs, std::vector<T>& tArgs, std::vector<int>& iArgs, bool isInplace) {
            VariableSpace<T> variableSpace;

            int cnt = -1;
            std::vector<int> in;
            for (auto v: inputs) {
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

            Block<T> block(1, &variableSpace, false);
            block.fillInputs(in);
            block.markInplace(isInplace);

            for (int e = 0; e < tArgs.size(); e++)
                block.getTArguments()->emplace_back(tArgs.at(e));


            for (int e = 0; e < iArgs.size(); e++)
                block.getIArguments()->emplace_back(iArgs.at(e));

            Nd4jStatus result = this->execute(&block);

            return result;
        }

        template <typename T>
        nd4j::ArrayList<T>* nd4j::ops::DeclarableOp<T>::execute(std::vector<NDArray<T>*>& inputs, std::vector<T>& tArgs, std::vector<int>& iArgs, bool isInplace) {
            VariableSpace<T> variableSpace;
            auto arrayList = new ArrayList<T>();
            //ArrayList<T> arrayList;

            int cnt = -1;
            std::vector<int> in;
            for (auto v: inputs) {
                auto var = new Variable<T>(v);
                var->markRemovable(false);
                in.push_back(cnt);
                variableSpace.putVariable(cnt--, var);
            }

            Block<T> block(1, &variableSpace, false);
            block.fillInputs(in);
            block.markInplace(isInplace);

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
                    var->markRemovable(false);
                    arrayList->push_back(var->getNDArray());
                } else
                    break;
            }

            return arrayList;
        }

        template <typename T>
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInputDimensionsMatch(Block<T>& block) {
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
        Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInputLengthMatch(Block<T>& block) {
            if (block.width() == 0)
                return ND4J_STATUS_OK;


            Nd4jIndex l0 = block.variable(0)->getNDArray()->lengthOf();
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