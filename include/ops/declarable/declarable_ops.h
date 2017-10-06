//
// @author raver119@gmail.com
//

#ifndef LIBND4J_DECLARABLE_OPS_H
#define LIBND4J_DECLARABLE_OPS_H

#include <sstream>
#include <types/float16.h>
#include <pointercast.h>
#include <NDArray.h>
#include <Variable.h>
#include <Block.h>
#include <Stash.h>
#include "OpDescriptor.h"
#include <helpers/helper_hash.h>
#include <memory/Workspace.h>
#include <memory/MemoryRegistrator.h>
#include <ShapeList.h>
#include <ArrayList.h>

#include <chrono>
#include <ctime>

using namespace nd4j::graph;

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
        Nd4jStatus resultHelper(T status, const char *func, const char *file, int line) {
            if (status) {
                //  TODO: fill out error codes here
                fprintf(stderr, "Validation error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                        static_cast<unsigned int>(status), "", func);

                return ND4J_STATUS_BAD_INPUT;
            }

            return ND4J_STATUS_OK;
        }

        template <typename T>
        class DeclarableOp {
        protected:
            Block<T> *_block;
            OpDescriptor *_descriptor;

            /**
             * This method executes this Op
             */
            virtual Nd4jStatus validateAndExecute(Block<T>& block) = 0;


            /**
             * This method ensures that target variable has enough space for op execution
             *
             * TODO: we want workspaces support right here
             */
            bool allocateResult(Block<T>& block, std::initializer_list<int>& shape, char order = 'c');
            bool allocateResult(Block<T>& block, int* shape);
            void storeResult(Block<T> &block, int outputNumber, NDArray<T>& array);
            nd4j::NDArray<T> *getZ(Block<T>& block, int inputId = 0);

            bool prepareOutputs(Block<T>& block);

            //std::vector<int>* calculateOutputShape(std::vector<int>* inputShape, nd4j::graph::Block<T>& block);
        public:
            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace) {
                _descriptor = new OpDescriptor(numInputs, numOutputs, opName, allowsInplace);
            }

            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, bool divergent) {
                _descriptor = new OpDescriptor(numInputs, numOutputs, opName, allowsInplace, divergent);
            }

            DeclarableOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs) {
                _descriptor = new OpDescriptor(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs);
            }

            ~DeclarableOp() {
                if (_descriptor != nullptr)
                    delete _descriptor;
            }


            OpDescriptor *getOpDescriptor() {
                return _descriptor;
            }

            virtual ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Block<T>& block) = 0;

            /**
             * Returns opName
             *
             * @return
             */
            std::string *getOpName() {
                return _descriptor->getOpName();
            }

            Nd4jIndex getOpHash() {
                return _descriptor->getHash();
            }

            /**
             * This method sets arguments for op
             */
            void setArguments();

            /**
             * This method returns pointer to results
             */
            void getResults();

            /**
             * This method executes everything
             * @param block
             * @return
             */
            Nd4jStatus execute(Block<T>* block);

            nd4j::ArrayList<T>* execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<T> tArgs = {}, std::initializer_list<int> iArgs = {});
            Nd4jStatus execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs = {}, std::initializer_list<T> tArgs = {}, std::initializer_list<int> iArgs = {});

            // There methods provide various validation options
            Nd4jStatus validateNonEmptyInput(Block<T>& block);
            Nd4jStatus validateInputLengthMatch(Block<T>& block);
            Nd4jStatus validateInputDimensionsMatch(Block<T>& block);
            Nd4jStatus validateOrdersMatch(Block<T>& block);
            Nd4jStatus validateInput2D(Block<T>& block);
            Nd4jStatus validateInput3D(Block<T>& block);
            Nd4jStatus validateInput4D(Block<T>& block);
            Nd4jStatus validateInputDimensions(Block<T>& block, int rank);

            Nd4jStatus validateArguments(Block<T>& block);
        };


        template <typename T>
        class DeclarableReductionOp : public nd4j::ops::DeclarableOp<T> {
        protected:
            /**
             * This method executes this Op
             */
            virtual Nd4jStatus validateAndExecute(Block<T>& block) = 0;
        public:
            DeclarableReductionOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs) : nd4j::ops::DeclarableOp<T>(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs) {
                //
            }

            ~DeclarableReductionOp()  {
                //
            }

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Block<T>& block);
        };

        template <typename T>
        class DeclarableCustomOp : public nd4j::ops::DeclarableOp<T> {
        protected:
            /**
             * This method executes this Op
             */
            virtual Nd4jStatus validateAndExecute(Block<T>& block) = 0;
        public:
            DeclarableCustomOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs) : nd4j::ops::DeclarableOp<T>(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs) {
                //
            }

            ~DeclarableCustomOp()  {
                //
            }

            virtual ShapeList* calculateOutputShape(ShapeList* inputShapes, nd4j::graph::Block<T>& block) = 0;
        };

        class OpRegistrator {
        private:
            static OpRegistrator* _INSTANCE;
            OpRegistrator() {};
            ~OpRegistrator() {};

            std::map<Nd4jIndex, std::string> _msvc;

            std::map<Nd4jIndex, nd4j::ops::DeclarableOp<float> *> _declarablesLF;
            std::map<std::string, nd4j::ops::DeclarableOp<float> *> _declarablesF;

            std::map<Nd4jIndex, nd4j::ops::DeclarableOp<double> *> _declarablesLD;
            std::map<std::string, nd4j::ops::DeclarableOp<double> *> _declarablesD;

            std::map<Nd4jIndex, nd4j::ops::DeclarableOp<float16> *> _declarablesLH;
            std::map<std::string, nd4j::ops::DeclarableOp<float16> *> _declarablesH;

            std::mutex _locker;
            std::string _opsList;
            bool isInit = false;
        public:
            static OpRegistrator* getInstance() {
                if (!_INSTANCE)
                    _INSTANCE = new nd4j::ops::OpRegistrator();

                return _INSTANCE;
            }

            void updateMSVC(Nd4jIndex newHash, std::string& oldName) {
                std::pair<Nd4jIndex, std::string> pair(newHash, oldName);
                _msvc.insert(pair);
            }

            template <typename T>
            std::string local_to_string(T value)
            {
                //create an output string stream
                std::ostringstream os ;

                //throw the value into the string stream
                os << value ;

                //convert the string stream into a string and return
                return os.str() ;
            }

            const char * getAllCustomOperations() {
                _locker.lock();

                if (!isInit) {
                    for (std::map<std::string, nd4j::ops::DeclarableOp<float>*>::iterator it=_declarablesF.begin(); it!=_declarablesF.end(); ++it) {
                         std::string op = it->first + ":"
                                    + local_to_string(it->second->getOpDescriptor()->getHash()) + ":"
                                    + local_to_string(it->second->getOpDescriptor()->getNumberOfInputs()) + ":"
                                    + local_to_string(it->second->getOpDescriptor()->getNumberOfOutputs()) + ":"
                                    + local_to_string(it->second->getOpDescriptor()->allowsInplace())  + ":"
                                    + local_to_string(it->second->getOpDescriptor()->getNumberOfTArgs())  + ":"
                                    + local_to_string(it->second->getOpDescriptor()->getNumberOfIArgs())  + ":"
                                    + ";" ;
                        _opsList += op;
                    }

                    isInit = true;
                }

                _locker.unlock();

                return _opsList.c_str();
            }

            /**
             * This method registers operation
             *
             * @param op
             */
            bool registerOperationFloat(nd4j::ops::DeclarableOp<float>* op) {
                return registerOperationFloat(op->getOpName()->c_str(), op);
            }

            bool registerOperationFloat(const char* name, nd4j::ops::DeclarableOp<float>* op) {
                auto str = new std::string(name);
                std::pair<std::string, nd4j::ops::DeclarableOp<float>*> pair(*str, op);
                _declarablesF.insert(pair);

                auto hash = nd4j::ops::HashHelper::getInstance()->getLongHash(*str);
                std::pair<Nd4jIndex, nd4j::ops::DeclarableOp<float>*> pair2(hash, op);
                _declarablesLF.insert(pair2);
                return true;
            }

            bool registerOperationDouble(const char* name, nd4j::ops::DeclarableOp<double>* op) {
                auto str = new std::string(name);
                std::pair<std::string, nd4j::ops::DeclarableOp<double>*> pair(*str, op);
                _declarablesD.insert(pair);

                auto hash = nd4j::ops::HashHelper::getInstance()->getLongHash(*str);
                std::pair<Nd4jIndex, nd4j::ops::DeclarableOp<double>*> pair2(hash, op);
                _declarablesLD.insert(pair2);
                return true;
            }

            bool registerOperationHalf(const char* name, nd4j::ops::DeclarableOp<float16>* op) {
                auto str = new std::string(name);
                std::pair<std::string, nd4j::ops::DeclarableOp<float16>*> pair(*str, op);
                _declarablesH.insert(pair);

                auto hash = nd4j::ops::HashHelper::getInstance()->getLongHash(*str);
                std::pair<Nd4jIndex, nd4j::ops::DeclarableOp<float16>*> pair2(hash, op);
                _declarablesLH.insert(pair2);
                return true;
            }

            bool registerOperationHalf(nd4j::ops::DeclarableOp<float16> *op) {
                return registerOperationHalf(op->getOpName()->c_str(), op);
            }

            bool registerOperationDouble(nd4j::ops::DeclarableOp<double> *op) {
                return registerOperationDouble(op->getOpName()->c_str(), op);
            }

            nd4j::ops::DeclarableOp<float>* getOperationFloat(const char *name) {
                std::string str(name);
                return getOperationFloat(str);
            }

            /**
             * This method returns registered Op by name
             *
             * @param name
             * @return
             */
             nd4j::ops::DeclarableOp<float> *getOperationFloat(std::string& name) {
                if (!_declarablesF.count(name)) {
                    nd4j_printf("Unknown operation requested: [%s]\n", name.c_str());
                    return nullptr;
                }

                return _declarablesF.at(name);
            }


            nd4j::ops::DeclarableOp<float> *getOperationFloat(Nd4jIndex hash) {
                if (!_declarablesLF.count(hash)) {
                    if (!_msvc.count(hash)) {
                        nd4j_printf("Unknown operation requested by hash: [%lld]\n", hash);
                        return nullptr;
                    } else {
                        _locker.lock();

                        auto str = _msvc.at(hash);
                        auto op = _declarablesF.at(str);
                        auto oHash = op->getOpDescriptor()->getHash();

                        std::pair<Nd4jIndex, nd4j::ops::DeclarableOp<float>*> pair(oHash, op);
                        _declarablesLF.insert(pair);

                        _locker.unlock();
                    }
                }

                return _declarablesLF.at(hash);
            }


            nd4j::ops::DeclarableOp<float16> *getOperationHalf(Nd4jIndex hash) {
                if (!_declarablesLH.count(hash)) {
                    if (!_msvc.count(hash)) {
                        nd4j_printf("Unknown operation requested by hash: [%lld]\n", hash);
                        return nullptr;
                    } else {
                        _locker.lock();

                        auto str = _msvc.at(hash);
                        auto op = _declarablesH.at(str);
                        auto oHash = op->getOpDescriptor()->getHash();

                        std::pair<Nd4jIndex, nd4j::ops::DeclarableOp<float16>*> pair(oHash, op);
                        _declarablesLH.insert(pair);

                        _locker.unlock();
                    }
                }

                return _declarablesLH.at(hash);
            }


            nd4j::ops::DeclarableOp<float16>* getOperationHalf(const char *name) {
                std::string str(name);
                return getOperationHalf(str);
            }


            nd4j::ops::DeclarableOp<float16> *getOperationHalf(std::string& name) {
                if (!_declarablesH.count(name)) {
                    nd4j_printf("Unknown operation requested: [%s]\n", name.c_str());
                    return nullptr;
                }

                return _declarablesH.at(name);
            }


            nd4j::ops::DeclarableOp<double >* getOperationDouble(const char *name) {
                std::string str(name);
                return getOperationDouble(str);
            }


            nd4j::ops::DeclarableOp<double> *getOperationDouble(Nd4jIndex hash) {
                if (!_declarablesLD.count(hash)) {
                    if (!_msvc.count(hash)) {
                        nd4j_printf("Unknown operation requested by hash: [%lld]\n", hash);
                        return nullptr;
                    } else {
                        _locker.lock();

                        auto str = _msvc.at(hash);
                        auto op = _declarablesD.at(str);
                        auto oHash = op->getOpDescriptor()->getHash();

                        std::pair<Nd4jIndex, nd4j::ops::DeclarableOp<double>*> pair(oHash, op);
                        _declarablesLD.insert(pair);

                        _locker.unlock();
                    }
                }

                return _declarablesLD.at(hash);
            }

            nd4j::ops::DeclarableOp<double> *getOperationDouble(std::string& name) {
                if (!_declarablesD.count(name)) {
                    nd4j_printf("Unknown operation requested: [%s]\n", name.c_str());
                    return nullptr;
                }

                return _declarablesD.at(name);
            }
        };

        template <typename OpName>
        struct __registratorSynonymFloat {
            __registratorSynonymFloat(const char *name, const char *oname) {
                OpName *ptr = (OpName *) OpRegistrator::getInstance()->getOperationFloat(oname);
                if (ptr == nullptr) {
                    std::string newName(name);
                    std::string oldName(oname);

                    OpRegistrator::getInstance()->updateMSVC(nd4j::ops::HashHelper::getInstance()->getLongHash(newName), oldName);
                    return;
                }
                OpRegistrator::getInstance()->registerOperationFloat(name, ptr);
            }
        };

        template <typename OpName>
        struct __registratorSynonymHalf {
            __registratorSynonymHalf(const char *name, const char *oname) {
                OpName *ptr = (OpName *) OpRegistrator::getInstance()->getOperationHalf(oname);
                if (ptr == nullptr) {
                    std::string newName(name);
                    std::string oldName(oname);

                    OpRegistrator::getInstance()->updateMSVC(nd4j::ops::HashHelper::getInstance()->getLongHash(newName), oldName);
                    return;
                }
                OpRegistrator::getInstance()->registerOperationHalf(name, ptr);
            }
        };

        template <typename OpName>
        struct __registratorSynonymDouble {
            __registratorSynonymDouble(const char *name, const char *oname) {
                OpName *ptr = (OpName *) OpRegistrator::getInstance()->getOperationDouble(oname);
                if (ptr == nullptr) {
                    std::string newName(name);
                    std::string oldName(oname);

                    OpRegistrator::getInstance()->updateMSVC(nd4j::ops::HashHelper::getInstance()->getLongHash(newName), oldName);
                    return;
                }
                OpRegistrator::getInstance()->registerOperationDouble(name, ptr);
            }
        };

        template <typename OpName>
        struct __registratorFloat {
            __registratorFloat() {
                OpName *ptr = new OpName();
                OpRegistrator::getInstance()->registerOperationFloat(ptr);
            }
        };

        template <typename OpName>
        struct __registratorHalf {
            __registratorHalf() {
                OpName *ptr = new OpName();
                OpRegistrator::getInstance()->registerOperationHalf(ptr);
            }
        };

        template <typename OpName>
        struct __registratorDouble {
            __registratorDouble() {
                OpName *ptr = new OpName();
                OpRegistrator::getInstance()->registerOperationDouble(ptr);
            }
        };
    }
}

nd4j::ops::OpRegistrator* nd4j::ops::OpRegistrator::_INSTANCE = 0;

template <typename T>
nd4j::ShapeList* nd4j::ops::DeclarableReductionOp<T>::calculateOutputShape(nd4j::ShapeList* inputShape, nd4j::graph::Block<T>& block)  {
    int numDims = block.getIArguments()->at(0);
    std::vector<int> dims;
    for (int e = 0; e < numDims; e++)
        dims.push_back(block.getIArguments()->at(e+1));

    if (numDims > 1)
        std::sort(dims.begin(), dims.end());

    // special case - output is scalar
    if (numDims == 1 && dims.at(0) == MAX_INT) {
        int* newShape;
        ALLOCATE(newShape, block.getWorkspace(), 8, int);

        newShape[0] = 2;
        newShape[1] = 1;
        newShape[2] = 1;
        newShape[3] = 1;
        newShape[4] = 1;
        newShape[5] = 0;
        newShape[6] = 1;
        newShape[7] = 99;

        return new ShapeList(newShape);
    }

    shape::TAD tad(inputShape->at(0), dims.data(), numDims);
    tad.createTadOnlyShapeInfo();

    Nd4jIndex tadLength = shape::tadLength(inputShape->at(0), dims.data(), numDims);
    Nd4jIndex numTads = shape::length(inputShape->at(0)) /  tadLength;

    int* newShape;
    ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);

    newShape[0] = 2;
    newShape[1] = 1;
    newShape[2] = numTads;
    newShape[3] = numTads;
    newShape[4] = 1;
    newShape[5] = 0;
    newShape[6] = 1;
    newShape[7] = 99;

    return new ShapeList(newShape);
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

template <typename T>
nd4j::NDArray<T>* nd4j::ops::DeclarableOp<T>::getZ(Block<T>& block, int inputId) {
    NDArray<T>* z = nullptr;

    if (block.isInplace()) {
        z = block.getVariables().at(inputId)->getNDArray();
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
        for (auto var: block.getVariables()) {
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

    if (debug) {
        T mean = array.meanNumber();
        if (mean == (T) 0.0f || (mean < (T) 1e-5f && mean > (T) -1e-5f))
            nd4j_debug("node_%i:%i result has 0.0 as mean\n", block.getNodeId(), outputNumber);
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
        auto variable = block.getVariableSpace()->getVariable(pair);
        variable->setNDArray(&array);
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
        if ((int) block.getTArguments()->size() != _descriptor->getNumberOfTArgs()) {
            nd4j_debug("% T args expected, but %i received", _descriptor->getNumberOfTArgs(), block.getTArguments()->size());
            return ND4J_STATUS_BAD_PARAMS;
        }
    } else
        if (_descriptor->getNumberOfTArgs() == -1)
            if (block.getTArguments()->size() == 0)
                return ND4J_STATUS_BAD_PARAMS;

    if (_descriptor->getNumberOfIArgs() > 0) {
        if ((int) block.getIArguments()->size() != _descriptor->getNumberOfIArgs()) {
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
    if (block.getVariables().size() == 0)
        return ND4J_STATUS_OK;

    for (auto v: block.getVariables()) {
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
    if (block.getVariables().size() < 1)
        return ND4J_STATUS_BAD_INPUT;


    int cnt = 0;
    for (auto v: block.getVariables()) {
        NDArray<T> *aV = v->getNDArray();

        if (aV == nullptr || !aV->nonNull())
            return ND4J_STATUS_BAD_INPUT;

        cnt++;
    }

    return ND4J_STATUS_OK;
}

template <typename T>
Nd4jStatus nd4j::ops::DeclarableOp<T>::validateOrdersMatch(Block<T>& block) {
    if (block.getVariables().size() == 0)
        return ND4J_STATUS_OK;

    NDArray<T> *a0 = block.getVariables().at(0)->getNDArray();
    for (auto v: block.getVariables()) {
        NDArray<T> *aV = v->getNDArray();
        if (a0->ordering() != aV->ordering())
            return ND4J_STATUS_BAD_ORDER;
    }

    return ND4J_STATUS_OK;
}

template <typename T>
Nd4jStatus nd4j::ops::DeclarableOp<T>::execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<NDArray<T>*> outputs, std::initializer_list<T> tArgs, std::initializer_list<int> iArgs) {
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

    std::vector<T> tt(tArgs);
    for (int e = 0; e < tt.size(); e++)
        block.getTArguments()->push_back(tt.at(e));


    std::vector<int> ii(iArgs);
    for (int e = 0; e < ii.size(); e++)
        block.getIArguments()->push_back(ii.at(e));

    Nd4jStatus result = this->execute(&block);

    return result;
}

template <typename T>
nd4j::ArrayList<T>* nd4j::ops::DeclarableOp<T>::execute(std::initializer_list<NDArray<T>*> inputs, std::initializer_list<T> tArgs, std::initializer_list<int> iArgs) {
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

    std::vector<T> tt(tArgs);
    for (int e = 0; e < tt.size(); e++)
        block.getTArguments()->push_back(tt.at(e));


    std::vector<int> ii(iArgs);
    for (int e = 0; e < ii.size(); e++)
        block.getIArguments()->push_back(ii.at(e));

    this->execute(&block);

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
    if (block.getVariables().size() == 0)
        return ND4J_STATUS_OK;


    NDArray<T> *a0 = block.getVariables().at(0)->getNDArray();
    for (auto v: block.getVariables()) {
        NDArray<T> *aV = v->getNDArray();
        if (!shape::equalsSoft(a0->getShapeInfo(), aV->getShapeInfo()))
            return ND4J_STATUS_BAD_DIMENSIONS;
    }

    return ND4J_STATUS_OK;
}

template <typename T>
Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInputLengthMatch(Block<T>& block) {
    if (block.getVariables().size() == 0)
        return ND4J_STATUS_OK;


    Nd4jIndex l0 = block.getVariables().at(0)->getNDArray()->lengthOf();
    for (uint32_t e = 0; e < block.getVariables().size(); e++) {
        if (l0 != block.getVariables().at(e)->getNDArray()->lengthOf())
            return ND4J_STATUS_BAD_LENGTH;
    }

    return ND4J_STATUS_OK;
}

#endif //LIBND4J_DECLARABLE_OPS_H
