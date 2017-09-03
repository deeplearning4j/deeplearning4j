//
// @author raver119@gmail.com
//

#ifndef LIBND4J_DECLARABLE_OPS_H
#define LIBND4J_DECLARABLE_OPS_H

#include <pointercast.h>
#include <NDArray.h>
#include <Variable.h>
#include <Block.h>
#include "OpDescriptor.h"
#include <helpers/helper_hash.h>


using namespace nd4j::graph;

namespace nd4j {
    namespace ops {

        template<typename T>
        void resultHelper(T status, const char *func, const char *file, int line) {
            if (status) {
                //  TODO: fill out error codes here
                fprintf(stderr, "Validation error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                        static_cast<unsigned int>(status), "", func);

                throw "Validation failed";
            }
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

            /**
             * Returns opName
             *
             * @return
             */
            std::string *getOpName() {
                return _descriptor->getOpName();
            }

            Nd4jIndex getOpHash() {
                return 0;
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

            // There methods provide various validation options
            Nd4jStatus validateNonEmptyInput(Block<T>& block);
            Nd4jStatus validateInputLengthMatch(Block<T>& block);
            Nd4jStatus validateInputDimensionsMatch(Block<T>& block);
            Nd4jStatus validateOrdersMatch(Block<T>& block);
            Nd4jStatus validateInput2D(Block<T>& block);
            Nd4jStatus validateInput3D(Block<T>& block);
            Nd4jStatus validateInput4D(Block<T>& block);
            Nd4jStatus validateInputDimensions(Block<T>& block, int rank);
        };


        class OpRegistrator {
        private:
            static OpRegistrator* _INSTANCE;
            OpRegistrator() {};
            ~OpRegistrator() {};

            std::map<Nd4jIndex, nd4j::ops::DeclarableOp<float> *> _declarablesLF;
            std::map<std::string, nd4j::ops::DeclarableOp<float> *> _declarablesF;

            std::map<Nd4jIndex, nd4j::ops::DeclarableOp<double> *> _declarablesLD;
            std::map<std::string, nd4j::ops::DeclarableOp<double> *> _declarablesD;

            std::mutex _locker;
            std::string _opsList;
            bool isInit = false;
        public:
            static OpRegistrator* getInstance() {
                if (!_INSTANCE)
                    _INSTANCE = new nd4j::ops::OpRegistrator();

                return _INSTANCE;
            }

            const char * getAllCustomOperations() {
                _locker.lock();

                if (!isInit) {
                    for (std::map<std::string, nd4j::ops::DeclarableOp<float>*>::iterator it=_declarablesF.begin(); it!=_declarablesF.end(); ++it) {
                        _opsList += it->first + ":"
                                    + std::to_string(it->second->getOpDescriptor()->getNumberOfInputs()) + ":"
                                    + std::to_string(it->second->getOpDescriptor()->getNumberOfOutputs()) + ":"
                                    + std::to_string(it->second->getOpDescriptor()->allowsInplace())  + ":"
                                    + std::to_string(it->second->getOpDescriptor()->getNumberOfTArgs())  + ":"
                                    + std::to_string(it->second->getOpDescriptor()->getNumberOfIArgs())  + ":"
                                    + ";" ;
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

            bool registerOperationDouble(nd4j::ops::DeclarableOp<double > *op) {
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
                    nd4j_verbose("Unknown operation requested: [%s]\n", name.c_str())
                    return nullptr;
                }

                return _declarablesF.at(name);
            }


            nd4j::ops::DeclarableOp<float> *getOperationFloat(Nd4jIndex hash) {
                if (!_declarablesLF.count(hash)) {
                    nd4j_verbose("Unknown operation requested by hash: [%lld]\n", hash);
                    return nullptr;
                }

                return _declarablesLF.at(hash);
            }


            nd4j::ops::DeclarableOp<double >* getOperationDouble(const char *name) {
                std::string str(name);
                return getOperationDouble(str);
            }


            nd4j::ops::DeclarableOp<double> *getOperationDouble(Nd4jIndex hash) {
                if (!_declarablesLD.count(hash)) {
                    nd4j_verbose("Unknown operation requested by hash: [%lld]\n", hash);
                    return nullptr;
                }

                return _declarablesLD.at(hash);
            }

            nd4j::ops::DeclarableOp<double> *getOperationDouble(std::string& name) {
                if (!_declarablesD.count(name)) {
                    nd4j_verbose("Unknown operation requested: [%s]\n", name.c_str())
                    return nullptr;
                }

                return _declarablesD.at(name);
            }
        };

        template <typename OpName>
        struct __registratorSynonymFloat {
            __registratorSynonymFloat(const char *name, const char *oname) {
                OpName *ptr = (OpName *) OpRegistrator::getInstance()->getOperationFloat(oname);
                OpRegistrator::getInstance()->registerOperationFloat(name, ptr);
            }
        };

        template <typename OpName>
        struct __registratorSynonymDouble {
            __registratorSynonymDouble(const char *name, const char *oname) {
                OpName *ptr = (OpName *) OpRegistrator::getInstance()->getOperationDouble(oname);
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
void nd4j::ops::DeclarableOp<T>::storeResult(Block<T> &block, int outputNumber, NDArray<T>& array) {
    // if that's the only output - treat it as singular variable
    if (outputNumber == 0 && this->getOpDescriptor()->getNumberOfOutputs() == 1) {
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

    Nd4jIndex len = shape::length(shape);
    int* __shape = new int[shape[0] * 2 + 4];
    memcpy(__shape, shape, sizeof(int) * (shape[0] * 2 + 4));

    // if that's first run - we probably have nothing here
    if (var->getNDArray() == nullptr) {
        T* buffer = new T[len];
        var ->setNDArray(new NDArray<T>(buffer, __shape));
        var->getNDArray()->_isShapeAlloc = true;
        var->getNDArray()->_isBuffAlloc = true;
    } else if(var->getNDArray()->lengthOf() != len) {
        // if length not match - lets reallocate array
        delete var->getNDArray();
        T* buffer = new T[len];
        var ->setNDArray(new NDArray<T>(buffer, __shape));
        var->getNDArray()->_isShapeAlloc = true;
        var->getNDArray()->_isBuffAlloc = true;
    }

    return true;
}

template <typename T>
bool nd4j::ops::DeclarableOp<T>::allocateResult(Block<T>& block, std::initializer_list<int>& shape, char order) {
    auto var = block.getVariableSpace()->getVariable(block.getNodeId());

    Nd4jIndex len = shape::length(shape);
    // if that's first run - we probably have nothing here
    if (var->getNDArray() == nullptr) {
        var ->setNDArray(new NDArray<T>(order, shape));
        var->getNDArray()->_allocated = true;
    } else if(var->getNDArray()->lengthOf() != len) {
        // if length not match - lets reallocate array
        delete var->getNDArray();
        var ->setNDArray(new NDArray<T>(order, shape));
        var->getNDArray()->_allocated = true;
    }

    return true;
}

template <typename T>
Nd4jStatus nd4j::ops::DeclarableOp<T>::execute(Block<T>* block) {
    if (block != nullptr)
        _block = block;
    else
        throw std::invalid_argument("Block is NULL");

    return this->validateAndExecute(*block);
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


    for (auto v: block.getVariables()) {
        NDArray<T> *aV = v->getNDArray();

        if (aV == nullptr || !aV->nonNull())
            return ND4J_STATUS_BAD_INPUT;
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
Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInputDimensionsMatch(Block<T>& block) {
    if (block.getVariables().size() == 0)
        return ND4J_STATUS_OK;


    NDArray<T> *a0 = block.getVariables().at(0)->getNDArray();
    for (auto v: block.getVariables()) {
        NDArray<T> *aV = v->getNDArray();
        if (!shape::equalsSoft(a0->shapeOf(), aV->shapeOf()))
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
