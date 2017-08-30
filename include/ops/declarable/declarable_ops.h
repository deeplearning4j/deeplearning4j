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
#include "OpRegistrator.h"

#define REQUIRE_OK(A) nd4j::ops::resultHelper( (A), #A, __FILE__, __LINE__ );


#define DECLARE_OP(NAME, NIN, NOUT)   DECLARE_OP_UNIQ(__COUNTER__, NAME, NIN, NOUT)
#define DECLARE_OP_UNIQ(CTR, NAME, NIN, NOUT)   template <typename T> \
                                                class NAME: public nd4j::ops::DeclarableOp<T> { \
                                                public:\
                                                    NAME() : nd4j::ops::DeclarableOp<T>(NIN, NOUT, #NAME) { } \
                                                protected: \
                                                    Nd4jStatus validateAndExecute(Block<T>& block); \
                                                };\
                                                static nd4j::ops::__registratorFloat<NAME<float>> register_opf_##NAME; \
                                                static nd4j::ops::__registratorDouble<NAME<double>> register_opd_##NAME; \
                                                template <typename T> \
                                                Nd4jStatus nd4j::ops::NAME<T>::validateAndExecute(Block<T>& block)

//#define END_OP(NAME) };


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
        public:
            DeclarableOp(int numInputs, int numOutputs, const char *opName) {
                _descriptor = new OpDescriptor(numInputs, numOutputs, opName);
            }

            ~DeclarableOp() {
                if (_descriptor != nullptr)
                    delete _descriptor;
            }


            OpDescriptor *getOpDescriptor() {
                return &_descriptor;
            }

            /**
             * Returns opName
             *
             * @return
             */
            std::string *getOpName() {
                return _descriptor->getOpName();
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

            std::map<std::string, nd4j::ops::DeclarableOp<float> *> _declarablesF;
            std::map<std::string, nd4j::ops::DeclarableOp<double> *> _declarablesD;
        public:
            static OpRegistrator* getInstance() {
                if (!_INSTANCE)
                    _INSTANCE = new nd4j::ops::OpRegistrator();

                return _INSTANCE;
            }

            /**
             * This method registers operation
             *
             * @param op
             */
            bool registerOperationFloat(nd4j::ops::DeclarableOp<float>* op) {
                std::pair<std::string, nd4j::ops::DeclarableOp<float>*> pair(*(op->getOpName()), op);
                _declarablesF.insert(pair);
                return true;
            }

            bool registerOperationDouble(nd4j::ops::DeclarableOp<double > *op) {
                std::pair<std::string, nd4j::ops::DeclarableOp<double>*> pair(*(op->getOpName()), op);
                _declarablesD.insert(pair);
                return true;
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

            nd4j::ops::DeclarableOp<double> *getOperationDouble(std::string& name) {
                if (!_declarablesD.count(name)) {
                    nd4j_verbose("Unknown operation requested: [%s]\n", name.c_str())
                    return nullptr;
                }

                return _declarablesD.at(name);
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
bool nd4j::ops::DeclarableOp<T>::allocateResult(Block<T>& block, int* shape) {
    auto var = block.getVariableSpace()->getVariable(block.getNodeId());

    Nd4jIndex len = shape::length(shape);
    int* __shape = new int[shape[0] * 2 + 4];
    memcpy(__shape, shape, sizeof(int) * (shape[0] * 2 + 4));

    // if that's first run - we probably have nothing here
    if (var->getNDArray() == nullptr) {
        T* buffer = new T[len];
        var ->setNDArray(new NDArray<T>(buffer, __shape));
        var->getNDArray()->_allocated = true;
    } else if(var->getNDArray()->lengthOf() != len) {
        // if length not match - lets reallocate array
        delete var->getNDArray();
        T* buffer = new T[len];
        var ->setNDArray(new NDArray<T>(buffer, __shape));
        var->getNDArray()->_allocated = true;
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
    for (int e = 0; e < block.getVariables().size(); e++) {
        if (l0 != block.getVariables().at(e)->getNDArray()->lengthOf())
            return ND4J_STATUS_BAD_LENGTH;
    }

    return ND4J_STATUS_OK;
}

#endif //LIBND4J_DECLARABLE_OPS_H
