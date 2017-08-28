//
// @author raver119@gmail.com
//

#ifndef LIBND4J_DECLARABLE_OPS_H
#define LIBND4J_DECLARABLE_OPS_H

#define REQUIRE_OK(A) nd4j::ops::resultHelper( (A), #A, __FILE__, __LINE__ );

#include <pointercast.h>
#include <NDArray.h>
#include <Variable.h>
#include <Block.h>
#include "OpDescriptor.h"

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
             * This method executes this Op b
             */
            virtual Nd4jStatus validateAndExecute(Block<T>& block) = 0;
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
        };
    }
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
    if (block.getVariables()->size() == 0)
        return ND4J_STATUS_OK;


    NDArray<T> *a0 = block.getVariables()->at(0)->getNDArray();
    for (auto v: block.getVariables()) {
        NDArray<T> *aV = v->getNDArray();
        if (!shape::equalsSoft(a0->shapeOf(), aV->shapeOf()))
            return ND4J_STATUS_BAD_DIMENSIONS;
    }

    return ND4J_STATUS_OK;
}

template <typename T>
Nd4jStatus nd4j::ops::DeclarableOp<T>::validateInputLengthMatch(Block<T>& block) {
    if (block.getVariables()->size() == 0)
        return ND4J_STATUS_OK;


    Nd4jIndex l0 = block.getVariables()->at(0);
    for (int e = 0; e < block.getVariables()->size(); e++) {
        if (l0 != block.getVariables()->at(e)->getNDArray()->lengthOf())
            return ND4J_STATUS_BAD_LENGTH;
    }

    return ND4J_STATUS_OK;
}


namespace nd4j {
    namespace ops {

        template <typename T>
        class Concat: public nd4j::ops::DeclarableOp<T> {
        public:
            Concat() : nd4j::ops::DeclarableOp<T>(-1, 1, "Concat") {

            }

        protected:
            Nd4jIndex _length;
            int _dimension = 0;

            // do something here
            Nd4jStatus validateAndExecute(Block<T>& block) {
                // basic checks are happening here
                REQUIRE_OK(this->validateNonEmptyInput(block));

                // we want to ensure that all
                NDArray<T> *first = block.getVariables().at(0)->getNDArray();

                int *shape = new int[first->_shapeInfo[0] * 2 + 4];
                std::memcpy(shape, first->_shapeInfo, (first->_shapeInfo[0] * 2 + 4) * sizeof(int));
                _length = shape::length(shape);

                Nd4jPointer *buffers = new Nd4jPointer[block.getVariables().size()];
                Nd4jPointer *shapes = new Nd4jPointer[block.getVariables().size()];

                buffers[0] = (Nd4jPointer) first->_buffer;
                shapes[0] = (Nd4jPointer) first->_shapeInfo;

                for (int e = 1; e < block.getVariables().size(); e++) {
                    Variable<T> *var = block.getVariables().at(e);
                    _length += var->getNDArray()->lengthOf();

                    shape[_dimension + 1] += var->getNDArray()->shapeOf()[_dimension];

                    buffers[e] = (Nd4jPointer) var->getNDArray()->_buffer;
                    shapes[e] = (Nd4jPointer) var->getNDArray()->_shapeInfo;
                }

                if (!block.getVariableSpace()->hasVariable(block.getNodeId()))
                    throw "VariableSpace has no registered node";

                auto variable = block.getVariableSpace()->getVariable(block.getNodeId());

                if (variable->getNDArray() == nullptr) {
                    Nd4jIndex len = shape::length(shape);
                    T *buffer = new T[len];
                    variable ->setNDArray(new NDArray<T>(buffer, shape));
                    variable->getNDArray()->_allocated = true;
                } else {

                    delete[] shape;
                }

                concatCpuGeneric(_dimension, block.getVariables().size(), buffers, shapes, variable->getNDArray()->_buffer, variable->getNDArray()->_shapeInfo);

                return ND4J_STATUS_OK;
            }
        };
    }
}


#endif //LIBND4J_DECLARABLE_OPS_H
