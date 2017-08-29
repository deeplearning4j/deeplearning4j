//
// These ops are provided for features parity with TF
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_PARITY_OPS_H
#define LIBND4J_PARITY_OPS_H

#include <ops/ops.h>
#include <ops/declarable/declarable_ops.h>

namespace nd4j {
    namespace ops {

        DECLARE_OP(Concat, -1, 1){
            // do something here{
                Nd4jIndex _length;
                int _dimension = 0;

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

                Nd4jIndex len = shape::length(shape);
                if (variable->getNDArray() == nullptr) {
                    T *buffer = new T[len];
                    variable ->setNDArray(new NDArray<T>(buffer, shape));
                    variable->getNDArray()->_allocated = true;
                } else if(variable->getNDArray()->lengthOf() != len) {
                    delete variable->getNDArray();
                    T *buffer = new T[len];
                    variable ->setNDArray(new NDArray<T>(buffer, shape));
                    variable->getNDArray()->_allocated = true;
                } else {
                    delete[] shape;
                }

                concatCpuGeneric(_dimension, block.getVariables().size(), buffers, shapes, variable->getNDArray()->_buffer, variable->getNDArray()->_shapeInfo);

                delete[] buffers;
                delete[] shapes;
                return ND4J_STATUS_OK;
            }


        DECLARE_OP(BiasAdd, 2, 1) {
                REQUIRE_OK(this->validateNonEmptyInput(block));
                REQUIRE_OK(this->validateInput2D(block));

                NDArray<T> *x = block.getVariables().at(0)->getNDArray();
                NDArray<T> *y = block.getVariables().at(1)->getNDArray();

                if (x->isMatrix() && y->isVector()) {
                    x->addiRowVector(y);
                } else if (y->isMatrix() && x->isVector()) {
                    y->addiRowVector(x);
                }

                return ND4J_STATUS_OK;
            }

        DECLARE_OP(MatMul, 2, 1) {

                REQUIRE_OK(this->validateNonEmptyInput(block));

                // FIXME: we might want to have gemv/dot fallback here
                REQUIRE_OK(this->validateInput2D(block));


                NDArray<T> *x = block.getVariables().at(0)->getNDArray();
                NDArray<T> *y = block.getVariables().at(1)->getNDArray();

                if (x->isMatrix() && y->isMatrix()) {
                    // gemm
                } else if (x->isMatrix() && y->isVector()) {
                    // gemv
                } else if (x->isVector() && y->isMatrix()) {
                    // gemv
                } else if (x->isVector() && y->isVector()) {
                    // dot
                } else if (x->isVector() && y->isScalar()) {
                    // elementwise mul
                } else if (x->isScalar() && y->isVector()) {
                    // elementwise mul, reverse op
                }

                return ND4J_STATUS_OK;
            }


        DECLARE_OP(Conv2D, 2, 1) {
                return ND4J_STATUS_OK;
            }


        DECLARE_OP(Conv3D, 2, 1) {
                return ND4J_STATUS_OK;
            }



        DECLARE_OP(Relu, 1, 1) {
                REQUIRE_OK(this->validateNonEmptyInput(block));

                NDArray<T> *first = block.getVariables().at(0)->getNDArray();
                first->template applyTransform<simdOps::RELU<T>>();

                return ND4J_STATUS_OK;
            }



        DECLARE_OP(Identity, 1, 1) {
                REQUIRE_OK(this->validateNonEmptyInput(block));

                NDArray<T> *first = block.getVariables().at(0)->getNDArray();
                first->template applyTransform<simdOps::Identity<T>>();

                return ND4J_STATUS_OK;
            }
    }
}

#endif //LIBND4J_PARITY_OPS_H
