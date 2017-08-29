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
        };


        template <typename T>
        class BiasAdd: public nd4j::ops::DeclarableOp<T> {
        public:
            BiasAdd() : nd4j::ops::DeclarableOp<T>(2, 1, "BiasAdd") {

            }

        protected:

            Nd4jStatus validateAndExecute(Block<T>& block) {
                REQUIRE_OK(validateNonEmptyInput(block));
                REQUIRE_OK(validateInput2D(block));

                NDArray<T> *x = block.getVariables().at(0)->getNDArray();
                NDArray<T> *y = block.getVariables().at(1)->getNDArray();

                if (x->isMatrix() && y->isVector()) {
                    x->addiRowVector(y);
                } else if (y->isMatrix() && x->isVector()) {
                    y->addiRowVector(x);
                }

                return ND4J_STATUS_OK;
            }
        };

        /**
         * gemm
         * @tparam T
         */
        template <typename T>
        class MatMul: public nd4j::ops::DeclarableOp<T> {
        public:
            MatMul() : nd4j::ops::DeclarableOp<T>(2, 1, "MatMul") {
                //
            }

        protected:

            Nd4jStatus validateAndExecute(Block<T>& block) {
                REQUIRE_OK(validateNonEmptyInput(block));

                // FIXME: we might want to have gemv/dot fallback here
                REQUIRE_OK(validateInput2D(block));


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
        };


        template <typename T>
        class Conv2D: public nd4j::ops::DeclarableOp<T> {
        public:
            Conv2D() : nd4j::ops::DeclarableOp<T>(2, 1, "Conv2D") {
                //
            }

        protected:

            Nd4jStatus validateAndExecute(Block<T>& block) {
                return ND4J_STATUS_OK;
            }
        };


        template <typename T>
        class Conv3D: public nd4j::ops::DeclarableOp<T> {
        public:
            Conv3D() : nd4j::ops::DeclarableOp<T>(2, 1, "Conv3D") {
                //
            }

        protected:

            Nd4jStatus validateAndExecute(Block<T>& block) {
                return ND4J_STATUS_OK;
            }
        };


        template <typename T>
        class Relu: public nd4j::ops::DeclarableOp<T> {
        public:
            Relu() : nd4j::ops::DeclarableOp<T>(1, 1, "Relu") {
                //
            }

        protected:

            Nd4jStatus validateAndExecute(Block<T>& block) {
                REQUIRE_OK(validateNonEmptyInput(block));

                NDArray<T> *first = block.getVariables().at(0)->getNDArray();
                first->template applyTransform<simdOps::RELU<T>>();

                return ND4J_STATUS_OK;
            }
        };



        template <typename T>
        class Identity: public nd4j::ops::DeclarableOp<T> {
        public:
            Identity() : nd4j::ops::DeclarableOp<T>(1, 1, "Identity") {
                //
            }

        protected:

            Nd4jStatus validateAndExecute(Block<T>& block) {
                REQUIRE_OK(validateNonEmptyInput(block));

                NDArray<T> *first = block.getVariables().at(0)->getNDArray();
                first->template applyTransform<simdOps::Identity<T>>();

                return ND4J_STATUS_OK;
            }
        };
    }
}

#endif //LIBND4J_PARITY_OPS_H
