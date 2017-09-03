//
// These ops are provided for features parity with TF
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_PARITY_OPS_H
#define LIBND4J_PARITY_OPS_H

#include <memory>
#include <shape.h>
#include <ops/ops.h>
#include <NDArray.h>
#include <ops/declarable/declarable_ops.h>

namespace nd4j {
    namespace ops {

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(concat, -1, 1, false){
            // do something here{
            Nd4jIndex _length;
            int _dimension = 0;

            // basic checks are happening here
            REQUIRE_OK(this->validateNonEmptyInput(block));

            // we want to ensure that all
            NDArray<T> *first = block.getVariables().at(0)->getNDArray();

            std::unique_ptr<int> shapePtr(new int[first->_shapeInfo[0] * 2 + 4]);

            std::memcpy(shapePtr.get(), first->_shapeInfo, (first->_shapeInfo[0] * 2 + 4) * sizeof(int));
            _length = shape::length(shapePtr.get());

            std::unique_ptr<Nd4jPointer> buffers(new Nd4jPointer[block.getVariables().size()]);
            std::unique_ptr<Nd4jPointer> shapes(new Nd4jPointer[block.getVariables().size()]);

            buffers.get()[0] = (Nd4jPointer) first->_buffer;
            shapes.get()[0] = (Nd4jPointer) first->_shapeInfo;

            for (int e = 1; e < block.getVariables().size(); e++) {
                Variable<T> *var = block.getVariables().at(e);
                _length += var->getNDArray()->lengthOf();

                shapePtr.get()[_dimension + 1] += var->getNDArray()->shapeOf()[_dimension];

                buffers.get()[e] = (Nd4jPointer) var->getNDArray()->_buffer;
                shapes.get()[e] = (Nd4jPointer) var->getNDArray()->_shapeInfo;
            }

            if (!block.getVariableSpace()->hasVariable(block.getNodeId()))
                throw "VariableSpace has no registered node";

            if (!this->allocateResult(block, shapePtr.get())){
                nd4j_printf("Allocation failed: %i\n", block.getNodeId());
                throw "Allocation failed";
            }

            auto variable = block.getVariableSpace()->getVariable(block.getNodeId());

            concatCpuGeneric(_dimension, block.getVariables().size(), buffers.get(), shapes.get(), variable->getNDArray()->_buffer, variable->getNDArray()->_shapeInfo);

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(biasAdd, 2, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));
            REQUIRE_OK(this->validateInput2D(block));

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();
            NDArray<T> *z = this->getZ(block);

            if (x->isMatrix() && y->isVector()) {
                x->addiRowVector(y);
            } else if (y->isMatrix() && x->isVector()) {
                y->addiRowVector(x);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(matMul, 2, 1, false) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            // FIXME: we might want to have gemv/dot fallback here
            REQUIRE_OK(this->validateInput2D(block));


            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();
            NDArray<T> *z = this->getZ(block);

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

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(mMul, matMul);

//////////////////////////////////////////////////////////////////////////
        DECLARE_CONFIGURABLE_OP(conv2d, 2, 1, false, 0, 7) {
            // basically im2col + gemm
            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_CONFIGURABLE_OP(conv3d, 2, 1, false, 0, 7) {
            // cubic convo
            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(maxPool, 2, 1, true) {
            // MaxPooling
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(MaxPool2D, maxPool);

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(avgPool, 2, 1, true) {
            // AvgPooling
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(AvgPool2D, avgPool);

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(lrn, 2, 1, true) {
            // LocalResponseNormalization
            return ND4J_STATUS_OK;
        }


///////////////////////
        DECLARE_OP(randomUniform, 1, 1, true) {
            // uniform distribution
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(shape, 2, 1, true) {
            // ?
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(floor, 1, 1, true) {
            // ?
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(realDiv, 2, 1, true) {
            // ?
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(merge, -1, 1, true) {
            // basically hstack
            return ND4J_STATUS_OK;
        }


        DECLARE_DIVERGENT_OP(Switch, 2, 2, true) {
            // conditional op !!!
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(switch, Switch);

        DECLARE_DIVERGENT_OP(noOp, -1, -1, true) {
            // Fastest op ever.
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(BroadcastGradientArgs, 2, 2, true) {

            return ND4J_STATUS_OK;
        }

        // test op, non-divergent
        DECLARE_OP(testop2i2o, 2, 2, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();

            x->template applyScalar<simdOps::Add<T>>(1.0);
            y->template applyScalar<simdOps::Add<T>>(2.0);

            STORE_2_RESULTS(*x, *y);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TestOp2i2o, testop2i2o);

        DECLARE_OP(assign, 2, 1, true) {
            // NDArray->assign(NDArray)
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(set, assign);

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(softmax, 2, 1, false) {
            // YaY
            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(relu, 1, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            first->template applyTransform<simdOps::RELU<T>>();

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(identity, 1, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            first->template applyTransform<simdOps::Identity<T>>();

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////		
		DECLARE_OP(add, 2, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();
            NDArray<T> *z = this->getZ(block);

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				x->template applyPairwiseTransform<simdOps::Add<T>>(y, z, nullptr);
            
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::Add<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::Add<T>>(*x, z);
            }						
			else { // x->isScalar() && y->isScalar()
				z->putScalar(0, x->getScalar(0) + y->getScalar(0));
			}

            STORE_RESULT(*z);

			return ND4J_STATUS_OK;
        }


//////////////////////////////////////////////////////////////////////////
		DECLARE_OP(subtract, 2, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();
            NDArray<T> *z = this->getZ(block);

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				x->template applyPairwiseTransform<simdOps::Subtract<T>>(y, z, nullptr);
            
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::Subtract<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::Subtract<T>>(*x, z);

            }						
			else { // x->isScalar() && y->isScalar()
				z->putScalar(0, x->getScalar(0) - y->getScalar(0));
			}

            STORE_RESULT(*z);

			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(Sub, subtract);
        DECLARE_SYN(sub, subtract);

//////////////////////////////////////////////////////////////////////////		
		DECLARE_OP(reverseSubtract, 2, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();
            NDArray<T> *z = this->getZ(block);

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				x->template applyPairwiseTransform<simdOps::ReverseSubtract<T>>(y, z, nullptr);
            
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::ReverseSubtract<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::ReverseSubtract<T>>(*x, z);

            }						
			else { // x->isScalar() && y->isScalar()
				z->putScalar(0, y->getScalar(0) - x->getScalar(0));
			}

            STORE_RESULT(*z);

			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(RSub, reverseSubtract);

//////////////////////////////////////////////////////////////////////////		
		DECLARE_OP(multiply, 2, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();
            NDArray<T> *z = this->getZ(block);

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				// REQUIRE_OK(this->validateInputDimensionsMatch(block));
				x->template applyPairwiseTransform<simdOps::Multiply<T>>(y, z, nullptr);
	
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::Multiply<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::Multiply<T>>(*z, y);

            }						
			else { // (x->isScalar() && y->isScalar())
				z->putScalar(0, x->getScalar(0) * y->getScalar(0));
            }

            STORE_RESULT(*z);

			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(Mul, multiply);

//////////////////////////////////////////////////////////////////////////		
		DECLARE_OP(divide, 2, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();
            NDArray<T> *z = this->getZ(block);

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				// REQUIRE_OK(this->validateInputDimensionsMatch(block));
				x->template applyPairwiseTransform<simdOps::Divide<T>>(y, z, nullptr);
	
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::Divide<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::Divide<T>>(*x, z);
            }						
			else { // (x->isScalar() && y->isScalar())
				z->putScalar(0, x->getScalar(0) / y->getScalar(0));
            }

            STORE_RESULT(*z);

			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(Div, divide);

//////////////////////////////////////////////////////////////////////////				
		DECLARE_OP(reverseDivide, 2, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();
            NDArray<T> *z = this->getZ(block);

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				// REQUIRE_OK(this->validateInputDimensionsMatch(block));
				x->template applyPairwiseTransform<simdOps::ReverseDivide<T>>(y, z, nullptr);
	
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::ReverseDivide<T>>(*y, z);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::ReverseDivide<T>>(*x, z);

            }						
			else { // (x->isScalar() && y->isScalar())
				z->putScalar(0, y->getScalar(0) / x->getScalar(0));
            }

            STORE_RESULT(*z);

			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(RDiv, reverseDivide);

//////////////////////////////////////////////////////////////////////////
		DECLARE_OP(reshape, 2, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();	
			
			std::vector<int> newShape(y->shapeOf(), y->shapeOf() + y->rankOf());
			char order = y->ordering();
			if (x->reshape(order, newShape))
				return ND4J_STATUS_OK;

            STORE_RESULT(*x);
			
			return ND4J_STATUS_BAD_INPUT;
        }
		
    }
}

#endif //LIBND4J_PARITY_OPS_H

