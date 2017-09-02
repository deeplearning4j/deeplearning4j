//
// These ops are provided for features parity with TF
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_PARITY_OPS_H
#define LIBND4J_PARITY_OPS_H

#include <memory>
#include <ops/ops.h>
#include <NDArray.h>
#include <ops/declarable/declarable_ops.h>

namespace nd4j {
    namespace ops {

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(concat, -1, 1){
            // do something here{
            Nd4jIndex _length;
            int _dimension = 0;

            // basic checks are happening here
            REQUIRE_OK(this->validateNonEmptyInput(block));

            // we want to ensure that all
            NDArray<T> *first = block.getVariables().at(0)->getNDArray();

            std::unique_ptr<int> shape(new int[first->_shapeInfo[0] * 2 + 4]);

            std::memcpy(shape.get(), first->_shapeInfo, (first->_shapeInfo[0] * 2 + 4) * sizeof(int));
            _length = shape::length(shape.get());

            std::unique_ptr<Nd4jPointer> buffers(new Nd4jPointer[block.getVariables().size()]);
            std::unique_ptr<Nd4jPointer> shapes(new Nd4jPointer[block.getVariables().size()]);

            buffers.get()[0] = (Nd4jPointer) first->_buffer;
            shapes.get()[0] = (Nd4jPointer) first->_shapeInfo;

            for (int e = 1; e < block.getVariables().size(); e++) {
                Variable<T> *var = block.getVariables().at(e);
                _length += var->getNDArray()->lengthOf();

                shape.get()[_dimension + 1] += var->getNDArray()->shapeOf()[_dimension];

                buffers.get()[e] = (Nd4jPointer) var->getNDArray()->_buffer;
                shapes.get()[e] = (Nd4jPointer) var->getNDArray()->_shapeInfo;
            }

            if (!block.getVariableSpace()->hasVariable(block.getNodeId()))
                throw "VariableSpace has no registered node";

            if (!this->allocateResult(block, shape.get())){
                nd4j_printf("Allocation failed: %i\n", block.getNodeId());
                throw "Allocation failed";
            }

            auto variable = block.getVariableSpace()->getVariable(block.getNodeId());

            concatCpuGeneric(_dimension, block.getVariables().size(), buffers.get(), shapes.get(), variable->getNDArray()->_buffer, variable->getNDArray()->_shapeInfo);

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(biasAdd, 2, 1) {
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

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(matMul, 2, 1) {
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
        DECLARE_SYN(mMul, matMul);

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(conv2D, 2, 1) {
            // basically im2col + gemm
            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(conv3D, 2, 1) {
            // cubic convo
            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(maxPool, 2, 1) {
            // MaxPooling
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(MaxPool2D, maxPool);

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(avgPool, 2, 1) {
            // AvgPooling
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(AvgPool2D, avgPool);

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(lrn, 2, 1) {
            // LocalResponseNormalization
            return ND4J_STATUS_OK;
        }


///////////////////////
        DECLARE_OP(randomUniform, 1, 1) {
            // uniform distribution
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(shape, 2, 1) {
            // ?
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(floor, 1, 1) {
            // ?
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(realDiv, 2, 1) {
            // ?
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(merge, -1, 1) {
            // basically hstack
            return ND4J_STATUS_OK;
        }


        DECLARE_DIVERGENT_OP(Switch, 2, 2) {
            // conditional op !!!
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(switch, Switch);

        DECLARE_DIVERGENT_OP(noOp, -1, 2) {
            // Fastest op ever.
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(BroadcastGradientArgs, 2, 2) {

            return ND4J_STATUS_OK;
        }

        // test op, non-divergent
        DECLARE_OP(testop2i2o, 2, 2) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();

            x->template applyScalar<simdOps::Add<T>>(1.0);
            y->template applyScalar<simdOps::Add<T>>(2.0);

            STORE_2_RESULTS(*x, *y);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TestOp2i2o, testop2i2o);

        DECLARE_OP(assign, 2, 1) {
            // NDArray->assign(NDArray)
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(set, assign);

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(softmax, 2, 1) {
            // YaY
            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(relu, 1, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            first->template applyTransform<simdOps::RELU<T>>();

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(identity, 1, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            first->template applyTransform<simdOps::Identity<T>>();

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////		
		DECLARE_OP(add, 2, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();

            NDArray<T> *z = x;
            if (block.getVariableSpace()->hasVariable(block.getNodeId())) {
                auto var = block.getVariableSpace()->getVariable(block.getNodeId());
                if (var->getNDArray() != nullptr) {
                    z = var->getNDArray();
                }
            }

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

			return ND4J_STATUS_OK;
        }


//////////////////////////////////////////////////////////////////////////
		DECLARE_OP(subtract, 2, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();			

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				x->template applyPairwiseTransform<simdOps::Subtract<T>>(y, nullptr);                
            
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::Subtract<T>>(*y, x);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::Subtract<T>>(*x, y);

            }						
			else { // x->isScalar() && y->isScalar()
				x->putScalar(0, x->getScalar(0) - y->getScalar(0));

			}
			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(Sub, subtract);
        DECLARE_SYN(sub, subtract);

//////////////////////////////////////////////////////////////////////////		
		DECLARE_OP(reverseSubtract, 2, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();			

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				x->template applyPairwiseTransform<simdOps::ReverseSubtract<T>>(y, nullptr);                
            
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::ReverseSubtract<T>>(*y, x);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::ReverseSubtract<T>>(*x, y);

            }						
			else { // x->isScalar() && y->isScalar()
				x->putScalar(0, y->getScalar(0) - x->getScalar(0));

			}
			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(RSub, reverseSubtract);

//////////////////////////////////////////////////////////////////////////		
		DECLARE_OP(multiply, 2, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();			

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				// REQUIRE_OK(this->validateInputDimensionsMatch(block));
				x->template applyPairwiseTransform<simdOps::Multiply<T>>(y, nullptr);                
	
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::Multiply<T>>(*y, x);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::Multiply<T>>(*x, y);

            }						
			else { // (x->isScalar() && y->isScalar())
				x->putScalar(0, x->getScalar(0) * y->getScalar(0));
			
            }
			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(Mul, multiply);

//////////////////////////////////////////////////////////////////////////		
		DECLARE_OP(divide, 2, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();			

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				// REQUIRE_OK(this->validateInputDimensionsMatch(block));
				x->template applyPairwiseTransform<simdOps::Divide<T>>(y, nullptr);                
	
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::Divide<T>>(*y, x);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::Divide<T>>(*x, y);

            }						
			else { // (x->isScalar() && y->isScalar())
				x->putScalar(0, x->getScalar(0) / y->getScalar(0));
			
            }
			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(Div, divide);

//////////////////////////////////////////////////////////////////////////				
		DECLARE_OP(reverseDivide, 2, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();			

			if (!x->isScalar() && !y->isScalar()) {
				REQUIRE_OK(this->validateInputLengthMatch(block));
				// REQUIRE_OK(this->validateInputDimensionsMatch(block));
				x->template applyPairwiseTransform<simdOps::ReverseDivide<T>>(y, nullptr);                
	
            } else if (!x->isScalar() && y->isScalar()) {
               x->template applyScalar<simdOps::ReverseDivide<T>>(*y, x);

            } else if (x->isScalar() && !y->isScalar()) {
                y->template applyScalar<simdOps::ReverseDivide<T>>(*x, y);

            }						
			else { // (x->isScalar() && y->isScalar())
				x->putScalar(0, y->getScalar(0) / x->getScalar(0));
			
            }
			return ND4J_STATUS_OK;
        }
        DECLARE_SYN(RDiv, reverseDivide);

//////////////////////////////////////////////////////////////////////////				
		DECLARE_OP(reshape, 2, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));            

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();	
			
			std::vector<int> newShape(y->shapeOf(), y->shapeOf() + y->rankOf());
			char order = y->ordering();
			if (x->reshape(order, newShape))
				return ND4J_STATUS_OK;
			
			return ND4J_STATUS_BAD_INPUT;
        }
		
    }
}

#endif //LIBND4J_PARITY_OPS_H

