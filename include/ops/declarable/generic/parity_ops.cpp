//
// These ops are provided for features parity with TF
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_PARITY_OPS_H
#define LIBND4J_PARITY_OPS_H

#include <climits>
#include <op_boilerplate.h>
#include <memory>
#include <shape.h>
#include <ops/ops.h>
#include <loops/random.h>
#include <NDArray.h>
#include <graph/Variable.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/DeclarableReductionOp.h>
#include <ops/declarable/DeclarableCustomOp.h>
#include <NDArrayFactory.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/Block.h>
#include <ops/declarable/OpRegistrator.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {

//////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(concat, -1, 1, false, 0, 1){
            // do something here{

            int _dimension = block.getIArguments()->at(0);

            // we want to ensure that all
            NDArray<T> *first = INPUT_VARIABLE(0);
            NDArray<T> *output = this->getZ(block);

            Nd4jPointer* buffers = new Nd4jPointer[block.width()];
            Nd4jPointer* shapes = new Nd4jPointer[block.width()];

            buffers[0] = (Nd4jPointer) first->getBuffer();
            shapes[0] = (Nd4jPointer) first->getShapeInfo();

            if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                printf("Shape %i: ", 0);
                shape::printShapeInfoLinear((int *) shapes[0]);
            }

            for (int e = 1; e < (int) block.width(); e++) {
                Variable<T> *var = block.variable(e);

                buffers[e] = (Nd4jPointer) var->getNDArray()->getBuffer();
                shapes[e] = (Nd4jPointer) var->getNDArray()->getShapeInfo();

                if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                    printf("Shape %i: ", e);
                    shape::printShapeInfoLinear((int *) shapes[e]);
                }
            }
            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                fflush(stdout);

            nd4j::SpecialMethods<T>::concatCpuGeneric(_dimension, block.width(), buffers, shapes, output->getBuffer(), output->getShapeInfo());

            STORE_RESULT(*output);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                output->printShapeInfo("Concat result shape");

            delete[] buffers;
            delete[] shapes;

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(concat_v2, concat);
        DECLARE_SYN(concatv2, concat);
        DECLARE_SHAPE_FN(concat) {
            int* inp = inputShape->at(0);
            int _dimension = block.getIArguments()->at(0);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inp), int);

            std::memcpy(newShape, inp, shape::shapeInfoByteLength(inp));
            for (int i = 1; i < inputShape->size(); i++) {
                newShape[_dimension + 1] += shape::shapeOf(inputShape->at(i))[_dimension];
            }

            shape::updateStrides(newShape, shape::order(inp));

            return new ShapeList(newShape);
        }

//////////////////////////////////////////////////////////////////////////
        OP_IMPL(biasadd, 2, 1, true) {
            //REQUIRE_OK(this->validateInput2D(block));

            NDArray<T> *input = INPUT_VARIABLE(0);
            NDArray<T> *bias = INPUT_VARIABLE(1);

            REQUIRE_TRUE(bias->isRowVector(), 0, "Bias array should be a vector");

            NDArray<T> *z = this->getZ(block);

            if (input->isMatrix())
                input->addiRowVector(bias);
            else {
                std::vector<int> shape({-1, (int) bias->lengthOf()});
                nd4j_debug("Reshaping to: [%i, %i]\n", -1, (int) bias->lengthOf());
                auto tArr = input->reshape(input->ordering(), shape);
                auto zArr = z->reshape(z->ordering(), shape);
                tArr->addRowVector(bias, zArr);

                delete tArr;
                delete zArr;
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(bias_add, biasadd);

//////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(matmul, 2, 1, false, -2, 0) {
            // FIXME: we might want to have gemv/dot fallback here
            REQUIRE_OK(this->validateInput2D(block));


            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
            NDArray<T> *z = this->getZ(block);

            T alpha = (T) 1.0f;
            T beta = (T) 0.0f;
            if (block.getTArguments()->size() > 0)
                alpha = block.getTArguments()->at(0);

            if (block.getTArguments()->size() > 1)
                beta = block.getTArguments()->at(1);


            if (x->isMatrix() && y->isVector()) {
                // gemv
                nd4j::NDArrayFactory<T>::mmulHelper(x, y, z, alpha, beta);

            } else if (x->isVector() && y->isMatrix()) {
                // gemm
                nd4j::NDArrayFactory<T>::mmulHelper(x, y, z, alpha, beta);
            }  else if (x->isVector() && y->isVector()) {
                // dot
                nd4j::NDArrayFactory<T>::mmulHelper(x, y, z, alpha, beta);
            } else if (x->isMatrix() && y->isMatrix()) {
                // gemm
                nd4j::NDArrayFactory<T>::mmulHelper(x, y, z, alpha, beta);
            } else if (x->isVector() && y->isScalar()) {
                // elementwise mul

                x->template applyScalar<simdOps::Multiply<T>>(y->getScalar(0), z, nullptr);
             } else if (x->isScalar() && y->isVector()) {
                // elementwise mul, reverse op

                y->template applyScalar<simdOps::Multiply<T>>(x->getScalar(0), z, nullptr);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(mMul, matmul);
        DECLARE_SYN(mmul, matmul);
        DECLARE_SYN(gemm, matmul);
        DECLARE_SYN(gemv, matmul);
        DECLARE_SYN(dot, matmul);
        DECLARE_SHAPE_FN(matmul) {
            int *inA = inputShape->at(0);
            int *inB = inputShape->at(1);
            int *shape;
            ALLOCATE(shape, block.getWorkspace(), 2, int);

            if (shape::isScalar(inA) && shape::isScalar(inB)) {
                // just scalar vs scalar
                shape[0] = 1;
                shape[1] = 1;
            } else if ((shape::isVector(inA) && shape::isScalar(inB)) || (shape::isScalar(inA) && shape::isVector(inB))) {
                // element-wise
                shape[0] = 1;
                shape[1] = (int) nd4j::math::nd4j_max<Nd4jIndex>(shape::length(inA), shape::length(inB));
            } else if (shape::isVector(inA) && shape::isVector(inB)) {
                // dot case
                shape[0] = 1;
                shape[1] = 1;
            } else if (shape::isMatrix(inA) && shape::isVector(inB)) {
                // gemv case
                shape[0] = 1;
                shape[1] = (int) shape::length(inB);
            } else if ((shape::isMatrix(inA) && shape::isMatrix(inB)) || (shape::isVector(inA) && shape::isMatrix(inB))) {
                // gemv case
                shape[0] = inA[1];
                shape[1] = inB[2];
            }

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), int);
            shape::shapeBufferFortran(2, shape, newShape);

            RELEASE(shape, block.getWorkspace());
            return new ShapeList(newShape);
        }


        OP_IMPL(floor, 1, 1, true) {
            NDArray<T> *first = INPUT_VARIABLE(0);
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::Floor<T>>(z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        OP_IMPL(realdiv, 2, 1, true) {
            // ?
            return ND4J_STATUS_OK;
        }

        OP_IMPL(merge, -1, 1, true) {
            // basically hstack
            return ND4J_STATUS_OK;
        }


        OP_IMPL(noop, -1, -1, true) {
            // Fastest op ever.
            return ND4J_STATUS_OK;
        }

        OP_IMPL(broadcastgradientargs, 2, 2, true) {

            return ND4J_STATUS_OK;
        }

       //////////////////////////////////////////////////////////////////////////
        /**
         * tensorMmul/tensorDot operation
         * takes 2 ndarrays, and 2 sets of axes
         *
         * Integer argumens map:
         * IArgs[0] - number of axes along for first array
         * IArgs[1]... axes values for first array
         * IArgs[] - number of axes along for second array
         * IArgs[1]... axes values for second array
         */
        CUSTOM_OP_IMPL(tensormmul, 2, 1, false, 0, -1) {
            NDArray<T>* a = INPUT_VARIABLE(0);
            NDArray<T>* b = INPUT_VARIABLE(1);

            NDArray<T>* c = OUTPUT_VARIABLE(0);                // 

            // building axes
            int axe0_size = block.getIArguments()->at(0);
            int axe1_size = block.getIArguments()->at(axe0_size+1);
            std::vector<int> axes_0, axes_1;
            for (int e = 0; e < axe0_size; e++)
                axes_0.emplace_back((int) block.getIArguments()->at(e+1));

            for (int e = 0; e < axe1_size; e++)
                axes_1.emplace_back((int) block.getIArguments()->at(e + axe0_size + 2));

            nd4j_verbose("axe0: %i; axe1: %i;\n", axes_0.size(), axes_1.size());

            nd4j::NDArrayFactory<T>::tensorDot(a, b, c, axes_0, axes_1);

            STORE_RESULT(*c);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(tensordot, tensormmul);


        DECLARE_SHAPE_FN(tensormmul) {               
        
            NDArray<T> *a = INPUT_VARIABLE(0);
            NDArray<T> *b = INPUT_VARIABLE(1);  
            // building axes
            int axe0_size = block.getIArguments()->at(0);
            int axe1_size = block.getIArguments()->at(axe0_size+1);
            std::vector<int> axes_0, axes_1;
            for (int e = 0; e < axe0_size; e++)
                axes_0.emplace_back((int) block.getIArguments()->at(e+1));
            for (int e = 0; e < axe1_size; e++)
                axes_1.emplace_back((int) block.getIArguments()->at(e + axe0_size + 2));        

            // evaluate shapes 
            std::vector<int> permutAt, permutBt, shapeAt, shapeBt;
            std::vector<int> outShape = nd4j::ShapeUtils<T>::evalShapeForTensorDot(a, b, axes_0, axes_1, permutAt, permutBt, shapeAt, shapeBt);
            
            int rank = outShape.size();

            int* newShapeInfo = nullptr; 
            ALLOCATE(newShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int); 
            newShapeInfo[0] = rank;
            copy(outShape.begin(), outShape.end(), newShapeInfo+1);
            shape::updateStrides(newShapeInfo, 'c');

            return new ShapeList(newShapeInfo);
        }

        //////////////////////////////////////////////////////////////////////////
        // test op, non-divergent
        OP_IMPL(testop2i2o, 2, 2, true) {
            nd4j_printf("CPU op used!\n","");
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);

            x->template applyScalar<simdOps::Add<T>>(1.0);
            y->template applyScalar<simdOps::Add<T>>(2.0);

            STORE_2_RESULTS(*x, *y);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TestOp2i2o, testop2i2o);

        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(testcustom, 1, 1, false, 0, -1) {
            auto z = this->getZ(block);

            //new NDArray<T>('c', {100, 100});

            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(testcustom) {
            // this test op will just return back original shape doubled
            int *shapeOf;
            ALLOCATE(shapeOf, block.getWorkspace(), shape::rank(inputShape->at(0)), int);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);

            for (int e = 0; e < shape::rank(inputShape->at(0)); e++)
                shapeOf[e] = inputShape->at(0)[e+1] * 2;


            shape::shapeBuffer(shape::rank(inputShape->at(0)), shapeOf, newShape);

            RELEASE(shapeOf, block.getWorkspace());

            return new ShapeList(newShape);
        }

        REDUCTION_OP_IMPL(testreduction, 1, 1, false, 0, -1) {
            auto z = OUTPUT_VARIABLE(0);

            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }

/////////////////////////////////////////
        OP_IMPL(assign, 2, 1, false) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);

            if (y->isScalar()) {

                x->assign(y->getScalar(0));
            } else {
                REQUIRE_OK(this->validateInputLengthMatch(block));
                REQUIRE_OK(this->validateInputDimensionsMatch(block));

                x->assign(y);
            }


            STORE_RESULT(*x);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(set, assign);
        DECLARE_SYN(copy, assign);


        OP_IMPL(mergemax, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.width();
            NDArray<T> *x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T max = -MAX_FLOAT;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = INPUT_VARIABLE(i);
                    T v = o->getIndexedScalar(e);
                    if (v > max)
                        max = v;
                }
                z->putIndexedScalar(e, max);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(MergeMax, mergemax);

        OP_IMPL(mergemaxindex, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.width();
            NDArray<T> *x = INPUT_VARIABLE(0);
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T max = -MAX_FLOAT;
                Nd4jIndex idx = 0;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = INPUT_VARIABLE(i);
                    T v = o->getIndexedScalar(e);
                    if (v > max) {
                        max = v;
                        idx = i;
                    }
                }
                z->putIndexedScalar(e, idx);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(MergeMaxIndex, mergemaxindex);

        OP_IMPL(mergeadd, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.width();
            NDArray<T> *x = INPUT_VARIABLE(0);
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T sum = (T) 0.0f;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = INPUT_VARIABLE(i);
                    T v = o->getIndexedScalar(e);
                    sum += v;
                }
                z->putIndexedScalar(e, sum);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(mergesum, mergeadd);
        DECLARE_SYN(add_n, mergeadd);

        OP_IMPL(mergeavg, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.width();
            NDArray<T> *x = INPUT_VARIABLE(0);
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T sum = (T) 0.0f;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = INPUT_VARIABLE(i);
                    T v = o->getIndexedScalar(e);
                    sum += v;
                }
                z->putIndexedScalar(e, sum / numArgs);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        CONFIGURABLE_OP_IMPL(clipbyvalue, 1, 1, true, 2, 0) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* output = this->getZ(block);

            input->template applyTransform<simdOps::ClipByValue<T>>(output, block.getTArguments()->data());

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(ClipByValue, clipbyvalue);



        /**
         * scatter update operation
         *
         * IArgs map:
         * IArgs[0] - update operation: 0 - add; 1 - sub; 2 - mul; 3 - div; 4 - rsub; 5 - rdiv; 6 - assign
         * IArgs[1] - number of dimensions
         * IArgs[...] - dimensions
         * IArgs[...] - number of indices
         * IArgs[...] - indices
         *
         * @tparam T
         */
        CONFIGURABLE_OP_IMPL(scatter_update, 2, 1, true, 0, -1) {
            NDArray<T> *operand = INPUT_VARIABLE(0);
            NDArray<T> *updates = INPUT_VARIABLE(1);
            NDArray<T> *z = this->getZ(block);

            int opCode = block.getIArguments()->at(0);
            int dimSize = block.getIArguments()->at(1);
            std::vector<int> tadDimension;
            unsigned long e;
            unsigned long limg = 2 + dimSize;
            for (e = 2; e < limg; e++)
                tadDimension.push_back((int) block.getIArguments()->at(e));

            // increasing counter to skip numIndices
            e++;
            std::vector<int> indices;
            std::vector<int> indicesU;
            int cnt = 0;
            for (; e< block.getIArguments()->size(); e++) {
                indices.push_back((int) block.getIArguments()->at(e));
                indicesU.push_back(cnt++);
            }

            std::unique_ptr<ResultSet<T>> tadsOperand(nd4j::NDArrayFactory<T>::multipleTensorsAlongDimension(operand, indices, tadDimension));
            std::unique_ptr<ResultSet<T>> tadsUpdate(nd4j::NDArrayFactory<T>::multipleTensorsAlongDimension(updates, indicesU, tadDimension));

//#pragma omp parallel for schedule(dynamic) proc_bind(close) shared(tadsOperand, tadsUpdate)
            for (unsigned long x = 0; x < indices.size(); x++) {
                NDArray<T> *tad = tadsOperand->at(x);
                NDArray<T> *tadUpdates = tadsUpdate->at(x);

                if (tad->lengthOf() != tadUpdates->lengthOf())
                    continue;

                switch (opCode) {
                    case 0:
                        tad->template applyPairwiseTransform<simdOps::Add<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 1:
                        tad->template applyPairwiseTransform<simdOps::Subtract<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 2:
                        tad->template applyPairwiseTransform<simdOps::Multiply<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 3:
                        tad->template applyPairwiseTransform<simdOps::Divide<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 4:
                        tad->template applyPairwiseTransform<simdOps::ReverseSubtract<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 5:
                        tad->template applyPairwiseTransform<simdOps::ReverseDivide<T>>(tadUpdates, tad, nullptr);
                        break;
                    case 6:
                        tad->template applyPairwiseTransform<simdOps::Copy<T>>(tadUpdates, tad, nullptr);
                        break;
                    default:
                        continue;
                        //return ND4J_STATUS_BAD_PARAMS;
                }
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(scatterupdate, scatter_update);

//////////////////////////////////////////////////////////////////////////
        CONFIGURABLE_OP_IMPL(relu, 1, 1, true, 1, 0) {
            NDArray<T> *first = INPUT_VARIABLE(0);
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::RELU<T>>(z, &block.getTArguments()->at(0));

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }


//////////////////////////////////////////////////////////////////////////
        OP_IMPL(identity, 1, 1, true) {
            NDArray<T> *first = INPUT_VARIABLE(0);
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::Identity<T>>(z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////		
		OP_IMPL(add, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
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
		OP_IMPL(subtract, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
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
		OP_IMPL(reversesubtract, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
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
        DECLARE_SYN(RSub, reversesubtract);

//////////////////////////////////////////////////////////////////////////		
		OP_IMPL(multiply, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
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
		OP_IMPL(divide, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
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
		OP_IMPL(reversedivide, 2, 1, true) {
            NDArray<T> *x = INPUT_VARIABLE(0);
            NDArray<T> *y = INPUT_VARIABLE(1);
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
        DECLARE_SYN(RDiv, reversedivide);

		//////////////////////////////////////////////////////////////////////////
		// here iArgs is int vector of repeats at the beginning and last element in iArgs is dimension
		CUSTOM_OP_IMPL(repeat, 1, 1, true, 0, -1) {			

			NDArray<T>* x   = INPUT_VARIABLE(0);
            NDArray<T>* ret = OUTPUT_VARIABLE(0);

			x->repeat(block.getIArguments()->back(), *ret);
			STORE_RESULT(*ret);

			return ND4J_STATUS_OK;				
        }
		
        DECLARE_SHAPE_FN(repeat) {                               
            
            NDArray<T>* x   = INPUT_VARIABLE(0);
            std::vector<int>* argumets = block.getIArguments();
            int argsSize = argumets->size();
            int dimension = (*argumets)[argsSize-1];
            std::vector<int> repeats = *argumets;
            repeats.pop_back();
            
            std::vector<int> outShape = ShapeUtils<T>::evalRepeatShape(dimension, repeats, *x);
            int rank = outShape.size();

            int* newShapeInfo = nullptr; 
            ALLOCATE(newShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), int); 
            newShapeInfo[0] = rank;
            copy(outShape.begin(), outShape.end(), newShapeInfo+1);
            shape::updateStrides(newShapeInfo, x->ordering());

            return new ShapeList(newShapeInfo);
        }

		//////////////////////////////////////////////////////////////////////////
		CONFIGURABLE_OP_IMPL(sum, 1, 1, false, 0, -1) {

			std::vector<int> argI = *(block.getIArguments());
			std::vector<int> argItrunc(argI.size()-1);
			for(int i=0; i< (int) argItrunc.size(); ++i)
				argItrunc[i] = argI[i+1];	
			NDArray<T>* x = INPUT_VARIABLE(0);
			NDArray<T> *z = this->getZ(block);

			if((argItrunc.size()==1 && argItrunc[0]==INT_MAX) || argItrunc.size()==0) {
				z->putScalar(0, 0, x->template reduceNumber<simdOps::Sum<T>>(nullptr));
				STORE_RESULT(*z);
			}
			else {
				z = x->template reduceAlongDimension<simdOps::Sum<T>>(argItrunc);
				STORE_RESULT(*z);
			}

			return ND4J_STATUS_OK;
		}
    }
}

#endif //LIBND4J_PARITY_OPS_H

