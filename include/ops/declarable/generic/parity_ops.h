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
#include <ops/declarable/declarable_ops.h>
#include <NDArrayFactory.h>
#include <ops/declarable/generic/third_party.h>
#include <ops/declarable/generic/convo/convo_ops.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {

//////////////////////////////////////////////////////////////////////////
        DECLARE_CUSTOM_OP(concat, -1, 1, false, 0, 1){
            // do something here{

            int _dimension = block.getIArguments()->at(0);

            // we want to ensure that all
            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            NDArray<T> *output = this->getZ(block);

            std::unique_ptr<Nd4jPointer> buffers(new Nd4jPointer[block.getVariables().size()]);
            std::unique_ptr<Nd4jPointer> shapes(new Nd4jPointer[block.getVariables().size()]);

            buffers.get()[0] = (Nd4jPointer) first->getBuffer();
            shapes.get()[0] = (Nd4jPointer) first->getShapeInfo();

            if (debug && verbose) {
                printf("Shape %i: ", 0);
                shape::printShapeInfoLinear((int *) shapes.get()[0]);
            }

            for (int e = 1; e < (int) block.getVariables().size(); e++) {
                Variable<T> *var = block.getVariables().at(e);

                buffers.get()[e] = (Nd4jPointer) var->getNDArray()->getBuffer();
                shapes.get()[e] = (Nd4jPointer) var->getNDArray()->getShapeInfo();

                if (debug && verbose) {
                    printf("Shape %i: ", e);
                    shape::printShapeInfoLinear((int *) shapes.get()[e]);
                }
            }
            if (debug && verbose)
                fflush(stdout);

            concatCpuGeneric(_dimension, block.getVariables().size(), buffers.get(), shapes.get(), output->getBuffer(), output->getShapeInfo());

            STORE_RESULT(*output);

            if (debug && verbose)
                output->printShapeInfo("Concat result shape");

            return ND4J_STATUS_OK;
        }
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
        DECLARE_OP(biasadd, 2, 1, true) {
            //REQUIRE_OK(this->validateInput2D(block));

            NDArray<T> *input = block.getVariables().at(0)->getNDArray();
            NDArray<T> *bias = block.getVariables().at(1)->getNDArray();

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
        DECLARE_CUSTOM_OP(matmul, 2, 1, false, -2, 0) {
            // FIXME: we might want to have gemv/dot fallback here
            REQUIRE_OK(this->validateInput2D(block));


            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();
            NDArray<T> *z = this->getZ(block);

            T alpha = (T) 1.0f;
            T beta = (T) 0.0f;
            if (block.getTArguments()->size() > 0)
                alpha = block.getTArguments()->at(0);

            if (block.getTArguments()->size() > 1)
                beta = block.getTArguments()->at(1);


            if (x->isMatrix() && y->isVector()) {
                // gemv
                nd4j::NDArrayFactory::mmulHelper<T>(x, y, z, alpha, beta);

            } else if (x->isVector() && y->isMatrix()) {
                // gemm
                nd4j::NDArrayFactory::mmulHelper<T>(x, y, z, alpha, beta);
            }  else if (x->isVector() && y->isVector()) {
                // dot
                nd4j::NDArrayFactory::mmulHelper<T>(x, y, z, alpha, beta);
            } else if (x->isMatrix() && y->isMatrix()) {
                // gemm
                nd4j::NDArrayFactory::mmulHelper<T>(x, y, z, alpha, beta);
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

//////////////////////////////////////////////////////////////////////////
        DECLARE_CUSTOM_OP(lrn, 1, 3, true, 4, 0) {
            // LocalResponseNormalization

            NDArray<T>* input = block.getVariables().at(0)->getNDArray();
            NDArray<T>* z = this->getZ(block);
            NDArray<T>* unitScale = this->getZ(block, 1);
            NDArray<T>* scale = this->getZ(block, 2);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input rank of 4 expected, but got %i instead", input->rankOf());

            T alpha = block.getTArguments()->at(0);
            T beta = block.getTArguments()->at(1);
            T bias = block.getTArguments()->at(2);
            T depth = block.getTArguments()->at(3);

            int halfDepth = (int) (depth / (T) 2.f);

            const int channel =  input->sizeAt(1);

            auto activitySqr = NDArrayFactory::createUninitialized<T>(input);
            input->template applyPairwiseTransform<simdOps::Multiply<T>>(input, activitySqr, nullptr);
            auto sumPart = activitySqr->dup('c');

            for (int i = 1; i < halfDepth + 1; i++) {
                IndicesList indA({NDIndex::all(), NDIndex::interval(i, channel), NDIndex::all(), NDIndex::all()});
                IndicesList indB({NDIndex::all(), NDIndex::interval(0, channel - i), NDIndex::all(), NDIndex::all()});

                std::unique_ptr<NDArray<T>> tmp(sumPart->subarray(indA));
                std::unique_ptr<NDArray<T>> addVal(activitySqr->subarray(indB));

                tmp.get()->template applyPairwiseTransform<simdOps::Add<T>>(addVal.get(), nullptr);


                std::unique_ptr<NDArray<T>> tmp2(sumPart->subarray(indB));
                std::unique_ptr<NDArray<T>> addVal2(activitySqr->subarray(indA));

                tmp2.get()->template applyPairwiseTransform<simdOps::Add<T>>(addVal2.get(), nullptr);
            }

            /*
             *  // taken from java
                unitScale = sumPart.mul(alpha).addi(k).leverageTo(ComputationGraph.workspaceExternal);
                // y = x * unitScale**-beta
                scale = Transforms.pow(unitScale, -beta).leverageTo(ComputationGraph.workspaceExternal);
                activations = input.mul(scale).leverageTo(ComputationGraph.workspaceExternal);
             */

            sumPart->template applyScalar<simdOps::Multiply<T>>(alpha, unitScale, nullptr);
            unitScale->template applyScalar<simdOps::Add<T>>(bias);

            T p = -beta;
            unitScale->template applyTransform<simdOps::Pow<T>>(scale, &p);
            input->template applyPairwiseTransform<simdOps::Multiply<T>>(scale, z, nullptr);

            STORE_3_RESULTS(*z, *unitScale, *scale);

            delete activitySqr;
            delete sumPart;

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(LRN, lrn);

        DECLARE_SHAPE_FN(lrn) {
            int *inp = inputShape->at(0);

            auto shapeList = new ShapeList();
            for(int e = 0; e < 3; e++) {
                int *newShape;
                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inp), int);
                memcpy(newShape, inp, shape::shapeInfoByteLength(inp));

                shapeList->push_back(newShape);
            }

            return shapeList;
        }

///////////////////////
        /**
         * uniform distribution
         * takes 1 ndarray
         *
         * T argumens map:
         * TArgs[0] - min for rng
         * TArgs[1] - max for rng
         */
        DECLARE_CONFIGURABLE_OP(randomuniform, 1, 1, true, 2, 0) {
            // uniform distribution
            auto rng = block.getRNG();

            if (rng == nullptr)
                return ND4J_STATUS_BAD_RNG;

            if (block.getTArguments()->size() != 2)
                return ND4J_STATUS_BAD_ARGUMENTS;

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            auto z = x;
            if (!block.isInplace())
                z = new NDArray<T>(x);

            functions::random::RandomFunction<T>::template execTransform<randomOps::UniformDistribution<T>>(block.getRNG(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data());

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }


        DECLARE_OP(floor, 1, 1, true) {
            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::Floor<T>>(z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        DECLARE_OP(realdiv, 2, 1, true) {
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

        DECLARE_DIVERGENT_OP(noop, -1, -1, true) {
            // Fastest op ever.
            return ND4J_STATUS_OK;
        }

        DECLARE_OP(broadcastgradientargs, 2, 2, true) {

            return ND4J_STATUS_OK;
        }

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
        DECLARE_CONFIGURABLE_OP(tensormmul, 2, 1, false, 0, -1) {
            NDArray<T> *a = block.getVariables().at(0)->getNDArray();
            NDArray<T> *b = block.getVariables().at(1)->getNDArray();

            // building axes
            int axe0_size = block.getIArguments()->at(0);
            int axe1_size = block.getIArguments()->at(axe0_size+1);
            std::vector<int> axes_0, axes_1;
            for (int e = 0; e < axe0_size; e++)
                axes_0.push_back((int) block.getIArguments()->at(e+1));

            for (int e = 0; e < axe1_size; e++)
                axes_1.push_back((int) block.getIArguments()->at(e + axe0_size + 2));

            nd4j_verbose("axe0: %i; axe1: %i;\n", axes_0.size(), axes_1.size());

            // validating axes
            int validationLength = nd4j::math::nd4j_min<int>(axe0_size, axe1_size);
            for (int i = 0; i < validationLength; i++) {
                if (a->sizeAt(axes_0[i]) != b->sizeAt(axes_1[i]))
                    throw "Size of the given axes at each dimension must be the same size.";
                if (axes_0[i] < 0)
                    axes_0[i] += a->rankOf();
                if (axes_1[i] < 0)
                    axes_1[i] += b->rankOf();
            }


            std::vector<int> list_A, list_B;
            for (int i = 0; i < a->rankOf(); i++)
                if (std::find(axes_0.begin(), axes_0.end(), i) == axes_0.end())
                    list_A.push_back(i);

            for (int i = 0; i < b->rankOf(); i++)
                if (std::find(axes_1.begin(), axes_1.end(), i) == axes_1.end())
                    list_B.push_back(i);


            std::vector<int> newAxesA(list_A);
            std::vector<int> newAxesB(list_B);
            for (auto v: axes_0)
                newAxesA.push_back(v);

            for (auto v: axes_1)
                newAxesB.push_back(v);

            int n2 = 1;
            int aLength = nd4j::math::nd4j_min<int>(a->rankOf(), axes_0.size());
            for (int i = 0; i < aLength; i++)
                n2 *= a->sizeAt(axes_0[i]);

            std::vector<int> newShapeA({-1, n2});
            std::vector<int> oldShapeA;
            if (list_A.size() == 0) {
                oldShapeA.push_back(1);
            } else {
                for (auto v: list_A)
                    oldShapeA.push_back(v);

                for (int i = 0; i < (int) oldShapeA.size(); i++)
                    oldShapeA[i] = a->sizeAt(oldShapeA[i]);
            }

            int n3 = 1;
            int bNax = nd4j::math::nd4j_min<int>(b->rankOf(), axes_1.size());
            for (int i = 0; i < bNax; i++)
                n3 *= b->sizeAt(axes_1[i]);

            std::vector<int> newShapeB({n3, -1});
            std::vector<int> oldShapeB;
            if (list_B.size() == 0) {
                oldShapeB.push_back(1);
            } else {
                for (auto v: list_B)
                    oldShapeB.push_back(v);
                for (int i = 0; i < (int) oldShapeB.size(); i++)
                    oldShapeB[i] = b->sizeAt(oldShapeB[i]);
            }

            // FIXME: when we'll bring proper gemm, this probably won't be needed
            auto aT = a->ordering() == 'c' ? a : a->dup('c');
            auto bT = b->ordering() == 'c' ? b : b->dup('c');

            aT->permutei(newAxesA);
            aT->reshapei('c', newShapeA);

            bT->permutei(newAxesB);
            bT->reshapei('f', newShapeB);

            auto c = nd4j::NDArrayFactory::mmulHelper<T>(aT, bT, nullptr, 1.0, 0.0);

            std::vector<int> aPlusB(oldShapeA);
            for (auto v: oldShapeB)
                aPlusB.push_back(v);

            c->reshapei('f', aPlusB);

            STORE_RESULT(*c);

            if (aT != a)
                delete aT;

            if (bT != b)
                delete bT;

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(tensordot, tensormmul);


        // test op, non-divergent
        DECLARE_OP(testop2i2o, 2, 2, true) {
            nd4j_printf("CPU op used!\n","");
            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();

            x->template applyScalar<simdOps::Add<T>>(1.0);
            y->template applyScalar<simdOps::Add<T>>(2.0);

            STORE_2_RESULTS(*x, *y);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TestOp2i2o, testop2i2o);


        DECLARE_REDUCTION_OP(testreduction, 1, 1, false, 0, -1) {
            auto z = this->getZ(block);

            STORE_RESULT(*z);
            return ND4J_STATUS_OK;
        }

/////////////////////////////////////////
        DECLARE_CUSTOM_OP(testcustom, 1, 1, false, 0, -1) {
            auto z = this->getZ(block);

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

/////////////////////////////////////////
        DECLARE_OP(assign, 2, 1, false) {
            REQUIRE_OK(this->validateInputLengthMatch(block));
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();

            x->assign(y);

            STORE_RESULT(*x);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(set, assign);
        DECLARE_SYN(copy, assign);


        DECLARE_OP(mergemax, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.getVariables().size();
            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T max = -MAX_FLOAT;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = block.getVariables().at(i)->getNDArray();
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

        DECLARE_OP(mergemaxindex, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.getVariables().size();
            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T max = -MAX_FLOAT;
                Nd4jIndex idx = 0;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = block.getVariables().at(i)->getNDArray();
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

        DECLARE_OP(mergeadd, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.getVariables().size();
            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T sum = (T) 0.0f;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = block.getVariables().at(i)->getNDArray();
                    T v = o->getIndexedScalar(e);
                    sum += v;
                }
                z->putIndexedScalar(e, sum);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(mergesum, mergeadd);

        DECLARE_OP(mergeavg, -1, 1, false) {
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

            Nd4jIndex numArgs = block.getVariables().size();
            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            auto z = this->getZ(block);


#pragma omp parallel for proc_bind(close)
            for (Nd4jIndex e = 0; e < x->lengthOf(); e++) {
                T sum = (T) 0.0f;
                for (int i = 0; i < numArgs; i++){
                    NDArray<T> *o = block.getVariables().at(i)->getNDArray();
                    T v = o->getIndexedScalar(e);
                    sum += v;
                }
                z->putIndexedScalar(e, sum / numArgs);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        DECLARE_CONFIGURABLE_OP(clipbyvalue, 1, 1, true, 2, 0) {
            NDArray<T>* input = block.getVariables().at(0)->getNDArray();
            NDArray<T>* output = this->getZ(block);

            input->template applyTransform<simdOps::ClipByValue<T>>(output, block.getTArguments()->data());

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(ClipByValue, clipbyvalue);



//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(softmax, 2, 1, false) {
            // YaY
            return ND4J_STATUS_OK;
        }


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
        DECLARE_CONFIGURABLE_OP(scatter_update, 2, 1, true, 0, -1) {
            NDArray<T> *operand = block.getVariables().at(0)->getNDArray();
            NDArray<T> *updates = block.getVariables().at(1)->getNDArray();
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

            std::unique_ptr<ArrayList<T>> tadsOperand(nd4j::NDArrayFactory::multipleTensorsAlongDimension(operand, indices, tadDimension));
            std::unique_ptr<ArrayList<T>> tadsUpdate(nd4j::NDArrayFactory::multipleTensorsAlongDimension(updates, indicesU, tadDimension));

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
        DECLARE_CONFIGURABLE_OP(relu, 1, 1, true, 1, 0) {
            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::RELU<T>>(z, &block.getTArguments()->at(0));

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }


//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(identity, 1, 1, true) {
            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::Identity<T>>(z, nullptr);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

//////////////////////////////////////////////////////////////////////////		
		DECLARE_OP(add, 2, 1, true) {
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
		DECLARE_OP(reversesubtract, 2, 1, true) {
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
        DECLARE_SYN(RSub, reversesubtract);

//////////////////////////////////////////////////////////////////////////		
		DECLARE_OP(multiply, 2, 1, true) {
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
		DECLARE_OP(reversedivide, 2, 1, true) {
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
        DECLARE_SYN(RDiv, reversedivide);

//////////////////////////////////////////////////////////////////////////
		DECLARE_OP(reshapeas, 2, 1, true) {
            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();	
			
			NDArray<T>* z = this->getZ(block);
			std::vector<int> shapeNew(y->shapeOf(), y->shapeOf() + y->rankOf());
			char order = y->ordering();
			
			if (x->reshapei(order, shapeNew)) {
				*z = *x;
				STORE_RESULT(*z);
				return ND4J_STATUS_OK;				
			}			
			
			return ND4J_STATUS_BAD_INPUT;
        }
        DECLARE_SYN(shape, reshapeas);

		//////////////////////////////////////////////////////////////////////////
		// here iArgs is vector with shape dimensions at the beginning and last element in iArgs is order
		DECLARE_CUSTOM_OP(reshape, 1, 1, true, 0, -1) {
			std::vector<int>* argumets = block.getIArguments();
			int argsSize = argumets->size();

            REQUIRE_TRUE(argsSize >= 3, 0, "Reshape arguments should have order and at least 2 dimensions");

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();

			char order = (char)(*argumets)[0];
            if (order != 'c' && order != 'f')
                order = x->ordering();

			std::vector<int> shapeNew;

            for (int e = 1; e < argumets->size(); e++)
                shapeNew.push_back((int) argumets->at(e));

            nd4j::Logger::printv("shapeNew: ", shapeNew);

			if(block.isInplace()) {
				if (x->reshapei(order, shapeNew)) {
					STORE_RESULT(*x);
                    x->printShapeInfo("New shape");
					return ND4J_STATUS_OK;				
				}
			}
			else {
				auto ret = new NDArray<T>(*x);
				if (ret->reshapei(order, shapeNew)) {
					STORE_RESULT(*ret);
                    ret->printShapeInfo("New shape");
					return ND4J_STATUS_OK;				
				}
			}			
			return ND4J_STATUS_BAD_INPUT;
        }
        DECLARE_SHAPE_FN(reshape) {
            int *inp = inputShape->at(0);

            std::vector<int>* arguments = block.getIArguments();

            char order = (char)(*arguments)[0];
            if (order != 'c' && order != 'f')
                order = shape::order(inp);

            std::vector<int> shapeNew;

            for (int e = 1; e < arguments->size(); e++)
                shapeNew.push_back((int) arguments->at(e));

            int* newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength((int) shapeNew.size()), int);

            int numberNegativesOnes = 0;

            int* shape_ = shapeNew.data();
            for (int i = 0; i < (int) shapeNew.size(); i++) {
                if (shapeNew[i] < 0) {
                    if (numberNegativesOnes >= 1)
                        throw "Only one dimension can be negative ones";

                    numberNegativesOnes++;

                    int shapeLength = 1;
                    for (int j = 0; j < (int) shapeNew.size(); j++)
                        if (shape_[j] >= 1)
                            shapeLength *= shape_[j];

                    // FIXME: use workspace here
                    int realShape = nd4j::math::nd4j_abs<int>((int) shape::length(inp) / shapeLength);
                    int* thisNewShape = new int[shapeNew.size()];

                    for (int j = 0; j < (int) shapeNew.size(); j++) {
                        if (i != j) {
                            thisNewShape[j] = shape_[j];
                        } else
                            thisNewShape[j] = realShape;
                    }

                    shape_ = thisNewShape;
                    break;
                }
            }

            for (int e = 0; e < (int) shapeNew.size(); e++) {
                shapeNew[e] = shape_[e];
            }

            if (numberNegativesOnes > 0)
                delete[] shape_;

            newShape[0] = shapeNew.size();
            int cnt = 1;
            for (auto v: shapeNew)
                newShape[cnt++] = v;

            shape::updateStrides(newShape, order);

            return new ShapeList(newShape);
        }

		//////////////////////////////////////////////////////////////////////////
		// here iArgs is int vector of repeats at the beginning and last element in iArgs is dimension
		DECLARE_CONFIGURABLE_OP(repeat, 1, 1, true, 0, -1) {
			std::vector<int>* argumets = block.getIArguments();
			int argsSize = argumets->size();
			int dimension = (*argumets)[argsSize-1];
			std::vector<int> repeats = *argumets;
			repeats.pop_back();

			NDArray<T> *x = block.getVariables().at(0)->getNDArray();            			
			NDArray<T>* ret = x->repeat(dimension, repeats);
			STORE_RESULT(*ret);

			return ND4J_STATUS_OK;				
        }
		
		//////////////////////////////////////////////////////////////////////////
		DECLARE_OP(transpose, 1, 1, true) {
			NDArray<T> *x = block.getVariables().at(0)->getNDArray();            			
			
			if(block.isInplace()) {
				x->transposei();
				STORE_RESULT(*x);
			}
			else {
				NDArray<T>* ret = x->transpose();
				STORE_RESULT(*ret);
			}
			return ND4J_STATUS_OK;
        }

		//////////////////////////////////////////////////////////////////////////
		// here iArgs is int vector of ordered set of dimensions to be permuted
		DECLARE_CONFIGURABLE_OP(permute, 1, 1, true, 0, -1) {
			std::vector<int>* argumets = block.getIArguments();
			NDArray<T> *x = block.getVariables().at(0)->getNDArray();            			
			
			if(block.isInplace()) {		// in-place
				x->permutei(*argumets);				
				STORE_RESULT(*x);
			}
			else {						// not-in-place
				NDArray<T>* ret = x->permute(*argumets);
				STORE_RESULT(*ret);
			}
			return ND4J_STATUS_OK;
        }

		//////////////////////////////////////////////////////////////////////////
		DECLARE_CONFIGURABLE_OP(sum, 1, 1, false, 0, -1) {

			std::vector<int> argI = *(block.getIArguments());
			std::vector<int> argItrunc(argI.size()-1);
			for(int i=0; i<argItrunc.size(); ++i)
				argItrunc[i] = argI[i+1];	
			NDArray<T>* x = block.getVariables().at(0)->getNDArray();
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

