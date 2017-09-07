//
// These ops are provided for features parity with TF
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_PARITY_OPS_H
#define LIBND4J_PARITY_OPS_H

#include <op_boilerplate.h>
#include <memory>
#include <shape.h>
#include <ops/ops.h>
#include <loops/random.h>
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
            NDArray<T> *z = nullptr;

            if (x->isMatrix() && y->isVector()) {
                // gemv
                z = NDArray<T>::mmulHelper(x, y);
            } else if (x->isVector() && y->isMatrix()) {
                // gemm
                z = NDArray<T>::mmulHelper(x, y);
            }  else if (x->isVector() && y->isVector()) {
                // dot
                z = NDArray<T>::mmulHelper(x, y);
            } else if (x->isMatrix() && y->isMatrix()) {
                // gemm
                z = NDArray<T>::mmulHelper(x, y);
            } else if (x->isVector() && y->isScalar()) {
                // elementwise mul
                z = this->getZ(block);

                x->template applyScalar<simdOps::Multiply<T>>(y->getScalar(0), z, nullptr);
             } else if (x->isScalar() && y->isVector()) {
                // elementwise mul, reverse op
                z = this->getZ(block, 1);

                y->template applyScalar<simdOps::Multiply<T>>(x->getScalar(0), z, nullptr);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(mMul, matMul);
        DECLARE_SYN(mmul, matMul);
        DECLARE_SYN(gemm, matMul);
        DECLARE_SYN(gemv, matMul);
        DECLARE_SYN(dot, matMul);

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
        /**
         * uniform distribution
         * takes 1 ndarray
         *
         * T argumens map:
         * TArgs[0] - min for rng
         * TArgs[1] - max for rng
         */
        DECLARE_CONFIGURABLE_OP(randomuniform, 1, 1, true, 2, 0) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

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

            functions::random::RandomFunction<T>::template execTransform<randomOps::UniformDistribution<T>>(block.getRNG(), z->_buffer, z->_shapeInfo, block.getTArguments()->data());

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }


        DECLARE_OP(floor, 1, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::Floor<T>>(z, nullptr);

            STORE_RESULT(*z);

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
            REQUIRE_OK(this->validateNonEmptyInput(block));

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

                for (int i = 0; i < oldShapeA.size(); i++)
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
                for (int i = 0; i < oldShapeB.size(); i++)
                    oldShapeB[i] = b->sizeAt(oldShapeB[i]);
            }

            // FIXME: when we'll bring proper gemm, this probably won't be needed
            auto aT = a->ordering() == 'c' ? a : a->dup('c');
            auto bT = b->ordering() == 'c' ? b : b->dup('c');

            aT->permutei(newAxesA);
            aT->reshape('c', newShapeA);

            bT->permutei(newAxesB);
            bT->reshape('f', newShapeB);

            auto c = NDArray<T>::mmulHelper(aT, bT);

            std::vector<int> aPlusB(oldShapeA);
            for (auto v: oldShapeB)
                aPlusB.push_back(v);

            c->reshape('f', aPlusB);

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
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();

            x->template applyScalar<simdOps::Add<T>>(1.0);
            y->template applyScalar<simdOps::Add<T>>(2.0);

            STORE_2_RESULTS(*x, *y);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TestOp2i2o, testop2i2o);



        DECLARE_OP(assign, 2, 1, false) {
            REQUIRE_OK(this->validateNonEmptyInput(block));
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
            REQUIRE_OK(this->validateNonEmptyInput(block));
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
            REQUIRE_OK(this->validateNonEmptyInput(block));
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
            REQUIRE_OK(this->validateNonEmptyInput(block));
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
            REQUIRE_OK(this->validateNonEmptyInput(block));
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

        /**
         * Upsampling implementation, based on pytorch
         *
         * IArgs map:
         * IArgs[0] - scale factor
         */
        DECLARE_CONFIGURABLE_OP(upsampling, 1, 1, false, 0, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T>* input = block.getVariables().at(0)->getNDArray();
            NDArray<T>* output = this->getZ(block);
            int scale_factor = block.getIArguments()->at(0);

            int inputHeight = input->sizeAt(2);
            int inputWidth  = input->sizeAt(3);

            int dW = scale_factor;
            int dH = scale_factor;
            int outputHeight = inputHeight * scale_factor;
            int outputWidth = inputWidth * scale_factor;
            int xDim = input->rankOf() - 2;
            int yDim = input->rankOf() - 1;

            int osz0 = output->sizeAt(0);
            int osz1 = output->sizeAt(1);
            int osz2 = output->sizeAt(2);
            int osz3 = osz3 = output->sizeAt(3);

            int i0, i1, i2, i3, isrc, idst;
            int iout[4];  // Output indices
            int iin[4];  // Input indices

            for (i0 = 0; i0 < osz0; i0++) {
                iout[0] = i0;
                iin[0] = i0;
                for (i1 = 0; i1 < osz1; i1++) {
                    iout[1] = i1;
                    iin[1] = i1;
                    for (i2 = 0; i2 < osz2; i2++) {
                        iout[2] = i2;
                        iin[2] = i2;
                        for (i3 = 0; i3 < osz3; i3++) {
                            iout[3] = i3;
                            iin[3] = i3;

                            // set the indices for the upsampled dimensions
                            iin[xDim] = iout[xDim] / dW;
                            iin[yDim] = iout[yDim] / dH;

                            idst = i0 * output->stridesOf()[0] + i1 * output->stridesOf()[1] + i2 * output->stridesOf()[2];
                            isrc = iin[0] * input->stridesOf()[0] + iin[1] * input->stridesOf()[1] + iin[2] * input->stridesOf()[2];

                            // in our case rank of input is always 4
                            idst += i3 * output->stridesOf()[3];
                            isrc += iin[3]* input->stridesOf()[3];


                            output->_buffer[idst] = input->_buffer[isrc];
                        }
                    }
                }
            }

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }

        /**
         * Upsampling backprop implementation, based on pytorch
         *
         * Input[0] - preoutput result
         * Input[1] - gradients from next node/layer
         *
         * Output[0] - gradient for this node
         *
         * IArgs map:
         * IArgs[0] - scale factor
         */
        DECLARE_CONFIGURABLE_OP(upsampling_bp, 2, 1, false, 0, 1) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T>* input = block.getVariables().at(0)->getNDArray();
            NDArray<T>* gradientNext = block.getVariables().at(1)->getNDArray();
            NDArray<T>* output = this->getZ(block);
            int scale_factor = block.getIArguments()->at(0);


            int dW = scale_factor;
            int dH = scale_factor;
            int xDim = output->rankOf() - 2;
            int yDim = output->rankOf() - 1;

            // dims
            int idim = output->rankOf();  // Guaranteed to be between 3 and 5
            int isz0 = output->sizeAt(0);
            int isz1 = output->sizeAt(1);
            int isz2 = output->sizeAt(2);
            int isz3 = 1;
            if (idim > 3) {
                isz3 = output->sizeAt(3);
            }

            output->assign(0.0);

            // perform the upsampling
            int i0, i1, i2, i3, isrc, idst, x, y;
            int iin[4];  // Input indices
            int iout[4];  // Output indices

            for (i0 = 0; i0 < isz0; i0++) {
                iin[0] = i0;
                iout[0] = i0;
                for (i1 = 0; i1 < isz1; i1++) {
                    iin[1] = i1;
                    iout[1] = i1;
                    for (i2 = 0; i2 < isz2; i2++) {
                        iin[2] = i2;
                        iout[2] = i2;
                        for (i3 = 0; i3 < isz3; i3++) {
                            iin[3] = i3;
                            iout[3] = i3;

                            idst = i0 * output->stridesOf()[0] + i1 * output->stridesOf()[1] + i2 * output->stridesOf()[2];
                            if (idim > 3) {
                                idst += i3 * output->stridesOf()[3];
                            }

                            // Now accumulate the gradients from gradOutput
                            for (y = 0; y < dH; y++) {
                                for (x = 0; x < dW; x++) {
                                    iout[xDim] = dW * iin[xDim] + x;
                                    iout[yDim] = dH * iin[yDim] + y;
                                    isrc = iout[0] * gradientNext->stridesOf()[0] + iout[1] * gradientNext->stridesOf()[1] + iout[2] * gradientNext->stridesOf()[2];
                                    if (idim > 3) {
                                        isrc += iout[3] * gradientNext->stridesOf()[3];
                                    }
                                    output->_buffer[idst] += gradientNext->_buffer[isrc];
                                }
                            }
                        }
                    }
                }
            }

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }

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
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *operand = block.getVariables().at(0)->getNDArray();
            NDArray<T> *updates = block.getVariables().at(1)->getNDArray();
            NDArray<T> *z = this->getZ(block);

            int opCode = block.getIArguments()->at(0);
            int dimSize = block.getIArguments()->at(1);
            std::vector<int> tadDimension;
            unsigned long e;
            for (e = 2; e < 2 + dimSize; e++)
                tadDimension.push_back((int) block.getIArguments()->at(e));

            int numIndices = block.getIArguments()->at(e++);
            std::vector<int> indices;
            for (; e< block.getIArguments()->size(); e++)
                indices.push_back((int) block.getIArguments()->at(e));

            for (unsigned long x = 0; x < indices.size(); x++) {
                NDArray<T> *tad = operand->tensorAlongDimension(indices.at(x), tadDimension);
                NDArray<T> *tadUpdates = updates->tensorAlongDimension(x, tadDimension);

                if (tad->lengthOf() != tadUpdates->lengthOf())
                    return ND4J_STATUS_BAD_DIMENSIONS;

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
                        return ND4J_STATUS_BAD_PARAMS;
                }

                delete tad;
                delete tadUpdates;
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(scatterupdate, scatter_update);


//////////////////////////////////////////////////////////////////////////
        DECLARE_CONFIGURABLE_OP(relu, 1, 1, true, 1, 0) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::RELU<T>>(z, &block.getTArguments()->at(0));

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }


//////////////////////////////////////////////////////////////////////////
        DECLARE_OP(identity, 1, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *first = block.getVariables().at(0)->getNDArray();
            auto z = this->getZ(block);

            first->template applyTransform<simdOps::Identity<T>>(z, nullptr);

            STORE_RESULT(*z);

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
		DECLARE_OP(reshapeas, 2, 1, true) {
            REQUIRE_OK(this->validateNonEmptyInput(block));

            NDArray<T> *x = block.getVariables().at(0)->getNDArray();
            NDArray<T> *y = block.getVariables().at(1)->getNDArray();	
			
			std::vector<int> shapeNew(y->shapeOf(), y->shapeOf() + y->rankOf());
			char order = y->ordering();
			
			if (x->reshape(order, shapeNew)) {
				STORE_RESULT(*x);
				return ND4J_STATUS_OK;				
			}			
			
			return ND4J_STATUS_BAD_INPUT;
        }
        DECLARE_SYN(shape, reshapeas);

		//////////////////////////////////////////////////////////////////////////
		// here iArgs is vector with shape dimensions at the beginning and last element in iArgs is order
		DECLARE_CONFIGURABLE_OP(reshape, 1, 1, true, 0, -1) {
			REQUIRE_OK(this->validateNonEmptyInput(block));			
			std::vector<int>* argumets = block.getIArguments();
			int argsSize = argumets->size();
			char order = (*argumets)[argsSize-1];
			std::vector<int> shapeNew = *argumets;
			shapeNew.pop_back();

			NDArray<T> *x = block.getVariables().at(0)->getNDArray();            			
			if(block.isInplace()) {
				if (x->reshape(order, shapeNew)) {
					STORE_RESULT(*x);
					return ND4J_STATUS_OK;				
				}
			}
			else {
				auto ret = new NDArray<T>(*x);
				if (ret->reshape(order, shapeNew)) {
					STORE_RESULT(*ret);
					return ND4J_STATUS_OK;				
				}
			}			
			return ND4J_STATUS_BAD_INPUT;
        }

		//////////////////////////////////////////////////////////////////////////
		// here iArgs is int vector of repeats at the beginning and last element in iArgs is dimension
		DECLARE_CONFIGURABLE_OP(repeat, 1, 1, true, 0, -1) {
			REQUIRE_OK(this->validateNonEmptyInput(block));			
			
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
		DECLARE_CONFIGURABLE_OP(transpose, 1, 1, true, 0, -1) {
			REQUIRE_OK(this->validateNonEmptyInput(block));			

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
			REQUIRE_OK(this->validateNonEmptyInput(block));						
			
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
    }
}

#endif //LIBND4J_PARITY_OPS_H

