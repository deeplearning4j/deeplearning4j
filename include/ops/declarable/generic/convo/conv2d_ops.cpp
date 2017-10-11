//
// 3D convolutions are based on pytorch - https://github.com/pytorch/pytorch
//

#ifndef LIBND4J_CONVO_OPS_H
#define LIBND4J_CONVO_OPS_H

#include <op_boilerplate.h>

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpRegistrator.h>
#include <declarable/generic/helpers/convolutions.h>



namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(conv2d, 2, 1, false, 0, 9) {
            // basically im2col + gemm
            NDArray<T>* input = block.getVariables().at(0)->getNDArray();
            NDArray<T>* weights = block.getVariables().at(1)->getNDArray();
            NDArray<T>* bias = nullptr;

            if (block.getVariables().size() == 3)
                bias = block.getVariables().at(2)->getNDArray();

            const int kY = block.getIArguments()->at(0);
            const int kX = block.getIArguments()->at(1);
            const int sY = block.getIArguments()->at(2);
            const int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            const int dY = block.getIArguments()->at(6);
            const int dX = block.getIArguments()->at(7);
            const bool isSameMode = block.getIArguments()->at(8) != 0;

            int oY = 0;
            int oX = 0;

            const int batchSize = input->shapeOf()[0];
            const int outDepth = weights->shapeOf()[0];
            const int inDepth = weights->shapeOf()[1];
            const int inY = input->shapeOf()[2];
            const int inX = input->shapeOf()[3];

            REQUIRE_TRUE(weights->shapeOf()[2] == kY && weights->shapeOf()[3] == kX, 0, "Kernels should have dimensions of [%i, %i], but got [%i, %i] instead", kY, kX, weights->sizeAt(2), weights->sizeAt(3));

            ConvolutionUtils<T>::calcOutHWpool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode) {
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);
            }

            NDArray<T>* output = this->getZ(block);

            Nd4jIndex prod = batchSize * outDepth * oY * oX;
            REQUIRE_TRUE(output->lengthOf() == prod, 0, "Z should have total length of %i, but got %i instead", prod, output->lengthOf());

            //INDArray col = Nd4j.createUninitialized(new int[] {miniBatch, outH, outW, inDepth, kH, kW}, 'c');
            std::unique_ptr<NDArray<T>> col(new NDArray<T>('c', {batchSize, oY, oX, inDepth, kY, kX}));
            std::unique_ptr<NDArray<T>> col2(col.get()->permute({0, 3, 4, 5, 1, 2}));

//            std::unique_ptr<NDArray<T>> col2(new NDArray<T>('c', {batchSize, inDepth, kY, kX, oY, oX }));

            std::unique_ptr<T> extrasIm2Col(new T[9]{(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

            input->template applyTransform<simdOps::Im2col<T>>(col2.get(), extrasIm2Col.get());

            std::unique_ptr<NDArray<T>> im2col2d(col->reshape('c', {batchSize * oY * oX, inDepth * kY * kX}));
            std::unique_ptr<NDArray<T>> permutedW(weights->permute({3, 2, 1, 0}));
            std::unique_ptr<NDArray<T>> reshapedW(permutedW.get()->reshape('f', {kX * kY * inDepth, outDepth}));

            output->reshapei('f', {im2col2d.get()->rows(), reshapedW.get()->columns()});

            auto res = NDArrayFactory<T>::mmulHelper(im2col2d.get(), reshapedW.get(), output, 1.0, 0.0);

            // bias addition is optional
            if (bias != nullptr) {
                if (!bias->isRowVector())
                    bias->transposei();

                // FIXME: do we really want transposei() above?
                output->addiRowVector(bias);
            }

            output->reshapei('f', {oX, oY, input->sizeAt(0),outDepth});
            output->permutei({2, 3, 1, 0});

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                output->printShapeInfo("Conv2D result shape");

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(conv2d) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);

            const int kY = block.getIArguments()->at(0);
            const int kX = block.getIArguments()->at(1);
            const int sY = block.getIArguments()->at(2);
            const int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            const int dY = block.getIArguments()->at(6);
            const int dX = block.getIArguments()->at(7);
            const bool isSameMode = block.getIArguments()->at(8) != 0;

            int oY = 0;
            int oX = 0;

            const int batchSize = inShape[1];
            const int outDepth = wShape[1];
            const int inY = inShape[3];
            const int inX = inShape[4];

            ConvolutionUtils<T>::calcOutHWpool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode) {
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);
            }

            //z = Shape.newShapeNoCopy(z, new int[] {outW, outH, miniBatch, outDepth}, true);
            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), int);
            std::vector<int> shape({batchSize, outDepth, oY, oX});
            shape::shapeBuffer(4, shape.data(), newShape);

            return new ShapeList(newShape);
        }



        CUSTOM_OP_IMPL(conv2d_bp, 3, 2, false, 0, 9) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* weights = INPUT_VARIABLE(1);
            NDArray<T>* bias = nullptr;
            NDArray<T>* epsilonNext;

            NDArray<T>* epsilon = OUTPUT_VARIABLE(0);
            NDArray<T>* gradW = OUTPUT_VARIABLE(1);
            NDArray<T>* gradB = nullptr;
            if (block.getVariables().size() == 3)
                epsilonNext = INPUT_VARIABLE(2);
            else {
                bias = INPUT_VARIABLE(2);
                epsilonNext = INPUT_VARIABLE(3);
                gradB = OUTPUT_VARIABLE(2);
            }

            REQUIRE_TRUE(input->rankOf() == 4, 0, "Conv2D expects 4D input, but got %i instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Conv2D expects 4D weights, but got %i instead", weights->rankOf());
            REQUIRE_TRUE(epsilonNext->rankOf() == 4, 0, "Conv2D expects 4D epsilons, but got %i instead", epsilonNext->rankOf());


            const int kY = block.getIArguments()->at(0);
            const int kX = block.getIArguments()->at(1);
            const int sY = block.getIArguments()->at(2);
            const int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            const int dY = block.getIArguments()->at(6);
            const int dX = block.getIArguments()->at(7);
            const bool isSameMode = block.getIArguments()->at(8) != 0;

            int oY, oX;

            const int batchSize = input->sizeAt(0);
            const int outDepth = weights->sizeAt(0);
            const int inDepth = weights->sizeAt(1);
            const int inY = input->sizeAt(2);
            const int inX = input->sizeAt(3);

            ConvolutionUtils<T>::calcOutHWpool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode)
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);

            auto epsilonNext2d = epsilonNext->permute({1, 0, 2, 3});
            epsilonNext2d->reshapei('c', {outDepth, batchSize * oY * oX});

            // gradW
            // we expect that activation was already calculated in next node
            auto col = new NDArray<T>('c', {batchSize, oY, oX, inDepth, kY, kX});
            auto col2 = col->permute({0, 3, 4, 5, 1, 2});
            std::unique_ptr<T> extrasIm2Col(new T[9]{(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

            input->template applyTransform<simdOps::Im2col<T>>(col2, extrasIm2Col.get());
            auto im2col2d = col->reshape('c', {batchSize * oY * oX, inDepth * kY * kX});
            delete col2;


            auto gradW2d = gradW->reshape('c', {outDepth, inDepth * kY * kX});
            gradW2d->transposei();

            im2col2d->transposei();
            auto eN2dT = epsilonNext2d->transpose();

            nd4j::NDArrayFactory<T>::mmulHelper(im2col2d, eN2dT, gradW2d);

            delete gradW2d;
            delete col;
            delete im2col2d;
            delete eN2dT;

            // epsilon
            auto pWeights = weights->permute({3, 2, 1, 0});
            pWeights->reshapei('f', {inDepth * kY * kX, outDepth});

            auto eps2d = nd4j::NDArrayFactory<T>::mmulHelper(pWeights, epsilonNext2d);

            auto eps6d = eps2d->reshape('f', {kX, kY, inDepth, oX, oY, batchSize});
            eps6d->permutei({5, 2, 1, 0, 4, 3});

            std::unique_ptr<T> extrasCol2Im(new T[9]{(T) sY, (T) sX, (T) pY, (T) pX, (T) inY, (T) inX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

            eps6d->template applyTransform<simdOps::Col2Im<T>>(epsilon, extrasCol2Im.get());

            if (bias == nullptr) {
                STORE_2_RESULTS(*epsilon, *gradW);
            } else {
                // bias is optional
                auto sum = epsilonNext2d->sum({1});
                gradB->assign(sum);
                delete sum;

                STORE_3_RESULTS(*epsilon, *gradW, *gradB);
            }

            delete pWeights;
            delete eps2d;
            delete eps6d;
            delete epsilonNext2d;

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(conv2d_bp) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);

            int *bShape = nullptr;
            // if conv2d op has bias provided, we'll have > 3 inputs (input, weights, _bias_, epsilonNext)
            if (inputShape->size() > 3)
                bShape = inputShape->at(2);

            int *newIShape;
            ALLOCATE(newIShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            memcpy(newIShape, inShape, shape::shapeInfoByteLength(inShape));

            int *newWShape;
            ALLOCATE(newWShape, block.getWorkspace(), shape::shapeInfoLength(wShape), int);
            memcpy(newWShape, wShape, shape::shapeInfoByteLength(wShape));

            auto shapeList = new ShapeList({newIShape, newWShape});

            if (bShape != nullptr) {
                int *newBShape;
                ALLOCATE(newBShape, block.getWorkspace(), shape::shapeInfoLength(bShape), int);
                memcpy(newBShape, bShape, shape::shapeInfoByteLength(bShape));

                shapeList->push_back(newBShape);
            }

            return shapeList;
        }

        /**
         * Depthwise convolution2d
         */
//////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(sconv2d, 2, 1, false, 0, 9) {
            NDArray<T> *input = block.getVariables().at(0)->getNDArray();
            NDArray<T> *weights = block.getVariables().at(1)->getNDArray();
            NDArray<T> *bias = nullptr;
            if (block.getVariables().size() == 3)
                bias = block.getVariables().at(2)->getNDArray();

            NDArray<T> *z = this->getZ(block);

            const int kY = block.getIArguments()->at(0);
            const int kX = block.getIArguments()->at(1);
            const int sY = block.getIArguments()->at(2);
            const int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            const int dY = block.getIArguments()->at(6);
            const int dX = block.getIArguments()->at(7);
            const bool isSameMode = block.getIArguments()->at(8) != 0;

            int oY = 0;
            int oX = 0;

            const int batchSize = input->shapeOf()[0];
            const int outDepth = weights->shapeOf()[0];
            const int inDepth = weights->shapeOf()[1];
            const int inY = input->shapeOf()[2];
            const int inX = input->shapeOf()[3];

            REQUIRE_TRUE(weights->shapeOf()[2] == kY && weights->shapeOf()[3] == kX, 0, "Kernels should have dimensions of [%i, %i], but got [%i, %i] instead", kY, kX, weights->sizeAt(2), weights->sizeAt(3));

            if (input->sizeAt(1) == 1) {
                nd4j_debug("Separable conv2d for 1 channel equals to standard conv2d\n","");
                nd4j::ops::conv2d<T> c2d;
                return c2d.execute(&block);
            }

            ConvolutionUtils<T>::calcOutHWpool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode) {
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);
            }

            std::unique_ptr<NDArray<T>> col2(new NDArray<T>('c', {batchSize, inDepth, kY, kX, oY, oX}));

            // col2d now has shape of [bS, inDepth, kY, kX, oY, oX]
            std::unique_ptr<T> extrasIm2Col(new T[9]{(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

            input->template applyTransform<simdOps::Im2col<T>>(col2.get(), extrasIm2Col.get());

            NDArray<T>* c_ = col2.get()->permute({1, 0, 4, 5, 2, 3});
            NDArray<T>* w_ = weights->permute({1, 2, 3, 0});

            c_->reshapei('c', {inDepth, batchSize * oY * oX, kY * kX});
            w_->reshapei('c', {inDepth, kY * kX, outDepth});

            // matmul here
            z->reshapei('c', {inDepth, batchSize * oY * oX, outDepth});
            NDArrayFactory<T>::mmulHelper(c_, w_, z);

            if (bias != nullptr) {
                z->reshapei('c', {-1, (int) bias->lengthOf()});
                z->addiRowVector(bias);
            }

            z->reshapei('c', {input->sizeAt(0),outDepth * inDepth, oY, oX });


            STORE_RESULT(*z);

            delete c_;
            delete w_;

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(sconv2d) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);

            const int kY = block.getIArguments()->at(0);
            const int kX = block.getIArguments()->at(1);
            const int sY = block.getIArguments()->at(2);
            const int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            const int dY = block.getIArguments()->at(6);
            const int dX = block.getIArguments()->at(7);
            const bool isSameMode = block.getIArguments()->at(8) != 0;

            int oY = 0;
            int oX = 0;

            const int batchSize = inShape[1];
            const int inDepth = inShape[2];
            const int outDepth = wShape[1];
            const int inY = inShape[3];
            const int inX = inShape[4];

            ConvolutionUtils<T>::calcOutHWpool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode) {
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);
            }

            //z = Shape.newShapeNoCopy(z, new int[] {outW, outH, miniBatch, outDepth}, true);
            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), int);
            std::vector<int> shape({batchSize, outDepth * inDepth, oY, oX});
            shape::shapeBuffer(4, shape.data(), newShape);

            return new ShapeList(newShape);
        }

        /**
         *
         *
         */
        CUSTOM_OP_IMPL(sconv2d_bp, 4, 2, false, 0, 9) {
            NDArray<T> *input = INPUT_VARIABLE(0);
            NDArray<T> *weights = INPUT_VARIABLE(1);
            NDArray<T> *epsilonNext = INPUT_VARIABLE(2);
            NDArray<T> *bias = nullptr;

            // bias is still optional
            if (block.getVariables().size() > 3)
                bias = INPUT_VARIABLE(3);

            //epsilonNext->rankOf() == 4 && weights->rankOf() == 4
            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should be 4D, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Weights should be 4D, but got %iD instead", weights->rankOf());
            REQUIRE_TRUE(epsilonNext->rankOf() == 4, 0, "Epsilon should be 4D, but got %iD instead", epsilonNext->rankOf());

            NDArray<T> * epsilon = this->getZ(block);
            NDArray<T> * gradW = this->getZ(block, 1);
            NDArray<T> * gradB = nullptr;
            if (bias != nullptr)
                gradB = this->getZ(block, 2);

            const int kY = block.getIArguments()->at(0);
            const int kX = block.getIArguments()->at(1);
            const int sY = block.getIArguments()->at(2);
            const int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            const int dY = block.getIArguments()->at(6);
            const int dX = block.getIArguments()->at(7);
            const bool isSameMode = block.getIArguments()->at(8) != 0;

            int oY = epsilonNext->sizeAt(2);
            int oX = epsilonNext->sizeAt(3);

            const int batchSize = input->shapeOf()[0];
            const int outDepth = weights->shapeOf()[0];
            const int inDepth = weights->shapeOf()[1];
            const int inY = input->shapeOf()[2];
            const int inX = input->shapeOf()[3];

            bool hasCol = CHECK_STASH("im2col");
            NDArray<T> *col = nullptr;
            if (hasCol)
                col = UNSTASH("im2col")
            else {
                std::unique_ptr<NDArray<T>> col2(new NDArray<T>('c', {batchSize, inDepth, kY, kX, oY, oX}));

                // col2d now has shape of [bS, inDepth, kY, kX, oY, oX]
                std::unique_ptr<T> extrasIm2Col(new T[9]{(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

                input->template applyTransform<simdOps::Im2col<T>>(col2.get(), extrasIm2Col.get());
            }

//            epsilonNext->printShapeInfo("eps next");

            /*
             gy_ = gy.reshape((B, C, D, IY * IX)).transpose(1, 2, 0, 3).reshape((C, D, B * IY * IX))
             */
            auto eN_ = epsilonNext->reshape('c', {batchSize, inDepth, outDepth, oY * oX});
            eN_->permutei({1, 2, 0, 3});
            eN_->reshapei('c', {inDepth, outDepth, batchSize * oY * oX});

            auto col_ = col->permute({1, 0, 4, 5, 2, 3});
            col_->reshapei('c', {inDepth, batchSize * oY * oX, kY * kX});

            /*
             # (C, D, B*IY*IX), (C, B*IY*IX, KY*KX) -> (C, D, KY*KX)
                gW_ = _matmul(gy_, c_, xp)
             */

            // calculating wieghts gradients here
            //auto gW_ = gradW->reshape('c', {inDepth, outDepth, kY * kX});
            auto gW_ = NDArrayFactory<T>::mmulHelper(eN_, col_);

            gW_->reshapei('c',{inDepth, outDepth, kY, kX});
            gW_->permutei({1, 0, 2, 3});
            gradW->assign(gW_);

            delete gW_;
            delete col_;
            if (!hasCol)
                delete col;

            // calculating epsilon here
            auto w_ = weights->permute({1, 2, 3, 0});
            w_->reshapei('c', {inDepth, kY * kX, outDepth});

            auto gcol = NDArrayFactory<T>::mmulHelper(w_, eN_);
            gcol->reshapei('c', {inDepth, kY, kX, batchSize, oY, oX});
            gcol->permutei({3, 0, 1, 2, 4, 5});

            std::unique_ptr<T> extrasCol2Im(new T[9]{(T) sY, (T) sX, (T) pY, (T) pX, (T) inY, (T) inX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

            // we're sure that col2im result will have the same size as original image
            //auto rCol = new NDArray<T>('c', {batchSize, inDepth, inY, inX});
            gcol->template applyTransform<simdOps::Col2Im<T>>(epsilon, extrasCol2Im.get());


            delete eN_;
            delete gcol;
            delete w_;



            if (bias != nullptr) {
                // calculating gradB, if defined
                auto eN_ = epsilonNext->permute({0, 2, 3, 1});
                auto sum = eN_->template reduceAlongDimension<simdOps::Sum<T>>({0, 1, 2});
                gradB->assign(sum);
                delete sum;

                STORE_3_RESULTS(*epsilon, *gradW, *gradB);
            } else {
                STORE_2_RESULTS(*epsilon, *gradW);
            }

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(sconv2d_bp) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);
            auto eShape = inputShape->at(2);
            int *bShape = nullptr;

            // bias is optional thing, and might be absent
            if (inputShape->size() == 4)
                bShape = inputShape->at(3);

            int *newInShape;
            int *newWShape;
            ALLOCATE(newInShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            ALLOCATE(newWShape, block.getWorkspace(), shape::shapeInfoLength(wShape), int);

            memcpy(newInShape, inShape, shape::shapeInfoByteLength(inShape));
            memcpy(newWShape, wShape, shape::shapeInfoByteLength(wShape));

            auto shapes = new ShapeList({newInShape, newWShape});

            if (bShape != nullptr) {
                int *newBShape;
                ALLOCATE(newBShape, block.getWorkspace(), shape::shapeInfoLength(bShape), int);
                memcpy(newBShape, bShape, shape::shapeInfoByteLength(bShape));

                shapes->push_back(newBShape);
            }

            return shapes;
        }

//////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(deconv2d, 2, 1, false, 0, 9) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* weights = INPUT_VARIABLE(1);
            NDArray<T>* bias = nullptr;
            if (block.getVariables().size() > 2)
                bias = INPUT_VARIABLE(2);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should be 4D, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Weights should be 4D, but got %iD instead", weights->rankOf());

            int oD = weights->sizeAt(0);

            if (bias != nullptr) {
                REQUIRE_TRUE(bias->isVector(), 0, "Bias should be vector");
                REQUIRE_TRUE(bias->lengthOf() == oD, 0, "Bias length be equal to outpuDepth, but got %i instead", bias->lengthOf());
            }

            int iY = input->sizeAt(2);
            int iX = input->sizeAt(3);

            int kY = block.getIArguments()->at(0);
            int kX = block.getIArguments()->at(1);
            int sY = block.getIArguments()->at(2);
            int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            int dY = block.getIArguments()->at(6);
            int dX = block.getIArguments()->at(7);
            const bool isSameMode = block.getIArguments()->at(8) != 0;

            NDArray<T> *z = this->getZ(block);

            int oY = z->sizeAt(2);
            int oX = z->sizeAt(3);

            auto gcol = nd4j::NDArrayFactory<T>::tensorDot(weights, input, nullptr, {0}, {1});
            gcol->permutei({3, 0, 1, 2, 4, 5});

            std::unique_ptr<T> extrasCol2Im(new T[9]{(T) sY, (T) sX, (T) pY, (T) pX, (T) oY, (T) oX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

            gcol->template applyTransform<simdOps::Col2Im<T>>(z, extrasCol2Im.get());

            delete gcol;

            if (bias != nullptr) {
                z->template applyBroadcast<simdOps::Add<T>>({1}, bias);
            }

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(deconv2d) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);

            int B = shape::shapeOf(inShape)[0];
            int iC = shape::shapeOf(inShape)[1];
            int iY = shape::shapeOf(inShape)[2];
            int iX = shape::shapeOf(inShape)[3];

            int oC = shape::shapeOf(wShape)[0];
            int kY = block.getIArguments()->at(0);
            int kX = block.getIArguments()->at(1);
            int sY = block.getIArguments()->at(2);
            int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);

            int oY = sY * (iY - 1) + kY - 2 * pY;
            int oX = sX * (iX - 1) + kX - 2 * pX;

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), int);
            std::vector<int> shape({B, oC, oY, oX});
            shape::shapeBuffer(4, shape.data(), newShape);

            return new ShapeList(newShape);
        }


        CUSTOM_OP_IMPL(deconv2d_bp, 4, 2, false, 0, 9) {
            NDArray<T> *input = INPUT_VARIABLE(0);
            NDArray<T> *weights = INPUT_VARIABLE(1);
            NDArray<T> *epsilonNext = INPUT_VARIABLE(2);
            NDArray<T> *bias = nullptr;

            // bias is still optional
            if (block.getVariables().size() > 3)
                bias = INPUT_VARIABLE(3);

            //epsilonNext->rankOf() == 4 && weights->rankOf() == 4
            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should be 4D, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Weights should be 4D, but got %iD instead", weights->rankOf());
            REQUIRE_TRUE(epsilonNext->rankOf() == 4, 0, "Epsilon should be 4D, but got %iD instead", epsilonNext->rankOf());

            int kY = block.getIArguments()->at(0);
            int kX = block.getIArguments()->at(1);
            int sY = block.getIArguments()->at(2);
            int sX = block.getIArguments()->at(3);
            int pY = block.getIArguments()->at(4);
            int pX = block.getIArguments()->at(5);
            int dY = block.getIArguments()->at(6);
            int dX = block.getIArguments()->at(7);
            const bool isSameMode = block.getIArguments()->at(8) != 0;

            NDArray<T>* epsilon = this->getZ(block);
            NDArray<T>* gradW = this->getZ(block, 1);
            NDArray<T>* gradB = nullptr;

            if (bias != nullptr)
                gradB = this->getZ(block, 2);

            // epsilon for deconv2d is FF conv pass

            nd4j::ops::conv2d<T> op;
            Nd4jStatus r1 = op.execute({input, weights}, {epsilon}, {}, {kY, kX, sY, sX, pY, pX, dY, dX, block.getIArguments()->at(8)});
            if (r1 != ND4J_STATUS_OK)
                return r1;

            // gradW is im2col + tensorDot
            /*
              col = conv.im2col_cpu(
            x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw,
            cover_all=self.cover_all)
             gW = numpy.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))).
             */

            int oY = 0;
            int oX = 0;
            int inY = epsilonNext->sizeAt(2);
            int inX = epsilonNext->sizeAt(3);

            ConvolutionUtils<T>::calcOutHWpool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode) {
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);
            }

            std::unique_ptr<T> extrasIm2Col(new T[9]{(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});
            auto gcol = new NDArray<T>('c', {input->sizeAt(0), input->sizeAt(1), kY, kX, oY, oX });
            epsilonNext->template applyTransform<simdOps::Im2col<T>>(gcol, extrasIm2Col.get());

            /*
            gW = numpy.tensordot(
                    gy, col, ((0, 2, 3), (0, 4, 5))).
            */

            auto gW = NDArrayFactory<T>::tensorDot(input, gcol, nullptr, {0, 2, 3}, {0, 4, 5});
            gradW->assign(gW);

            delete gW;
            delete gcol;

            if (gradB != nullptr) {
                auto sum = epsilon->template reduceAlongDimension<simdOps::Sum<T>>({0, 2, 3});
                gradB->assign(sum);
                delete sum;

                STORE_3_RESULTS(*epsilon, *gradW, *gradB);
            } else {
                STORE_2_RESULTS(*epsilon, *gradW);
            }



            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(deconv2d_bp) {
            auto inShape = inputShape->at(0);
            auto wShape = inputShape->at(1);
            auto eShape = inputShape->at(2);
            int *bShape = nullptr;

            // bias is optional thing, and might be absent
            if (inputShape->size() == 4)
                bShape = inputShape->at(3);

            int *newInShape;
            int *newWShape;
            ALLOCATE(newInShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            ALLOCATE(newWShape, block.getWorkspace(), shape::shapeInfoLength(wShape), int);

            memcpy(newInShape, inShape, shape::shapeInfoByteLength(inShape));
            memcpy(newWShape, wShape, shape::shapeInfoByteLength(wShape));

            auto shapes = new ShapeList({newInShape, newWShape});

            if (bShape != nullptr) {
                int *newBShape;
                ALLOCATE(newBShape, block.getWorkspace(), shape::shapeInfoLength(bShape), int);
                memcpy(newBShape, bShape, shape::shapeInfoByteLength(bShape));

                shapes->push_back(newBShape);
            }

            return shapes;
        }

//////////////////////////////////////////////////////////////////////////
        /**
         * Upsampling implementation, based on pytorch
         *
         * IArgs map:
         * IArgs[0] - scale factor
         */
        CONFIGURABLE_OP_IMPL(upsampling2d, 1, 1, false, 0, 1) {
            NDArray<T>* input = block.getVariables().at(0)->getNDArray();
            NDArray<T>* output = this->getZ(block);
            int scale_factor = block.getIArguments()->at(0);

//            int inputHeight = input->sizeAt(2);
//            int inputWidth  = input->sizeAt(3);

            int dW = scale_factor;
            int dH = scale_factor;
//            int outputHeight = inputHeight * scale_factor;
//            int outputWidth = inputWidth * scale_factor;
            int xDim = input->rankOf() - 2;
            int yDim = input->rankOf() - 1;

            int osz0 = output->sizeAt(0);
            int osz1 = output->sizeAt(1);
            int osz2 = output->sizeAt(2);
            int osz3 = output->sizeAt(3);

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


                            output->getBuffer()[idst] = input->getBuffer()[isrc];
                        }
                    }
                }
            }

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(upsampling, upsampling2d);

//////////////////////////////////////////////////////////////////////////
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
        CONFIGURABLE_OP_IMPL(upsampling2d_bp, 2, 1, false, 0, 1) {
            //NDArray<T>* input = block.getVariables().at(0)->getNDArray();
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
                                    output->getBuffer()[idst] += gradientNext->getBuffer()[isrc];
                                }
                            }
                        }
                    }
                }
            }

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(upsampling_bp, upsampling2d_bp);

//////////////////////////////////////////////////////////////////////////



		//////////////////////////////////////////////////////////////////////////
		CONFIGURABLE_OP_IMPL(ismax, 1, 1, false, 0, -1) {

			REQUIRE_OK(this->validateInputLengthMatch(block));
            REQUIRE_OK(this->validateInputDimensionsMatch(block));

			NDArray<T>* x = block.getVariables().at(0)->getNDArray();			
			NDArray<T>* z = this->getZ(block);
			std::vector<int> dimensions = *(block.getIArguments());			// argI

			if (x->isVector()) {
				int dimensionsLength = dimensions.size();
				int length = x->lengthOf();
				if ((x->shapeOf())[dimensions[0]] == 1) {
					for (int i = 0; i < length; i++)
						z->putScalar(i, 1.f);
				}
				else {
					int eleStride = shape::elementWiseStride(x->getShapeInfo());
					if (eleStride == 1) {
						int maxIdx = 0;
						T currMax = x->getScalar(0);
						if (length < ELEMENT_THRESHOLD) {

//#pragma omp simd reduction(max:maxIdx,currMax)
							for (int i = 0; i < length; i++) {
								if (currMax < x->getScalar(i)) {
									currMax = x->getScalar(i);
									maxIdx = i;
								}
								x->putScalar(i, 0.f);
							}
						}
						else {
#pragma omp parallel proc_bind(AFFINITY) default(shared)
{
							int maxIdxLocal = maxIdx;
							T currMaxLocal = currMax;
//#pragma omp simd reduction(max:maxIdxLocal,currMaxLocal)
							for (int i = 0; i < length; i++) {
								if (currMaxLocal < x->getScalar(i)) {
									currMaxLocal = x->getScalar(i);
									maxIdxLocal = i;
								}
								z->putScalar(i, 0.f);
							}
#pragma omp critical
                            {
							    if (currMax < currMaxLocal) {
								    currMax = currMaxLocal;
								    maxIdx = maxIdxLocal;
							    }
                            }
}
						}
						z->putScalar(maxIdx, 1.f);
					}
					else {
						int maxIdx = 0;
						T currMax = x->getScalar(0);
						if (length < ELEMENT_THRESHOLD) {
//#pragma omp parallel for reduction(max:maxIdx,currMax) proc_bind(AFFINITY)
							for (int i = 0; i < length; i++) {
								if (currMax < x->getScalar(i*eleStride)) {
									currMax = x->getScalar(i*eleStride);
									maxIdx = i;
								}
								z->putScalar(i, 0.f);
							}
						}
						else {
#pragma omp parallel proc_bind(AFFINITY) default(shared)
{
							int maxIdxLocal = maxIdx;
							T currMaxLocal = currMax;
//#pragma omp parallel for reduction(max:maxIdx,currMax)  proc_bind(AFFINITY)
							for (int i = 0; i < length; i++) {
								if (currMaxLocal < x->getScalar(i*eleStride)) {
									currMaxLocal = x->getScalar(i*eleStride);
									maxIdxLocal = i;
								}
								z->putScalar(i, 0.f);
							}
#pragma omp critical
{
							if (currMax < currMaxLocal) {
								currMax = currMaxLocal;
								maxIdx = maxIdxLocal;
							}
}
}
						}
						z->putScalar(maxIdx, 1.f);
					}
				}
			}
			else {
                int dimensionsLength = dimensions.size();
//                int tads = tad.numTads;
                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem
                shape::TAD tad(x->getShapeInfo(), dimensions.data(), dimensionsLength);
                tad.createTadOnlyShapeInfo();
                tad.createOffsets();
						
                int *tadShapeShapeInfo = tad.tadOnlyShapeInfo;
				Nd4jIndex* tadOffsets = tad.tadOffsets;

                int tadLength = shape::tadLength(x->getShapeInfo(), dimensions.data(), dimensionsLength);
                int tads = x->lengthOf() / tadLength;

                int tadsPerThread = tads / TAD_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                int tadEWS = shape::elementWiseStride(tadShapeShapeInfo);
                int zEWS = tadEWS;

                int span = (tads / num_threads) + 8;

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY)
                {
                    int tid = omp_get_thread_num();
                    int start = span * tid;
                    int end = span * (tid + 1);
                    if (end > tads) end = tads;

                    for (int r = start; r < end; r++) {
                        if (tadEWS > 0 && zEWS > 0 && dimensionsLength == 1) {
                            T *rX = x->getBuffer() + tadOffsets[r];
                            T *rZ = z->getBuffer() + tadOffsets[r];

                            T maxValue = rX[0];
                            int maxIdx = 0;
                            if (tadEWS == 1 && zEWS == 1) {
//#pragma omp simd reduction(max:maxValue,maxIdx)
                                for (int i = 0; i < tadLength; i++) {
                                    if (rX[i] > maxValue) {
                                        maxIdx = i;
                                        maxValue = rX[i];
                                    }
                                }

#pragma omp simd
                                for (int i = 0; i < tadLength; i++) {
                                    rZ[i] = maxIdx == i ? (T) 1.0 : (T) 0.0;
                                }

                            } else {

//#pragma omp parallel for reduction(max:maxValue,maxIdx) default(shared)
                                for (int i = 0; i < tadLength; i++) {
                                    if (rX[i * tadEWS] > maxValue) {
                                        maxIdx = i;
                                        maxValue = rX[i * tadEWS];
                                    }
                                }

#pragma omp simd
                                for (int i = 0; i < tadLength; i++) {
                                    rZ[i * zEWS] = maxIdx == i ? (T) 1.0 : (T) 0.0;
                                }
                            }
                        } else {
                            int tadsPerThread = tads / TAD_THRESHOLD;
                            int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                            num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                            int offset = tadOffsets[r];
                            int shapeIter[MAX_RANK];
                            int coord[MAX_RANK];
                            int dim;
                            int xStridesIter[MAX_RANK];
                            int resultStridesIter[MAX_RANK];
                            int *xShape = shape::shapeOf(tadShapeShapeInfo);
                            int *xStride = shape::stride(tadShapeShapeInfo);
                            int *resultStride = shape::stride(tadShapeShapeInfo);
                            int rank = shape::rank(tadShapeShapeInfo);
                            T *xPointer = x->getBuffer() + offset;
                            T *resultPointer = z->getBuffer() + offset;
                            T maxValue = xPointer[0];

                            T *maxCursor = resultPointer;
                            Nd4jPointer maxCursorLong = reinterpret_cast<Nd4jPointer>(maxCursor);
                            if (PrepareTwoRawArrayIter<T>(rank,
                                                             xShape,
                                                             xPointer,
                                                             xStride,
                                                             resultPointer,
                                                             resultStride,
                                                             &rank,
                                                             shapeIter,
                                                             &xPointer,
                                                             xStridesIter,
                                                             &resultPointer,
                                                             resultStridesIter) >= 0) {
                                   ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                       if (maxValue < xPointer[0]) {
                                           maxCursor = resultPointer;
                                           maxCursorLong = reinterpret_cast<Nd4jPointer>(resultPointer);
                                           maxValue = xPointer[0];
                                       }

                                       resultPointer[0] = 0.0;
                                   }
                                   ND4J_RAW_ITER_TWO_NEXT(dim,
                                                          rank,
                                                          coord,
                                                          shapeIter,
                                                          xPointer,
                                                          xStridesIter,
                                                          resultPointer,
                                                          resultStridesIter);
                                   maxCursor = reinterpret_cast<T *>(maxCursorLong);
                                   maxCursor[0] = 1.0;
                            }
                        }
                    }
                }
            }
			return ND4J_STATUS_OK;
		}
        DECLARE_SYN(IsMax, ismax);        

		//////////////////////////////////////////////////////////////////////////        
        CUSTOM_OP_IMPL(pooling2d, 1, 1, false, 0, 11) {

			NDArray<T> *x = block.getVariables().at(0)->getNDArray();			
			REQUIRE_TRUE(x->rankOf() == 4, 0, "Input should have rank of 4, but got %i instead", x->rankOf());            
            std::vector<int> argI = *(block.getIArguments());				// 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode; 9 - pooling mode; 10 - divisor extraParam0 for pnorm case			
            auto z = this->getZ(block);
			
			int kH = argI[0];
			int kW = argI[1];
			int sH = argI[2];
			int sW = argI[3];
			int pH = argI[4];
			int pW = argI[5];
			int dH = argI[6];			//Dilation, height dimension
			int dW = argI[7];			//Dilation, width dimension
			int poolingMode = argI[9];
			T extraParam0 = (int)argI[10];

			int kSize = kW * kH;

			int *inShape = shape::shapeOf(x->getShapeInfo());
			int *inStride = shape::stride(x->getShapeInfo());

			int samples = inShape[0];
			int depth = inShape[1];
			int height = inShape[2];
			int width = inShape[3];

			int strideex = inStride[0];
			int stridech = inStride[1];
			int strideh = inStride[2];
			int stridew = inStride[3];

			int outH = (z->getShapeInfo())[3];
			int outW = (z->getShapeInfo())[4];			
            int *im2colShapeInfo = new int[16] {6, samples, depth, kH, kW, outH, outW, depth*kH*kW*outH*outW, kH*kW*outH*outW, kW*outH*outW, outH*outW, outW, 1, 0, 1, 99};

            int *outShape = shape::shapeOf(im2colShapeInfo);
            int *outStride = shape::stride(im2colShapeInfo);

			int height_col = outShape[4];
			int width_col = outShape[5];

			int n = samples * depth * height_col * width_col;

			int _threads = omp_get_max_threads();
			int span = (n / _threads) + 1;


#pragma omp parallel num_threads(_threads) proc_bind(close)
            {
				int tid = omp_get_thread_num();
				int start = span * tid;
				int end = span * (tid + 1);
				if (end > n) end = n;
                T res;

                for (int index = start; index < end; index++) {
                    int h_index = index / width_col;
                    int h_col = h_index % height_col;
                    int w_col = index % width_col;

                    int c_im = h_index / height_col;
                    int c_col = c_im * kSize;

                    int depth_im = c_im % depth;
                    int num_im = c_im / depth;
                    int h_offset = h_col * sH - pH;
                    int w_offset = w_col * sW - pW;

                    T *data_col_ptr = z->getBuffer();

                    int i_c = (c_col * height_col + h_col) * width_col + w_col;
                    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

                    T *data_im_ptr = x->getBuffer();

                    data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset * stridew;
                    res = poolingMode == 0 ? (T) -MAX_FLOAT : (T) 0.0f;

                    for (int i = 0; i < kH; ++i) {
                        for (int j = 0; j < kW; ++j) {
                            int h_im = h_offset + i * dH;
                            int w_im = w_offset + j * dW;
                            int i_f = 0;
                            int i_c_temp = i_c;
                            for (int dim = 5; dim >= 0; dim--) {
                                i_f += (i_c_temp % outShape[dim]) * outStride[dim];
                                i_c_temp = i_c_temp / outShape[dim];
                            }

                            T val;
                            if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                                val = data_im_ptr[i * dH * strideh + j * dW * stridew];
                            else
                                val = (T) 0.0f;

                            //kernel[i * kH + j] = val;
                            // max
                            if (poolingMode == 0) {
                                if (res < val)
                                    res = val;
                            // avg
                            } else if (poolingMode == 1) {
                                res += val;

                            // phorm
                            } else if (poolingMode == 2) {
                                res += nd4j::math::nd4j_pow<T>(nd4j::math::nd4j_abs<T>(val), extraParam0);
                            }

                            //result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
                            data_col_ptr += height_col * width_col;
                            i_c += height_col * width_col;
                        }
                    }

                    // avg final step
                    if (poolingMode == 1) {
                        res /= kSize;

                    // pnorm final step
                    } else if (poolingMode == 2) {
                        res = nd4j::math::nd4j_pow<T>(res, (T) 1.0f /  extraParam0);
                    }

                    z->putScalar(index,res);
                }
            }
			delete im2colShapeInfo;
			return ND4J_STATUS_OK;
		}
		DECLARE_SYN(Pooling2D, pooling2d);
		
		//////////////////////////////////////////////////////////////////////////
		DECLARE_SHAPE_FN(pooling2d) {
			int* inShape = inputShape->at(0);            
			// 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode; 9 - pooling mode; 
			std::vector<int> argI = *(block.getIArguments());			
			int kH = argI[0];
			int kW = argI[1];
			int sH = argI[2];
			int sW = argI[3];
			int pH = argI[4];
			int pW = argI[5];
			int dH = argI[6];
			int dW = argI[7];
			int isSameMode = argI[8];

            int bS = inShape[1];
            int iD = inShape[2];
            int iH = inShape[3];
            int iW = inShape[4];
			
			char order = shape::order(inShape); // output order must be equal to input order

			// calculate output Height/Width
			int oH, oW;
			ConvolutionUtils<T>::calcOutHWpool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);
			// allocate memory for new shape
            int* newShapeInfo = nullptr;
            ALLOCATE(newShapeInfo, block.getWorkspace(), 12, int);
			newShapeInfo[0] = 4;		// rank
			newShapeInfo[1] = bS;
			newShapeInfo[2] = iD;
			newShapeInfo[3] = oH;
			newShapeInfo[4] = oW;
            shape::updateStrides(newShapeInfo, order);

			return new ShapeList(newShapeInfo);
		}
		
		//////////////////////////////////////////////////////////////////////////
		CUSTOM_OP_IMPL(avgpool2d_bp, 2, 1, false, 0, 9) {
			
            NDArray<T>* input = block.getVariables().at(0)->getNDArray();
			REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should have rank of 4, but got %i instead", input->rankOf());
			NDArray<T>* epsilon = block.getVariables().at(1)->getNDArray();
			NDArray<T>* outEpsilon = this->getZ(block);
			std::vector<int> argI = *(block.getIArguments());
			
			int kH = argI[0];
			int kW = argI[1];
			int sH = argI[2];
			int sW = argI[3];
			int pH = argI[4];
			int pW = argI[5];
			int dH = argI[6];
			int dW = argI[7];
			int isSameMode = argI[8];
			
			int bS = input->getShapeInfo()[1];
			int iD = input->getShapeInfo()[2];
			int iH = input->getShapeInfo()[3];
			int iW = input->getShapeInfo()[4];
						
			// calculate output Height/Width
			int oH, oW;
			ConvolutionUtils<T>::calcOutHWpool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

			bool cOrderStrides = false;
			bool isEpsilonDup = false;
			if (epsilon->ordering() != 'c') {
				epsilon = epsilon->dup('c');
				cOrderStrides = true;
				isEpsilonDup = true;
			}

			int strideToCompare[] = {oH*oW, iD*oH*oW, oW, 1};
			if (!cOrderStrides && shape::strideDescendingCAscendingF(epsilon->getShapeInfo())) {
				cOrderStrides = true;
			}
			else if (!shape::strideEquals(strideToCompare, 4, epsilon->stridesOf(), epsilon->rankOf())) {
				epsilon = epsilon->dup('c');
				cOrderStrides = true;
				isEpsilonDup = true;
			}

			NDArray<T>* col6d = nullptr;
			NDArray<T>* col6dPermuted = nullptr;
			NDArray<T>* epsilon1d = nullptr;

			if (cOrderStrides) {
				col6d = new NDArray<T>('c', {bS, iD, oH, oW, kH, kW}, block.getWorkspace());
				col6dPermuted = col6d->permute({0, 1, 4, 5, 2, 3});
				epsilon1d = epsilon->reshape('c', {(int) epsilon->lengthOf(), 1}); //zero copy reshape
			}
			else {
				col6d = new NDArray<T>('c', {iD, bS, oH, oW, kH, kW}, block.getWorkspace());
				col6dPermuted = col6d->permute({1, 0, 4, 5, 2, 3});
				NDArray<T>* epsilonTemp = epsilon->permute({1, 0, 2, 3});
				epsilon1d = epsilonTemp->reshape('c', {(int) epsilon->lengthOf(), 1}); //Should be a zero-copy reshape always
				delete epsilonTemp;
			}

			// NDArray<T>* col2d = col6d->reshape('c', {bS*iD*oH*oW, kH*kW}, block.getWorkspace());			
			// col2d->addiColumnVector(epsilon1d);		

			// NDArray<T>* tempEpsilon = new NDArray<T>('c', {iD, bS, iH, iW}, block.getWorkspace());
			// NDArray<T>* outEpsilon = tempEpsilon.permute({1, 0, 2, 3});
			T extraParams3[] = {(T)sW, (T)sH, (T)pW, (T)pH, (T)iH, (T)iW, (T)dW, (T)dH};   			// ??? zeros
			col6dPermuted->template applyTransform<simdOps::Col2Im<T>>(outEpsilon, extraParams3);
            outEpsilon->template applyScalar<simdOps::Divide<T>>((T) kH*kW, outEpsilon);

			STORE_RESULT(*outEpsilon);

			if(isEpsilonDup)
				delete epsilon;
			delete col6d;
			delete col6dPermuted;
			delete epsilon1d;
            // delete col2d;

			return ND4J_STATUS_OK;
        }

		//////////////////////////////////////////////////////////////////////////
		DECLARE_SHAPE_FN(avgpool2d_bp) {
			int* inShape = inputShape->at(0);            
            int bS = inShape[1];
            int iD = inShape[2];
            int iH = inShape[3];
            int iW = inShape[4];

			// calculate output Height/Width
            int* newShapeInfo = nullptr;
            ALLOCATE(newShapeInfo, block.getWorkspace(), 12, int);
			newShapeInfo[0] = 4;		// rank
			newShapeInfo[1] = iD;
			newShapeInfo[2] = bS;
			newShapeInfo[3] = iH;
			newShapeInfo[4] = iW;
            shape::updateStrides(newShapeInfo, 'c');
			int dimensions[] = {1, 0, 2, 3};
			shape::doPermuteShapeBuffer(4, newShapeInfo, dimensions);	
			return new ShapeList(newShapeInfo);
		}	

		//////////////////////////////////////////////////////////////////////////
		CUSTOM_OP_IMPL(pnormpool2d_bp, 2, 1, false, 1, 10) {
			
            NDArray<T>* input = block.getVariables().at(0)->getNDArray();
			NDArray<T>* epsilon = block.getVariables().at(1)->getNDArray();
			NDArray<T>* outEpsilon = this->getZ(block);
			std::vector<int> argI = *(block.getIArguments());
			std::vector<T>   argT = *(block.getTArguments());
			
			int kH = argI[0];
			int kW = argI[1];
			int sH = argI[2];
			int sW = argI[3];
			int pH = argI[4];
			int pW = argI[5];
			int dH = argI[6];
			int dW = argI[7];			
			int isSameMode = argI[8];
			int pnorm = argI[9];
			T eps = argT[0];
		
			int bS = input->getShapeInfo()[1];
			int iD = input->getShapeInfo()[2];
			int iH = input->getShapeInfo()[3];
			int iW = input->getShapeInfo()[4];
			
			// calculate output Height/Width
			int oH, oW;
            ConvolutionUtils<T>::calcOutHWpool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

			bool cOrderStrides = false;
			bool isEpsilonDup = false;
			if (epsilon->ordering() != 'c') {
				epsilon = epsilon->dup('c');
				cOrderStrides = true;
				isEpsilonDup = true;
			}

			int strideToCompare[] = {oH*oW, iD*oH*oW, oW, 1};
			if (!cOrderStrides && shape::strideDescendingCAscendingF(epsilon->getShapeInfo())) {
				cOrderStrides = true;
			}
			else if (!shape::strideEquals(strideToCompare, 4, epsilon->stridesOf(), epsilon->rankOf())) {
				epsilon = epsilon->dup('c');
				cOrderStrides = true;
				isEpsilonDup = true;
			}

			NDArray<T>* col6d = nullptr;
			NDArray<T>* col6dPermuted = nullptr;
			NDArray<T>* epsilon1d = nullptr;

			if (cOrderStrides) {
				col6d = new NDArray<T>('c', {bS, iD, oH, oW, kH, kW}, block.getWorkspace());
				col6dPermuted = col6d->permute({0, 1, 4, 5, 2, 3});
				epsilon1d = epsilon->reshape('c', {(int) epsilon->lengthOf(), 1}); //zero copy reshape
			}
			else {
				col6d = new NDArray<T>('c', {iD, bS, oH, oW, kH, kW}, block.getWorkspace());
				col6dPermuted = col6d->permute({1, 0, 4, 5, 2, 3});
				NDArray<T>* epsilonTemp = epsilon->permute({1, 0, 2, 3});
				epsilon1d = epsilonTemp->reshape('c', {(int) epsilon->lengthOf(), 1}); //Should be a zero-copy reshape always
				delete epsilonTemp;
			}

			NDArray<T>* col2d = col6d->reshape('c', {bS*iD*oH*oW, kH*kW});

			T extraParams1[] = {(T)kW, (T)kH, (T)sW, (T)sH, (T)pW, (T)pH, (T)dW, (T)dH};
			input->template applyTransform<simdOps::Im2col<T>>(col6dPermuted, extraParams1);
						
			NDArray<T>* pNorm = new NDArray<T>(col2d->getShapeInfo(), block.getWorkspace()); 
            col2d->template applyTransform<simdOps::Abs<T>>(pNorm, nullptr); 
			
			T extraParams11[] = {(T)pnorm};
			pNorm->template applyTransform<simdOps::Pow<T>>(extraParams11);
			*pNorm = *(pNorm->sum({1}));
			T extraParams2[] = {1.f/pnorm};
			pNorm->template applyTransform<simdOps::Pow<T>>(extraParams2);
			
			NDArray<T>* numerator = new NDArray<T>(col2d->getShapeInfo(), block.getWorkspace());
            if (pnorm != 2) {
				NDArray<T>* absp2 = new NDArray<T>(col2d->getShapeInfo(), block.getWorkspace());
				col2d->template applyTransform<simdOps::Abs<T>>(absp2, nullptr);
				T extraParams3[] = {(T) (pnorm - 2)};
				absp2->template applyTransform<simdOps::Pow<T>>(extraParams3);
				nd4j::NDArrayFactory<T>::mmulHelper(col2d, absp2, numerator, (T)1.f, (T)0.f);
				delete absp2;
            }
			NDArray<T>* denom = new NDArray<T>(pNorm->getShapeInfo(), block.getWorkspace()); 
			T extraParams4[] = {(T) (pnorm - 1)};

			pNorm->template applyTransform<simdOps::Pow<T>>(denom, extraParams4);
            denom->template applyScalar<simdOps::Max<T>>(eps); // in case of 0
			denom->template applyPairwiseTransform<simdOps::Divide<T>>(epsilon1d, denom, nullptr);
			numerator->muliColumnVector(denom);

			// NDArray<T>* tempEpsilon = new NDArray<T>('c', {iD, bS, iH, iW});
			// NDArray<T>* outEpsilon = tempEpsilon.permute({1, 0, 2, 3});
			T extraParams5[] = {(T)sW, (T)sH, (T)pW, (T)pH, (T)iH, (T)iW, (T)dW, (T)dH};   			// ??? zeros
			col6dPermuted->template applyTransform<simdOps::Col2Im<T>>(outEpsilon, extraParams5);

			STORE_RESULT(*outEpsilon);

			if(isEpsilonDup)
				delete epsilon;
			delete col6d;
			delete col6dPermuted;
			delete epsilon1d;
			delete pNorm;	
			delete numerator;
			delete denom;
            delete col2d;

			return ND4J_STATUS_OK;
        }

		//////////////////////////////////////////////////////////////////////////
		DECLARE_SHAPE_FN(pnormpool2d_bp) {

			int* inShape = inputShape->at(0);            
            int bS = inShape[1];
            int iD = inShape[2];
            int iH = inShape[3];
            int iW = inShape[4];

			// calculate output Height/Width
            int* newShapeInfo = nullptr;
            ALLOCATE(newShapeInfo, block.getWorkspace(), 12, int);
			newShapeInfo[0] = 4;		// rank
			newShapeInfo[1] = iD;
			newShapeInfo[2] = bS;
			newShapeInfo[3] = iH;
			newShapeInfo[4] = iW;
            shape::updateStrides(newShapeInfo, 'c');
			int dimensions[] = {1, 0, 2, 3};
			shape::doPermuteShapeBuffer(4, newShapeInfo, dimensions);	
			return new ShapeList(newShapeInfo);
		}	
		
		

    }
}

#endif //LIBND4J_CONVO_OPS_H
