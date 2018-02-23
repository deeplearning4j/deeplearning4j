//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>
#include <memory>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(sconv2d, 2, 1, false, 0, 9) {
            auto input = INPUT_VARIABLE(0);
            auto weightsDepth = INPUT_VARIABLE(1);
            NDArray<T> *weightsPoint = nullptr;
            NDArray<T> *bias = nullptr;
            if (block.width() == 3) {
                auto tmp = INPUT_VARIABLE(2);
                if (tmp->rankOf() == 4)
                    weightsPoint = tmp;
                else
                    bias = tmp;
            } else if (block.width() == 4) {
                weightsPoint = INPUT_VARIABLE(2);
                bias = INPUT_VARIABLE(3);
            }

            auto z = OUTPUT_VARIABLE(0);

            const int kY = INT_ARG(0);
            const int kX = INT_ARG(1);
            const int sY = INT_ARG(2);
            const int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            const int dY = INT_ARG(6);
            const int dX = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;

            int oY = 0;
            int oX = 0;

            const int batchSize = input->shapeOf()[0];
            const int outDepth = weightsDepth->shapeOf()[0];
            const int inDepth = weightsDepth->shapeOf()[1];
            const int inY = input->shapeOf()[2];
            const int inX = input->shapeOf()[3];

            REQUIRE_TRUE(inDepth == input->sizeAt(1), 0, "Sconv2d: input number of channels shout match weightsDepth dim 1, but got %i vs %i instead", input->sizeAt(1), inDepth);

            if (weightsPoint != nullptr) {
                REQUIRE_TRUE(weightsPoint->sizeAt(2) == 1  && weightsPoint->sizeAt(3) == 1, 0, "Sconv2d: for sconv2d point-wise kernelHeight and kernelWidth should be equal to 1");
            }

            REQUIRE_TRUE(weightsDepth->shapeOf()[2] == kY && weightsDepth->shapeOf()[3] == kX, 0, "Sconv2d: kernels should have dimensions of [%i, %i], but got [%i, %i] instead", kY, kX, weightsDepth->sizeAt(2), weightsDepth->sizeAt(3));

            if (input->sizeAt(1) == 1) {
                nd4j_debug("Separable conv2d for 1 channel equals to standard conv2d\n","");
                nd4j::ops::conv2d<T> c2d;
                return c2d.execute(&block);
            }

            ConvolutionUtils<T>::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode) {
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);
            }

            std::unique_ptr<NDArray<T>> col2(new NDArray<T>('c', {batchSize, inDepth, kY, kX, oY, oX}));

            // col2d now has shape of [bS, inDepth, kY, kX, oY, oX]
            std::vector<T> extrasIm2Col({(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f, 0.0});

            input->template applyTransform<simdOps::Im2col<T>>(col2.get(), extrasIm2Col.data());

            NDArray<T>* c_ = col2.get()->permute({1, 0, 4, 5, 2, 3});
            NDArray<T>* w_ = weightsDepth->permute({1, 2, 3, 0});

            c_->reshapei('c', {inDepth, batchSize * oY * oX, kY * kX});
            w_->reshapei('c', {inDepth, kY * kX, outDepth});

            // if weightsPoint is null, we'll be doing only depthwise step
            if (weightsPoint == nullptr) {
                // matmul here
                z->reshapei('c', {inDepth, batchSize * oY * oX, outDepth});
                NDArrayFactory<T>::mmulHelper(c_, w_, z);


                if (bias != nullptr) {
                    z->reshapei('c', {-1, (int) bias->lengthOf()});
                    z->addiRowVector(bias);
                }

                z->reshapei('c', {inDepth, batchSize, oY * oX, outDepth});
                z->permutei({1, 0, 3, 2});
                z->reshapei('c', {batchSize, inDepth * outDepth, oY, oX});
            } else {
                // if we have weightsPoint, it means we'll be doing point-wise convolution too now
                auto z_ = NDArrayFactory<T>::mmulHelper(c_, w_);

                z_->reshapei('c', {inDepth, batchSize, oY * oX, outDepth});
                z_->permutei({1, 0, 3, 2});
                z_->reshapei('c', {batchSize, inDepth * outDepth, oY, oX});


                // now we'll be using conv2d op
                nd4j::ops::conv2d<T> op;
                if (bias == nullptr)
                    op.execute({z_, weightsPoint}, {z}, {}, {1, 1, 1, 1, 0, 0, 1, 1, isSameMode ? 1 : 0, 0});
                else
                    op.execute({z_, weightsPoint, bias}, {z}, {}, {1, 1, 1, 1, 0, 0, 1, 1, isSameMode ? 1 : 0, 0});

                delete z_;
            }

            STORE_RESULT(*z);

            delete c_;
            delete w_;

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(sconv2d) {
            auto inShape = inputShape->at(0);
            auto wdShape = inputShape->at(1);
            int *wpShape = nullptr;
            if (inputShape->size() == 3) {
                auto tmp = inputShape->at(2);
                if (shape::rank(tmp) == 4)
                    wpShape = tmp;
            } else if (inputShape->size() == 4) {
                wpShape = inputShape->at(2);
            }


            const int kY = INT_ARG(0);
            const int kX = INT_ARG(1);
            const int sY = INT_ARG(2);
            const int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            const int dY = INT_ARG(6);
            const int dX = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;

            int oY = 0;
            int oX = 0;
            int *newShape;

            // just a depth-wise step
            if (wpShape == nullptr) {

                const int batchSize = inShape[1];
                const int inDepth = inShape[2];
                const int outDepth = wdShape[1];
                const int inY = inShape[3];
                const int inX = inShape[4];

                ConvolutionUtils<T>::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

                if (isSameMode)
                    ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);

                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), int);
                std::vector<int> shape({batchSize, outDepth * inDepth, oY, oX});
                shape::shapeBuffer(4, shape.data(), newShape);

            } else {
                const int batchSize = inShape[1];
                const int inDepth = inShape[2];
                const int outDepth = wpShape[1];
                const int inY = inShape[3];
                const int inX = inShape[4];

                ConvolutionUtils<T>::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

                if (isSameMode)
                    ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);

                ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), int);
                std::vector<int> shape({batchSize, outDepth, oY, oX});
                shape::shapeBuffer(4, shape.data(), newShape);
            }


            return new ShapeList(newShape);
        }


        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(sconv2d_bp, 4, 2, false, 0, 9) {
            NDArray<T> *input = INPUT_VARIABLE(0);
            NDArray<T> *epsilonNext = INPUT_VARIABLE(1);
            NDArray<T> *weightsDepth = INPUT_VARIABLE(2);
            NDArray<T> *weightsPoint = nullptr;
            NDArray<T> *bias = nullptr;

            // bias is still optional
            if (block.width() == 4) {
                auto tmp = INPUT_VARIABLE(3);
                if (tmp->rankOf() == 4)
                    weightsPoint = tmp;
                else
                    bias = tmp;
            } else if (block.width() == 5) {
                weightsPoint = INPUT_VARIABLE(3);
                bias = INPUT_VARIABLE(4);
            }

            //epsilonNext->rankOf() == 4 && weights->rankOf() == 4
            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should be 4D, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(weightsDepth->rankOf() == 4, 0, "Weights should be 4D, but got %iD instead",
                         weightsDepth->rankOf());
            REQUIRE_TRUE(epsilonNext->rankOf() == 4, 0, "Epsilon should be 4D, but got %iD instead",
                         epsilonNext->rankOf());

            if (weightsPoint != nullptr) {
                REQUIRE_TRUE(weightsPoint->rankOf() == 4, 0,
                             "Weights for point-wise convolution should be 4d, but got %D instead",
                             weightsPoint->rankOf());
                REQUIRE_TRUE(weightsPoint->sizeAt(2) == 1 && weightsPoint->sizeAt(3) == 1, 1,
                             "Point-wise weights should be [1, 1], but got [%i, %i] instead", weightsPoint->sizeAt(2),
                             weightsPoint->sizeAt(3));
            }

            NDArray<T> *epsilon = OUTPUT_VARIABLE(0);
            NDArray<T> *gradWD = OUTPUT_VARIABLE(1);
            NDArray<T> *gradWP = nullptr;
            NDArray<T> *gradB = nullptr;

            if (weightsPoint != nullptr)
                gradWP = OUTPUT_VARIABLE(2);

            if (bias != nullptr)
                gradB = OUTPUT_VARIABLE(3);


            // now we're just launching depth-wise bp step

            const int kY = INT_ARG(0);
            const int kX = INT_ARG(1);
            const int sY = INT_ARG(2);
            const int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            const int dY = INT_ARG(6);
            const int dX = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;

            int oY = epsilonNext->sizeAt(2);
            int oX = epsilonNext->sizeAt(3);

            const int batchSize = input->shapeOf()[0];
            const int outDepth = weightsDepth->shapeOf()[0];
            const int inDepth = weightsDepth->shapeOf()[1];
            const int inY = input->shapeOf()[2];
            const int inX = input->shapeOf()[3];

            // if weightsPont are defiend - then we're going to do point-wise backprop first
            NDArray<T> *epsilon_;
            if (weightsPoint != nullptr) {
                nd4j::ops::sconv2d<T> opFF;
                auto result = opFF.execute({input, weightsDepth}, {}, {kY, kX, sY, sX, pY, pX, dY, dX, isSameMode ? 1 : 0});
                auto depthInput = result->at(0);

                nd4j::ops::conv2d_bp<T> opBP;

                epsilon_ = new NDArray<T>('c', {batchSize, weightsDepth->sizeAt(0) * weightsDepth->sizeAt(1), oY, oX});

                if (bias == nullptr)
                    opBP.execute({depthInput, weightsPoint, epsilonNext}, {epsilon_, gradWP}, {}, {1, 1, 1, 1, 0, 0, 1, 1, isSameMode ? 1 : 0});
                else
                    opBP.execute({depthInput, weightsPoint, bias, epsilonNext}, {epsilon_, gradWP, gradB}, {}, {1, 1, 1, 1, 0, 0, 1, 1, isSameMode ? 1 : 0});

                epsilonNext = epsilon_;

                delete result;
            }


            bool hasCol = CHECK_STASH("im2col");
            NDArray<T> *col = nullptr;
            if (hasCol)
                col = UNSTASH("im2col")
            else {
                col = new NDArray<T>('c', {batchSize, inDepth, kY, kX, oY, oX});

                // col2d now has shape of [bS, inDepth, kY, kX, oY, oX]
                std::vector<T> extrasIm2Col({(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX,
                                             isSameMode ? (T) 1.0f : (T) 0.0f, 0.0});

                input->template applyTransform<simdOps::Im2col<T>>(col, extrasIm2Col.data());
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

            gW_->reshapei('c', {inDepth, outDepth, kY, kX});
            gW_->permutei({1, 0, 2, 3});
            gradWD->assign(gW_);

            delete gW_;
            delete col_;
            if (!hasCol)
                delete col;

            // calculating epsilon here
            auto w_ = weightsDepth->permute({1, 2, 3, 0});
            w_->reshapei('c', {inDepth, kY * kX, outDepth});

            auto gcol = NDArrayFactory<T>::mmulHelper(w_, eN_);
            gcol->reshapei('c', {inDepth, kY, kX, batchSize, oY, oX});
            gcol->permutei({3, 0, 1, 2, 4, 5});

            std::vector<T> extrasCol2Im({(T) sY, (T) sX, (T) pY, (T) pX, (T) inY, (T) inX, (T) dY, (T) dX,
                                         isSameMode ? (T) 1.0f : (T) 0.0f});

            // we're sure that col2im result will have the same size as original image
            //auto rCol = new NDArray<T>('c', {batchSize, inDepth, inY, inX});
            gcol->template applyTransform<simdOps::Col2Im<T>>(epsilon, extrasCol2Im.data());


            delete eN_;
            delete gcol;
            delete w_;


            if (weightsPoint == nullptr) {
                if (bias != nullptr) {
                    // calculating gradB, if defined
                    auto eN_ = epsilonNext->permute({0, 2, 3, 1});
                    auto sum = eN_->template reduceAlongDimension<simdOps::Sum<T>>({0, 1, 2});
                    gradB->assign(sum);
                    delete sum;

                    STORE_3_RESULTS(*epsilon, *gradWD, *gradB);
                } else {
                    STORE_2_RESULTS(*epsilon, *gradWD);
                }
            } else {
                if (bias != nullptr) {
                    STORE_4_RESULTS(*epsilon, *gradWD, *gradWP, *gradB);
                } else {
                    STORE_3_RESULTS(*epsilon, *gradWD, *gradWP);
                }
            }

            if (weightsPoint != nullptr)
                delete epsilonNext;

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(sconv2d_bp) {
            auto inShape = inputShape->at(0);
            auto eShape = inputShape->at(1);
            auto wdShape = inputShape->at(2);
            int *wpShape = nullptr;
            int *bShape = nullptr;

            // bias is optional thing, and might be absent
            if (inputShape->size() == 5) {
                wpShape = inputShape->at(3);
                bShape = inputShape->at(4);
            } else if (inputShape->size() == 4) {
                auto tmp = inputShape->at(3);
                if (shape::rank(tmp) == 4)
                    wpShape = tmp;
                else
                    bShape = tmp;
            }

            int *newInShape;
            int *newWdShape;
            ALLOCATE(newInShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            ALLOCATE(newWdShape, block.getWorkspace(), shape::shapeInfoLength(wdShape), int);

            memcpy(newInShape, inShape, shape::shapeInfoByteLength(inShape));
            memcpy(newWdShape, wdShape, shape::shapeInfoByteLength(wdShape));

            auto shapes = new ShapeList({newInShape, newWdShape});

            if (wpShape != nullptr) {
                int *newWpShape;
                ALLOCATE(newWpShape, block.getWorkspace(), shape::shapeInfoLength(wpShape), int);
                memcpy(newWpShape, wpShape, shape::shapeInfoByteLength(wpShape));

                shapes->push_back(newWpShape);
            }

            if (bShape != nullptr) {
                int *newBShape;
                ALLOCATE(newBShape, block.getWorkspace(), shape::shapeInfoLength(bShape), int);
                memcpy(newBShape, bShape, shape::shapeInfoByteLength(bShape));

                shapes->push_back(newBShape);
            }

            return shapes;
        }
    }
}