//
// 3D convolutions are based on pytorch - https://github.com/pytorch/pytorch
//

#ifndef LIBND4J_CONVO_OPS_H
#define LIBND4J_CONVO_OPS_H

#include <op_boilerplate.h>
#include <memory>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpRegistrator.h>
#include <declarable/generic/helpers/convolutions.h>



namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(conv2d, 2, 1, false, 0, 9) {
            // basically im2col + gemm
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* weights = INPUT_VARIABLE(1);
            NDArray<T>* bias = nullptr;


            REQUIRE_TRUE(input->rankOf() == 4, 0, "Conv2D: input should be 4D NDArray, but got %i instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Conv2D: weights should be 4D NDArray, but got %i instead", weights->rankOf());

            if (block.width() == 3)
                bias = INPUT_VARIABLE(2);

            const int kY = INT_ARG(0);
            const int kX = INT_ARG(1);
            const int sY = INT_ARG(2);
            const int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            const int dY = INT_ARG(6);
            const int dX = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;

            REQUIRE_TRUE(weights->sizeAt(2) == kY, 0, "Conv2D: weights dim 2 should be equal to %i, but got %i instead. Not a NCHW?", kY, weights->sizeAt(2));
            REQUIRE_TRUE(weights->sizeAt(3) == kX, 0, "Conv2D: weights dim 3 should be equal to %i, but got %i instead. Not a NCHW?", kX, weights->sizeAt(3));
            REQUIRE_TRUE(weights->sizeAt(1) == input->sizeAt(1), 0, "Conv2D: weights dim 1 should be equal to number of input channels. But got %i vs %i. Not a NCHW?", weights->sizeAt(1), input->sizeAt(1))

            if (bias != nullptr) {
                REQUIRE_TRUE(weights->sizeAt(0) == bias->lengthOf(), 0, "Conv2D: bias length should be equal to outChannels, but got %i instead", bias->lengthOf());
            }

            int oY = 0;
            int oX = 0;

            const int batchSize = input->shapeOf()[0];
            const int outDepth = weights->shapeOf()[0];
            const int inDepth = weights->shapeOf()[1];
            const int inY = input->shapeOf()[2];
            const int inX = input->shapeOf()[3];

            REQUIRE_TRUE(weights->shapeOf()[2] == kY && weights->shapeOf()[3] == kX, 0, "Kernels should have dimensions of [%i, %i], but got [%i, %i] instead", kY, kX, weights->sizeAt(2), weights->sizeAt(3));

            ConvolutionUtils<T>::calcOutHWpool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode)
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);

            NDArray<T>* output = OUTPUT_VARIABLE(0);

            Nd4jIndex prod = batchSize * outDepth * oY * oX;
            REQUIRE_TRUE(output->sizeAt(0) == batchSize && output->sizeAt(1) == outDepth && output->sizeAt(2) == oY && output->sizeAt(3) == oX, 0, "Expected output shape is [%i, %i, %i, %i] but got [%i, %i, %i, %i] instead", batchSize, outDepth, oY, oX, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3))
            REQUIRE_TRUE(output->lengthOf() == prod, 0, "Z should have total length of %i, but got %i instead", prod, output->lengthOf());

            std::unique_ptr<NDArray<T>> col(new NDArray<T>('c', {batchSize, oY, oX, inDepth, kY, kX}));
            std::unique_ptr<NDArray<T>> col2(col.get()->permute({0, 3, 4, 5, 1, 2}));

            std::vector<T> extrasIm2Col({(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

            input->template applyTransform<simdOps::Im2col<T>>(col2.get(), extrasIm2Col.data());

            std::unique_ptr<NDArray<T>> im2col2d(col->reshape('c', {batchSize * oY * oX, inDepth * kY * kX}));
            std::unique_ptr<NDArray<T>> permutedW(weights->permute({3, 2, 1, 0}));
            std::unique_ptr<NDArray<T>> reshapedW(permutedW.get()->reshape('f', {kX * kY * inDepth, outDepth}));

            output->reshapei('f', {im2col2d.get()->rows(), reshapedW.get()->columns()});

            NDArrayFactory<T>::mmulHelper(im2col2d.get(), reshapedW.get(), output, 1.0, 0.0);

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


        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(conv2d_bp, 3, 2, false, 0, 9) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* weights = INPUT_VARIABLE(1);
            NDArray<T>* bias = nullptr;
            NDArray<T>* epsilonNext;

            NDArray<T>* epsilon = OUTPUT_VARIABLE(0);
            NDArray<T>* gradW = OUTPUT_VARIABLE(1);
            NDArray<T>* gradB = nullptr;
            if (block.width() == 3)
                epsilonNext = INPUT_VARIABLE(2);
            else {
                bias = INPUT_VARIABLE(2);
                epsilonNext = INPUT_VARIABLE(3);
                gradB = OUTPUT_VARIABLE(2);
            }

            REQUIRE_TRUE(input->rankOf() == 4, 0, "Conv2D expects 4D input, but got %i instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Conv2D expects 4D weights, but got %i instead", weights->rankOf());
            REQUIRE_TRUE(epsilonNext->rankOf() == 4, 0, "Conv2D expects 4D epsilons, but got %i instead", epsilonNext->rankOf());


            const int kY = INT_ARG(0);
            const int kX = INT_ARG(1);
            const int sY = INT_ARG(2);
            const int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            const int dY = INT_ARG(6);
            const int dX = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;

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
            std::vector<T> extrasIm2Col({(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

            input->template applyTransform<simdOps::Im2col<T>>(col2, extrasIm2Col.data());
            auto im2col2d = col->reshape('c', {batchSize * oY * oX, inDepth * kY * kX});


            auto gradW2d = gradW->reshape('c', {outDepth, inDepth * kY * kX});
            gradW2d->transposei();

            im2col2d->transposei();
            auto eN2dT = epsilonNext2d->transpose();

            nd4j::NDArrayFactory<T>::mmulHelper(im2col2d, eN2dT, gradW2d);

            delete gradW2d;
            delete col;
            delete col2;
            delete im2col2d;
            delete eN2dT;


            // epsilon
            auto pWeights = weights->permute({3, 2, 1, 0});
            pWeights->reshapei('f', {inDepth * kY * kX, outDepth});

            auto eps2d = nd4j::NDArrayFactory<T>::mmulHelper(pWeights, epsilonNext2d);

            auto eps6d = eps2d->reshape('f', {kX, kY, inDepth, oX, oY, batchSize});
            eps6d->permutei({5, 2, 1, 0, 4, 3});

            std::vector<T> extrasCol2Im({(T) sY, (T) sX, (T) pY, (T) pX, (T) inY, (T) inX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});
            eps6d->template applyTransform<simdOps::Col2Im<T>>(epsilon, extrasCol2Im.data());

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
    }
}

#endif //LIBND4J_CONVO_OPS_H
