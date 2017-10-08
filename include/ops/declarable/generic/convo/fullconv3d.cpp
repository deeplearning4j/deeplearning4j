//
// Created by raver119 on 08.10.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(fullconv3d, 5, 1, false, 0, 13) {

            NDArray<T> *input = block.getVariables().at(0)->getNDArray();
            NDArray<T> *weights = block.getVariables().at(1)->getNDArray();
            NDArray<T> *bias = block.getVariables().at(2)->getNDArray();
            NDArray<T> *columns = block.getVariables().at(3)->getNDArray();
            NDArray<T> *ones = block.getVariables().at(4)->getNDArray();

            REQUIRE_TRUE(weights->rankOf() == 5, 0, "Weights should be 5D, got %i instead", weights->rankOf());
            REQUIRE_TRUE(input->rankOf() == 5, 0, "Input should be 5D, got %i instead", input->rankOf());

            int dT = block.getIArguments()->at(0);
            int dW = block.getIArguments()->at(1);
            int dH = block.getIArguments()->at(2);
            int pT = block.getIArguments()->at(3);
            int pW = block.getIArguments()->at(4);
            int pH = block.getIArguments()->at(5);
            int dilationT = block.getIArguments()->at(6);
            int dilationW = block.getIArguments()->at(7);
            int dilationH = block.getIArguments()->at(8);
            int aT = block.getIArguments()->at(9);
            int aW = block.getIArguments()->at(10);
            int aH = block.getIArguments()->at(11);
            bool biasUsed = block.getIArguments()->at(12) != 0;


            REQUIRE_TRUE(dT > 0 && dW > 0 && dH > 0, 11,
                         "stride should be greater than zero, but got dT: %d dH: %d dW: %d", dT, dH, dW);
            REQUIRE_TRUE(dilationT > 0 && dilationW > 0 && dilationH > 0, 15,
                         "dilation should be greater than zero, but got dilationT: %d, dilationH: %d, dilationW: %d",
                         dilationT, dilationH, dilationW);
            REQUIRE_TRUE((aT < dT || aT < dilationT)
                         && (aW < dW || aW < dilationW)
                         && (aH < dH || aH < dilationH), 15,
                         "output padding must be smaller than either stride or dilation,"
                                 " but got aT: %d aH: %d aW: %d dT: %d dH: %d dW: %d "
                                 "dilationT: %d dilationH: %d dilationW: %d",
                         aT, aH, aW, dT, dH, dW, dilationT, dilationH, dilationW);

            NDArray<T> *output = this->getZ(block);

            const int nInputPlane  = weights->shapeOf()[0];
            const int nOutputPlane = weights->shapeOf()[1];
            const int kT           = weights->shapeOf()[2];
            const int kH           = weights->shapeOf()[3];
            const int kW           = weights->shapeOf()[4];

            const Nd4jIndex inputWidth   = input->shapeOf()[4];
            const Nd4jIndex inputHeight  = input->shapeOf()[3];
            const Nd4jIndex inputDepth   = input->shapeOf()[2];
            const Nd4jIndex outputDepth  = (inputDepth - 1) * dT - 2*pT + (dilationT * (kT - 1) + 1) + aT;
            const Nd4jIndex outputHeight = (inputHeight - 1) * dH - 2*pH + (dilationH * (kH - 1) + 1) + aH;
            const Nd4jIndex outputWidth  = (inputWidth - 1) * dW - 2*pW + (dilationW * (kW - 1) + 1) + aW;

            const Nd4jIndex batchSize = input->shapeOf()[0];

            REQUIRE_TRUE(output->isSameShape({ (int) batchSize, (int)nOutputPlane, (int)outputDepth, (int)outputHeight, (int)outputWidth}), 0, "Output should have shape of [%i, %i, %i, %i, %i], but got [%i, %i, %i, %i, %i] instead", (int) batchSize, (int)nOutputPlane, (int)outputDepth, (int)outputHeight, (int)outputWidth, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3), output->sizeAt(4));

            std::unique_ptr<ArrayList<T>> inputs(NDArrayFactory<T>::allExamples(input));
            std::unique_ptr<ArrayList<T>> outputs(NDArrayFactory<T>::allExamples(output));
            for (int e = 0; e < batchSize; e++) {
                auto tadIn = inputs->at(e);
                auto tadOut = outputs->at(e);

                const int m = weights->shapeOf()[1] * weights->shapeOf()[2] * weights->shapeOf()[3] * weights->shapeOf()[4];
                const int n = columns->shapeOf()[1];
                const int k = weights->shapeOf()[0];

                nd4j::blas::GEMM<T>::op('c', 'n', 't', m, n, k,
                                        1.0,
                                        tadIn->getBuffer(), n,
                                        weights->getBuffer(), m,
                                        0.0,
                                        columns->getBuffer(), n);

                ConvolutionUtils<T>::_col2vol(columns->getBuffer(),
                                              nOutputPlane, outputDepth, outputHeight, outputWidth,
                                              inputDepth, inputHeight, inputWidth,
                                              kT, kH, kW,
                                              pT, pH, pW,
                                              dT, dH, dW,
                                              dilationT,  dilationH,  dilationW,
                                              tadOut->getBuffer());


                const int m_ = nOutputPlane;
                const int n_ = outputDepth * outputHeight * outputWidth;
                const int k_ = 1;

                if (biasUsed) {
                    nd4j::blas::GEMM<T>::op('c', 't', 'n', n_, m_, k_,
                                            1.0,
                                            ones->getBuffer(), k_,
                                            bias->getBuffer(), k_,
                                            1.0,
                                            tadOut->getBuffer(), n_);
                }
            }

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(fullconv3d) {

            int* input = inputShape->at(0);
            int* weights = inputShape->at(1);

            int dT = block.getIArguments()->at(0);
            int dW = block.getIArguments()->at(1);
            int dH = block.getIArguments()->at(2);
            int pT = block.getIArguments()->at(3);
            int pW = block.getIArguments()->at(4);
            int pH = block.getIArguments()->at(5);
            int dilationT = block.getIArguments()->at(6);
            int dilationW = block.getIArguments()->at(7);
            int dilationH = block.getIArguments()->at(8);
            int aT = block.getIArguments()->at(9);
            int aW = block.getIArguments()->at(10);
            int aH = block.getIArguments()->at(11);
            bool biasUsed = block.getIArguments()->at(12) != 0;

            int *shapeOf;
            int *newShape;
            ALLOCATE(shapeOf, block.getWorkspace(), 5, int);
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(5), int);

            const int nInputPlane  = weights[1];
            const int nOutputPlane = weights[2];
            const int kT           = weights[3];
            const int kH           = weights[4];
            const int kW           = weights[5];

            const int batchSize          = input[1];
            const Nd4jIndex inputWidth   = input[5];
            const Nd4jIndex inputHeight  = input[4];
            const Nd4jIndex inputDepth   = input[3];
            const Nd4jIndex outputDepth  = (inputDepth - 1) * dT - 2*pT + (dilationT * (kT - 1) + 1) + aT;
            const Nd4jIndex outputHeight = (inputHeight - 1) * dH - 2*pH + (dilationH * (kH - 1) + 1) + aH;
            const Nd4jIndex outputWidth  = (inputWidth - 1) * dW - 2*pW + (dilationW * (kW - 1) + 1) + aW;

            nd4j::ArrayUtils::toIntPtr({(int) batchSize, (int)nOutputPlane, (int)outputDepth, (int)outputHeight, (int)outputWidth}, shapeOf);

            shape::shapeBuffer(5, shapeOf, newShape);

            RELEASE(shapeOf, block.getWorkspace());

            return new ShapeList(newShape);
        }

//////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(fullconv3d_bp, 5, 1, false, 0, 13) {

            NDArray<T> *input = block.getVariables().at(0)->getNDArray();
            NDArray<T> *gradNext = block.getVariables().at(1)->getNDArray();
            NDArray<T> *weights = block.getVariables().at(2)->getNDArray();
            NDArray<T> *finput = block.getVariables().at(3)->getNDArray();

            // not used
            NDArray<T> *fgradInput = block.getVariables().at(4)->getNDArray();


            REQUIRE_TRUE(weights->rankOf() == 5, 0, "Weights should be 5D, got %i instead", weights->rankOf());
            REQUIRE_TRUE(input->rankOf() == 5, 0, "Input should be 5D, got %i instead", input->rankOf());

            NDArray<T> *output = this->getZ(block);

            int dT = block.getIArguments()->at(0);
            int dW = block.getIArguments()->at(1);
            int dH = block.getIArguments()->at(2);
            int pT = block.getIArguments()->at(3);
            int pW = block.getIArguments()->at(4);
            int pH = block.getIArguments()->at(5);
            int dilationT = block.getIArguments()->at(6);
            int dilationW = block.getIArguments()->at(7);
            int dilationH = block.getIArguments()->at(8);
            int aT = block.getIArguments()->at(9);
            int aW = block.getIArguments()->at(10);
            int aH = block.getIArguments()->at(11);
            bool biasUsed = block.getIArguments()->at(12) != 0;

            const int nInputPlane  = (int)weights->shapeOf()[0];
            const int nOutputPlane = (int)weights->shapeOf()[1];
            const int kT           = (int)weights->shapeOf()[2];
            const int kH           = (int)weights->shapeOf()[3];
            const int kW           = (int)weights->shapeOf()[4];

            const Nd4jIndex inputWidth   = input->shapeOf()[4];
            const Nd4jIndex inputHeight  = input->shapeOf()[3];
            const Nd4jIndex inputDepth   = input->shapeOf()[2];
            const Nd4jIndex outputDepth  = (inputDepth - 1) * dT - 2*pT + (dilationT * (kT - 1) + 1) + aT;
            const Nd4jIndex outputHeight = (inputHeight - 1) * dH - 2*pH + (dilationH * (kH - 1) + 1) + aH;
            const Nd4jIndex outputWidth  = (inputWidth - 1) * dW - 2*pW + (dilationW * (kW - 1) + 1) + aW;

            const Nd4jIndex batchSize = input->shapeOf()[0];


            REQUIRE_TRUE(output->isSameShape({(int) batchSize, (int) nInputPlane, (int) inputDepth, (int) inputHeight, (int) inputWidth}) ,0, "Output should have shape of [%i, %i, %i, %i, %i], but got [%i, %i, %i, %i, %i] instead", (int) batchSize, (int) nInputPlane, (int) inputDepth, (int) inputHeight, (int) inputWidth, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3), output->sizeAt(4));

            output->assign(0.0);

            // FIXME: non-inplace reshape!!!!
            NDArray<T> *gradColumns;
            //auto gradColumns = finput->reshape('c', {nOutputPlane*kW*kH*kT, inputDepth*inputHeight*inputWidth });

            std::unique_ptr<ArrayList<T>> tadsNext(NDArrayFactory<T>::allExamples(gradNext));
            std::unique_ptr<ArrayList<T>> tadsOutput(NDArrayFactory<T>::allExamples(output));
            for (int e = 0; e < tadsNext->size(); e++) {
                auto tadNext = tadsNext->at(e);
                auto tadOutput = tadsOutput->at(e);

                ConvolutionUtils<T>::_vol2col(
                        tadNext->getBuffer(),
                        nOutputPlane, outputDepth, outputHeight, outputWidth,
                        kT, kH, kW,
                        pT, pH, pW,
                        dT, dH, dW,
                        dilationT,  dilationH,  dilationW,
                        gradColumns->getBuffer());

                const long m = weights->shapeOf()[0];
                const long n = gradColumns->shapeOf()[1];
                const long k = weights->shapeOf()[1] * weights->shapeOf()[2] * weights->shapeOf()[3] * weights->shapeOf()[4];

                nd4j::blas::GEMM<T>::op('f', 'n', 'n',
                                        n, m, k,
                                        1.0f,
                                        gradColumns->getBuffer(), n,
                                        weights->getBuffer(), k,
                                        0,
                                        tadOutput->getBuffer(), n
                );
            }


            STORE_RESULT(*output);

            delete gradColumns;
            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(fullconv3d_bp) {
            // output shape equals to input shape, all out of sudden
            int* newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);
            memcpy(newShape, inputShape->at(0), shape::shapeInfoByteLength(inputShape->at(0)));
            return new ShapeList(newShape);
        }

//////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(fullconv3d_grad, 4, 2, false, 1, 13) {

            NDArray<T> *input = block.getVariables().at(0)->getNDArray();
            NDArray<T> *epsilon = block.getVariables().at(1)->getNDArray();
            NDArray<T> *columns = block.getVariables().at(2)->getNDArray();
            NDArray<T> *ones = block.getVariables().at(3)->getNDArray();

            REQUIRE_TRUE(input->rankOf() == epsilon->rankOf(), 0, "Rank of input (%i) & epsilon (%i) should be equal", input->rankOf(), epsilon->rankOf());
            REQUIRE_TRUE(input->sizeAt(0) == epsilon->sizeAt(0), 1, "Batch size should be equal for input and epsilon");

            NDArray<T> *gradWeight = this->getZ(block);
            NDArray<T> *gradBias = this->getZ(block, 1);

            REQUIRE_TRUE(gradBias->sizeAt(0) == gradWeight->sizeAt(1), 0, "Bias shape mismatch");

            int dT = block.getIArguments()->at(0);
            int dW = block.getIArguments()->at(1);
            int dH = block.getIArguments()->at(2);
            int pT = block.getIArguments()->at(3);
            int pW = block.getIArguments()->at(4);
            int pH = block.getIArguments()->at(5);
            int dilationT = block.getIArguments()->at(6);
            int dilationW = block.getIArguments()->at(7);
            int dilationH = block.getIArguments()->at(8);
            int aT = block.getIArguments()->at(9);
            int aW = block.getIArguments()->at(10);
            int aH = block.getIArguments()->at(11);
            bool biasUsed = block.getIArguments()->at(12) != 0;

            T scale = block.getTArguments()->at(0);

            int nInputPlane  = (int)gradWeight->shapeOf()[0];
            int nOutputPlane = (int)gradWeight->shapeOf()[1];
            int kT           = (int)gradWeight->shapeOf()[2];
            int kH           = (int)gradWeight->shapeOf()[3];
            int kW           = (int)gradWeight->shapeOf()[4];


            const Nd4jIndex inputWidth   = input->shapeOf()[4];
            const Nd4jIndex inputHeight  = input->shapeOf()[3];
            const Nd4jIndex inputDepth   = input->shapeOf()[2];
            const Nd4jIndex outputDepth  = (inputDepth - 1) * dT - 2*pT + (dilationT * (kT - 1) + 1) + aT;
            const Nd4jIndex outputHeight = (inputHeight - 1) * dH - 2*pH + (dilationH * (kH - 1) + 1) + aH;
            const Nd4jIndex outputWidth  = (inputWidth - 1) * dW - 2*pW + (dilationW * (kW - 1) + 1) + aW;


            REQUIRE_TRUE(gradWeight->isContiguous(), 0, "gradWight should be continuous");
            REQUIRE_TRUE(gradBias->isContiguous(), 0, "gradBias should be continuous");
            REQUIRE_TRUE(ones->rankOf() == 3, 0, "Ones should have rank 3, got %i instead", ones->rankOf());

            REQUIRE_TRUE(ones->isSameShape({outputDepth, outputHeight, outputWidth}), 0, "");

            ones->assign(1.0);

            std::unique_ptr<ArrayList<T>> tadsInput(NDArrayFactory<T>::allExamples(input));
            std::unique_ptr<ArrayList<T>> tadsEpsilon(NDArrayFactory<T>::allExamples(epsilon));

            for (int e = 0; e < tadsInput->size(); e++) {
                auto tadInput = tadsInput->at(e);
                auto tadEpsilon = tadsEpsilon->at(e);

                ConvolutionUtils<T>::_vol2col(
                        tadEpsilon->getBuffer(), nOutputPlane,
                        outputDepth, outputHeight, outputWidth,
                        kT, kH, kW,
                        pT, pH, pW,
                        dT, dH, dW,
                        dilationT,  dilationH,  dilationW,
                        columns->getBuffer()
                );

                const Nd4jIndex n = columns->shapeOf()[0];   // nOutputPlane * kt * kh * kw
                const Nd4jIndex m = tadInput->shapeOf()[0];   // nInputPlane
                const Nd4jIndex k = columns->shapeOf()[1];

                nd4j::blas::GEMM<T>::op('f', 't', 'n',
                                        n, m, k,
                                        scale,
                                        columns->getBuffer(), k,
                                        tadInput->getBuffer(), k,
                                        1,
                                        gradWeight->getBuffer(), n);

                const Nd4jIndex m_ = nOutputPlane;
                const Nd4jIndex k_ = outputDepth * outputHeight * outputWidth;


                if (gradBias) {
                    nd4j::blas::GEMV<T>::op('t',
                                            k_, m_,
                                            scale,
                                            tadEpsilon->getBuffer(), k_,
                                            ones->getBuffer(), 1, (T)1.0f,
                                            gradBias->getBuffer(), 1);
                }
            }


            STORE_2_RESULTS(*gradWeight, *gradBias);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(fullconv3d_grad) {
            auto list = new ShapeList();

            // _grad ops MUST have output arrays provided

            return list;
        }

    }
}