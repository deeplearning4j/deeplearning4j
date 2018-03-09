//
// Created by raver119 on 29/10/17.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
  
        // CUSTOM_OP_IMPL(deconv2d, 2, 1, false, 0, 9) {
            
        //     NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
        //     NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
        //     NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
        //     NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
    
        //     REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM CONV2D OP: rank of input array must be equal to 4 !");
        //     REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM CONV2D OP: rank of weights array must be equal to 4 !");
                                     
        //     int kH = INT_ARG(0);                                                        // filter(kernel) height
        //     int kW = INT_ARG(1);                                                        // filter(kernel) width
        //     int sH = INT_ARG(2);                                                        // strides height
        //     int sW = INT_ARG(3);                                                        // strides width
        //     int pH = INT_ARG(4);                                                        // paddings height
        //     int pW = INT_ARG(5);                                                        // paddings width
        //     int dH = INT_ARG(6);                                                        // dilations height
        //     int dW = INT_ARG(7);                                                        // dilations width
        //     int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
        //     int isNCHW     = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // 0-NCHW,  1-NHWC

        //     if(!isNCHW) {
        //         input   = input->permute({0, 3, 1, 2});                                 // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        //         weights = weights->permute({4, 3, 1, 2});                               // [kH, kW, iC, oC] -> [oC, iC, kH, kW]                 
        //         output  = output->permute({0, 3, 1, 2});                                // [bS, oH, oW, oC] -> [bS, oC, oH, oW] 
        //     }

        //     int bS = input->sizeAt(0);           // batch size
        //     int iC = input->sizeAt(1);           // input channels        
        //     int iH = input->sizeAt(2);           // input height
        //     int iW = input->sizeAt(3);           // input width
        //     int oC = weights->sizeAt(0);         // output channels        
        //     int oH = output->sizeAt(2);          // output height
        //     int oW = output->sizeAt(3);          // output width    
    
        //     REQUIRE_TRUE(weights->sizeAt(0) == oC && weights->sizeAt(2) == kH && weights->sizeAt(3) == kW, 0, "CUSTOM DECONV2D OP: wrong shape of weights array !");    
        //     if (bias) {
        //         REQUIRE_TRUE(bias->rankOf() <= 2 ,   0, "CUSTOM DECONV2D OP: rank of biases array must be equal to 1 or 2!");
        //         REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM DECONV2D OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
        //     }            
    
        //     if(isSameMode)                       // SAME        
        //         ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

        //     NDArray<T>* columns = nd4j::NDArrayFactory<T>::tensorDot(weights, input, {0}, {1});
        //     columns->permutei({3, 0, 1, 2, 4, 5});            
        //     std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
        //     columns->template applyTransform<simdOps::Col2Im<T>>(output, extrasCol2Im.data());
            
        //     if(bias)
        //         output->template applyBroadcast<simdOps::Add<T>>({1}, bias);

        //     if(!isNCHW) {
        //         delete input;                
        //         delete weights;
        //         delete output;    
        //     }    
            
        //     delete columns;        
    
        //     return Status::OK();

        // }

        // DECLARE_SHAPE_FN(deconv2d) {

        //     int kH = INT_ARG(0);                                                        // filter(kernel) height
        //     int kW = INT_ARG(1);                                                        // filter(kernel) width
        //     int sH = INT_ARG(2);                                                        // strides height
        //     int sW = INT_ARG(3);                                                        // strides width
        //     int pH = INT_ARG(4);                                                        // paddings height
        //     int pW = INT_ARG(5);                                                        // paddings width
        //     int dH = INT_ARG(6);                                                        // dilations height
        //     int dW = INT_ARG(7);                                                        // dilations width
        //     int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
        //     int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // 0-NDHWC, 1-NCDHW

        //     int indIH = isNCHW == 1 ? 2 : 1;
        //     int indIC = isNCHW == 1 ? 1 : 3;
        //     int indOC = isNCHW == 1 ? 0 : 3;

        //     int* inputShapeInfo   = inputShape->at(0);
        //     int* weightsShapeInfo = inputShape->at(1);

        //     int bS = inputShapeInfo[1];                         // batch size
        //     int iH = inputShapeInfo[indIH+1];                   // input height
        //     int iW = inputShapeInfo[indIH+2];                   // input width
        //     int iC = inputShapeInfo[indIC+1];                   // input channels        
        //     int oC = weightsShapeInfo[indOC+1];                 // output channels

        //     int oH, oW;                                         // output height, width
        //     ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);
    
        //     int* outputShapeInfo = nullptr;
        //     ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);

        //     outputShapeInfo[0] = 4;
        //     outputShapeInfo[1] = bS;

        //     if (isNCHW) {
        //         outputShapeInfo[2] = oC;
        //         outputShapeInfo[3] = oH;
        //         outputShapeInfo[4] = oW;
        //     } else {
        //         outputShapeInfo[2] = oH;
        //         outputShapeInfo[3] = oW;
        //         outputShapeInfo[4] = oC;
        //     }
    
        //     shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

        //     return SHAPELIST(outputShapeInfo);
        // }

           CUSTOM_OP_IMPL(deconv2d, 2, 1, false, 0, 10) {
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* weights = INPUT_VARIABLE(1);
            NDArray<T>* bias = nullptr;
            if (block.width() > 2)
                bias = INPUT_VARIABLE(2);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should be 4D, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Weights should be 4D, but got %iD instead", weights->rankOf());

            int oD = weights->sizeAt(1);

            if (bias != nullptr) {
                REQUIRE_TRUE(bias->isVector(), 0, "Bias should be vector");
                REQUIRE_TRUE(bias->lengthOf() == oD, 0, "Bias length be equal to outpuDepth, but got %i instead", bias->lengthOf());
            }

            int iY = input->sizeAt(2);
            int iX = input->sizeAt(3);

            int kY = INT_ARG(0);
            int kX = INT_ARG(1);
            int sY = INT_ARG(2);
            int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            int dY = INT_ARG(6);
            int dX = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;
            bool isNCHW = true;
            if (block.getIArguments()->size() > 9)
                isNCHW = INT_ARG(9) == 0;

            NDArray<T> *z = OUTPUT_VARIABLE(0);

            int oY = z->sizeAt(2);
            int oX = z->sizeAt(3);

            //weights->printShapeInfo("weights");
            //input->printShapeInfo("input");
            //auto wP = weights->permute({1, 0, 2, 3});
            auto gcol = nd4j::NDArrayFactory<T>::tensorDot(weights, input, {0}, {1});
            gcol->permutei({3, 0, 1, 2, 4, 5});

            std::vector<T> extrasCol2Im({(T) sY, (T) sX, (T) pY, (T) pX, (T) oY, (T) oX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f});

            gcol->template applyTransform<simdOps::Col2Im<T>>(z, extrasCol2Im.data());

            delete gcol;
            //delete wP;

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

            int oC = shape::shapeOf(wShape)[1];
            int kY = INT_ARG(0);
            int kX = INT_ARG(1);
            int sY = INT_ARG(2);
            int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            int dY = INT_ARG(6);
            int dX = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;
            bool isNCHW = true;
            if (block.getIArguments()->size() > 9)
                isNCHW = INT_ARG(9) == 0;

            int oY, oX;

            if (isSameMode) {
                oY = sY * iY;
                oX = sX * iX;
            } else {
                int ekY, ekX;
                if (dY == 1 && dX == 1) {
                    ekY = kY;
                    ekX = kX;
                } else {
                    ekY = kY + (kY - 1) * (dY - 1);
                    ekX = kX + (kX - 1) * (dX - 1);
                }

                oY = sY * (iY - 1) + ekY - 2 * pY;
                oX = sX * (iX - 1) + ekX - 2 * pX;
            }

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), int);
            std::vector<int> shape({B, oC, oY, oX});
            shape::shapeBuffer(4, shape.data(), newShape);

            return SHAPELIST(newShape);
        }

        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(deconv2d_bp, 4, 2, false, 0, 9) {
            NDArray<T> *input = INPUT_VARIABLE(0);
            NDArray<T> *weights = INPUT_VARIABLE(1);
            NDArray<T> *bias = nullptr;
            NDArray<T> *epsilonNext = nullptr; //INPUT_VARIABLE(2);

            REQUIRE_TRUE(block.width() >= 3, 0, "deconv2d_bp: Number of input variables should be 3 or 4, but got %i instead", block.width());

            // bias is still optional
            if (block.width() == 4) {
                bias = INPUT_VARIABLE(2);
                epsilonNext = INPUT_VARIABLE(3);
            } else if (block.width() == 3) {
                epsilonNext = INPUT_VARIABLE(2);
            }

            REQUIRE_TRUE(epsilonNext->rankOf() == 4, 0, "epsilon should have rank of 4, but got %i instead", epsilonNext->rankOf());

            //epsilonNext->rankOf() == 4 && weights->rankOf() == 4
            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should be 4D, but got %iD instead", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "Weights should be 4D, but got %iD instead", weights->rankOf());
            REQUIRE_TRUE(epsilonNext->rankOf() == 4, 0, "Epsilon should be 4D, but got %iD instead", epsilonNext->rankOf());

            int kY = INT_ARG(0);
            int kX = INT_ARG(1);
            int sY = INT_ARG(2);
            int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            int dY = INT_ARG(6);
            int dX = INT_ARG(7);
            const bool isSameMode = INT_ARG(8) != 0;
            bool isNCHW = true;
            if (block.getIArguments()->size() > 9)
                isNCHW = INT_ARG(9) == 0;

            NDArray<T>* epsilon = OUTPUT_VARIABLE(0);
            NDArray<T>* gradW = OUTPUT_VARIABLE(1);
            NDArray<T>* gradB = nullptr;

            if (bias != nullptr)
                gradB = OUTPUT_VARIABLE(2);

            // epsilon for deconv2d is FF conv pass

            nd4j::ops::conv2d<T> op;
            Nd4jStatus r1 = op.execute({epsilonNext, weights}, {epsilon}, {}, {kY, kX, sY, sX, pY, pX, dY, dX, INT_ARG(8), 0});
            if (r1 != ND4J_STATUS_OK)
                return r1;

            int oY = 0;
            int oX = 0;
            int inY = epsilonNext->sizeAt(2);
            int inX = epsilonNext->sizeAt(3);

            ConvolutionUtils<T>::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode) {
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);
            }

            std::vector<T> extrasIm2Col({(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T) dX, isSameMode ? (T) 1.0f : (T) 0.0f, 0.0});
            auto columns = new NDArray<T>('c', {input->sizeAt(0), weights->sizeAt(1), kY, kX, oY, oX });
            epsilonNext->template applyTransform<simdOps::Im2col<T>>(columns, extrasIm2Col.data());

            auto gW = NDArrayFactory<T>::tensorDot(input, columns, {0, 2, 3}, {0, 4, 5});
            gradW->assign(gW);

            delete gW;
            delete columns;

            if (gradB != nullptr) {
                auto sum = epsilonNext->template reduceAlongDimension<simdOps::Sum<T>>({0, 2, 3});
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
            int* eShape = nullptr;
            int* bShape = nullptr;

            // bias is optional thing, and might be absent
            if (inputShape->size() == 4) {
                bShape = inputShape->at(2);
                eShape = inputShape->at(3);
            } else {
                eShape = inputShape->at(2);
            }

            int *newInShape;
            int *newWShape;
            ALLOCATE(newInShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            ALLOCATE(newWShape, block.getWorkspace(), shape::shapeInfoLength(wShape), int);

            memcpy(newInShape, inShape, shape::shapeInfoByteLength(inShape));
            memcpy(newWShape, wShape, shape::shapeInfoByteLength(wShape));

            auto shapes = SHAPELIST(newInShape, newWShape);

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