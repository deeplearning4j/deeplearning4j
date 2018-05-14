//
// @author raver119@gmail.com, created on 29/10/17.
// @author Yurii Shyrma (iuriish@yahoo.com), changed on 14.05.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_avgpool2d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(avgpool2d, 1, 1, false, 0, 11) {

            NDArray<T>* input = INPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should have rank of 4, but got %i instead", input->rankOf());

            // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;
            std::vector<int> argI = *(block.getIArguments());
            NDArray<T>* output = OUTPUT_VARIABLE(0);

            int kY = argI[0];
            int kX = argI[1];

            int sY = argI[2];
            int sX = argI[3];

            int pY = argI[4];
            int pX = argI[5];

            int dY = argI[6];
            int dX = argI[7];

            int oY = 0;
            int oX = 0;

            bool isNCHW = true;
            if (block.getIArguments()->size() > 10)
                isNCHW = INT_ARG(10) == 0;

            const int inY = isNCHW ? input->sizeAt(2) : input->sizeAt(1);
            const int inX = isNCHW ? input->sizeAt(3) : input->sizeAt(2);

            if (!isNCHW) {
                input  = input->permute({0, 3, 1, 2});                // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
                output = output->permute({0, 3, 1, 2});               // [bS, oH, oW, iC] -> [bS, iC, oH, oW]
            }

            const bool isSameMode = INT_ARG(8) > 0;

            ConvolutionUtils<T>::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode)
                ConvolutionUtils<T>::calcPadding2D(pY, pX, oY, oX, inY, inX, argI[0], argI[1], argI[2], argI[3], argI[6], argI[7]);
            
            // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8,9 - poolingMode; 10 - divisor;
            std::vector<T> argT = {(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T)dX, (T)1.f, (T)1.f, (T) argI[9], (T) oY, (T) oX};
            input->template applyTransform<simdOps::Pooling2D<T>>(output, argT.data());

            if (!isNCHW) {
                delete input;
                delete output;
            }

            return Status::OK();
        }

        DECLARE_SYN(AvgPool2D, avgpool2d);
        DECLARE_SYN(AvgPool, avgpool2d);
        DECLARE_SYN(avgpool, avgpool2d);

        DECLARE_SHAPE_FN(avgpool2d) {
            int* inShape = inputShape->at(0);
            int* shapeOf = shape::shapeOf(inShape);

            // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;
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

            bool isNCHW = true;
            if (block.getIArguments()->size() > 10)
                isNCHW = INT_ARG(10) == 0;

            int bS = shapeOf[0];
            int iD = isNCHW ? shapeOf[1] : shapeOf[3];
            int iH = isNCHW ? shapeOf[2] : shapeOf[1];
            int iW = isNCHW ? shapeOf[3] : shapeOf[2];


            char order = shape::order(inShape); // output order must be equal to input order

            // calculate output Height/Width
            int oH, oW;
            ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

            // allocate memory for new shape
            int* newShapeInfo = nullptr;
            ALLOCATE(newShapeInfo, block.getWorkspace(), 12, int);
            if (isNCHW) {
                newShapeInfo[0] = 4;        // rank
                newShapeInfo[1] = bS;
                newShapeInfo[2] = iD;
                newShapeInfo[3] = oH;
                newShapeInfo[4] = oW;
            } else {
                newShapeInfo[0] = 4;        // rank
                newShapeInfo[1] = bS;
                newShapeInfo[2] = oH;
                newShapeInfo[3] = oW;
                newShapeInfo[4] = iD;
            }
            shape::updateStrides(newShapeInfo, order);

            return SHAPELIST(newShapeInfo);
        }


        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(avgpool2d_bp, 2, 1, false, 0, 9) {

            NDArray<T>* input = INPUT_VARIABLE(0);
            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should have rank of 4, but got %i instead", input->rankOf());
            NDArray<T>* epsilon = INPUT_VARIABLE(1);
            NDArray<T>* outEpsilon = OUTPUT_VARIABLE(0);
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
            ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

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
            col2d->addiColumnVector(epsilon1d);

            T extraParams3[] = {(T)sH, (T)sW, (T)pH, (T)pW, (T)iH, (T)iW, (T)dH, (T)dW};   			// ??? zeros
            col6dPermuted->template applyTransform<simdOps::Col2Im<T>>(outEpsilon, extraParams3);
            outEpsilon->template applyScalar<simdOps::Divide<T>>((T) kH*kW, outEpsilon);

            STORE_RESULT(*outEpsilon);

            if(isEpsilonDup)
                delete epsilon;
            delete col6d;
            delete col6dPermuted;
            delete epsilon1d;
            delete col2d;

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(avgpool2d_bp) {
            // FIXME: memcpy should be removed
            int* newShapeInfo = nullptr;
            ALLOCATE(newShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);
            memcpy(newShapeInfo, inputShape->at(0), shape::shapeInfoByteLength(inputShape->at(0)));
            return SHAPELIST(newShapeInfo);
        }
    }
}

#endif