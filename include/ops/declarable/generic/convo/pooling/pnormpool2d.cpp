//
// @author raver119@gmail.com, created on 29/10/17.
// @author Yurii Shyrma (iuriish@yahoo.com), changed on 14.05.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_pnormpool2d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(pnormpool2d, 1, 1, false, 0, 11) {

            REQUIRE_OK(this->validateInputLengthMatch(block));
            REQUIRE_OK(this->validateInputDimensionsMatch(block));
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should have rank of 4, but got %i instead", input->rankOf());

            std::vector<int> argI = *(block.getIArguments()); // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode; 9 - extraParam0 for pnorm case;

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

            const bool isSameMode = INT_ARG(8) > 0;

            bool isNCHW = true;
            if (block.getIArguments()->size() > 10)
                isNCHW = INT_ARG(10) == 0;            

            if (!isNCHW) {
                input  = input->permute({0, 3, 1, 2});                  // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
                output = output->permute({0, 3, 1, 2});                 // [bS, oH, oW, iC] -> [bS, iC, oH, oW]
            }

            const int inY = input->sizeAt(2);
            const int inX = input->sizeAt(3);

            ConvolutionUtils<T>::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode)
                ConvolutionUtils<T>::calcPadding2D(pY, pX, oY, oX, inY, inX, argI[0], argI[1], argI[2], argI[3], argI[6], argI[7]);

            std::vector<T> argT = {(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T)dX, (T)1.f, (T)2.f, (T) argI[9]};

            input->template applyTransform<simdOps::Pooling2D<T>>(output, argT.data());

            if (!isNCHW) {
                delete input;
                delete output;
            }

            return Status::OK();
        }
        DECLARE_SYN(PnormPool2D, pnormpool2d);
        DECLARE_SYN(PnormPool, pnormpool2d);
        DECLARE_SYN(pnormpool, pnormpool2d);


        DECLARE_SHAPE_FN(pnormpool2d) {
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
        CUSTOM_OP_IMPL(pnormpool2d_bp, 2, 1, false, 1, 10) {

            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto outEpsilon = OUTPUT_VARIABLE(0);
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

            T extraParams1[] = {(T)kH, (T)kW, (T)sH, (T)sW, (T)pH, (T)pW, (T)dH, (T)dW, (T)0.0};
            input->template applyTransform<simdOps::Im2col<T>>(col6dPermuted, extraParams1);

            NDArray<T>* pNorm = new NDArray<T>(col2d->getShapeInfo(), block.getWorkspace());
            col2d->template applyTransform<simdOps::Abs<T>>(pNorm, nullptr);

            T extraParams11[] = {(T)pnorm};
            pNorm->template applyTransform<simdOps::Pow<T>>(extraParams11);
            NDArray<T>* sumArr = pNorm->sum({1});
            *pNorm = *sumArr;
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

            T extraParams5[] = {(T)sH, (T)sW, (T)pH, (T)pW, (T)iH, (T)iW, (T)dH, (T)dW};   			// ??? zeros
            col6dPermuted->template applyTransform<simdOps::Col2Im<T>>(outEpsilon, extraParams5);

            STORE_RESULT(*outEpsilon);

            if(isEpsilonDup)
                delete epsilon;
            delete sumArr;
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

            // FIXME: remove memcpy here
            int* newShapeInfo = nullptr;
            ALLOCATE(newShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);
            memcpy(newShapeInfo, inputShape->at(0), shape::shapeInfoByteLength(inputShape->at(0)));
            return SHAPELIST(newShapeInfo);
        }
    }
}

#endif