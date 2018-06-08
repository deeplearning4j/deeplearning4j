//
// Created by raver119 on 17.10.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_im2col)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(im2col, 1, 1, false, 0, 9) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);


            REQUIRE_TRUE(x->rankOf() == 4, 0, "im2col input should be 4D, but got %i instead", x->rankOf());
            REQUIRE_TRUE(z->rankOf() == 6, 0, "im2col output should be 6D, but got %i instead", z->rankOf());

            int kernelHeight = INT_ARG(0);
            int kernelWidth = INT_ARG(1);
            int strideY = INT_ARG(2);
            int strideX = INT_ARG(3);
            int padHeight = INT_ARG(4);
            int padWidth = INT_ARG(5);
            int dY = INT_ARG(6);			//Dilation, height/y dimension
            int dX = INT_ARG(7);			//Dilation, width/x dimension
            bool isSameMode = INT_ARG(8) > 0;
            T zeroPadVal = 0.0;
            if (block.getTArguments()->size() > 0)
                zeroPadVal = T_ARG(0);

            LaunchContext ctx;
            nd4j::ops::helpers::_im2col(ctx, z->buffer(), x->buffer(), z->shapeInfo(), x->shapeInfo(), kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth, dY, dX, isSameMode, zeroPadVal);

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(im2col) {
            auto inShape = inputShape->at(0);

            int bS = shape::shapeOf(inShape)[0];
            int iD = shape::shapeOf(inShape)[1];
            int inY = shape::shapeOf(inShape)[2];
            int inX = shape::shapeOf(inShape)[3];

            int kY = INT_ARG(0);
            int kX = INT_ARG(1);
            int sY = INT_ARG(2);
            int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            int dY = INT_ARG(6);			//Dilation, height/y dimension
            int dX = INT_ARG(7);			//Dilation, width/x dimension
            bool isSameMode = INT_ARG(8) > 0;

            // output is always 6d for im2col
            Nd4jLong* zShape;
            ALLOCATE(zShape, block.getWorkspace(), shape::shapeInfoLength(6), Nd4jLong);

            int oY = 0;
            int oX = 0;

            ConvolutionUtils<T>::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode)
                ConvolutionUtils<T>::_calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);

            zShape[0] = 6;
            zShape[1] = bS;
            zShape[2] = iD;
            zShape[3] = kY;
            zShape[4] = kX;
            zShape[5] = oY;
            zShape[6] = oX;

            zShape[shape::shapeInfoLength(zShape) - 3] = 0;
            zShape[shape::shapeInfoLength(zShape) - 2] = 1;
            zShape[shape::shapeInfoLength(zShape) - 1] = 99;

            shape::updateStrides(zShape, 'c');

            return SHAPELIST(zShape);
        }
    }
}

#endif