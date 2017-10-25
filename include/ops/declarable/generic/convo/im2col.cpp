//
// Created by raver119 on 17.10.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>


namespace nd4j {
    namespace ops {

        CUSTOM_OP_IMPL(im2col, 1, 1, false, 0, 9) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);


            REQUIRE_TRUE(x->rankOf() == 4, 0, "im2col input should be 4D, but got %i instead", x->rankOf());
            REQUIRE_TRUE(z->rankOf() == 6, 0, "im2col output should be 6D, but got %i instead", z->rankOf());

            T* dx = x->buffer();
            T* result = z->buffer();

            int kernelHeight = INT_ARG(0);
            int kernelWidth = INT_ARG(1);
            int strideY = INT_ARG(2);
            int strideX = INT_ARG(3);
            int padHeight = INT_ARG(4);
            int padWidth = INT_ARG(5);
            int dY = INT_ARG(6);			//Dilation, height/y dimension
            int dX = INT_ARG(7);			//Dilation, width/x dimension
            bool isSameMode = INT_ARG(8) > 0;
            int kSize = kernelWidth * kernelHeight;

            int *outShape = z->shapeOf();
            char resultOrder = z->ordering();
            int *outStride = z->stridesOf();

            int *inShape = x->shapeOf();
            int *inStride = x->stridesOf();

            int samples = inShape[0];
            int depth = inShape[1];
            int height = inShape[2];
            int width = inShape[3];


            int strideex = inStride[0];
            int stridech = inStride[1];
            int strideh = inStride[2];
            int stridew = inStride[3];

            int height_col = outShape[4];
            int width_col = outShape[5];

            int n = samples * depth * height_col * width_col;

#pragma omp parallel for schedule(guided) proc_bind(close)
            for (int index = 0; index < n; index++) {
                int h_index = index / width_col;
                int h_col = h_index % height_col;
                int w_col = index % width_col;

                int c_im = h_index / height_col;
                int c_col = c_im * kSize;

                int depth_im = c_im % depth;
                int num_im = c_im / depth;
                int h_offset = h_col * strideY - padHeight;
                int w_offset = w_col * strideX - padWidth;

                T* data_col_ptr = result;

                int i_c = (c_col * height_col + h_col) * width_col + w_col;
                data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;

                T* data_im_ptr = dx;

                data_im_ptr += num_im * strideex + depth_im * stridech + h_offset * strideh + w_offset*stridew;

                for (int i = 0; i < kernelHeight; ++i) {
                    for (int j = 0; j < kernelWidth; ++j) {
                        int h_im = h_offset + i * dY;
                        int w_im = w_offset + j * dX;
                        int i_f = 0;
                        int i_c_temp = i_c;
                        for (int dim = 5; dim >= 0; dim--) {
                            i_f += (i_c_temp % outShape[dim])  * outStride[dim];
                            i_c_temp = i_c_temp / outShape[dim];
                        }
                        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width){
                            result[i_f] = data_im_ptr[i * dY * strideh + j * dX * stridew];
                        } else result[i_f] = 0;

                        //result[i_f] = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * strideh + j*stridew] : 0;
                        data_col_ptr += height_col * width_col;
                        i_c += height_col * width_col;
                    }
                }
            }

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
            int* zShape;
            ALLOCATE(zShape, block.getWorkspace(), shape::shapeInfoLength(6), int);

            int oY = 0;
            int oX = 0;

            ConvolutionUtils<T>::calcOutHWpool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

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

            return new ShapeList(zShape);
        }
    }
}