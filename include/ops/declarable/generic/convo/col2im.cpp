//
// Created by raver119 on 17.10.2017.
//

#include <ops/declarable/CustomOperations.h>


namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(col2im, 1, 1, false, 0, 9) {
            auto x = INPUT_VARIABLE(0);
            auto z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->rankOf() == 6, 0, "col2im input should be 6D, but got %i instead", x->rankOf());
            REQUIRE_TRUE(z->rankOf() == 4, 0, "col2im output should be 4D, but got %i instead", z->rankOf());

            T* dx = x->buffer();
            T* result = z->buffer();

            int strideY = INT_ARG(0);
            int strideX = INT_ARG(1);
            int padHeight = INT_ARG(2);
            int padWidth = INT_ARG(3);
            int imgHeight = INT_ARG(4);
            int imgWidth = INT_ARG(5);
            int dY = INT_ARG(6);			//Dilation in height/y dimension
            int dX = INT_ARG(7);			//Dilation in width/x dimension

            int *inShape = x->shapeOf();
            int *inStride = x->stridesOf();

            int strideex = inStride[0];
            int stridech = inStride[1];
            int stridekrow = inStride[2];
            int stridekcol = inStride[3];
            int striderow = inStride[4];
            int stridecol = inStride[5];

            int kernelHeight = inShape[2];
            int kernelWidth = inShape[3];

            int *outShape = z->shapeOf();
            char resultOrder = z->ordering();
            int *outStride = z->stridesOf();

            int samples = outShape[0];
            int depth = outShape[1];
            int imgH = outShape[2];
            int imgW = outShape[3];

            int height_col = inShape[4];//(imgHeight + 2 * padHeight - kernelHeight) / strideX + 1;
            int width_col = inShape[5];//(imgWidth + 2 * padWidth - kernelWidth) / strideY + 1;

            int n = samples * depth * imgHeight * imgWidth;

            //Effective kernel size, accounting for dilation
            int kEffectiveW = kernelWidth + (kernelWidth - 1) * (dX - 1);
            int kEffectiveH = kernelHeight + (kernelHeight - 1) * (dY - 1);

#pragma omp parallel for schedule(guided) proc_bind(close)
            for (int i = 0; i < n; i++) {
                T val = 0;
                int w_im = i % imgWidth + padWidth;
                int h_im = (i / imgWidth) % imgHeight + padHeight;
                int c_im = i / (imgWidth * imgHeight);

                int num_im = c_im / depth;
                int depth_im = c_im % depth;

                // compute the start and end of the output
                // These are the indexes for dimensions ??? in the 6d col matrix
                int w_col_start = (w_im < kEffectiveW) ? 0 : (w_im - kEffectiveW) / strideX + 1;
                int w_col_end = nd4j::math::nd4j_min<int>(w_im / strideX + 1, width_col);

                int h_col_start = (h_im < kEffectiveH) ? 0 : (h_im - kEffectiveH) / strideY + 1;
                int h_col_end = nd4j::math::nd4j_min<int>(h_im / strideY + 1, height_col);


                //Iterate over col entries in the 6d array... these are added up
                for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
                    for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
                        int h_k = (h_im - h_col * strideY);
                        int w_k = (w_im - w_col * strideX);

                        if(h_k % dY == 0 && w_k % dX == 0){
                            h_k /= dY;
                            w_k /= dX;

                            int data_col_index = num_im * strideex + depth_im * stridech + h_k * stridekrow + w_k * stridekcol + h_col * striderow + w_col * stridecol;
                            val += dx[data_col_index];
                        }
                    }
                }
                int i_f = 0;
                int i_c = i;
                for (int dim = 3; dim >= 0; dim--)
                {
                    i_f += (i_c % outShape[dim])  * outStride[dim];
                    i_c = i_c / outShape[dim];
                }
                result[i_f] += val;
            }


            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(col2im) {
            auto inShape = inputShape->at(0);

            int bS = shape::shapeOf(inShape)[0];
            int iD = shape::shapeOf(inShape)[1];

            int sY = INT_ARG(0);
            int sX = INT_ARG(1);
            int pY = INT_ARG(2);
            int pX = INT_ARG(3);
            int inY = INT_ARG(4);
            int inX = INT_ARG(5);
            int dY = INT_ARG(6);			//Dilation, height/y dimension
            int dX = INT_ARG(7);			//Dilation, width/x dimension
            bool isSameMode = INT_ARG(8) > 0;

            int* zShape;
            ALLOCATE(zShape, block.getWorkspace(), shape::shapeInfoLength(4), int);

            zShape[0] = 4;
            zShape[1] = bS;
            zShape[2] = iD;
            zShape[3] = inY;
            zShape[4] = inX;

            zShape[shape::shapeInfoLength(zShape) - 3] = 0;
            zShape[shape::shapeInfoLength(zShape) - 2] = 1;
            zShape[shape::shapeInfoLength(zShape) - 1] = 99;

            shape::updateStrides(zShape, 'c');

            return new ShapeList(zShape);
        }
    }
}