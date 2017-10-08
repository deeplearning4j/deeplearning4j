//
// Created by raver119 on 08.10.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        DECLARE_SHAPE_FN(avgpool3d) {
            int* input = inputShape->at(0);

            int kT = block.getIArguments()->at(0);
            int kW = block.getIArguments()->at(1);
            int kH = block.getIArguments()->at(2);
            int dT = block.getIArguments()->at(3);
            int dW = block.getIArguments()->at(4);
            int dH = block.getIArguments()->at(5);
            int padT = block.getIArguments()->at(6);
            int padW = block.getIArguments()->at(7);
            int padH = block.getIArguments()->at(8);
            bool ceil_mode = block.getIArguments()->at(9) != 0;

            Nd4jIndex nslices;
            Nd4jIndex itime;
            Nd4jIndex iheight;
            Nd4jIndex iwidth;
            Nd4jIndex otime;
            Nd4jIndex oheight;
            Nd4jIndex owidth;

            int dimN = 2;
            int dimt = 3;
            int dimh = 4;
            int dimw = 5;

            int nBatch = input[1];
            nslices = input[dimN];
            itime   = input[dimt];
            iheight = input[dimh];
            iwidth  = input[dimw];

            if (ceil_mode)
            {
                otime   = (Nd4jIndex)(nd4j::math::nd4j_ceil<T>((T)(itime   - kT + 2*padT) / dT)) + 1;
                oheight = (Nd4jIndex)(nd4j::math::nd4j_ceil<T>((T)(iheight - kH + 2*padH) / dH)) + 1;
                owidth  = (Nd4jIndex)(nd4j::math::nd4j_ceil<T>((T)(iwidth  - kW + 2*padW) / dW)) + 1;
            }
            else
            {
                otime   = (Nd4jIndex)(nd4j::math::nd4j_floor<T>((T)(itime   - kT + 2*padT) / dT)) + 1;
                oheight = (Nd4jIndex)(nd4j::math::nd4j_floor<T>((T)(iheight - kH + 2*padH) / dH)) + 1;
                owidth  = (Nd4jIndex)(nd4j::math::nd4j_floor<T>((T)(iwidth  - kW + 2*padW) / dW)) + 1;
            }
            if (padT || padH || padW)
            {
                // ensure that the last pooling starts inside the image
                // needed to avoid problems in ceil mode
                if ((otime   - 1)*dT >= itime   + padT)
                    --otime;
                if ((oheight - 1)*dH >= iheight + padH)
                    --oheight;
                if ((owidth  - 1)*dW >= iwidth  + padW)
                    --owidth;
            }

            int *shapeOf;
            int *newShape;
            ALLOCATE(shapeOf, block.getWorkspace(), 5, int);
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(5), int);

            nd4j::ArrayUtils::toIntPtr({nBatch, (int) nslices, (int)otime, (int)oheight, (int)owidth}, shapeOf);

            shape::shapeBuffer(5, shapeOf, newShape);

            RELEASE(shapeOf, block.getWorkspace());
            return new ShapeList(newShape);
        }


//////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(avgpool3d_bp, 2, 1, true, 0, 11) {
            NDArray<T> *input = block.getVariables().at(0)->getNDArray();
            NDArray<T> *gradNext = block.getVariables().at(1)->getNDArray();

            NDArray<T> *output = this->getZ(block);

            REQUIRE_TRUE(input->rankOf() == 5, 0, "Input should be 5D, got %i instead", input->rankOf());

            Nd4jIndex nslices;
            Nd4jIndex itime;
            Nd4jIndex iheight;
            Nd4jIndex iwidth;
            Nd4jIndex otime;
            Nd4jIndex oheight;
            Nd4jIndex owidth;
            T *gradInput_data;
            T *gradOutput_data;
            int kT = block.getIArguments()->at(0);
            int kW = block.getIArguments()->at(1);
            int kH = block.getIArguments()->at(2);
            int dT = block.getIArguments()->at(3);
            int dW = block.getIArguments()->at(4);
            int dH = block.getIArguments()->at(5);
            int padT = block.getIArguments()->at(6);
            int padW = block.getIArguments()->at(7);
            int padH = block.getIArguments()->at(8);
            bool ceil_mode = block.getIArguments()->at(9) != 0;
            bool count_include_pad  = block.getIArguments()->at(10) != 0;

            REQUIRE_TRUE(output->isSameShape(input), 0, "Output gradients should have the same dimensionality as input");

            int dimN = 1;
            int dimt = 2;
            int dimh = 3;
            int dimw = 4;

            output->assign(0.0);

            nslices = input->sizeAt(dimN);
            itime = input->shapeOf()[dimt];
            iheight = input->shapeOf()[dimh];
            iwidth = input->shapeOf()[dimw];
            otime = gradNext->shapeOf()[dimt];
            oheight = gradNext->shapeOf()[dimh];
            owidth = gradNext->shapeOf()[dimw];


            gradInput_data = output->getBuffer();
            gradOutput_data = gradNext->getBuffer();

            long nBatch = input->sizeAt(0);

            long istride = nslices * itime * iwidth * iheight;
            long ostride = nslices * otime * owidth * oheight;

            for (int p = 0; p < nBatch; p++)
            {
                ConvolutionUtils<T>::_avgPool3D_bp(
                        gradInput_data  + p * istride,
                        gradOutput_data + p * ostride,
                        nslices,
                        itime, iwidth, iheight,
                        otime, owidth, oheight,
                        kT, kW, kH,
                        dT, dW, dH,
                        padT, padW, padH,
                        count_include_pad
                );
            }

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(avgpool3d_bp) {
            // output shape equals to input shape, all out of sudden
            int* newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);
            memcpy(newShape, inputShape->at(0), shape::shapeInfoByteLength(inputShape->at(0)));
            return new ShapeList(newShape);
        }


        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(avgpool3d, 1, 1, true, 0, 11) {

            NDArray<T> *input = block.getVariables().at(0)->getNDArray();
            NDArray<T> *output = this->getZ(block);

            REQUIRE_TRUE(input->rankOf() == 5, 0, "Input should be 5D, got %i instead", input->rankOf());

            int kT = block.getIArguments()->at(0);
            int kW = block.getIArguments()->at(1);
            int kH = block.getIArguments()->at(2);
            int dT = block.getIArguments()->at(3);
            int dW = block.getIArguments()->at(4);
            int dH = block.getIArguments()->at(5);
            int padT = block.getIArguments()->at(6);
            int padW = block.getIArguments()->at(7);
            int padH = block.getIArguments()->at(8);
            bool ceil_mode = block.getIArguments()->at(9) != 0;
            bool count_include_pad  = block.getIArguments()->at(10) != 0;


            Nd4jIndex nslices;
            Nd4jIndex itime;
            Nd4jIndex iheight;
            Nd4jIndex iwidth;
            Nd4jIndex otime;
            Nd4jIndex oheight;
            Nd4jIndex owidth;
            T *input_data;
            T *output_data;

            int dimN = 1;
            int dimt = 2;
            int dimh = 3;
            int dimw = 4;

            nslices = input->sizeAt(dimN);
            itime   = input->sizeAt(dimt);
            iheight = input->sizeAt(dimh);
            iwidth  = input->sizeAt(dimw);

            if (ceil_mode)
            {
                otime   = (Nd4jIndex)(nd4j::math::nd4j_ceil<T>((T)(itime   - kT + 2*padT) / dT)) + 1;
                oheight = (Nd4jIndex)(nd4j::math::nd4j_ceil<T>((T)(iheight - kH + 2*padH) / dH)) + 1;
                owidth  = (Nd4jIndex)(nd4j::math::nd4j_ceil<T>((T)(iwidth  - kW + 2*padW) / dW)) + 1;
            }
            else
            {
                otime   = (Nd4jIndex)(nd4j::math::nd4j_floor<T>((T)(itime   - kT + 2*padT) / dT)) + 1;
                oheight = (Nd4jIndex)(nd4j::math::nd4j_floor<T>((T)(iheight - kH + 2*padH) / dH)) + 1;
                owidth  = (Nd4jIndex)(nd4j::math::nd4j_floor<T>((T)(iwidth  - kW + 2*padW) / dW)) + 1;
            }
            if (padT || padH || padW)
            {
                // ensure that the last pooling starts inside the image
                // needed to avoid problems in ceil mode
                if ((otime   - 1)*dT >= itime   + padT)
                    --otime;
                if ((oheight - 1)*dH >= iheight + padH)
                    --oheight;
                if ((owidth  - 1)*dW >= iwidth  + padW)
                    --owidth;
            }

            int nBatch = input->sizeAt(0);

            Nd4jIndex istride = nslices * itime * iwidth * iheight;
            Nd4jIndex ostride = nslices * otime * owidth * oheight;

            REQUIRE_TRUE(output->isSameShape({nBatch, (int) nslices, (int)otime, (int)oheight, (int)owidth}), 0, "Output should have shape of [%i, %i, %i, %i, %i], but got [%i, %i, %i, %i, %i] instead", nBatch, nslices, otime, oheight, owidth, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3), output->sizeAt(4));

            input_data = input->getBuffer();
            output_data = output->getBuffer();

            for (int p=0; p < nBatch; p++)
            {

                ConvolutionUtils<T>::_avgPool3D(
                        input_data + p * istride, output_data + p * ostride, nslices,
                        itime, iwidth, iheight,
                        otime, owidth, oheight,
                        kT, kW, kH,
                        dT, dW, dH,
                        padT, padW, padH,
                        count_include_pad
                );

            }

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }




//////////////////////////////////////////////////////////////////////////
        CONFIGURABLE_OP_IMPL(maxpool3d, 1, 2, true, 0, 13) {

            NDArray<T> *input = block.getVariables().at(0)->getNDArray();

            NDArray<T> *output = this->getZ(block);
            NDArray<T> *indices = this->getZ(block, 1);

            REQUIRE_TRUE(input->sizeOfT() > 2, 0, "MaxPool3D can't be used in HALF precision")
            REQUIRE_TRUE(input->rankOf() == 5, 0, "Input should be 5D, got rank %i instead", input->rankOf());
            REQUIRE_TRUE(output->rankOf() == 5, 0, "Output should be 5D, got rank %i instead", output->rankOf());

            int kT = block.getIArguments()->at(0);
            int kW = block.getIArguments()->at(1);
            int kH = block.getIArguments()->at(2);
            int dT = block.getIArguments()->at(3);
            int dW = block.getIArguments()->at(4);
            int dH = block.getIArguments()->at(5);
            int pT = block.getIArguments()->at(6);
            int pW = block.getIArguments()->at(7);
            int pH = block.getIArguments()->at(8);
            int dilationT = block.getIArguments()->at(9);
            int dilationW = block.getIArguments()->at(10);
            int dilationH = block.getIArguments()->at(11);
            bool ceilMode = block.getIArguments()->at(12) != 0;


            REQUIRE_TRUE(kT > 0 && kW > 0 && kH > 0, 0,
                         "Kernel size should be greater than zero, but got kT: %d kH: %d kW: %d",
                         kT, kH, kW);

            REQUIRE_TRUE(dT > 0 && dW > 0 && dH > 0, 8,
                         "stride should be greater than zero, but got dT: %d dH: %d dW: %d",
                         dT, dH, dW);

            REQUIRE_TRUE(dilationT > 0 && dilationW > 0 && dilationH > 0, 14,
                         "dilation should be greater than 0, but got dilationT: %d dilationH: %d dilationW: %d",
                         dilationT, dilationH, dilationW);

            REQUIRE_TRUE(kT/2 >= pT && kW/2 >= pW && kH/2 >= pH, 2,
                         "pad should be smaller than half of kernel size, but got "
                                 "kT: %d kW: %d, kH: %d, padT: %d, padW: %d, padH: %d",
                         kT, kW, kH, pT, pW, pH);

            Nd4jIndex nslices;
            Nd4jIndex itime;
            Nd4jIndex iheight;
            Nd4jIndex iwidth;
            Nd4jIndex otime;
            Nd4jIndex oheight;
            Nd4jIndex owidth;
            T *input_data;
            T *output_data;

            ////////////
            T *indices_data;


            int dimN = 1;
            int dimt = 2;
            int dimh = 3;
            int dimw = 4;


            nslices = input->sizeAt(dimN);
            itime   = input->sizeAt(dimt);
            iheight = input->sizeAt(dimh);
            iwidth  = input->sizeAt(dimw);

            if (ceilMode) {
                otime = (int)(nd4j::math::nd4j_ceil<T>((T)(itime - (dilationT * (kT - 1) + 1) + 2*pT) / dT)) + 1;
                oheight = (int)(nd4j::math::nd4j_ceil<T>((T)(iheight - (dilationH * (kH - 1) + 1) + 2*pH) / dH)) + 1;
                owidth  = (int)(nd4j::math::nd4j_ceil<T>((T)(iwidth  - (dilationW * (kW - 1) + 1) + 2*pW) / dW)) + 1;
            } else {
                otime = (int)(nd4j::math::nd4j_floor<T>((T)(itime - (dilationT * (kT - 1) + 1) + 2*pT) / dT)) + 1;
                oheight = (int)(nd4j::math::nd4j_floor<T>((T)(iheight - (dilationH * (kH - 1) + 1) + 2*pH) / dH)) + 1;
                owidth  = (int)(nd4j::math::nd4j_floor<T>((T)(iwidth  - (dilationW * (kW - 1) + 1) + 2*pW) / dW)) + 1;
            }

            if (pT > 0 || pW > 0 || pH > 0) {
                // ensure that the last pooling starts inside the image
                if ((otime - 1)*dT >= itime + pT)
                    --otime;
                if ((oheight - 1)*dH >= iheight + pH)
                    --oheight;
                if ((owidth  - 1)*dW >= iwidth  + pW)
                    --owidth;
            }


            REQUIRE_TRUE(otime >= 1 && owidth >= 1 && oheight >= 1, 0, "Output size is too small: [%i, %i, %i]", otime, oheight, owidth);

            NDArray<T>* _input;
            if (!input->isContiguous())
                _input = input->dup(input->ordering());
            else
                _input = input;

            Nd4jIndex istride = nslices * itime * iwidth * iheight;
            Nd4jIndex ostride = nslices * otime * owidth * oheight;

            REQUIRE_TRUE(output->sizeAt(0) == input->sizeAt(0) && output->sizeAt(1) == nslices && output->sizeAt(2) == otime && output->sizeAt(3) == oheight && output->sizeAt(4) == owidth, 0,
                         "Output shape expected to be [%i, %i, %i, %i, %i], but got [%i, %i, %i, %i, %i] instead", input->sizeAt(0), nslices, otime, oheight, owidth, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3), output->sizeAt(4));

            REQUIRE_TRUE(indices->isSameShape(output), 0, "Output and Indices shapes should be equal");

            input_data = _input->getBuffer();
            output_data = output->getBuffer();
            indices_data = indices->getBuffer();

            for (int n = 0; n < input->sizeAt(0); n++) {
                ConvolutionUtils<T>::_dilatedMaxPool3D(
                        input_data   + n * istride,
                        output_data  + n * ostride,
                        indices_data + n * ostride,
                        nslices,
                        itime, iwidth, iheight,
                        otime, owidth, oheight,
                        kT, kW, kH,
                        dT, dW, dH,
                        pT, pW, pH,
                        dilationT, dilationW, dilationH);
            }

            if (_input != input)
                delete _input;

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(MaxPool3D, maxpool3d);
        DECLARE_SYN(MaxPool3d, maxpool3d);

//////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(maxpool3d_bp, 3, 1, true, 0, 13) {

            NDArray<T> *input = block.getVariables().at(0)->getNDArray();
            NDArray<T> *gradNext = block.getVariables().at(1)->getNDArray();
            NDArray<T> *indices = block.getVariables().at(2)->getNDArray();

            NDArray<T> *output = this->getZ(block);

            REQUIRE_TRUE(input->rankOf() == 5, 0, "Input should be 5D, got %i instead", input->rankOf());
            REQUIRE_TRUE(indices->isSameShape(input), 1, "Indices should have the same dimensionality as input");
            REQUIRE_TRUE(output->isSameShape(input), 1, "Output gradient should have the same dimensionality as input");


            int kT = block.getIArguments()->at(0);
            int kW = block.getIArguments()->at(1);
            int kH = block.getIArguments()->at(2);
            int dT = block.getIArguments()->at(3);
            int dW = block.getIArguments()->at(4);
            int dH = block.getIArguments()->at(5);
            int pT = block.getIArguments()->at(6);
            int pW = block.getIArguments()->at(7);
            int pH = block.getIArguments()->at(8);
            int dilationT = block.getIArguments()->at(9);
            int dilationW = block.getIArguments()->at(10);
            int dilationH = block.getIArguments()->at(11);
            bool ceilMode = block.getIArguments()->at(12) != 0;


            REQUIRE_TRUE(kT > 0 && kW > 0 && kH > 0, 0,
                         "Kernel size should be greater than zero, but got kT: %d kH: %d kW: %d",
                         kT, kH, kW);

            REQUIRE_TRUE(dT > 0 && dW > 0 && dH > 0, 8,
                         "stride should be greater than zero, but got dT: %d dH: %d dW: %d",
                         dT, dH, dW);

            REQUIRE_TRUE(dilationT > 0 && dilationW > 0 && dilationH > 0, 14,
                         "dilation should be greater than 0, but got dilationT: %d dilationH: %d dilationW: %d",
                         dilationT, dilationH, dilationW);

            REQUIRE_TRUE(kT/2 >= pT && kW/2 >= pW && kH/2 >= pH, 2,
                         "pad should be smaller than half of kernel size, but got "
                                 "kT: %d kW: %d, kH: %d, padT: %d, padW: %d, padH: %d",
                         kT, kW, kH, pT, pW, pH);


            int nslices;
            int itime;
            int iheight;
            int iwidth;
            int otime;
            int oheight;
            int owidth;
            T *gradInput_data;
            T *gradOutput_data;
            T *indices_data;

            int dimN = 1;
            int dimt = 2;
            int dimh = 3;
            int dimw = 4;

            /* sizes */
            nslices = input->sizeAt(dimN);
            itime = input->sizeAt(dimt);
            iheight = input->sizeAt(dimh);
            iwidth = input->sizeAt(dimw);
            otime = gradNext->sizeAt(dimt);
            oheight = gradNext->sizeAt(dimh);
            owidth = gradNext->sizeAt(dimw);

            /* get raw pointers */
            gradInput_data = output->getBuffer();
            gradOutput_data = gradNext->getBuffer();
            indices_data = indices->getBuffer();

            int nBatch = input->sizeAt(0);

            Nd4jIndex istride = nslices * itime * iwidth * iheight;
            Nd4jIndex ostride = nslices * otime * owidth * oheight;

            for (int p = 0; p < nBatch; p++) {
                ConvolutionUtils<T>::_dilatedMaxPool3D_bp(
                        gradInput_data + p * istride,
                        gradOutput_data + p * ostride,
                        indices_data + p * ostride,
                        nslices,
                        itime, iwidth, iheight,
                        otime, owidth, oheight,
                        dT, dW, dH,
                        pT, pW, pH,
                        dilationT, dilationW, dilationH
                );
            }

            STORE_RESULT(*output);

            return ND4J_STATUS_OK;
        }
        DECLARE_SHAPE_FN(maxpool3d_bp) {
            // output shape equals to input shape, all out of sudden
            int* newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inputShape->at(0)), int);
            memcpy(newShape, inputShape->at(0), shape::shapeInfoByteLength(inputShape->at(0)));
            return new ShapeList(newShape);
        }
    }
}