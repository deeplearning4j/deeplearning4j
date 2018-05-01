//
// Created by raver119 on 30.11.17.
//

#include <ops/declarable/helpers/col2im.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            FORCEINLINE bool is_a_ge_zero_and_a_lt_b(int a, int b) {
                return static_cast<unsigned>(a) < static_cast<unsigned>(b);
            }

            // input [bS, iC, kH, kW, oH, oW] is de-convoluted to output [bS, iC, iH, iW]
            template <typename T>
            void _col2im(nd4j::graph::LaunchContext& context, T *out, T *in, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW) {

                const int *inShape = shape::shapeOf(inShapeInfo);
                const int *inStride = shape::stride(inShapeInfo);
                const int *outShape = shape::shapeOf(outShapeInfo);
                const int *outStride = shape::stride(outShapeInfo);

                const int kH = inShape[2];
                const int kW = inShape[3];        
                const int bS = outShape[0];
                const int iC = outShape[1];
                const int oH = inShape[4];                            // (iH + 2 * pH- kH) / sH + 1;
                const int oW = inShape[5];                            // (iW + 2 * pW- kW) / sW + 1;
                const int inStride0  = inStride[0];
                const int inStride1  = inStride[1];
                const int inStride2  = inStride[2];
                const int inStride3  = inStride[3];
                const int inStride4  = inStride[4];
                const int inStride5  = inStride[5];
                const int outStride0 = outStride[0];
                const int outStride1 = outStride[1];
                const int outStride2 = outStride[2];
                const int outStride3 = outStride[3];

                const int inStepOW = oW * inStride5;
                int inRowStart, inColStart, inRow, inCol;
                T *out0, *out1, *out2;

                if (shape::order(inShapeInfo) == 'c' &&  shape::order(outShapeInfo) == 'c' && shape::strideDescendingCAscendingF(inShapeInfo) && shape::strideDescendingCAscendingF(outShapeInfo)) {

#pragma omp parallel for schedule(guided) proc_bind(close) private(out0, out1, out2, inRowStart, inColStart, inRow, inCol)
                    for (int b = 0; b < bS; b++) {
                        T *input = in + (b * inStride0);
                        out0 = out + (b * outStride0);

                        for (int channel = 0; channel < iC; ++channel, out0 += outStride1) {

                            for (int kRow = 0; kRow < kH; ++kRow) {                                
                                inRowStart = -pH + kRow * dH;
                                
                                for (int kCol = 0; kCol < kW; ++kCol) {
                                    inRow = inRowStart;
                                    inColStart = -pW + kCol * dW;

                                    for (int outRow = 0; outRow < oH; ++outRow, inRow += sH) {
                                        
                                        if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) {
                                            input += inStepOW;
                                        } 
                                        else {
                                            inCol = inColStart;
                                            out1 = out0 + inRow * outStride2;

                                            // if (channel == iC && is_a_ge_zero_and_a_lt_b(inCol, iW))
                                            //     *(out1 + inCol * outStride3) = (T) 0.0f;

                                            for (int outCol = 0; outCol < oW; ++outCol, inCol += sW, input += inStride5) {
                                                if (is_a_ge_zero_and_a_lt_b(inCol, iW)) {
                                                    out2 = out1 + inCol * outStride3;
                                                    *out2 += *input;                                                    
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } 
                else {
                    
                    T *in0, *in1, *in2, *in3, *in4;
#pragma omp parallel for schedule(guided) proc_bind(close) private(in0, in1, in2, in3, in4, out0, out1, out2, inRowStart, inColStart, inRow, inCol)
                    for (int b = 0; b < bS; b++) {                        
                        out0 = out + (b * outStride0);
                        in0 = in + b * inStride0;

                        for (int channel = 0; channel < iC; ++channel, out0+=outStride1, in0+=inStride1) {
                            in1 = in0;

                            for (int kRow = 0; kRow < kH; ++kRow, in1+=inStride2) {  
                                in2 = in1;
                                inRowStart = -pH + kRow * dH;
                                
                                for (int kCol = 0; kCol < kW; ++kCol, in2+=inStride3) {
                                    in3 = in2;
                                    inRow = inRowStart;
                                    inColStart = -pW + kCol * dW;

                                    for (int outRow = 0; outRow < oH; ++outRow, inRow+=sH, in3+=inStride4) {
                                        in4 = in3;

                                        if (!is_a_ge_zero_and_a_lt_b(inRow, iH)) {
                                            in4 += inStepOW;
                                        } 
                                        else {
                                            inCol = inColStart;
                                            out1 = out0 + inRow * outStride2;

                                            // if (channel == iC && is_a_ge_zero_and_a_lt_b(inCol, iW))
                                            //     *(out1 + inCol * outStride3) = (T) 0.0f;

                                            for (int outCol = 0; outCol < oW; ++outCol, inCol+=sW, in4+=inStride5) {
                                                if (is_a_ge_zero_and_a_lt_b(inCol, iW)) {
                                                    out2 = out1 + inCol * outStride3;
                                                    *out2 += *in4;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            };

            template void _col2im<float>(nd4j::graph::LaunchContext& context, float *in, float *output, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW);
            template void _col2im<float16>(nd4j::graph::LaunchContext& context, float16 *in, float16 *output, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW);
            template void _col2im<double>(nd4j::graph::LaunchContext& context, double *in, double *output, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW);
        }
    }
}
