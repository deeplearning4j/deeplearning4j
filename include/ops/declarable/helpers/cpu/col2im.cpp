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
            void _col2im(nd4j::graph::LaunchContext& context, T *im, T *col, int *imShapeInfo, int *colShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW) {

                const int *colShape = shape::shapeOf(colShapeInfo);
                const int *colStride = shape::stride(colShapeInfo);
                const int *imShape = shape::shapeOf(imShapeInfo);
                const int *imStride = shape::stride(imShapeInfo);

                const int kH = colShape[2];
                const int kW = colShape[3];        
                const int bS = imShape[0];
                const int iC = imShape[1];
                const int oH = colShape[4];                            
                const int oW = colShape[5];                            
                const int colStride0  = colStride[0];
                const int colStride1  = colStride[1];
                const int colStride2  = colStride[2];
                const int colStride3  = colStride[3];
                const int colStride4  = colStride[4];
                const int colStride5  = colStride[5];
                const int imStride0 = imStride[0];
                const int imStride1 = imStride[1];
                const int imStride2 = imStride[2];
                const int imStride3 = imStride[3];

                const T* im0End = im + imStride1 * iC;
                const int kRowEnd = -pH + kH * dH;
                const int colStepOW = oW * colStride5;
                const int kColEnd = -pW + kW * dW;
                const int colRowEnd = oH * sH;
                const int colColEnd = oW * sW;

                T *im0, *im1, *im2;

                if (shape::order(colShapeInfo) == 'c' &&  shape::order(imShapeInfo) == 'c' && shape::strideDescendingCAscendingF(colShapeInfo) && shape::strideDescendingCAscendingF(imShapeInfo)) {
                    int i = 0;
#pragma omp parallel for schedule(guided) proc_bind(close) private(im0, im1, im2)
                    for (int b = 0; b < bS; b++) {
                        T *col0 = col + (b * colStride0);
                        
                        for (im0 = im + (b * imStride0); im0 < (b * imStride0) + im0End; im0 += imStride1) {
                            for (int kRow = -pH; kRow < kRowEnd; kRow+=dH) {                               
                                
                                for (int kCol = -pW; kCol < kColEnd; kCol+=dW) {
                
                                    for (int colRow = kRow; colRow < kRow + colRowEnd; colRow+=sH) {                                                                            

                                        if (!is_a_ge_zero_and_a_lt_b(colRow, iH)) {
                                            col0 += colStepOW;
                                        } 
                                        else {                                            
                                            im1 = im0 + colRow * imStride2;                                            
                                            
                                            if(kRow == -pH && kCol == -pW) {        // first pass
                                                for (int colCol = kCol; colCol < kCol + colColEnd; colCol+=sW, col0+=colStride5) 
                                                    if (is_a_ge_zero_and_a_lt_b(colCol, iW)) 
                                                        *(im1 + colCol * imStride3) = *col0;                                                                                                                                                        
                                            }
                                            else {
                                                for (int colCol = kCol; colCol < kCol + colColEnd; colCol+=sW, col0+=colStride5) {
                                                    if (is_a_ge_zero_and_a_lt_b(colCol, iW)) {
                                                        im2 = im1 + colCol * imStride3;                                                    
                                                        *im2 += *col0;                                                    
                                                    }
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
                    
                    T *col0, *col1, *col2, *col3, *col4;
#pragma omp parallel for schedule(guided) proc_bind(close) private(col0, col1, col2, col3, col4, im0, im1, im2)
                    for (int b = 0; b < bS; b++) {                                                
                        col0 = col + b * colStride0;

                        for (im0 = im + (b * imStride0); im0 < (b * imStride0) + im0End; im0+=imStride1, col0+=colStride1) {
                            col1 = col0;

                            for (int kRow = -pH; kRow < kRowEnd; kRow+=dH, col1+=colStride2) {                            
                                col2 = col1;
                                
                                for (int kCol = -pW; kCol < kColEnd; kCol+=dW, col2+=colStride3) {                                
                                    col3 = col2;

                                    for (int colRow = kRow; colRow < kRow + colRowEnd; colRow+=sH, col3+=colStride4) {                                    
                                        col4 = col3;

                                        if (!is_a_ge_zero_and_a_lt_b(colRow, iH)) {
                                            col4 += colStepOW;
                                        } 
                                        else {                                            
                                            im1 = im0 + colRow * imStride2;

                                            if(kRow == -pH && kCol == -pW) {        // first pass
                                                for (int colCol = kCol; colCol < kCol + colColEnd; colCol+=sW, col4+=colStride5) 
                                                   if (is_a_ge_zero_and_a_lt_b(colCol, iW)) 
                                                        *(im1 + colCol * imStride3) = *col4;
                                            }
                                            else { 
                                                for (int colCol = kCol; colCol < kCol + colColEnd; colCol+=sW, col4+=colStride5) {
                                                   if (is_a_ge_zero_and_a_lt_b(colCol, iW)) {
                                                        im2 = im1 + colCol * imStride3;
                                                        *im2 += *col4;                                                    
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            template void _col2im<float>(nd4j::graph::LaunchContext& context, float *in, float *output, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW);
            template void _col2im<float16>(nd4j::graph::LaunchContext& context, float16 *in, float16 *output, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW);
            template void _col2im<double>(nd4j::graph::LaunchContext& context, double *in, double *output, int *outShapeInfo, int *inShapeInfo, int sH, int sW, int pH, int pW, int iH, int iW, int dH, int dW);
        }
    }
}
