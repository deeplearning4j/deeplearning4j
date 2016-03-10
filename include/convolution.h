//
// Created by agibsonccc on 3/9/16.
//

#ifndef NATIVEOPERATIONS_CONVOLUTION_H
#define NATIVEOPERATIONS_CONVOLUTION_H

#include <omp.h>

template <typename T>
class Im2col {
private:
    T *img;
    T *out;
    int kernelWidth;
    int kernelHeight;
    int strideY;
    int strideX;
    int padHeight;
    int padWidth;
    int exampleFrom;
    int exampleTo;
    int depthFrom;
    int depthTo;
    int yOutFrom;
    int yOutTo;
    int xOutFrom;
    int xOutTo;
    bool coverAll;

int opSize() {
    return (exampleTo - exampleFrom) * (depthTo - depthFrom) * (xOutTo - xOutFrom) * (yOutTo - yOutFrom) * kernelHeight * kernelWidth;
}
    
    void exec() {
         T * dbIn = img;
         T * dbOut = out;

            int outArrayOffset = out.offset();
            int * outShape = out.shape();
            int * outStride = out.stride();

            int inArrayOffset = img.offset();
            int * inShape = img.shape();
            int * inStride = img.stride();

            int * outIndices = new int[6];
            int * inIndices = new int[4];

            const int inStride2 = inStride[2];
            const int inStride3 = inStride[3];
            const int outStride2 = outStride[2];
            const int outStride3 = outStride[3];
            const int inShape2 = inShape[2];
            const int inShape3 = inShape[3];

            const boolean padding = padHeight > 0 || padWidth > 0;

            T dIn = dbIn;
            T dOut = dbOut;
#pragma omp parallel for collapse(4)
            for (int ex = exampleFrom; ex < exampleTo; ex++) {
                for (int d = depthFrom; d < depthTo; d++) {
                    inIndices[0] = ex;
                    inIndices[1] = d;
                    outIndices[0] = ex;
                    outIndices[1] = d;

                    for (int x = xOutFrom; x < xOutTo; x++) {  //Along width
                        for (int y = yOutFrom; y < yOutTo; y++) {  //along height
                            outIndices[4] = y;
                            outIndices[5] = x;
                            int baseOffsetOut = getOffsetUnsafe6(outArrayOffset, outShape, outStride, outIndices);

                            if(padding) {
                                int i = y * strideY - padHeight;    //index along height of first element of patch in original img
                                int j = x * strideX - padWidth;     //index along width of first element in patch in original img
                                inIndices[2] = i;   //along height
                                inIndices[3] = j;   //along width

                                int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride, inIndices);
                                if (outStride2 <= outStride3) {
                                    //Want dimension 2 (along height) in inner loop for cache reasons
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                        int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                        for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                            if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape2 || j + patchX >= inShape3)
                                                dOut[outBufferIdxX + patchY * outStride2] = 0; //padding
                                            else {
                                                dOut[outBufferIdxX + patchY * outStride2] = dIn[inBufferIdxX + patchY * inStride2];
                                            }
                                        }
                                    }
                                } else {
                                    //Want dimension 3 in inner loop for cache reasons
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                        int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                        for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                            if (i + patchY < 0 || j + patchX < 0 || i + patchY >= inShape[2] || j + patchX >= inShape[3])
                                                dOut[outBufferIdxY + patchX * outStride3] = 0f; //padding
                                            else {
                                                dOut[outBufferIdxY + patchX * outStride3] = dIn[inBufferIdxY + patchX * inStride3];
                                            }
                                        }
                                    }
                                }
                            } else {
                                //No padding
                                int i = y * strideY;    //index along height of first element of patch in original img
                                int j = x * strideX;     //index along width of first element in patch in original img
                                inIndices[2] = i;   //along height
                                inIndices[3] = j;   //along width

                                int baseOffsetIn = getOffsetUnsafe4(inArrayOffset, inShape, inStride, inIndices);
                                if (outStride2 <= outStride3) {
                                    //Want dimension 2 (along height) in inner loop for cache reasons
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        int outBufferIdxX = baseOffsetOut + patchX * outStride3;
                                        int inBufferIdxX = baseOffsetIn + patchX * inStride3;
                                        for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                            dOut[outBufferIdxX + patchY * outStride2] = dIn[inBufferIdxX + patchY * inStride2];
                                        }
                                    }
                                } else {
                                    //Want dimension 3 in inner loop for cache reasons
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        int outBufferIdxY = baseOffsetOut + patchY * outStride2;
                                        int inBufferIdxY = baseOffsetIn + patchY * inStride2;
                                        for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                            dOut[outBufferIdxY + patchX*outStride3] = dIn[inBufferIdxY + patchX*inStride3];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

        }

    /**
      * A version of Shape.getOffset without checking on input for negative indices etc
      * normally negative indices are bad, OK here because of other checks on input indices
      * Uses unrolled loop specifically for length 6, where indices[2] and indices[3] are zero (always are here)
      */
 int getOffsetUnsafe6(int baseOffset,  int* shape,  int* stride,  int* indices) {
        int offset = baseOffset;
        if(shape[0] != 1) offset += indices[0] * stride[0];
        if(shape[1] != 1) offset += indices[1] * stride[1];
        if(shape[4] != 1) offset += indices[4] * stride[4];
        if(shape[5] != 1) offset += indices[5] * stride[5];
        return offset;
    }

};

template <typename  T>
class Col2Im {
private:
    T *col;
    T *imgOut;
    int kernelHeight;
    int kernelWidth;
    int strideY;
    int strideX;
    int padHeight;
    int padWidth;
    int imgHeight;
    int imgWidth;
    int parallelThreshold;

    int exampleFrom;
    int exampleTo;
    int depthFrom;
    int depthTo;


    void exec() {
        T * dbCol = col;
        T * dbOut = imgOut;

        int outArrayOffset = 0;
        int* outShape = imgOut.shape();
            int* outStride = imgOut.stride();

            int inOffset = 0;
            int* inShape = col.shape();
            int* inStride = col.stride();

            int* outIndices = new int[4];
            int* inIndices = new int[6];

            const int inStride2 = inStride[2];
            const int inStride3 = inStride[3];
            const int outStride2 = outStride[2];
            const int outStride3 = outStride[3];
            const int outShape2 = outShape[2];
            const int outShape3 = outShape[3];

            const int yOutTo = inShape[4];
            const int xOutTo = inShape[5];


            const boolean padding = padHeight > 0 || padWidth > 0;

            T * fIn = dbCol;
            T * fOut = dbOut;
#pragma omp parallel for
            for (int ex = exampleFrom; ex < exampleTo; ex++) {
                for (int d = depthFrom; d < depthTo; d++) {
                    inIndices[0] = ex;
                    inIndices[1] = d;
                    outIndices[0] = ex;
                    outIndices[1] = d;

                    for (int x = 0; x < xOutTo; x++) {  //Patch number along width
                        for (int y = 0; y < yOutTo; y++) {  //Patch number along height
                            inIndices[4] = y;   //patch number (along height)
                            inIndices[5] = x;   //patch number (along width)
                            int baseOffsetIn = getOffsetUnsafe6(inOffset, inShape, inStride, inIndices);

                            if(padding){
                                int i = y * strideY - padHeight;    //index along height of first element of patch in original img
                                int j = x * strideX - padWidth;     //index along width of first element in patch in original img
                                outIndices[2] = i;  //along height
                                outIndices[3] = j;  //along width

                                int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride, outIndices);

                                if (inStride2 <= inStride3) {
                                    //Want dimension 2 (along height) in inner loop for cache efficiency
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        if( j + patchX < 0 || j + patchX >= outShape3 ) continue;

                                        for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                            if (i + patchY < 0 || i + patchY >= outShape2 ) continue;
                                            fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                    fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                        }
                                    }
                                } else {
                                    //Want dimension 3 (along width) in inner loop for cache efficiency
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        if(i + patchY < 0 || i + patchY >= outShape2) continue;
                                        for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                            if (j + patchX < 0 || j + patchX >= outShape3) continue;
                                            fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                    fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                        }
                                    }
                                }
                            } else {
                                //No padding
                                int i = y * strideY;    //index along height of first element of patch in output img
                                int j = x * strideX;     //index along width of first element in patch in output img

                                outIndices[2] = i;
                                outIndices[3] = j;

                                int baseOffsetOut = getOffsetUnsafe4(outArrayOffset, outShape, outStride, outIndices);

                                if (inStride2 <= inStride3) {
                                    //Want dimension 2 (along height) in inner loop for cache efficiency
                                    for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                        for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                            fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                    fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                        }
                                    }
                                } else {
                                    //Want dimension 3 (along width) in inner loop for cache efficiency
                                    for (int patchY = 0; patchY < kernelHeight; patchY++) {
                                        for (int patchX = 0; patchX < kernelWidth; patchX++) {
                                            fOut[baseOffsetOut + patchY * outStride2 + patchX * outStride3] +=
                                                    fIn[baseOffsetIn + patchY * inStride2 + patchX * inStride3];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    /** Calculate buffer offset (like Shape.getOffset) without checking on input for negative indices etc
 *  normally negative indices are bad, OK here because of other checks on input indices
 *  Uses unrolled loop specifically for length 4
 */
 getOffsetUnsafe4(int baseOffset, int* shape, int* stride, int* indices) {
        int offset = baseOffset;
        if(shape[0] != 1) offset += indices[0] * stride[0];
        if(shape[1] != 1) offset += indices[1] * stride[1];
        if(shape[2] != 1) offset += indices[2] * stride[2];
        if(shape[3] != 1) offset += indices[3] * stride[3];
        return offset;
    }

    /** A version of Shape.getOffset without checking on input for negative indices etc
     * normally negative indices are bad, OK here because of other checks on input indices
     * Uses unrolled loop specifically for length 6, where indices[2] and indices[3] are zero (always are here)
     */
 int getOffsetUnsafe6(int baseOffset, int* shape, int* stride, int* indices) {
        int offset = baseOffset;
        if(shape[0] != 1) offset += indices[0] * stride[0];
        if(shape[1] != 1) offset += indices[1] * stride[1];
        if(shape[4] != 1) offset += indices[4] * stride[4];
        if(shape[5] != 1) offset += indices[5] * stride[5];
        return offset;
    }

};
#endif //NATIVEOPERATIONS_CONVOLUTION_H
