//
// Created by Yurii Shyrma on 25.01.2018
//

#include <ops/declarable/helpers/reverseArray.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


/////////////////////////////////////////////////////////////////////////////////////
// this legacy op is written by raver119@gmail.com
template<typename T>
void reverseArray(T *inArr, int *inShapeBuffer, T *outArr, int *outShapeBuffer, int numOfElemsToReverse) {
            
            Nd4jIndex inLength = shape::length(inShapeBuffer);
            if(numOfElemsToReverse == 0)
                numOfElemsToReverse = inLength;
            int inEWS = shape::elementWiseStride(inShapeBuffer);
            char inOrder = shape::order(inShapeBuffer);
            Nd4jIndex sLength = numOfElemsToReverse - 1;

            // two step phase here
            if (inArr == outArr) {
                if (inEWS == 1) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jIndex e = 0; e < numOfElemsToReverse / 2; e++) {
                        Nd4jIndex idx = sLength - e;
                        T tmp = inArr[e];
                        inArr[e] = inArr[idx];
                        inArr[idx] = tmp;
                    }
                } 
                else if (inEWS > 1) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jIndex e = 0; e < numOfElemsToReverse / 2; e++) {
                        Nd4jIndex idx1 = (sLength - e) * inEWS;
                        Nd4jIndex idx2 =  e * inEWS;
                        T tmp = inArr[idx2];
                        inArr[idx2] = inArr[idx1];
                        inArr[idx1] = tmp;
                    }
                } 
                else {
                    int inRank = shape::rank(inShapeBuffer);
                    int *inShape = shape::shapeOf(inShapeBuffer);
                    int *inStride = shape::stride(inShapeBuffer);

                    int inCoord[MAX_RANK];
                    int outCoord[MAX_RANK];

#pragma omp parallel for private(inCoord, outCoord) schedule(guided)
                    for (Nd4jIndex e = 0; e < numOfElemsToReverse / 2; e++) {
                        if (inOrder == 'c') {
                            shape::ind2subC(inRank, inShape, e, inCoord);
                            shape::ind2subC(inRank, inShape, sLength - e, outCoord);
                        } else {
                            shape::ind2sub(inRank, inShape, e, inCoord);
                            shape::ind2sub(inRank, inShape, sLength - e, outCoord);
                        }

                        Nd4jIndex inOffset  = shape::getOffset(0, inShape, inStride, inCoord, inRank);
                        Nd4jIndex outOffset = shape::getOffset(0, inShape, inStride, outCoord, inRank);

                        outArr[outOffset] = inArr[inOffset];
                    }
                }
            } 
            else {
                // single step phase here
                int outEWS = shape::elementWiseStride(outShapeBuffer);
                char outOrder = shape::order(outShapeBuffer);

                if (inEWS == 1 && outEWS == 1 && inOrder == outOrder) {

#pragma omp parallel for schedule(guided)
                    for (Nd4jIndex e = 0; e < numOfElemsToReverse; e++) 
                        outArr[sLength - e] = inArr[e];                    

                    if(inLength != numOfElemsToReverse) {
#pragma omp parallel for schedule(guided)
                        for (Nd4jIndex e = numOfElemsToReverse; e < inLength; e++)
                            outArr[e] = inArr[e];
                    }
                } 
                else if (inEWS >= 1 && outEWS >= 1 && inOrder == outOrder) {

#pragma omp parallel for schedule(guided)
                    for (Nd4jIndex e = 0; e < numOfElemsToReverse; e++)
                        outArr[(sLength - e) * outEWS] = inArr[e * inEWS];

                    if(inLength != numOfElemsToReverse) {
#pragma omp parallel for schedule(guided)
                        for (Nd4jIndex e = numOfElemsToReverse; e < inLength; e++)
                            outArr[e * outEWS] = inArr[e * inEWS];
                    }
                } 
                else {

                    int inRank = shape::rank(inShapeBuffer);
                    int *inShape = shape::shapeOf(inShapeBuffer);
                    int *inStride = shape::stride(inShapeBuffer);

                    int outRank = shape::rank(outShapeBuffer);
                    int *outShape = shape::shapeOf(outShapeBuffer);
                    int *outStride = shape::stride(outShapeBuffer);

                    int inCoord[MAX_RANK];
                    int outCoord[MAX_RANK];

#pragma omp parallel for private(inCoord, outCoord) schedule(guided)
                    for (Nd4jIndex e = 0; e < numOfElemsToReverse; e++) {

                        if (inOrder == 'c')
                            shape::ind2subC(inRank, inShape, e, inCoord);
                        else
                            shape::ind2sub(inRank, inShape, e, inCoord);

                        if (outOrder == 'c')
                            shape::ind2subC(outRank, outShape, (sLength - e), outCoord);
                        else
                            shape::ind2sub(outRank, outShape, (sLength - e), outCoord);

                        Nd4jIndex inOffset = shape::getOffset(0, inShape, inStride, inCoord, inRank);
                        Nd4jIndex outOffset = shape::getOffset(0, outShape, outStride, outCoord, outRank);

                        outArr[outOffset] = inArr[inOffset];
                    }

                    if(inLength != numOfElemsToReverse) {

#pragma omp parallel for private(inCoord, outCoord) schedule(guided)
                        for (Nd4jIndex e = numOfElemsToReverse; e < inLength; e++) {
                             
                             if (inOrder == 'c')
                                shape::ind2subC(inRank, inShape, e, inCoord);
                            else
                                shape::ind2sub(inRank, inShape, e, inCoord);

                            if (outOrder == 'c')
                                shape::ind2subC(outRank, outShape, e, outCoord);
                            else
                                shape::ind2sub(outRank, outShape, e, outCoord);

                            Nd4jIndex inOffset = shape::getOffset(0, inShape, inStride, inCoord, inRank);
                            Nd4jIndex outOffset = shape::getOffset(0, outShape, outStride, outCoord, outRank);

                            outArr[outOffset] = inArr[inOffset];        
                        }
                    }
                }
            }
}

                            
template void reverseArray<float>(float *inArr, int *inShapeBuffer, float *outArr, int *outShapeBuffer, int numOfElemsToReverse);
template void reverseArray<float16>(float16 *inArr, int *inShapeBuffer, float16 *outArr, int *outShapeBuffer, int numOfElemsToReverse);
template void reverseArray<double>(double *inArr, int *inShapeBuffer, double *outArr, int *outShapeBuffer, int numOfElemsToReverse);


}
}
}