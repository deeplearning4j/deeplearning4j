//
// @author Yurii Shyrma, created on 16.04.2018
//

#include <ops/declarable/helpers/reverse.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>
#include <NDArrayFactory.h>


namespace nd4j    {
namespace ops     {
namespace helpers {


/////////////////////////////////////////////////////////////////////////////////////
// this legacy op is written by raver119@gmail.com
template<typename T>
void reverseArray(T *inArr, Nd4jLong *inShapeBuffer, T *outArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse) {
            
            Nd4jLong inLength = shape::length(inShapeBuffer);
            if(numOfElemsToReverse == 0)
                numOfElemsToReverse = inLength;
            int inEWS = shape::elementWiseStride(inShapeBuffer);
            char inOrder = shape::order(inShapeBuffer);
            Nd4jLong sLength = numOfElemsToReverse - 1;

            // two step phase here
            if (inArr == outArr) {
                if (inEWS == 1) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jLong e = 0; e < numOfElemsToReverse / 2; e++) {
                        Nd4jLong idx = sLength - e;
                        T tmp = inArr[e];
                        inArr[e] = inArr[idx];
                        inArr[idx] = tmp;
                    }
                } 
                else if (inEWS > 1) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jLong e = 0; e < numOfElemsToReverse / 2; e++) {
                        Nd4jLong idx1 = (sLength - e) * inEWS;
                        Nd4jLong idx2 =  e * inEWS;
                        T tmp = inArr[idx2];
                        inArr[idx2] = inArr[idx1];
                        inArr[idx1] = tmp;
                    }
                } 
                else {
                    int inRank = shape::rank(inShapeBuffer);
                    auto inShape = shape::shapeOf(inShapeBuffer);
                    auto inStride = shape::stride(inShapeBuffer);

                    Nd4jLong inCoord[MAX_RANK];
                    Nd4jLong outCoord[MAX_RANK];

#pragma omp parallel for private(inCoord, outCoord) schedule(guided)
                    for (Nd4jLong e = 0; e < numOfElemsToReverse / 2; e++) {
                        if (inOrder == 'c') {
                            shape::ind2subC(inRank, inShape, e, inCoord);
                            shape::ind2subC(inRank, inShape, sLength - e, outCoord);
                        } else {
                            shape::ind2sub(inRank, inShape, e, inCoord);
                            shape::ind2sub(inRank, inShape, sLength - e, outCoord);
                        }

                        Nd4jLong inOffset  = shape::getOffset(0, inShape, inStride, inCoord, inRank);
                        Nd4jLong outOffset = shape::getOffset(0, inShape, inStride, outCoord, inRank);

                        outArr[outOffset] = inArr[inOffset];
                    }
                }
            } 
            else {
                // single step phase here
                auto outEWS = shape::elementWiseStride(outShapeBuffer);
                char outOrder = shape::order(outShapeBuffer);

                if (inEWS == 1 && outEWS == 1 && inOrder == outOrder) {

#pragma omp parallel for schedule(guided)
                    for (Nd4jLong e = 0; e < numOfElemsToReverse; e++) 
                        outArr[sLength - e] = inArr[e];                    

                    if(inLength != numOfElemsToReverse) {
#pragma omp parallel for schedule(guided)
                        for (Nd4jLong e = numOfElemsToReverse; e < inLength; e++)
                            outArr[e] = inArr[e];
                    }
                } 
                else if (inEWS >= 1 && outEWS >= 1 && inOrder == outOrder) {

#pragma omp parallel for schedule(guided)
                    for (Nd4jLong e = 0; e < numOfElemsToReverse; e++)
                        outArr[(sLength - e) * outEWS] = inArr[e * inEWS];

                    if(inLength != numOfElemsToReverse) {
#pragma omp parallel for schedule(guided)
                        for (Nd4jLong e = numOfElemsToReverse; e < inLength; e++)
                            outArr[e * outEWS] = inArr[e * inEWS];
                    }
                } 
                else {

                    int inRank = shape::rank(inShapeBuffer);
                    auto inShape = shape::shapeOf(inShapeBuffer);
                    auto inStride = shape::stride(inShapeBuffer);

                    int outRank = shape::rank(outShapeBuffer);
                    auto outShape = shape::shapeOf(outShapeBuffer);
                    auto outStride = shape::stride(outShapeBuffer);

                    Nd4jLong inCoord[MAX_RANK];
                    Nd4jLong outCoord[MAX_RANK];

#pragma omp parallel for private(inCoord, outCoord) schedule(guided)
                    for (Nd4jLong e = 0; e < numOfElemsToReverse; e++) {

                        if (inOrder == 'c')
                            shape::ind2subC(inRank, inShape, e, inCoord);
                        else
                            shape::ind2sub(inRank, inShape, e, inCoord);

                        if (outOrder == 'c')
                            shape::ind2subC(outRank, outShape, (sLength - e), outCoord);
                        else
                            shape::ind2sub(outRank, outShape, (sLength - e), outCoord);

                        auto inOffset = shape::getOffset(0, inShape, inStride, inCoord, inRank);
                        auto outOffset = shape::getOffset(0, outShape, outStride, outCoord, outRank);

                        outArr[outOffset] = inArr[inOffset];
                    }

                    if(inLength != numOfElemsToReverse) {

#pragma omp parallel for private(inCoord, outCoord) schedule(guided)
                        for (Nd4jLong e = numOfElemsToReverse; e < inLength; e++) {
                             
                             if (inOrder == 'c')
                                shape::ind2subC(inRank, inShape, e, inCoord);
                            else
                                shape::ind2sub(inRank, inShape, e, inCoord);

                            if (outOrder == 'c')
                                shape::ind2subC(outRank, outShape, e, outCoord);
                            else
                                shape::ind2sub(outRank, outShape, e, outCoord);

                            Nd4jLong inOffset = shape::getOffset(0, inShape, inStride, inCoord, inRank);
                            Nd4jLong outOffset = shape::getOffset(0, outShape, outStride, outCoord, outRank);

                            outArr[outOffset] = inArr[inOffset];        
                        }
                    }
                }
            }
}


///////////////////////////////////////////////////////////////////
template <typename T>
void reverseSequence(const NDArray<T>* input, const NDArray<T>* seqLengths, NDArray<T>* output, int seqDim, const int batchDim){

    int posOfNonUnityDim = -1;
    if(input->isVector() || shape::isLikeVector(input->getShapeInfo(), posOfNonUnityDim)) {

        if((seqDim == 0 && input->sizeAt(0) == 1) || (batchDim == posOfNonUnityDim))
            output->assign(input);
        else
            helpers::reverseArray<T>(const_cast<NDArray<T>*>(input)->getBuffer(), const_cast<NDArray<T>*>(input)->getShapeInfo(), output->getBuffer(), output->getShapeInfo(), (int)(*seqLengths)(0.));
    }
    else {
            
        if(seqDim > batchDim)
            --seqDim;

        std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), {batchDim});       

        ResultSet<T>* inSubArrsSet  = NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions);
        ResultSet<T>* outSubArrsSet = NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions);

#pragma omp parallel for if(inSubArrsSet->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) 
        for(int i = 0; i < inSubArrsSet->size(); ++i) {

            int numOfElemsToReverse = (*seqLengths)(i);
        
            if(numOfElemsToReverse == 0 || numOfElemsToReverse == 1) {
                outSubArrsSet->at(i)->assign(inSubArrsSet->at(i));
            }
            else {
                ResultSet<T>* inInnerSet  = NDArrayFactory<T>::allTensorsAlongDimension(inSubArrsSet->at(i), {seqDim});
                ResultSet<T>* outInnerSet = NDArrayFactory<T>::allTensorsAlongDimension(outSubArrsSet->at(i), {seqDim});
                for(int j = 0; j < inInnerSet->size(); ++j)
                    helpers::reverseArray<T>(inInnerSet->at(j)->getBuffer(), inInnerSet->at(j)->getShapeInfo(), outInnerSet->at(j)->getBuffer(), outInnerSet->at(j)->getShapeInfo(), numOfElemsToReverse);
            
                delete inInnerSet;
                delete outInnerSet;
            }
        }
        delete inSubArrsSet;
        delete outSubArrsSet;
    }

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void reverse(const NDArray<T>* input, NDArray<T>* output, const std::vector<int>* intArgs) {

    std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), *intArgs);

    ResultSet<T>* listOut = NDArrayFactory<T>::allTensorsAlongDimension(output, dimensions);
    ResultSet<T>* listIn  = NDArrayFactory<T>::allTensorsAlongDimension(input, dimensions);
       
    NDArray<T>* subArrIn  = nullptr;
    NDArray<T>* subArrOut = nullptr;    

    for(int i = 0; i < listIn->size(); ++i) {               // listIn->size() = listOut->size()
        subArrIn   = listIn->at(i);
        subArrOut  = listOut->at(i);        
        helpers::reverseArray<T>(subArrIn->getBuffer(), subArrIn->getShapeInfo(), subArrOut->getBuffer(), subArrOut->getShapeInfo());
    }

    delete listOut;
    delete listIn;
}

template void reverseSequence<float>(const NDArray<float>* input, const NDArray<float>* seqLengths, NDArray<float>* output, int seqDim, const int batchDim);
template void reverseSequence<float16>(const NDArray<float16>* input, const NDArray<float16>* seqLengths, NDArray<float16>* output, int seqDim, const int batchDim);
template void reverseSequence<double>(const NDArray<double>* input, const NDArray<double>* seqLengths, NDArray<double>* output, int seqDim, const int batchDim);

template void reverseArray<float>(float *inArr, Nd4jLong *inShapeBuffer, float *outArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse);
template void reverseArray<float16>(float16 *inArr, Nd4jLong *inShapeBuffer, float16 *outArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse);
template void reverseArray<double>(double *inArr, Nd4jLong *inShapeBuffer, double *outArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse);

template void reverse<float>(const NDArray<float>* input, NDArray<float>* output, const std::vector<int>* intArgs);
template void reverse<float16>(const NDArray<float16>* input, NDArray<float16>* output, const std::vector<int>* intArgs);
template void reverse<double>(const NDArray<double>* input, NDArray<double>* output, const std::vector<int>* intArgs);


}
}
}

