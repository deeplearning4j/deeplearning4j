/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma, created on 16.04.2018
//

#include <ops/declarable/helpers/reverse.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>


namespace nd4j    {
namespace ops     {
namespace helpers {


/////////////////////////////////////////////////////////////////////////////////////
// this legacy op is written by raver119@gmail.com
template<typename T>
void reverseArray(void *vinArr, Nd4jLong *inShapeBuffer, void *voutArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse) {
            auto inArr = reinterpret_cast<T *>(vinArr);
            auto outArr = reinterpret_cast<T *>(voutArr);

            Nd4jLong inLength = shape::length(inShapeBuffer);
            if(numOfElemsToReverse == 0)
                numOfElemsToReverse = inLength;
            int inEWS = shape::elementWiseStride(inShapeBuffer);
            char inOrder = shape::order(inShapeBuffer);
            auto sLength = numOfElemsToReverse - 1;

            // two step phase here
            if (inArr == outArr) {
                if (inEWS == 1) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jLong e = 0; e < numOfElemsToReverse / 2; e++) {
                        auto idx = sLength - e;
                        T tmp = inArr[e];
                        inArr[e] = inArr[idx];
                        inArr[idx] = tmp;
                    }
                } 
                else if (inEWS > 1) {
#pragma omp parallel for schedule(guided)
                    for (Nd4jLong e = 0; e < numOfElemsToReverse / 2; e++) {
                        auto idx1 = (sLength - e) * inEWS;
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

                        auto inOffset  = shape::getOffset(0, inShape, inStride, inCoord, inRank);
                        auto outOffset = shape::getOffset(0, inShape, inStride, outCoord, inRank);

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

                            auto inOffset = shape::getOffset(0, inShape, inStride, inCoord, inRank);
                            auto outOffset = shape::getOffset(0, outShape, outStride, outCoord, outRank);

                            outArr[outOffset] = inArr[inOffset];        
                        }
                    }
                }
            }
}


///////////////////////////////////////////////////////////////////
template <typename T>
static void _reverseSequence(const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim){

    int posOfNonUnityDim = -1;
    if(input->isVector() || shape::isLikeVector(input->getShapeInfo(), posOfNonUnityDim)) {

        if((seqDim == 0 && input->sizeAt(0) == 1) || (batchDim == posOfNonUnityDim))
            output->assign(input);
        else
            helpers::reverseArray<T>(const_cast<NDArray*>(input)->getBuffer(), const_cast<NDArray*>(input)->getShapeInfo(), output->getBuffer(), output->getShapeInfo(), seqLengths->getScalar<int>(0));
    }
    else {
            
        if(seqDim > batchDim)
            --seqDim;

        std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {batchDim});

        auto inSubArrsSet  = input->allTensorsAlongDimension(dimensions);
        auto outSubArrsSet = output->allTensorsAlongDimension(dimensions);

// #pragma omp parallel for schedule(guided)  if(inSubArrsSet->size() > Environment::getInstance()->elementwiseThreshold()) 
        for(int i = 0; i < inSubArrsSet->size(); ++i) {

            Nd4jLong numOfElemsToReverse = seqLengths->getScalar<Nd4jLong>(i);
        
            if(numOfElemsToReverse == 0 || numOfElemsToReverse == 1) {
                outSubArrsSet->at(i)->assign(inSubArrsSet->at(i));
            }
            else {
                auto inInnerSet  = inSubArrsSet->at(i)->allTensorsAlongDimension({seqDim});
                auto outInnerSet = outSubArrsSet->at(i)->allTensorsAlongDimension({seqDim});
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

    void reverseSequence(const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim) {
        BUILD_SINGLE_SELECTOR(input->dataType(), _reverseSequence, (input, seqLengths, output, seqDim, batchDim), LIBND4J_TYPES);
    }

//////////////////////////////////////////////////////////////////////////
void reverse(const NDArray* input, NDArray* output, const std::vector<int>* intArgs, bool isLegacy) {

    // we need to reverse axis only if that's new op
    std::vector<int> dimensions = isLegacy ? *intArgs : ShapeUtils::evalDimsToExclude(input->rankOf(), *intArgs);

    auto listOut = output->allTensorsAlongDimension(dimensions);
    auto listIn  = input->allTensorsAlongDimension(dimensions);
       
    NDArray* subArrIn  = nullptr;
    NDArray* subArrOut = nullptr;

    for(int i = 0; i < listIn->size(); ++i) {               // listIn->size() = listOut->size()
        subArrIn   = listIn->at(i);
        subArrOut  = listOut->at(i);        
        BUILD_SINGLE_SELECTOR(input->dataType(), helpers::reverseArray, (subArrIn->getBuffer(), subArrIn->getShapeInfo(), subArrOut->getBuffer(), subArrOut->getShapeInfo()), LIBND4J_TYPES);
    }

    delete listOut;
    delete listIn;
}

BUILD_SINGLE_TEMPLATE(template void reverseArray, (void *inArr, Nd4jLong *inShapeBuffer, void *outArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse), LIBND4J_TYPES);


}
}
}

