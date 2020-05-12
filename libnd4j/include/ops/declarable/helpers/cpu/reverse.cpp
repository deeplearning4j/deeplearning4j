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
#include <execution/Threads.h>


namespace sd    {
namespace ops     {
namespace helpers {

template <typename T>
inline void swap(T* arr, Nd4jLong from, Nd4jLong to) {
    T tmp = arr[from];
    arr[from] = arr[to];
    arr[to] = tmp;
}
/////////////////////////////////////////////////////////////////////////////////////
// this legacy op is written by raver119@gmail.com

template<typename T>
static void reverseArray(sd::LaunchContext * context, void const* vinArr, Nd4jLong const*inShapeBuffer, void *voutArr, Nd4jLong const*outShapeBuffer, int numOfElemsToReverse = 0) {
            auto inArr = reinterpret_cast<T const*>(vinArr);
            auto outArr = reinterpret_cast<T *>(voutArr);

            Nd4jLong inLength  = shape::length(inShapeBuffer);
            Nd4jLong outLength = shape::length(outShapeBuffer);
            if(numOfElemsToReverse == 0)
                numOfElemsToReverse = inLength;
            int inEWS = shape::elementWiseStride(inShapeBuffer);
            char inOrder = shape::order(inShapeBuffer);
            auto sLength = numOfElemsToReverse - 1;

            // two step phase here
            if (inArr == outArr) {
                if (inEWS == 1) {
                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e++) {
                            auto idx = sLength - e;
                            swap(const_cast<T*>(inArr), e, idx);
                        }
                    };
                    samediff::Threads::parallel_for(func, 0, numOfElemsToReverse / 2);
                }
                else if (inEWS > 1) {
                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e++) {
                            auto idx1 = (sLength - e) * inEWS;
                            Nd4jLong idx2 = e * inEWS;
                            swap(const_cast<T*>(inArr), idx1, idx2);
                        }
                    };

                    samediff::Threads::parallel_for(func, 0, numOfElemsToReverse / 2);
                }
                else {

                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e++) {
                            auto inOffset = shape::getIndexOffset(e, inShapeBuffer);
                            auto outOffset = shape::getIndexOffset(sLength - e, inShapeBuffer);
                            swap(outArr, inOffset, outOffset);
                        }
                    };

                    samediff::Threads::parallel_for(func, 0, numOfElemsToReverse / 2);
                }
            }
            else {
                // single step phase here
                auto outEWS = shape::elementWiseStride(outShapeBuffer);
                char outOrder = shape::order(outShapeBuffer);

                if (inEWS == 1 && outEWS == 1 && inOrder == outOrder) {

                    auto func = PRAGMA_THREADS_FOR {
                        for (Nd4jLong e = start; e < stop; e++)
                            outArr[sLength - e] = inArr[e];
                    };
                    samediff::Threads::parallel_for(func, 0, numOfElemsToReverse);

                    if(inLength != numOfElemsToReverse) {
                        auto f2 = PRAGMA_THREADS_FOR {
                            for (auto e = start; e < stop; e++)
                                outArr[e] = inArr[e];
                        };
                        samediff::Threads::parallel_for(f2, numOfElemsToReverse, inLength);
                    }
                }
                else if (inEWS >= 1 && outEWS >= 1 && inOrder == outOrder) {

                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e++)
                            outArr[(sLength - e) * outEWS] = inArr[e * inEWS];
                    };
                    samediff::Threads::parallel_for(func, 0, numOfElemsToReverse);

                    if(inLength != numOfElemsToReverse) {
                        auto f2 = PRAGMA_THREADS_FOR {
                            for (auto e = start; e < stop; e++)
                                outArr[e * outEWS] = inArr[e * inEWS];
                        };
                        samediff::Threads::parallel_for(f2, numOfElemsToReverse, inLength);
                    }
                }
                else {

                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e++) {
                            auto inOffset = shape::getIndexOffset(e, inShapeBuffer);
                            auto outOffset = shape::getIndexOffset(sLength - e, outShapeBuffer);
                            outArr[outOffset] = inArr[inOffset];
                        }
                    };
                    samediff::Threads::parallel_for(func, 0, numOfElemsToReverse);

                    if(inLength != numOfElemsToReverse) {

                        auto f2 = PRAGMA_THREADS_FOR {
                            for (auto e = start; e < stop; e++) {
                                auto inOffset = shape::getIndexOffset(e, inShapeBuffer);
                                auto outOffset = shape::getIndexOffset(e, outShapeBuffer);
                                outArr[outOffset] = inArr[inOffset];
                            }
                        };
                        samediff::Threads::parallel_for(f2, numOfElemsToReverse, inLength);
                    }
                }
            }
}


///////////////////////////////////////////////////////////////////
template <typename T>
static void reverseSequence_(sd::LaunchContext * context, const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim){

    int posOfNonUnityDim = -1;
    if(input->isVector() || shape::isLikeVector(input->shapeInfo(), posOfNonUnityDim)) {

        if((seqDim == 0 && input->sizeAt(0) == 1) || (batchDim == posOfNonUnityDim))
            output->assign(input);
        else
            helpers::reverseArray<T>(context, const_cast<NDArray*>(input)->buffer(), const_cast<NDArray*>(input)->shapeInfo(), output->buffer(), output->shapeInfo(), seqLengths->e<int>(0));
    }
    else {

        if(seqDim > batchDim)
            --seqDim;

        std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {batchDim});

        auto inSubArrsSet  = input->allTensorsAlongDimension(dimensions);
        auto outSubArrsSet = output->allTensorsAlongDimension(dimensions);

        for(int i = 0; i < inSubArrsSet.size(); ++i) {

            Nd4jLong numOfElemsToReverse = seqLengths->e<Nd4jLong>(i);

            if(numOfElemsToReverse == 0 || numOfElemsToReverse == 1) {
                outSubArrsSet.at(i)->assign(inSubArrsSet.at(i));
            }
            else {
                auto inInnerSet  = inSubArrsSet.at(i)->allTensorsAlongDimension({seqDim});
                auto outInnerSet = outSubArrsSet.at(i)->allTensorsAlongDimension({seqDim});
                for(int j = 0; j < inInnerSet.size(); ++j)
                    helpers::reverseArray<T>(context, inInnerSet.at(j)->buffer(), inInnerSet.at(j)->shapeInfo(), outInnerSet.at(j)->buffer(), outInnerSet.at(j)->shapeInfo(), numOfElemsToReverse);
            }
        }
    }
}

    void reverseSequence(sd::LaunchContext * context, const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim) {
        BUILD_SINGLE_SELECTOR(input->dataType(), reverseSequence_, (context, input, seqLengths, output, seqDim, batchDim), LIBND4J_TYPES);
    }

//////////////////////////////////////////////////////////////////////////
void reverse(sd::LaunchContext * context, const NDArray* input, NDArray* output, const std::vector<int>* intArgs) {

    auto listOut = output->allTensorsAlongDimension(*intArgs);
    auto listIn  = input->allTensorsAlongDimension(*intArgs);

    NDArray *subArrIn, *subArrOut;

    for(int i = 0; i < listIn.size(); ++i) {               // listIn.size() = listOut.size()
        subArrIn   = listIn.at(i);
        subArrOut  = listOut.at(i);
        BUILD_SINGLE_SELECTOR(input->dataType(), helpers::reverseArray, (context, subArrIn->buffer(), subArrIn->shapeInfo(), subArrOut->buffer(), subArrOut->shapeInfo()), LIBND4J_TYPES);
    }
}

BUILD_SINGLE_TEMPLATE(template void reverseSequence_, (sd::LaunchContext * context, const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim), LIBND4J_TYPES);
BUILD_SINGLE_TEMPLATE(template void reverseArray, (sd::LaunchContext * context, void const*inArr, Nd4jLong const*inShapeBuffer, void* outArr, Nd4jLong const* outShapeBuffer, int numOfElemsToReverse), LIBND4J_TYPES);


}
}
}

