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
#include <TAD.h>
#include <PointersManager.h>
#include <ConstantTadHelper.h>


namespace nd4j    {
namespace ops     {
namespace helpers {

    template <typename T>
    static __global__ void reverseArrayKernel(void* input, Nd4jLong *inputShape, void* output, Nd4jLong *outputShape, Nd4jLong numOfElemsToReverse) {
        const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;
        __shared__ int linearStatus;
        __shared__ T* inputArr;
        __shared__ T* outputArr;
        __shared__ char inputOrder, outputOrder;

        if (threadIdx.x == 0) {
            linearStatus = (shape::elementWiseStride(inputShape) == shape::elementWiseStride(outputShape)) && (inputOrder == outputOrder)? shape::elementWiseStride(inputShape):0;

            char inputOrder = shape::order(inputShape);
            char outputOrder = shape::order(outputShape);
            inputArr = reinterpret_cast<T*>(input);
            outputArr = reinterpret_cast<T*>(output);
        }
        __syncthreads();

        auto odd = numOfElemsToReverse % 2 != 0;
        auto limit = numOfElemsToReverse / 2;

        for (Nd4jLong e = tid; e < limit; e += step) {
            // we're calculating offsets within input array
            auto fOffset = shape::getIndexOffset(e, inputShape);
            auto lOffset = shape::getIndexOffset(numOfElemsToReverse - e - 1, inputShape);

            // now we're storing input values
            auto v1 = inputArr[fOffset];
            auto v2 = inputArr[lOffset];

            // now we're calculating offsets within output array
            auto zfOffset = shape::getIndexOffset(e, outputShape);
            auto zlOffset = shape::getIndexOffset(numOfElemsToReverse - e - 1, outputShape);

            // and saving values to output arrays
            outputArr[zfOffset] = v2;
            outputArr[zlOffset] = v1;
        }

        // in case of odd array we'll have to move middle value
        if (odd && tid == 0) {
            auto xOffset = shape::getIndexOffset(limit, inputShape);
            auto zOffset = shape::getIndexOffset(limit, outputShape);

            outputArr[zOffset] = inputArr[xOffset];
        }
    }

    template<typename T>
    static void reverseArray(nd4j::LaunchContext * context, NDArray* input, NDArray* output, Nd4jLong numOfElemsToReverse) {
        auto stream = context->getCudaStream();
        Nd4jLong numOfReverse = numOfElemsToReverse;
        if (numOfElemsToReverse == 0)
            numOfReverse = input->lengthOf();

        reverseArrayKernel<T><<<256, 512, 8192, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), numOfReverse);
    }


    ///////////////////////////////////////////////////////////////////
    template <typename T>
    static void reverseSequence_(nd4j::LaunchContext * context, const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim){
        int posOfNonUnityDim = -1;
        seqLengths->syncToHost();
        auto stream = context->getCudaStream();

        if(input->isVector() || shape::isLikeVector(input->getShapeInfo(), posOfNonUnityDim) || seqLengths->lengthOf() == 1) {
            int numOfElemsToReverse = seqLengths->e<int>(0);
            if((seqDim == 0 && input->sizeAt(0) == 1) || (batchDim == posOfNonUnityDim))
                output->assign(input);
            else
                reverseArrayKernel<T><<<256, 512, 8192, *stream>>>(input->getSpecialBuffer(), input->getSpecialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), numOfElemsToReverse);//helpers::reverseArray<T>(context, const_cast<NDArray*>(input), output, numOfElemsToReverse);
        }
        else {

            if(seqDim > batchDim)
                --seqDim;

            std::vector<int> dimensions = ShapeUtils::evalDimsToExclude(input->rankOf(), {batchDim});

            auto inSubArrsSet  = input->allTensorsAlongDimension(dimensions);
            auto outSubArrsSet = output->allTensorsAlongDimension(dimensions);

            for(int i = 0; i < inSubArrsSet->size(); ++i) {

                int numOfElemsToReverse = seqLengths->e<int>(i);

                if(numOfElemsToReverse == 0 || numOfElemsToReverse == 1) {
                    outSubArrsSet->at(i)->assign(inSubArrsSet->at(i));
                }
                else {
                    auto inInnerSet  = inSubArrsSet->at(i)->allTensorsAlongDimension({seqDim});
                    auto outInnerSet = outSubArrsSet->at(i)->allTensorsAlongDimension({seqDim});
                    for(int j = 0; j < inInnerSet->size(); ++j)
                        reverseArray<T>(context, inInnerSet->at(j), outInnerSet->at(j), numOfElemsToReverse);

                    delete inInnerSet;
                    delete outInnerSet;
                }
            }
            delete inSubArrsSet;
            delete outSubArrsSet;
        }

    }

    void reverseSequence(nd4j::LaunchContext * context, const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim) {
        NDArray::prepareSpecialUse({output}, {input, seqLengths});

        // if op isn't inplace - copy original data into output array
        if (output->getSpecialBuffer() != input->getSpecialBuffer())
            output->assign(input);

        BUILD_SINGLE_SELECTOR(input->dataType(), reverseSequence_, (context, input, seqLengths, output, seqDim, batchDim), LIBND4J_TYPES);
        NDArray::registerSpecialUse({output}, {input, seqLengths});
    }

    //////////////////////////////////////////////////////////////////////////
    void reverse(nd4j::LaunchContext * context, const NDArray* input, NDArray* output, const std::vector<int>* intArgs, bool isBackProp) {
        // we need to reverse axis only if that's new op
        std::vector<int> dimensions = isBackProp ? ShapeUtils::evalDimsToExclude(input->rankOf(), *intArgs) : *intArgs;
        std::vector<int> axis = ShapeUtils::evalDimsToExclude(input->rankOf(), dimensions);
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), axis);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), axis);

        auto listOut = output->allTensorsAlongDimension(dimensions);
        auto listIn  = input->allTensorsAlongDimension(dimensions);

        NDArray *subArrIn, *subArrOut;

        NDArray::prepareSpecialUse({output}, {input});
        for(int i = 0; i < listIn->size(); ++i) {               // listIn->size() = listOut->size()
            subArrIn   = listIn->at(i);
            subArrOut  = listOut->at(i);
            BUILD_SINGLE_SELECTOR(input->dataType(), reverseArray, (context, subArrIn, subArrOut, 0), LIBND4J_TYPES);
        }
        //BUILD_SINGLE_SELECTOR(input->dataType(), reverseArray, (context, const_cast<NDArray*>(input), output, (int)0), LIBND4J_TYPES);
        NDArray::registerSpecialUse({output}, {input});
        delete listOut;
        delete listIn;
    }

BUILD_SINGLE_TEMPLATE(template void reverseArray, (nd4j::LaunchContext * context, NDArray *inArr, NDArray *outArr, Nd4jLong numOfElemsToReverse), LIBND4J_TYPES);

}
}
}

