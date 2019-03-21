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
    inline void __device__ indexSwap(T* arr, Nd4jLong idx1, Nd4jLong idx2) {
        T tmp = arr[idx1];
        arr[idx1] = arr[idx2];
        arr[idx2] = tmp;
    }
//    template <typename T>
//    void reverseArray(graph::LaunchContext* context, void* inArr, Nd4jLong *inShapeBuffer, void *result, Nd4jLong *zShapeBuffer, int numOfElemsToReverse = 0);

    /////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    static __global__ void reverseArrayInplaceKernel(void *input, Nd4jLong *inputShape, Nd4jLong numOfElemsToReverse) {
        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;
        __shared__ Nd4jLong length;
        __shared__ int linearStatus;
        __shared__ T* inputArr;
        if (threadIdx.x == 0) {
            length = shape::length(inputShape);
            linearStatus = shape::elementWiseStride(inputShape);
            inputArr = reinterpret_cast<T*>(input);
        }

        for (Nd4jLong e = tid; e < numOfElemsToReverse / 2; e += step) {
            if (linearStatus == 1) {
                auto idx = numOfElemsToReverse - e - 1;
                indexSwap(inputArr, e, idx);
            }
            else if (linearStatus > 1) {
                auto idx1 = (numOfElemsToReverse - e - 1) * linearStatus;
                Nd4jLong idx2 =  e * linearStatus;
                indexSwap(inputArr, idx1, idx2);
            }
            else {
                auto inOffset  = shape::getIndexOffset(e, inputShape, length);
                auto outOffset = shape::getIndexOffset(numOfElemsToReverse - e - 1, inputShape, length);
                indexSwap(inputArr, inOffset, outOffset);
            }
        }
    }

    template <typename T>
    static __global__ void reverseArrayKernel(void* input, Nd4jLong *inputShape, void* output, Nd4jLong *outputShape, Nd4jLong numOfElemsToReverse) {
        const auto tid = blockIdx.x * gridDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;
        __shared__ Nd4jLong length;
        __shared__ int linearStatus;
        __shared__ T* inputArr;
        __shared__ T* outputArr;
        __shared__ char inputOrder, outputOrder;

        if (threadIdx.x == 0) {
            length = shape::length(inputShape);
            linearStatus = (shape::elementWiseStride(inputShape) == shape::elementWiseStride(outputShape)) && (inputOrder == outputOrder)? shape::elementWiseStride(inputShape):0;

            char inputOrder = shape::order(inputShape);
            char outputOrder = shape::order(outputShape);
            inputArr = reinterpret_cast<T*>(input);
            outputArr = reinterpret_cast<T*>(output);
        }
        __syncthreads();

        for (Nd4jLong e = tid; e < numOfElemsToReverse; e += step) {
            if (linearStatus == 1) {
                auto idx = numOfElemsToReverse - e - 1;
                outputArr[idx] = inputArr[e];
            }
            else if (linearStatus > 1) {
                auto idx1 = (numOfElemsToReverse - e - 1) * linearStatus;
                Nd4jLong idx2 =  e * linearStatus;
                outputArr[idx1] = inputArr[idx2];
            }
            else {
                auto inOffset  = shape::getIndexOffset(e, inputShape, length);
                auto outOffset = shape::getIndexOffset(numOfElemsToReverse - e - 1, outputShape, length);
                outputArr[outOffset] = inputArr[inOffset];
            }
        }
        //printf("\n");
    }

    template<typename T>
    static void reverseArray(graph::LaunchContext* context, NDArray* input, NDArray* output, int numOfElemsToReverse) {
        auto stream = context->getCudaStream();
        Nd4jLong numOfReverse = numOfElemsToReverse;
        if (numOfElemsToReverse == 0)
            numOfReverse = input->lengthOf();
        if (input == output) {
            reverseArrayInplaceKernel<T><<<256, 512, 8192, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), numOfReverse);
        }
        else {
            reverseArrayKernel<T><<<256, 512, 8192, *stream>>>(input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), numOfReverse);
        }
    }


    ///////////////////////////////////////////////////////////////////
    template <typename T>
    static void _reverseSequence(const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim){

    }

    void reverseSequence(graph::LaunchContext* context, const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim) {
        BUILD_SINGLE_SELECTOR(input->dataType(), _reverseSequence, (input, seqLengths, output, seqDim, batchDim), LIBND4J_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    void reverse(graph::LaunchContext* context, const NDArray* input, NDArray* output, const std::vector<int>* intArgs, bool isBackProp) {
        // we need to reverse axis only if that's new op
        std::vector<int> dimensions = isBackProp ? ShapeUtils::evalDimsToExclude(input->rankOf(), *intArgs) : *intArgs;
        std::vector<int> axis = ShapeUtils::evalDimsToExclude(input->rankOf(), dimensions);
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), axis);
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), axis);

        auto listOut = output->allTensorsAlongDimension(dimensions);
        auto listIn  = input->allTensorsAlongDimension(dimensions);

        NDArray *subArrIn, *subArrOut;

        for(int i = 0; i < listIn->size(); ++i) {               // listIn->size() = listOut->size()
            subArrIn   = listIn->at(i);
            subArrOut  = listOut->at(i);
            BUILD_SINGLE_SELECTOR(input->dataType(), reverseArray, (context, subArrIn, subArrOut, 0), LIBND4J_TYPES);
        }
        //BUILD_SINGLE_SELECTOR(input->dataType(), reverseArray, (context, const_cast<NDArray*>(input), output, (int)0), LIBND4J_TYPES);
        input->tickReadDevice();
        output->tickWriteDevice();
        delete listOut;
        delete listIn;
    }

BUILD_SINGLE_TEMPLATE(template void reverseArray, (graph::LaunchContext* context, NDArray *inArr, NDArray *outArr, int numOfElemsToReverse), LIBND4J_TYPES);

}
}
}

