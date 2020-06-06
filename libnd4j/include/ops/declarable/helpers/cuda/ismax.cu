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
// @author Yurii Shyrma, created on 21.09.2018
// @author raver119@gmail.com
//


#include <helpers/TAD.h>
#include<ops/declarable/helpers/ismax.h>
#include<loops/special_kernels.h>
#include <helpers/DebugHelper.h>
#include <exceptions/cuda_exception.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

template <typename T>
static void ismax_(sd::LaunchContext * context, const NDArray* input, NDArray* output, const std::vector<int>& dimensions) {
    auto stream = context->getCudaStream();

    auto xRank = input->rankOf();
    auto zRank = output->rankOf();
    auto xType = input->dataType();
    auto zType = output->dataType();
    input->syncToDevice();
    Nd4jLong* special = nullptr;
    PointersManager manager(context, "IsMaxHelper");
    if (dimensions.size() == 0) {
        /**
        * In case of vector-input for IsMax, it just turns into IndexReduce call + subsequent filler call
        */
        auto indexMax = input->applyIndexReduce(indexreduce::IndexMax, dimensions);
        auto targetIdx = indexMax.e<Nd4jLong>(0);

        dim3 launchDims(128, 512, 1024);
        BUILD_SINGLE_SELECTOR(zType, fillIsMaxGeneric, (launchDims, stream, output->specialBuffer(), output->specialShapeInfo(), output->lengthOf(), targetIdx), LIBND4J_TYPES);
        manager.synchronize();

    } else {
        Nd4jLong* hostYShapeInfo  = nullptr;
        Nd4jLong* hostTShapeInfo  = nullptr;
        int* dimension = nullptr;
        int dimensionLength = dimensions.size();
        std::vector<int> copy(dimensions);

        auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), copy.data(), copy.size());

        // we launch legacy IndexMax op, to get indices of max values along dimension
        auto indexMaxArr = input->applyIndexReduce(indexreduce::IndexMax, dimensions);

        dim3 launchDims(256, 256, 16384);
        dimension = (int *) manager.replicatePointer(dimensions.data(), dimensions.size() * sizeof(int));

        // at this point, all IMax indexes are gathered, and we execute filler
        BUILD_SINGLE_SELECTOR(zType, fillDimensionalIsMaxGeneric, (launchDims, stream, indexMaxArr.specialBuffer(), output->specialBuffer(), output->specialShapeInfo(), packZ.specialShapeInfo(), dimension, dimensionLength, packZ.specialOffsets()), LIBND4J_TYPES);
        manager.synchronize();
    }
}


void ismax(sd::LaunchContext * context, const NDArray *input, NDArray *output, const std::vector<int>& dimensions) {
    NDArray::prepareSpecialUse({output}, {input});

    BUILD_SINGLE_SELECTOR(input->dataType(), ismax_, (context, input, output, dimensions), LIBND4J_TYPES);

    NDArray::registerSpecialUse({output}, {input});
}

BUILD_SINGLE_TEMPLATE(template void ismax_, (sd::LaunchContext * context, const NDArray *input, NDArray *output, const std::vector<int>& dimensions), LIBND4J_TYPES);

}
}
}

