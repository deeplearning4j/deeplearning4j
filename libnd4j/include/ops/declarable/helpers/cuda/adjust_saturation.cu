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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/adjust_saturation.h>
#include <helpers/ConstantTadHelper.h>


namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void _CUDA_G adjustSaturationSingleNHWCKernel(void *xBuffer, Nd4jLong *xShapeInfo,  void *zBuffer, Nd4jLong *zShapeInfo, Nd4jLong tuples, float delta) {
        int numChannels = 3;
        auto tid = threadIdx.x + blockIdx.x * blockDim.x;

        auto bIn = reinterpret_cast<T*>(xBuffer);
        auto bOut = reinterpret_cast<T*>(zBuffer);
        static const int kChannelRange = 6;

        for (Nd4jLong e = tid; e < tuples; e += blockDim.x * gridDim.x) {
            auto i = bIn + e * numChannels;
            auto o = bOut + e * numChannels;

            T h, s, v;
            // Convert the RGB color to Hue/V-range.
            helpers::rgb_to_hsv(i[0], i[1], i[2], &h, &s, &v);
            s = nd4j::math::nd4j_min<T>((T) 1.0f, nd4j::math::nd4j_max<T>((T) 0.0f, s * delta));

            // Convert the hue and v-range back into RGB.
            helpers::hsv_to_rgb(h, s, v, o, o + 1, o + 2);
        }
    }

    template <typename T>
    static void _CUDA_G adjustSaturationSingleNCHWKernel(void *xBuffer, Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, void *zBuffer, Nd4jLong *zTadShapeInfo, Nd4jLong *zOffsets, Nd4jLong tadLength, Nd4jLong tuples, float delta) {
        int numChannels = 3;
        auto tid = threadIdx.x + blockIdx.x * blockDim.x;
        static const int kChannelRange = 6;

        auto bufferR = reinterpret_cast<T *>(xBuffer) + xOffsets[0];
        auto bufferG = reinterpret_cast<T *>(xBuffer) + xOffsets[1];
        auto bufferB = reinterpret_cast<T *>(xBuffer) + xOffsets[2];

        auto outputR = reinterpret_cast<T *>(zBuffer) + zOffsets[0];
        auto outputG = reinterpret_cast<T *>(zBuffer) + zOffsets[1];
        auto outputB = reinterpret_cast<T *>(zBuffer) + zOffsets[2];

        for (Nd4jLong e = tid; e < tuples; e += blockDim.x * gridDim.x) {
            auto _ri = bufferR + shape::getIndexOffset(e, xTadShapeInfo, tadLength);
            auto _gi = bufferG + shape::getIndexOffset(e, xTadShapeInfo, tadLength);
            auto _bi = bufferB + shape::getIndexOffset(e, xTadShapeInfo, tadLength);

            auto _ro = outputR + shape::getIndexOffset(e, xTadShapeInfo, tadLength);
            auto _go = outputG + shape::getIndexOffset(e, xTadShapeInfo, tadLength);
            auto _bo = outputB + shape::getIndexOffset(e, xTadShapeInfo, tadLength);

            T h, s, v;
            // Convert the RGB color to Hue/V-range.
            helpers::rgb_to_hsv(_ri[0], _gi[0], _bi[0], &h, &s, &v);
            s = nd4j::math::nd4j_min<T>((T) 1.0f, nd4j::math::nd4j_max<T>((T) 0.0f, s * delta));
            // Convert the hue and v-range back into RGB.
            helpers::hsv_to_rgb(h, s, v, _ro, _go, _bo);
        }
    }

    template <typename T>
    static void _adjust_saturation_single(nd4j::LaunchContext * context, NDArray *array, NDArray *output, float delta, bool isNHWC) {
        // numChannels is always 3
        auto tuples = array->lengthOf() / 3;

        if (isNHWC) {
            adjustSaturationSingleNHWCKernel<T><<<256, 256, 1024, *context->getCudaStream()>>>(array->specialBuffer(), array->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), tuples, delta);
        } else {
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(array->getShapeInfo(), {1, 2});
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), {1, 2});

            auto tadLength = shape::length(packX.primaryShapeInfo());

            adjustSaturationSingleNCHWKernel<T><<<256, 256, 1024, *context->getCudaStream()>>>(array->specialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), output->specialBuffer(), packZ.platformShapeInfo(), packZ.platformOffsets(), tadLength, tuples, delta);
        }
    }

    template <typename T>
    static void _adjust_saturation_batch(nd4j::LaunchContext * context, NDArray *array, NDArray *output, float delta, bool isNHWC) {
        auto xType = array->dataType();

        // numChannels is always 3
        auto tuples = array->lengthOf() / 3;

        if (isNHWC) {
            // in case of nhwc batch, we don't really care about examples: it's still bunch of RGB values
            BUILD_SINGLE_SELECTOR(xType, _adjust_saturation_single, (context, array, output, delta, isNHWC);, FLOAT_TYPES);
        } else {
            // TODO: check this one
            auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(array->getShapeInfo(), {0, 2, 3});
            auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), {0, 2, 3});

            auto tadLength = shape::length(packX.primaryShapeInfo());

            adjustSaturationSingleNCHWKernel<T><<<256, 256, 1024, *context->getCudaStream()>>>(array->specialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), output->specialBuffer(), packZ.platformShapeInfo(), packZ.platformOffsets(), tadLength, tuples, delta);
        }
    }

    void adjust_saturation(nd4j::LaunchContext * context, NDArray *array, NDArray *output, NDArray* delta, bool isNHWC) {
        auto xType = array->dataType();

        float d = delta->e<float>(0);
        if (array->rankOf() == 4) {
            BUILD_SINGLE_SELECTOR(xType, _adjust_saturation_batch, (context, array, output, d, isNHWC);, FLOAT_TYPES);
        } else {
            BUILD_SINGLE_SELECTOR(xType, _adjust_saturation_single, (context, array, output, d, isNHWC);, FLOAT_TYPES);
        }
    }


}
}
}