/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/adjust_saturation.h>
#include <ops/declarable/helpers/adjust_hue.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>


namespace sd    {
namespace ops     {
namespace helpers {

template <typename T>
static void adjustSaturation_(const NDArray *input, const NDArray* factorScalarArr, NDArray *output, const int dimC) {

    const T factor = factorScalarArr->e<T>(0);
    const int rank = input->rankOf();

    const T* x = input->bufferAsT<T>();
          T* z = output->bufferAsT<T>();

    if(dimC == rank - 1 && input->ews() == 1 && output->ews() == 1 && input->ordering() == 'c' && output->ordering() == 'c') {

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i += increment) {
                T h, s, v;

                rgbToHsv<T>(x[i], x[i + 1], x[i + 2], h, s, v);

                s *= factor;
                if (s > 1.f)
                    s = 1.f;
                else if (s < 0.f)
                    s = 0.f;

                hsvToRgb<T>(h, s, v, z[i], z[i + 1], z[i + 2]);
            }
        };

        samediff::Threads::parallel_for(func, 0, input->lengthOf(), 3);
    } else {
        auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(),  dimC);
        auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimC);

        const Nd4jLong numOfTads   = packX.numberOfTads();
        const Nd4jLong xDimCstride = input->stridesOf()[dimC];
        const Nd4jLong zDimCstride = output->stridesOf()[dimC];

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
                const T *xTad = x + packX.platformOffsets()[i];
                T *zTad = z + packZ.platformOffsets()[i];

                T h, s, v;

                rgbToHsv<T>(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], h, s, v);

                s *= factor;
                if (s > 1.f)
                    s = 1.f;
                else if (s < 0.f)
                    s = 0.f;

                hsvToRgb<T>(h, s, v, zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);
            }
        };

        samediff::Threads::parallel_tad(func, 0, numOfTads);
    }
}


void adjustSaturation(sd::LaunchContext* context, const NDArray *input, const NDArray* factorScalarArr, NDArray *output, const int dimC) {

    BUILD_SINGLE_SELECTOR(input->dataType(), adjustSaturation_, (input, factorScalarArr, output, dimC), FLOAT_TYPES);
}

/*
template <typename T>
static void adjust_saturation_single_(sd::LaunchContext * context, NDArray *array, NDArray *output, float delta, bool isNHWC) {
    // we're 100% sure it's 3
    const int numChannels = 3;
    int tuples = array->lengthOf() /  numChannels;
    auto bIn = reinterpret_cast<T *>(array->buffer());
    auto bOut = reinterpret_cast<T *>(output->buffer());
    static const int kChannelRange = 6;

    if (isNHWC) {
        // for NHWC our rgb values are stored one by one
        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int e = 0; e < tuples; e++) {
            auto i = bIn + e * numChannels;
            auto o = bOut + e * numChannels;

            T h, s, v;
            // Convert the RGB color to Hue/V-range.
            helpers::rgb_to_hsv(i[0], i[1], i[2], &h, &s, &v);
            s = sd::math::nd4j_min<T>((T) 1.0f, sd::math::nd4j_max<T>((T) 0.0f, s * delta));
            // Convert the hue and v-range back into RGB.
            helpers::hsv_to_rgb(h, s, v, o, o + 1, o + 2);
        }
    } else {
        auto tadsChannelsIn = array->allTensorsAlongDimension({0});
        auto tadsChannelsOut = output->allTensorsAlongDimension({0});

        auto bufferR = reinterpret_cast<T *>(tadsChannelsIn->at(0)->buffer());
        auto bufferG = reinterpret_cast<T *>(tadsChannelsIn->at(1)->buffer());
        auto bufferB = reinterpret_cast<T *>(tadsChannelsIn->at(2)->buffer());

        auto outputR = reinterpret_cast<T *>(tadsChannelsOut->at(0)->buffer());
        auto outputG = reinterpret_cast<T *>(tadsChannelsOut->at(1)->buffer());
        auto outputB = reinterpret_cast<T *>(tadsChannelsOut->at(2)->buffer());

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int e = 0; e < tuples; e++) {
            auto _ri = bufferR + e;
            auto _gi = bufferG + e;
            auto _bi = bufferB + e;

            auto _ro = outputR + e;
            auto _go = outputG + e;
            auto _bo = outputB + e;

            T h, s, v;
            // Convert the RGB color to Hue/V-range.
            helpers::rgb_to_hsv(_ri[0], _gi[0], _bi[0], &h, &s, &v);
            s = sd::math::nd4j_min<T>((T) 1.0f, sd::math::nd4j_max<T>((T) 0.0f, s * delta));
            // Convert the hue and v-range back into RGB.
            helpers::hsv_to_rgb(h, s, v, _ro, _go, _bo);
        }

        delete tadsChannelsIn;
        delete tadsChannelsOut;
    }
}

void adjust_saturation(sd::LaunchContext * context, NDArray *array, NDArray *output, NDArray* delta, bool isNHWC) {
    auto xType = array->dataType();

    float d = delta->e<float>(0);
    if (array->rankOf() == 4) {
        auto tadsIn = array->allTensorsAlongDimension({0});
        auto tadsOut = output->allTensorsAlongDimension({0});
        int tSize = tadsIn->size();

        // FIXME: template selector should be moved out of loop
        PRAGMA_OMP_PARALLEL_FOR
        for (int e = 0; e < tSize; e++) {
            BUILD_SINGLE_SELECTOR(xType, adjust_saturation_single_, (context, tadsIn->at(e), tadsOut->at(e), d, isNHWC);, FLOAT_TYPES);
        }


        delete tadsIn;
        delete tadsOut;
    }
    else {
        BUILD_SINGLE_SELECTOR(xType, adjust_saturation_single_, (context, array, output, d, isNHWC);, FLOAT_TYPES);
    }
}

BUILD_SINGLE_TEMPLATE(template void adjust_saturation_single_, (sd::LaunchContext * context, NDArray *array, NDArray *output, float delta, bool isNHWC), FLOAT_TYPES);
*/

}
}
}