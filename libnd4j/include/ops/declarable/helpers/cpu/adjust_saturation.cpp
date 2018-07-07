//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/adjust_saturation.h>


namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    FORCEINLINE void _adjust_saturation_single(NDArray<T> *array, NDArray<T> *output, T delta, bool isNHWC) {
        // we're 100% sure it's 3
        const int numChannels = 3;
        int tuples = array->lengthOf() /  numChannels;
        auto bIn = array->buffer();
        auto bOut = output->buffer();
        static const int kChannelRange = 6;

        if (isNHWC) {
            // for NHWC our rgb values are stored one by one
            #pragma omp parallel for simd
            for (int e = 0; e < tuples; e++) {
                auto i = bIn + e * numChannels;
                auto o = bOut + e * numChannels;

                T h, s, v;
                // Convert the RGB color to Hue/V-range.
                helpers::rgb_to_hsv(i[0], i[1], i[2], &h, &s, &v);
                s = nd4j::math::nd4j_min<T>((T) 1.0f, nd4j::math::nd4j_max<T>((T) 0.0f, s * delta));
                // Convert the hue and v-range back into RGB.
                helpers::hsv_to_rgb(h, s, v, o, o + 1, o + 2);
            }
        } else {
            auto tadsChannelsIn = array->allTensorsAlongDimension({0});
            auto tadsChannelsOut = output->allTensorsAlongDimension({0});

            auto bufferR = tadsChannelsIn->at(0)->buffer();
            auto bufferG = tadsChannelsIn->at(1)->buffer();
            auto bufferB = tadsChannelsIn->at(2)->buffer();

            auto outputR = tadsChannelsOut->at(0)->buffer();
            auto outputG = tadsChannelsOut->at(1)->buffer();
            auto outputB = tadsChannelsOut->at(2)->buffer();

            #pragma omp parallel for simd 
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
                s = nd4j::math::nd4j_min<T>((T) 1.0f, nd4j::math::nd4j_max<T>((T) 0.0f, s * delta));
                // Convert the hue and v-range back into RGB.
                helpers::hsv_to_rgb(h, s, v, _ro, _go, _bo);
            }

            delete tadsChannelsIn;
            delete tadsChannelsOut;
        }
    }

    template <typename T>
    void _adjust_saturation(NDArray<T> *array, NDArray<T> *output, T delta, bool isNHWC) {
        if (array->rankOf() == 4) {
            auto tadsIn = array->allTensorsAlongDimension({0});
            auto tadsOut = output->allTensorsAlongDimension({0});

#pragma omp parallel for
            for (int e = 0; e < tadsIn->size(); e++)
                _adjust_saturation_single(tadsIn->at(e), tadsOut->at(e), delta, isNHWC);
            

            delete tadsIn;
            delete tadsOut;
        } else {
            _adjust_saturation_single(array, output, delta, isNHWC);
        }
    }

    template void _adjust_saturation<float>(NDArray<float> *array, NDArray<float> *output, float delta, bool isNHWC);
    template void _adjust_saturation<float16>(NDArray<float16> *array, NDArray<float16> *output, float16 delta, bool isNHWC);
    template void _adjust_saturation<double>(NDArray<double> *array, NDArray<double> *output, double delta, bool isNHWC);
}
}
}