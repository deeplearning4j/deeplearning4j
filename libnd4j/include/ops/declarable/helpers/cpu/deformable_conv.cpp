/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//

#include <execution/Threads.h>
#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/helpers/deformable_conv.h>

#include <cmath>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_deformable_conv2d)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_INLINE T bilinearInterpolateImpl(const T* input,
                                            sd::LongType batch,
                                            sd::LongType channel,
                                            double y, double x,
                                            sd::LongType height, sd::LongType width,
                                            sd::LongType batchStride,
                                            sd::LongType channelStride,
                                            sd::LongType heightStride,
                                            sd::LongType widthStride) {
    // Return 0 if outside bounds
    if (y < 0 || y >= height || x < 0 || x >= width) {
        return static_cast<T>(0);
    }

    // Get integer positions
    sd::LongType y0 = static_cast<sd::LongType>(std::floor(y));
    sd::LongType x0 = static_cast<sd::LongType>(std::floor(x));
    sd::LongType y1 = y0 + 1;
    sd::LongType x1 = x0 + 1;

    // Compute interpolation weights
    T ly = static_cast<T>(y - y0);
    T lx = static_cast<T>(x - x0);
    T hy = static_cast<T>(1.0) - ly;
    T hx = static_cast<T>(1.0) - lx;

    // Base offset for batch and channel
    const T* basePtr = input + batch * batchStride + channel * channelStride;

    // Gather values with boundary checking
    T v00 = (y0 >= 0 && y0 < height && x0 >= 0 && x0 < width) ?
            basePtr[y0 * heightStride + x0 * widthStride] : static_cast<T>(0);
    T v01 = (y0 >= 0 && y0 < height && x1 >= 0 && x1 < width) ?
            basePtr[y0 * heightStride + x1 * widthStride] : static_cast<T>(0);
    T v10 = (y1 >= 0 && y1 < height && x0 >= 0 && x0 < width) ?
            basePtr[y1 * heightStride + x0 * widthStride] : static_cast<T>(0);
    T v11 = (y1 >= 0 && y1 < height && x1 >= 0 && x1 < width) ?
            basePtr[y1 * heightStride + x1 * widthStride] : static_cast<T>(0);

    // Bilinear interpolation
    return hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11;
}

template <typename T>
static void deformableConv2d_(sd::LaunchContext* context,
                               NDArray* input,
                               NDArray* weights,
                               NDArray* offset,
                               NDArray* bias,
                               NDArray* mask,
                               NDArray* output,
                               int kH, int kW,
                               int sH, int sW,
                               int pH, int pW,
                               int dH, int dW,
                               int groups, int offsetGroups) {

    // Input: [batch, inChannels, inputH, inputW] (NCHW)
    const auto batchSize = input->sizeAt(0);
    const auto inChannels = input->sizeAt(1);
    const auto inputH = input->sizeAt(2);
    const auto inputW = input->sizeAt(3);

    // Weights: [outChannels, inChannels/groups, kH, kW]
    const auto outChannels = weights->sizeAt(0);

    // Output dimensions
    const auto outputH = (inputH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    const auto outputW = (inputW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;

    // Get raw buffers
    const T* inputBuffer = input->bufferAsT<T>();
    const T* weightsBuffer = weights->bufferAsT<T>();
    const T* offsetBuffer = offset->bufferAsT<T>();
    const T* biasBuffer = bias != nullptr ? bias->bufferAsT<T>() : nullptr;
    const T* maskBuffer = (mask != nullptr && !mask->isEmpty()) ? mask->bufferAsT<T>() : nullptr;
    T* outputBuffer = output->bufferAsT<T>();

    // Get strides
    const auto inputBatchStride = input->strideAt(0);
    const auto inputChannelStride = input->strideAt(1);
    const auto inputHeightStride = input->strideAt(2);
    const auto inputWidthStride = input->strideAt(3);

    const auto weightOutStride = weights->strideAt(0);
    const auto weightInStride = weights->strideAt(1);
    const auto weightKhStride = weights->strideAt(2);
    const auto weightKwStride = weights->strideAt(3);

    const auto offsetBatchStride = offset->strideAt(0);
    const auto offsetChannelStride = offset->strideAt(1);
    const auto offsetHeightStride = offset->strideAt(2);
    const auto offsetWidthStride = offset->strideAt(3);

    const auto outputBatchStride = output->strideAt(0);
    const auto outputChannelStride = output->strideAt(1);
    const auto outputHeightStride = output->strideAt(2);
    const auto outputWidthStride = output->strideAt(3);

    sd::LongType maskBatchStride = 0, maskChannelStride = 0, maskHeightStride = 0, maskWidthStride = 0;
    if (maskBuffer != nullptr) {
        maskBatchStride = mask->strideAt(0);
        maskChannelStride = mask->strideAt(1);
        maskHeightStride = mask->strideAt(2);
        maskWidthStride = mask->strideAt(3);
    }

    const auto kernelSize = kH * kW;
    const auto channelsPerGroup = inChannels / groups;
    const auto channelsPerOffsetGroup = inChannels / offsetGroups;
    const auto outChannelsPerGroup = outChannels / groups;

    // Initialize output to zero
    output->nullify();

    // Parallel over batch and output channels
    auto func = [&](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) {
        for (sd::LongType idx = start; idx < stop; idx += increment) {
            const sd::LongType b = idx / outChannels;
            const sd::LongType oc = idx % outChannels;

            const sd::LongType groupIdx = oc / outChannelsPerGroup;
            T* outPtr = outputBuffer + b * outputBatchStride + oc * outputChannelStride;

            // Add bias if present
            T biasVal = biasBuffer != nullptr ? biasBuffer[oc] : static_cast<T>(0);

            for (sd::LongType oh = 0; oh < outputH; oh++) {
                for (sd::LongType ow = 0; ow < outputW; ow++) {
                    T sum = biasVal;

                    // For each kernel position
                    for (sd::LongType kh = 0; kh < kH; kh++) {
                        for (sd::LongType kw = 0; kw < kW; kw++) {
                            const sd::LongType kernelIdx = kh * kW + kw;

                            // For each input channel in this group
                            for (sd::LongType ic = groupIdx * channelsPerGroup;
                                 ic < (groupIdx + 1) * channelsPerGroup; ic++) {

                                const sd::LongType offsetGroupIdx = ic / channelsPerOffsetGroup;
                                const sd::LongType offsetIdx = offsetGroupIdx * kernelSize + kernelIdx;

                                // Get learned offsets
                                const T* offsetPtr = offsetBuffer + b * offsetBatchStride;
                                T offsetY = offsetPtr[(2 * offsetIdx) * offsetChannelStride +
                                                       oh * offsetHeightStride + ow * offsetWidthStride];
                                T offsetX = offsetPtr[(2 * offsetIdx + 1) * offsetChannelStride +
                                                       oh * offsetHeightStride + ow * offsetWidthStride];

                                // Calculate sampling position with offset
                                double sampleY = static_cast<double>(oh * sH - pH + kh * dH) + offsetY;
                                double sampleX = static_cast<double>(ow * sW - pW + kw * dW) + offsetX;

                                // Bilinear interpolation
                                T value = bilinearInterpolateImpl<T>(
                                    inputBuffer, b, ic, sampleY, sampleX,
                                    inputH, inputW,
                                    inputBatchStride, inputChannelStride,
                                    inputHeightStride, inputWidthStride);

                                // Apply modulation mask if provided (Deformable Conv v2)
                                if (maskBuffer != nullptr) {
                                    const sd::LongType maskIdx = offsetGroupIdx * kernelSize + kernelIdx;
                                    T maskVal = maskBuffer[b * maskBatchStride +
                                                           maskIdx * maskChannelStride +
                                                           oh * maskHeightStride +
                                                           ow * maskWidthStride];
                                    value *= maskVal;
                                }

                                // Get weight and accumulate
                                T weight = weightsBuffer[oc * weightOutStride +
                                                          (ic - groupIdx * channelsPerGroup) * weightInStride +
                                                          kh * weightKhStride +
                                                          kw * weightKwStride];
                                sum += value * weight;
                            }
                        }
                    }

                    outPtr[oh * outputHeightStride + ow * outputWidthStride] = sum;
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize * outChannels, 1);
}

void deformableConv2d(sd::LaunchContext* context,
                       NDArray* input,
                       NDArray* weights,
                       NDArray* offset,
                       NDArray* bias,
                       NDArray* mask,
                       NDArray* output,
                       int kH, int kW,
                       int sH, int sW,
                       int pH, int pW,
                       int dH, int dW,
                       int groups, int offsetGroups) {

    BUILD_SINGLE_SELECTOR(input->dataType(), deformableConv2d_,
                          (context, input, weights, offset, bias, mask, output,
                           kH, kW, sH, sW, pH, pW, dH, dW, groups, offsetGroups),
                          SD_FLOAT_TYPES);
}


}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
