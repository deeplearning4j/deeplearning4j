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

#include <cuda_runtime.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/deformable_conv.h>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_deformable_conv2d)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
__device__ static T bilinearInterpolateDevice(const T* input,
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
    sd::LongType y0 = static_cast<sd::LongType>(floor(y));
    sd::LongType x0 = static_cast<sd::LongType>(floor(x));
    sd::LongType y1 = y0 + 1;
    sd::LongType x1 = x0 + 1;

    // Compute interpolation weights
    T ly = static_cast<T>(y - y0);
    T lx = static_cast<T>(x - x0);
    T hy = static_cast<T>(1.0) - ly;
    T hx = static_cast<T>(1.0) - lx;

    // Base offset
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

    return hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11;
}

template <typename T>
__global__ static void deformableConv2dKernel(
    const T* input,
    const T* weights,
    const T* offset,
    const T* bias,
    const T* mask,
    T* output,
    const sd::LongType batchSize,
    const sd::LongType inChannels,
    const sd::LongType inputH,
    const sd::LongType inputW,
    const sd::LongType outChannels,
    const sd::LongType outputH,
    const sd::LongType outputW,
    const int kH, const int kW,
    const int sH, const int sW,
    const int pH, const int pW,
    const int dH, const int dW,
    const int groups,
    const int offsetGroups,
    const sd::LongType inputBatchStride,
    const sd::LongType inputChannelStride,
    const sd::LongType inputHeightStride,
    const sd::LongType inputWidthStride,
    const sd::LongType weightOutStride,
    const sd::LongType weightInStride,
    const sd::LongType weightKhStride,
    const sd::LongType weightKwStride,
    const sd::LongType offsetBatchStride,
    const sd::LongType offsetChannelStride,
    const sd::LongType offsetHeightStride,
    const sd::LongType offsetWidthStride,
    const sd::LongType maskBatchStride,
    const sd::LongType maskChannelStride,
    const sd::LongType maskHeightStride,
    const sd::LongType maskWidthStride,
    const sd::LongType outputBatchStride,
    const sd::LongType outputChannelStride,
    const sd::LongType outputHeightStride,
    const sd::LongType outputWidthStride,
    const bool hasMask) {

    // Each thread computes one output element
    const sd::LongType totalOutputs = batchSize * outChannels * outputH * outputW;
    const sd::LongType idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalOutputs) return;

    // Decode indices
    const sd::LongType ow = idx % outputW;
    const sd::LongType oh = (idx / outputW) % outputH;
    const sd::LongType oc = (idx / (outputW * outputH)) % outChannels;
    const sd::LongType b = idx / (outputW * outputH * outChannels);

    const sd::LongType kernelSize = kH * kW;
    const sd::LongType channelsPerGroup = inChannels / groups;
    const sd::LongType channelsPerOffsetGroup = inChannels / offsetGroups;
    const sd::LongType outChannelsPerGroup = outChannels / groups;
    const sd::LongType groupIdx = oc / outChannelsPerGroup;

    // Start with bias
    T sum = (bias != nullptr) ? bias[oc] : static_cast<T>(0);

    // For each kernel position
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            const sd::LongType kernelIdx = kh * kW + kw;

            // For each input channel in this group
            for (sd::LongType ic = groupIdx * channelsPerGroup;
                 ic < (groupIdx + 1) * channelsPerGroup; ic++) {

                const sd::LongType offsetGroupIdx = ic / channelsPerOffsetGroup;
                const sd::LongType offsetIdx = offsetGroupIdx * kernelSize + kernelIdx;

                // Get learned offsets
                const T* offsetPtr = offset + b * offsetBatchStride;
                T offsetY = offsetPtr[(2 * offsetIdx) * offsetChannelStride +
                                       oh * offsetHeightStride + ow * offsetWidthStride];
                T offsetX = offsetPtr[(2 * offsetIdx + 1) * offsetChannelStride +
                                       oh * offsetHeightStride + ow * offsetWidthStride];

                // Calculate sampling position
                double sampleY = static_cast<double>(oh * sH - pH + kh * dH) + offsetY;
                double sampleX = static_cast<double>(ow * sW - pW + kw * dW) + offsetX;

                // Bilinear interpolation
                T value = bilinearInterpolateDevice<T>(
                    input, b, ic, sampleY, sampleX,
                    inputH, inputW,
                    inputBatchStride, inputChannelStride,
                    inputHeightStride, inputWidthStride);

                // Apply modulation mask if provided
                if (hasMask && mask != nullptr) {
                    const sd::LongType maskIdx = offsetGroupIdx * kernelSize + kernelIdx;
                    T maskVal = mask[b * maskBatchStride +
                                      maskIdx * maskChannelStride +
                                      oh * maskHeightStride +
                                      ow * maskWidthStride];
                    value *= maskVal;
                }

                // Get weight and accumulate
                T weight = weights[oc * weightOutStride +
                                    (ic - groupIdx * channelsPerGroup) * weightInStride +
                                    kh * weightKhStride +
                                    kw * weightKwStride];
                sum += value * weight;
            }
        }
    }

    output[b * outputBatchStride + oc * outputChannelStride +
           oh * outputHeightStride + ow * outputWidthStride] = sum;
}

template <typename T>
static void deformableConv2dCuda_(sd::LaunchContext* context,
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

    const auto batchSize = input->sizeAt(0);
    const auto inChannels = input->sizeAt(1);
    const auto inputH = input->sizeAt(2);
    const auto inputW = input->sizeAt(3);
    const auto outChannels = weights->sizeAt(0);
    const auto outputH = (inputH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    const auto outputW = (inputW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;

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
    bool hasMask = mask != nullptr && !mask->isEmpty();
    if (hasMask) {
        maskBatchStride = mask->strideAt(0);
        maskChannelStride = mask->strideAt(1);
        maskHeightStride = mask->strideAt(2);
        maskWidthStride = mask->strideAt(3);
    }

    const sd::LongType totalOutputs = batchSize * outChannels * outputH * outputW;
    const int blockSize = 256;
    const int numBlocks = (totalOutputs + blockSize - 1) / blockSize;

    PointersManager manager(context, "deformableConv2d");

    const T* inputBuffer = input->specialBufferasT<T>();
    const T* weightsBuffer = weights->specialBufferasT<T>();
    const T* offsetBuffer = offset->specialBufferasT<T>();
    const T* biasBuffer = bias != nullptr ? bias->specialBufferasT<T>() : nullptr;
    const T* maskBuffer = hasMask ? mask->specialBufferasT<T>() : nullptr;
    T* outputBuffer = output->specialBufferasT<T>();

    deformableConv2dKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        inputBuffer, weightsBuffer, offsetBuffer, biasBuffer, maskBuffer, outputBuffer,
        batchSize, inChannels, inputH, inputW,
        outChannels, outputH, outputW,
        kH, kW, sH, sW, pH, pW, dH, dW,
        groups, offsetGroups,
        inputBatchStride, inputChannelStride, inputHeightStride, inputWidthStride,
        weightOutStride, weightInStride, weightKhStride, weightKwStride,
        offsetBatchStride, offsetChannelStride, offsetHeightStride, offsetWidthStride,
        maskBatchStride, maskChannelStride, maskHeightStride, maskWidthStride,
        outputBatchStride, outputChannelStride, outputHeightStride, outputWidthStride,
        hasMask);

    manager.synchronize();
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

    BUILD_SINGLE_SELECTOR(input->dataType(), deformableConv2dCuda_,
                          (context, input, weights, offset, bias, mask, output,
                           kH, kW, sH, sW, pH, pW, dH, dW, groups, offsetGroups),
                          SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(void deformableConv2dCuda_,
                      (sd::LaunchContext* context, NDArray* input, NDArray* weights,
                       NDArray* offset, NDArray* bias, NDArray* mask, NDArray* output,
                       int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW,
                       int groups, int offsetGroups),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
