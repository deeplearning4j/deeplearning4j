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
// @author Adam Gibson
//
// Metal Performance Shaders - Image operations (resize, transform)
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Bilinear Resize using MPS
//////////////////////////////////////////////////////////////////////////

static void resizeBilinearMPS(const NDArray* input, NDArray* output, int newHeight, int newWidth, bool alignCorners) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        // Input: [N, C, H, W] or [N, H, W, C]
        // Assuming NCHW format
        NSUInteger batch = input->sizeAt(0);
        NSUInteger channels = input->sizeAt(1);
        NSUInteger inHeight = input->sizeAt(2);
        NSUInteger inWidth = input->sizeAt(3);

        // Create image descriptors
        MPSImageDescriptor* inputDesc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:inWidth
                                      height:inHeight
                             featureChannels:channels
                              numberOfImages:batch
                                       usage:MTLTextureUsageShaderRead];

        MPSImageDescriptor* outputDesc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:newWidth
                                      height:newHeight
                             featureChannels:channels
                              numberOfImages:batch
                                       usage:MTLTextureUsageShaderWrite];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:inputDesc];
        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:outputDesc];

        // Copy input data to MPS image
        MTLRegion inRegion = MTLRegionMake3D(0, 0, 0, inWidth, inHeight, 1);
        size_t bytesPerRow = inWidth * sizeof(float);

        for (NSUInteger n = 0; n < batch; n++) {
            for (NSUInteger c = 0; c < channels; c++) {
                const float* srcPtr = input->bufferAsT<float>() +
                                      n * channels * inHeight * inWidth +
                                      c * inHeight * inWidth;
                [inputImage.texture replaceRegion:inRegion
                                      mipmapLevel:0
                                            slice:n * channels + c
                                        withBytes:srcPtr
                                      bytesPerRow:bytesPerRow
                                    bytesPerImage:0];
            }
        }

        // Create bilinear scale kernel
        MPSImageBilinearScale* scaleKernel = [[MPSImageBilinearScale alloc] initWithDevice:device];

        // Create command buffer and execute
        id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();

        // Set up scale transform
        MPSScaleTransform transform;
        transform.scaleX = (double)inWidth / newWidth;
        transform.scaleY = (double)inHeight / newHeight;
        transform.translateX = 0;
        transform.translateY = 0;

        if (alignCorners && newWidth > 1 && newHeight > 1) {
            transform.scaleX = (double)(inWidth - 1) / (newWidth - 1);
            transform.scaleY = (double)(inHeight - 1) / (newHeight - 1);
        }

        [scaleKernel setScaleTransform:&transform];

        [scaleKernel encodeToCommandBuffer:commandBuffer
                               sourceImage:inputImage
                          destinationImage:outputImage];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy output back
        MTLRegion outRegion = MTLRegionMake3D(0, 0, 0, newWidth, newHeight, 1);
        size_t outBytesPerRow = newWidth * sizeof(float);

        for (NSUInteger n = 0; n < batch; n++) {
            for (NSUInteger c = 0; c < channels; c++) {
                float* dstPtr = output->bufferAsT<float>() +
                                n * channels * newHeight * newWidth +
                                c * newHeight * newWidth;
                [outputImage.texture getBytes:dstPtr
                                  bytesPerRow:outBytesPerRow
                                bytesPerImage:0
                                   fromRegion:outRegion
                                  mipmapLevel:0
                                        slice:n * channels + c];
            }
        }
    }
}

PLATFORM_IMPL(resize_bilinear, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int newHeight = output->sizeAt(2);
    int newWidth = output->sizeAt(3);

    bool alignCorners = false;
    if (block.getBArguments()->size() > 0) {
        alignCorners = B_ARG(0);
    }

    resizeBilinearMPS(input, output, newHeight, newWidth, alignCorners);

    return sd::Status::OK;
}

PLATFORM_CHECK(resize_bilinear, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS RESIZE_BILINEAR OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Nearest Neighbor Resize using MPS
//////////////////////////////////////////////////////////////////////////

static void resizeNearestMPS(const NDArray* input, NDArray* output, int newHeight, int newWidth) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        NSUInteger batch = input->sizeAt(0);
        NSUInteger channels = input->sizeAt(1);
        NSUInteger inHeight = input->sizeAt(2);
        NSUInteger inWidth = input->sizeAt(3);

        // Create image descriptors
        MPSImageDescriptor* inputDesc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:inWidth
                                      height:inHeight
                             featureChannels:channels
                              numberOfImages:batch
                                       usage:MTLTextureUsageShaderRead];

        MPSImageDescriptor* outputDesc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:newWidth
                                      height:newHeight
                             featureChannels:channels
                              numberOfImages:batch
                                       usage:MTLTextureUsageShaderWrite];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:inputDesc];
        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:outputDesc];

        // Copy input data
        MTLRegion inRegion = MTLRegionMake3D(0, 0, 0, inWidth, inHeight, 1);
        size_t bytesPerRow = inWidth * sizeof(float);

        for (NSUInteger n = 0; n < batch; n++) {
            for (NSUInteger c = 0; c < channels; c++) {
                const float* srcPtr = input->bufferAsT<float>() +
                                      n * channels * inHeight * inWidth +
                                      c * inHeight * inWidth;
                [inputImage.texture replaceRegion:inRegion
                                      mipmapLevel:0
                                            slice:n * channels + c
                                        withBytes:srcPtr
                                      bytesPerRow:bytesPerRow
                                    bytesPerImage:0];
            }
        }

        // Use MPSImageLanczosScale with nearest neighbor sampling
        // Note: MPS doesn't have a dedicated nearest neighbor kernel,
        // so we implement it manually for exact nearest neighbor behavior
        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();

        float scaleY = (float)inHeight / newHeight;
        float scaleX = (float)inWidth / newWidth;

        for (NSUInteger n = 0; n < batch; n++) {
            for (NSUInteger c = 0; c < channels; c++) {
                for (int y = 0; y < newHeight; y++) {
                    for (int x = 0; x < newWidth; x++) {
                        int srcY = (int)(y * scaleY);
                        int srcX = (int)(x * scaleX);

                        srcY = std::min(srcY, (int)(inHeight - 1));
                        srcX = std::min(srcX, (int)(inWidth - 1));

                        NSUInteger srcIdx = n * channels * inHeight * inWidth +
                                           c * inHeight * inWidth +
                                           srcY * inWidth + srcX;
                        NSUInteger dstIdx = n * channels * newHeight * newWidth +
                                           c * newHeight * newWidth +
                                           y * newWidth + x;

                        outPtr[dstIdx] = inPtr[srcIdx];
                    }
                }
            }
        }
    }
}

PLATFORM_IMPL(resize_nearest, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int newHeight = output->sizeAt(2);
    int newWidth = output->sizeAt(3);

    resizeNearestMPS(input, output, newHeight, newWidth);

    return sd::Status::OK;
}

PLATFORM_CHECK(resize_nearest, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS RESIZE_NEAREST OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Image Crop and Resize
//////////////////////////////////////////////////////////////////////////

static void cropAndResizeMPS(const NDArray* image, const NDArray* boxes, const NDArray* boxIndices,
                              NDArray* output, int cropHeight, int cropWidth, int method) {
    @autoreleasepool {
        // image: [batch, height, width, channels] or [batch, channels, height, width]
        // boxes: [num_boxes, 4] - normalized coordinates (y1, x1, y2, x2)
        // boxIndices: [num_boxes] - batch index for each box
        // output: [num_boxes, crop_height, crop_width, channels]

        NSUInteger numBoxes = boxes->sizeAt(0);
        NSUInteger batch = image->sizeAt(0);
        NSUInteger channels = image->sizeAt(1);
        NSUInteger inHeight = image->sizeAt(2);
        NSUInteger inWidth = image->sizeAt(3);

        const float* imgPtr = image->bufferAsT<float>();
        const float* boxPtr = boxes->bufferAsT<float>();
        const int* idxPtr = boxIndices->bufferAsT<int>();
        float* outPtr = output->bufferAsT<float>();

        for (NSUInteger b = 0; b < numBoxes; b++) {
            int batchIdx = idxPtr[b];

            // Get box coordinates (normalized)
            float y1 = boxPtr[b * 4 + 0];
            float x1 = boxPtr[b * 4 + 1];
            float y2 = boxPtr[b * 4 + 2];
            float x2 = boxPtr[b * 4 + 3];

            // Convert to pixel coordinates
            float startY = y1 * (inHeight - 1);
            float startX = x1 * (inWidth - 1);
            float endY = y2 * (inHeight - 1);
            float endX = x2 * (inWidth - 1);

            float heightScale = (endY - startY) / (cropHeight - 1);
            float widthScale = (endX - startX) / (cropWidth - 1);

            for (NSUInteger c = 0; c < channels; c++) {
                for (int y = 0; y < cropHeight; y++) {
                    for (int x = 0; x < cropWidth; x++) {
                        float srcY = startY + y * heightScale;
                        float srcX = startX + x * widthScale;

                        // Clamp to valid range
                        srcY = std::max(0.0f, std::min(srcY, (float)(inHeight - 1)));
                        srcX = std::max(0.0f, std::min(srcX, (float)(inWidth - 1)));

                        float value;
                        if (method == 0) {
                            // Bilinear interpolation
                            int y0 = (int)srcY;
                            int x0 = (int)srcX;
                            int y1 = std::min(y0 + 1, (int)(inHeight - 1));
                            int x1 = std::min(x0 + 1, (int)(inWidth - 1));

                            float dy = srcY - y0;
                            float dx = srcX - x0;

                            NSUInteger baseIdx = batchIdx * channels * inHeight * inWidth + c * inHeight * inWidth;
                            float v00 = imgPtr[baseIdx + y0 * inWidth + x0];
                            float v01 = imgPtr[baseIdx + y0 * inWidth + x1];
                            float v10 = imgPtr[baseIdx + y1 * inWidth + x0];
                            float v11 = imgPtr[baseIdx + y1 * inWidth + x1];

                            value = (1 - dy) * (1 - dx) * v00 +
                                    (1 - dy) * dx * v01 +
                                    dy * (1 - dx) * v10 +
                                    dy * dx * v11;
                        } else {
                            // Nearest neighbor
                            int nearY = (int)(srcY + 0.5f);
                            int nearX = (int)(srcX + 0.5f);
                            nearY = std::max(0, std::min(nearY, (int)(inHeight - 1)));
                            nearX = std::max(0, std::min(nearX, (int)(inWidth - 1)));

                            NSUInteger srcIdx = batchIdx * channels * inHeight * inWidth +
                                               c * inHeight * inWidth +
                                               nearY * inWidth + nearX;
                            value = imgPtr[srcIdx];
                        }

                        NSUInteger dstIdx = b * channels * cropHeight * cropWidth +
                                           c * cropHeight * cropWidth +
                                           y * cropWidth + x;
                        outPtr[dstIdx] = value;
                    }
                }
            }
        }
    }
}

PLATFORM_IMPL(crop_and_resize, ENGINE_CPU) {
    auto image = INPUT_VARIABLE(0);
    auto boxes = INPUT_VARIABLE(1);
    auto boxIndices = INPUT_VARIABLE(2);
    auto output = OUTPUT_VARIABLE(0);

    if (image->isEmpty()) return sd::Status::OK;

    int cropHeight = output->sizeAt(2);
    int cropWidth = output->sizeAt(3);
    int method = 0;  // 0 = bilinear, 1 = nearest

    if (block.getIArguments()->size() > 0) {
        method = INT_ARG(0);
    }

    cropAndResizeMPS(image, boxes, boxIndices, output, cropHeight, cropWidth, method);

    return sd::Status::OK;
}

PLATFORM_CHECK(crop_and_resize, ENGINE_CPU) {
    auto image = INPUT_VARIABLE(0);

    Requirements req("MPS CROP_AND_RESIZE OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(image->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectEq(makeInfoVariable(image->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectTrue(mpsUtils::isContiguous(*image), "Image must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Image Transpose (NHWC <-> NCHW)
//////////////////////////////////////////////////////////////////////////

static void transposeImageMPS(const NDArray* input, NDArray* output, bool toNCHW) {
    @autoreleasepool {
        if (toNCHW) {
            // NHWC -> NCHW: [N, H, W, C] -> [N, C, H, W]
            NSUInteger batch = input->sizeAt(0);
            NSUInteger height = input->sizeAt(1);
            NSUInteger width = input->sizeAt(2);
            NSUInteger channels = input->sizeAt(3);

            const float* inPtr = input->bufferAsT<float>();
            float* outPtr = output->bufferAsT<float>();

            for (NSUInteger n = 0; n < batch; n++) {
                for (NSUInteger h = 0; h < height; h++) {
                    for (NSUInteger w = 0; w < width; w++) {
                        for (NSUInteger c = 0; c < channels; c++) {
                            NSUInteger srcIdx = n * height * width * channels +
                                               h * width * channels +
                                               w * channels + c;
                            NSUInteger dstIdx = n * channels * height * width +
                                               c * height * width +
                                               h * width + w;
                            outPtr[dstIdx] = inPtr[srcIdx];
                        }
                    }
                }
            }
        } else {
            // NCHW -> NHWC: [N, C, H, W] -> [N, H, W, C]
            NSUInteger batch = input->sizeAt(0);
            NSUInteger channels = input->sizeAt(1);
            NSUInteger height = input->sizeAt(2);
            NSUInteger width = input->sizeAt(3);

            const float* inPtr = input->bufferAsT<float>();
            float* outPtr = output->bufferAsT<float>();

            for (NSUInteger n = 0; n < batch; n++) {
                for (NSUInteger c = 0; c < channels; c++) {
                    for (NSUInteger h = 0; h < height; h++) {
                        for (NSUInteger w = 0; w < width; w++) {
                            NSUInteger srcIdx = n * channels * height * width +
                                               c * height * width +
                                               h * width + w;
                            NSUInteger dstIdx = n * height * width * channels +
                                               h * width * channels +
                                               w * channels + c;
                            outPtr[dstIdx] = inPtr[srcIdx];
                        }
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
// Depthwise Convolution 2D
//////////////////////////////////////////////////////////////////////////

static void depthwiseConv2dMPS(const NDArray* input, const NDArray* weights, const NDArray* bias,
                                NDArray* output, int kH, int kW, int sH, int sW,
                                int pH, int pW, int dH, int dW) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        // Input: [N, C, H, W]
        // Weights: [C, 1, kH, kW] for depthwise
        NSUInteger batch = input->sizeAt(0);
        NSUInteger channels = input->sizeAt(1);
        NSUInteger inHeight = input->sizeAt(2);
        NSUInteger inWidth = input->sizeAt(3);

        NSUInteger outHeight = (inHeight + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        NSUInteger outWidth = (inWidth + 2 * pW - dW * (kW - 1) - 1) / sW + 1;

        // CPU fallback for depthwise convolution
        const float* inPtr = input->bufferAsT<float>();
        const float* wPtr = weights->bufferAsT<float>();
        const float* bPtr = bias ? bias->bufferAsT<float>() : nullptr;
        float* outPtr = output->bufferAsT<float>();

        output->nullify();

        for (NSUInteger n = 0; n < batch; n++) {
            for (NSUInteger c = 0; c < channels; c++) {
                for (NSUInteger oh = 0; oh < outHeight; oh++) {
                    for (NSUInteger ow = 0; ow < outWidth; ow++) {
                        float sum = 0.0f;

                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = (int)(oh * sH) - pH + kh * dH;
                                int iw = (int)(ow * sW) - pW + kw * dW;

                                if (ih >= 0 && ih < (int)inHeight && iw >= 0 && iw < (int)inWidth) {
                                    NSUInteger inIdx = n * channels * inHeight * inWidth +
                                                      c * inHeight * inWidth +
                                                      ih * inWidth + iw;
                                    NSUInteger wIdx = c * kH * kW + kh * kW + kw;
                                    sum += inPtr[inIdx] * wPtr[wIdx];
                                }
                            }
                        }

                        if (bPtr) {
                            sum += bPtr[c];
                        }

                        NSUInteger outIdx = n * channels * outHeight * outWidth +
                                           c * outHeight * outWidth +
                                           oh * outWidth + ow;
                        outPtr[outIdx] = sum;
                    }
                }
            }
        }
    }
}

PLATFORM_IMPL(depthwise_conv2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    int kH = INT_ARG(0);
    int kW = INT_ARG(1);
    int sH = INT_ARG(2);
    int sW = INT_ARG(3);
    int pH = INT_ARG(4);
    int pW = INT_ARG(5);
    int dH = INT_ARG(6);
    int dW = INT_ARG(7);

    depthwiseConv2dMPS(input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW);

    return sd::Status::OK;
}

PLATFORM_CHECK(depthwise_conv2d, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);

    Requirements req("MPS DEPTHWISE_CONV2D OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
