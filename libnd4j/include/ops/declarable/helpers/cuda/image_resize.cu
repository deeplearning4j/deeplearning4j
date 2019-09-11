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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/image_resize.h>
#include <cuda_exception.h>

namespace nd4j {
namespace ops {
namespace helpers {

    struct BilinearInterpolationData {
        Nd4jLong bottomIndex;  // Lower source index used in the interpolation
        Nd4jLong topIndex;  // Upper source index used in the interpolation
        // 1-D linear iterpolation scale (see:
        // https://en.wikipedia.org/wiki/Bilinear_interpolation)
        double interpolarValue;
    };
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// computeInterpolationWeights kernel
//      outSize - output length
//      inSize - input size
//      scale - input scale
//      interporationData - result
//
    static __global__ void computeInterpolationWeights(Nd4jLong outSize,
                                              Nd4jLong inSize,
                                              double scale,
                                              Nd4jLong channels,
                                              BilinearInterpolationData* interpolationData) {
        interpolationData[outSize].bottomIndex = 0;
        interpolationData[outSize].topIndex = 0;
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (Nd4jLong i = outSize - tid; i >= 0; i -= step) {
            double in = i * scale;
            interpolationData[i].bottomIndex = static_cast<Nd4jLong>(in);
            interpolationData[i].topIndex = nd4j::math::nd4j_min(interpolationData[i].bottomIndex + 1, inSize - 1);
            interpolationData[i].interpolarValue = in - interpolationData[i].bottomIndex;
            if (channels) {
                math::atomics::nd4j_atomicMul(&interpolationData[i].bottomIndex, channels);
                math::atomics::nd4j_atomicMul(&interpolationData[i].topIndex, channels);
            }
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize image with bilinear interpolation algorithm
//
    static void resizeImage(nd4j::LaunchContext* context, NDArray const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                     Nd4jLong outWidth, Nd4jLong channels,
                     BilinearInterpolationData* xs_,
                     BilinearInterpolationData* ys_,
                     NDArray* output);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize image with bilinear interpolation algorithm kernel
//
    template <typename T>
    static __global__ void resizeImageKernel(T const* input, Nd4jLong const* inputShape, T* outputYptr, Nd4jLong* outputShape, Nd4jLong batchSize,
                                             Nd4jLong outWidth, Nd4jLong outHeight, Nd4jLong channels, Nd4jLong inRowSize, Nd4jLong outRowSize, Nd4jLong inBatchNumValues,
                                             BilinearInterpolationData* xs_, BilinearInterpolationData* ys_) {

        if (blockIdx.x < batchSize) { // blockIdx.x as batch index
            auto pX = input + blockIdx.x * inBatchNumValues;

            auto channelStart = blockIdx.z * blockDim.z + threadIdx.z;
            auto step = blockDim.z * gridDim.z;
            for (Nd4jLong y = threadIdx.x; y < outHeight; y += blockDim.x) {
                const T *ys_input_lower_ptr = pX + ys_[y].bottomIndex * inRowSize;
                const T *ys_input_upper_ptr = pX + ys_[y].topIndex * inRowSize;
                double yVal = ys_[y].interpolarValue;
                auto pZ = outputYptr + y * outRowSize;
                for (Nd4jLong x = threadIdx.y; x < outWidth; x += blockDim.y) {
                    auto xsBottom = xs_[x].bottomIndex;
                    auto xsTop = xs_[x].topIndex;
                    auto xVal = xs_[x].interpolarValue;
                    // process interpolation for all channels
                    for (int c = channelStart; c < channels; c += step) {
                        double topLeft(ys_input_lower_ptr[xsBottom + c]);
                        double topRight(ys_input_lower_ptr[xsTop + c]);
                        double bottomLeft(ys_input_upper_ptr[xsBottom + c]);
                        double bottomRight(ys_input_upper_ptr[xsTop + c]);
                        double top = topLeft + (topRight - topLeft) * xVal;
                        double bottom = bottomLeft + (bottomRight - bottomLeft) * xVal;
                        pZ[x * channels + c] = T(top + (bottom - top) * yVal);
                    }
                }
            }
        }
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize image with
    template <typename T>
    static void resizeImage_(nd4j::LaunchContext* context, NDArray const* images, Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight,
                     Nd4jLong outWidth, Nd4jLong channels,
                     BilinearInterpolationData* xs_,
                     BilinearInterpolationData* ys_,
                     NDArray* output) {
        Nd4jLong inRowSize = inWidth * channels;
        Nd4jLong inBatchNumValues = inHeight * inRowSize;
        Nd4jLong outRowSize = outWidth * channels;
        auto stream = context->getCudaStream();
        T const *input_b_ptr = reinterpret_cast<T const *>(images->getSpecialBuffer()); // this works only with 'c' direction
        T *output_y_ptr = reinterpret_cast<T *>(output->specialBuffer());

        resizeImageKernel<T><<<batchSize, outHeight, 256, *stream>>>(input_b_ptr, images->getSpecialShapeInfo(), output_y_ptr, output->specialShapeInfo(), batchSize,
                outWidth, outHeight, channels, inRowSize, outRowSize, inBatchNumValues, xs_, ys_);
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        template <typename T>
    static int resizeBilinearFunctor_(nd4j::LaunchContext* context, NDArray const* images, int width, int height, bool center, NDArray* output) {
        const Nd4jLong batchSize = images->sizeAt(0);
        const Nd4jLong inHeight = images->sizeAt(1);
        const Nd4jLong inWidth = images->sizeAt(2);
        const Nd4jLong channels = images->sizeAt(3);

        const Nd4jLong outHeight = output->sizeAt(1);
        const Nd4jLong outWidth = output->sizeAt(2);

        // Handle no-op resizes efficiently.
        if (outHeight == inHeight && outWidth == inWidth) {
            output->assign(images);
            return ND4J_STATUS_OK;
        }

        // Special case for TF compatibility
        if((center && inHeight < 2) || (center && inWidth < 2)){
            center = false;
        }

        if ((center && inHeight < 2) || (inHeight < 1) || (outHeight < 1) || (center && outHeight < 2) ||
            (center && inWidth < 2) || (inWidth < 1) || (outWidth < 1) || (center && outWidth < 2)) {
            // wrong input data
            nd4j_printf("image.resize_bilinear: Wrong input or output size to resize\n", "");
            return ND4J_STATUS_BAD_ARGUMENTS;
        }
        float heightScale = center ? (inHeight - 1.f) / double(outHeight - 1.f) : (inHeight / float(outHeight));
        float widthScale = center ? (inWidth - 1.f) / double(outWidth - 1.f) : (inWidth / float(outWidth));

        BilinearInterpolationData* xs_;// = xs.data();
        BilinearInterpolationData* ys_;// = xs.data();

        cudaError_t err = cudaMalloc(&xs_, sizeof(BilinearInterpolationData) * (outWidth + 1));
        if (err != 0) {
            throw cuda_exception::build("helpers::resize_image: Cannot allocate memory for vertical parts rectangulars", err);
        }

        err = cudaMalloc(&ys_, sizeof(BilinearInterpolationData) * (outHeight + 1));
        if (err != 0) {
            throw cuda_exception::build("helpers::resize_image: Cannot allocate memory for horizontal parts rectangulars", err);
        }
        auto stream = context->getCudaStream();
        // Compute the cached interpolation weights on the x and y dimensions.
        computeInterpolationWeights<<<256, 512, 512, *stream>>>(outHeight, inHeight, heightScale, 0, ys_);
        computeInterpolationWeights<<<256, 512, 512, *stream>>>(outWidth, inWidth, widthScale, channels, xs_);

        NDArray::prepareSpecialUse({output}, {images});
        resizeImage(context, images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs_, ys_, output);
        NDArray::registerSpecialUse({output}, {images});

        err = cudaFree(xs_);
        if (err != 0) {
            throw cuda_exception::build("helpers::resize_image: Cannot deallocate memory for vertical parts rectangulars", err);
        }

        err = cudaFree(ys_);
        if (err != 0) {
            throw cuda_exception::build("helpers::resize_image: Cannot deallocate memory for horizontical parts rectangulars", err);
        }

        return Status::OK();
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resize by interpolation nearest neighbor algorithm kernel
//
    template <typename T>
    static __global__ void resizeNeighborKernel(T const* input, Nd4jLong* inputShape, T* output, Nd4jLong* outputShape,
            Nd4jLong batchSize, Nd4jLong inWidth, Nd4jLong inHeight, Nd4jLong outWidth, Nd4jLong outHeight, Nd4jLong channels, double widthScale, double heightScale, bool center) {

        //for (int b = blockIdx.x; b < batchSize; b += gridDim.x)
        if (blockIdx.x < batchSize)
        {
            auto b = blockIdx.x;
            for (int y = threadIdx.x; y < outHeight; y += blockDim.x) {
                Nd4jLong inY = nd4j::math::nd4j_min(
                        (center) ? static_cast<Nd4jLong>(nd4j::math::p_round<float>(y * heightScale)) : static_cast<Nd4jLong>(nd4j::math::p_floor<float>(
                                y * heightScale)), inHeight - 1);
                for (int x = threadIdx.y; x < outWidth; x += blockDim.y) {
                    Nd4jLong inX = nd4j::math::nd4j_min(
                            (center) ? static_cast<Nd4jLong>(nd4j::math::p_round<float>(x * widthScale)) : static_cast<Nd4jLong>(nd4j::math::p_floor<float>(
                                    x * widthScale)), inWidth - 1);
                    auto start = blockIdx.z * blockDim.z + threadIdx.z;
                    auto step = blockDim.z * gridDim.z;

                    for (Nd4jLong e = start; e < channels; e += step) {
                        Nd4jLong posX[] = {b, inY, inX, e};
                        Nd4jLong posZ[] = {b, y, x, e};
                        auto xIndex = shape::getOffset(inputShape, posX);
                        auto zIndex = shape::getOffset(outputShape, posZ);
                        output[zIndex] = input[xIndex];
                    }
                }
            }
        }

    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resizeNeighborFunctor - main algorithm by nearest neighbor
//
    template <typename T>
    int resizeNeighborFunctor_(nd4j::LaunchContext* context, NDArray const* images, int width, int height, bool center, NDArray* output) {
        const Nd4jLong batchSize = images->sizeAt(0);
        const Nd4jLong inHeight = images->sizeAt(1);
        const Nd4jLong inWidth = images->sizeAt(2);
        const Nd4jLong channels = images->sizeAt(3);

        const Nd4jLong outHeight = output->sizeAt(1);
        const Nd4jLong outWidth = output->sizeAt(2);

        // Handle no-op resizes efficiently.
        if (outHeight == inHeight && outWidth == inWidth) {
            output->assign(images);
            return ND4J_STATUS_OK;
        }

        if ((center && inHeight < 2) || (inHeight < 1) || (outHeight < 1) || (center && outHeight < 2) ||
            (center && inWidth < 2) || (inWidth < 1) || (outWidth < 1) || (center && outWidth < 2)) {
            // wrong input data
            nd4j_printf("image.resize_nearest_neighbor: Wrong input or output size to resize\n", "");
            return ND4J_STATUS_BAD_ARGUMENTS;
        }
        double heightScale = center ? (inHeight - 1.) / double(outHeight - 1.0) : (inHeight / double(outHeight));
        double widthScale = center ? (inWidth - 1.) / double(outWidth - 1.0) : (inWidth / double(outWidth));
        auto imagesBuffer = reinterpret_cast<T const*>(images->getSpecialBuffer());
        auto outputBuffer = reinterpret_cast<T*>(output->specialBuffer());
        auto stream = context->getCudaStream();

        //T const* input, Nd4jLong const* inputShape, T* output, Nd4jLong* outputShape,
        //            Nd4jLong batchSize, Nd4jLong inWidth, Nd4jLong inHeight, Nd4jLong outWidth, Nd4jLong outHeight, Nd4jLong channels, double widthScale, double heightScale, bool center
        //input, inputShape, output, outputShape,
        //            batchSize, inWidth, inHeight, outWidth, outHeight, channels, widthScale, heightScale, center
        NDArray::prepareSpecialUse({output}, {images});
        resizeNeighborKernel<T><<<batchSize, outHeight * outWidth, 512, *stream>>>(imagesBuffer, images->getSpecialShapeInfo(), outputBuffer, output->specialShapeInfo(),
                batchSize, inWidth, inHeight, outWidth, outHeight, channels, widthScale, heightScale, center);
        NDArray::registerSpecialUse({output}, {images});

        return Status::OK();
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// resizeImage - resize bilinear algorithm caller
//
    void resizeImage(nd4j::LaunchContext* context, NDArray const* images, Nd4jLong batchSize, Nd4jLong inHeight,
            Nd4jLong inWidth, Nd4jLong outHeight, Nd4jLong outWidth, Nd4jLong channels, BilinearInterpolationData* xs_,
            BilinearInterpolationData* ys_, NDArray* output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), resizeImage_, (context, images, batchSize, inHeight, inWidth, outHeight, outWidth, channels, xs_, ys_, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void resizeImage_,(nd4j::LaunchContext* context, NDArray const* images,
            Nd4jLong batchSize, Nd4jLong inHeight, Nd4jLong inWidth, Nd4jLong outHeight, Nd4jLong outWidth,
            Nd4jLong channels, BilinearInterpolationData* xs_, BilinearInterpolationData* ys_, NDArray* output), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int resizeBilinearFunctor(nd4j::LaunchContext* context, NDArray const* images, int width, int height, bool center, NDArray* output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), return resizeBilinearFunctor_, (context, images, width, height, center, output), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int resizeBilinearFunctor_, (nd4j::LaunchContext* context, NDArray const* images, int width, int height, bool center, NDArray* output), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int resizeNeighborFunctor(nd4j::LaunchContext* context, NDArray const* images, int width, int height, bool center, NDArray* output) {
        BUILD_SINGLE_SELECTOR(images->dataType(), return resizeNeighborFunctor_, (context, images, width, height, center, output), LIBND4J_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int resizeNeighborFunctor_, (nd4j::LaunchContext* context, NDArray const* images,
            int width, int height, bool center, NDArray* output), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // --------------------------------------------------------------------------------------------------------------- //
    // Crop and Resize helper implementation
    // --------------------------------------------------------------------------------------------------------------- //
    // cropAndResize kernel
    //
    template <typename T, typename Z, typename I>
    static __global__ void cropAndResizeKernel(T const *images, Nd4jLong* imagesShape, Z const* boxes, Nd4jLong* boxesShape,
            I const* indices, Nd4jLong* indexShape, I const* cropSize, Nd4jLong* cropShape, int method,
            double extrapolationVal, Z* output, Nd4jLong* outputShape, int numBoxes, int cropHeight, int cropWidth,
            int batchSize, int imageHeight, int imageWidth, int depth) {

        for (int b = blockIdx.x; b < numBoxes; b += gridDim.x)
        {
            Nd4jLong x1Pos[] = {b, 1};
            Nd4jLong y1Pos[] = {b, 0};
            Nd4jLong y2Pos[] = {b, 2};
            Nd4jLong x2Pos[] = {b, 3};
            Z y1 = boxes[shape::getOffset(boxesShape, y1Pos)];//->t<T>(b, 0)];
            Z x1 = boxes[shape::getOffset(boxesShape, x1Pos)];
            Z y2 = boxes[shape::getOffset(boxesShape, y2Pos)];
            Z x2 = boxes[shape::getOffset(boxesShape, x2Pos)];

            int bIn = indices[b];
            if (bIn >= batchSize) {
                continue;
            }

            Z heightScale = (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / Z(cropHeight - 1) : Z(0);
            Z widthScale = (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / Z(cropWidth - 1) : Z(0);

            for (int y = threadIdx.x; y < cropHeight; y += blockDim.x) {
                const float inY = (cropHeight > 1)
                                  ? y1 * (imageHeight - 1) + y * heightScale
                                  : 0.5 * (y1 + y2) * (imageHeight - 1);
                if (inY < 0 || inY > imageHeight - 1) {
                    for (int x = threadIdx.y; x < cropWidth; x += blockDim.y) {
                        auto start = blockIdx.z * blockDim.x + threadIdx.z;
                        auto step = blockDim.z * gridDim.z;
                        for (int d = start; d < depth; d += step) {
                            Nd4jLong zPos[] = {b, y, x, d};
                            auto zIndex = shape::getOffset(outputShape, zPos);
                            output[zIndex] = (Z)extrapolationVal;
                            //crops->p(b, y, x, d, extrapolationVal);
                        }
                    }
                    continue;
                }

                if (method == 0 /* bilinear */) {
                    const int topYIndex = nd4j::math::p_floor(inY);
                    const int bottomYIndex = nd4j::math::p_ceil(inY);
                    const float y_lerp = inY - topYIndex;

                    for (int x = 0; x < cropWidth; ++x) {
                        const float in_x = (cropWidth > 1)
                                           ? x1 * (imageWidth - 1) + x * widthScale
                                           : 0.5 * (x1 + x2) * (imageWidth - 1);
                        if (in_x < 0 || in_x > imageWidth - 1) {
                            auto start = blockIdx.z * blockDim.x + threadIdx.z;
                            auto step = blockDim.z * gridDim.z;
                            for (int d = start; d < depth; d += step) {
                                Nd4jLong zPos[] = {b, y, x, d};
                                auto zIndex = shape::getOffset(outputShape, zPos);
                                output[zIndex] = (Z)extrapolationVal;
//                                crops->p(b, y, x, d, extrapolationVal);
                            }
                            continue;
                        }
                        int left_x_index = math::p_floor(in_x);
                        int right_x_index = math::p_ceil(in_x);
                        T x_lerp = in_x - left_x_index;

                        auto start = blockIdx.z * blockDim.x + threadIdx.z;
                        auto step = blockDim.z * gridDim.z;
                        for (int d = start; d < depth; d += step) {
                            Nd4jLong topLeftPos[] = {bIn, topYIndex, left_x_index, d};
                            Nd4jLong topRightPos[] = {bIn, topYIndex, right_x_index, d};
                            Nd4jLong bottomLeftPos[] = {bIn, bottomYIndex, left_x_index, d};
                            Nd4jLong bottomRightPos[] = {bIn, bottomYIndex, right_x_index, d};
                            const T topLeft(images[shape::getOffset(imagesShape, topLeftPos)]); //->e<float>(bIn, topYIndex, left_x_index, d));
                            const T topRight(images[shape::getOffset(imagesShape, topRightPos)]); //->e<float>(bIn, topYIndex, right_x_index, d));
                            const T bottomLeft(images[shape::getOffset(imagesShape, bottomLeftPos)]);//->e<float>(bIn, bottomYIndex, left_x_index, d));
                            const T bottomRight(images[shape::getOffset(imagesShape, bottomRightPos)]); //->e<float>(bIn, bottomYIndex, right_x_index, d));
                            const T top = topLeft + (topRight - topLeft) * x_lerp;
                            const T bottom = bottomLeft + (bottomRight - bottomLeft) * x_lerp;
                            Nd4jLong zPos[] = {b, y, x, d};
                            auto zIndex = shape::getOffset(outputShape, zPos);
                            output[zIndex] = Z(top + (bottom - top) * y_lerp);
                        }
                    }
                } else {  // method is "nearest neighbor"
                    for (int x = 0; x < cropWidth; ++x) {
                        const float inX = (cropWidth > 1)
                                          ? x1 * (imageWidth - 1) + x * widthScale
                                          : 0.5 * (x1 + x2) * (imageWidth - 1);
                        if (inX < 0 || inX > imageWidth - 1) {
                            auto start = blockIdx.z * blockDim.x + threadIdx.z;
                            auto step = blockDim.z * gridDim.z;
                            for (int d = start; d < depth; d += step) {
                                Nd4jLong zPos[] = {b, y, x, d};
                                auto zIndex = shape::getOffset(outputShape, zPos);
                                output[zIndex] = (Z)extrapolationVal;
                            }
                            continue;
                        }
                        const int closestXIndex = roundf(inX);
                        const int closestYIndex = roundf(inY);
                        auto start = blockIdx.z * blockDim.x + threadIdx.z;
                        auto step = blockDim.z * gridDim.z;
                        for (int d = start; d < depth; d += step) {
                            Nd4jLong zPos[] = {b, y, x, d};
                            Nd4jLong xPos[] = {bIn, closestYIndex, closestXIndex, d};
                            auto zIndex = shape::getOffset(outputShape, zPos);
                            auto xIndex = shape::getOffset(imagesShape, xPos);
                            output[zIndex] = images[xIndex];
                        }
                    }
                }
            }
        }

    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cropAndResizeFunctor main algorithm
//      context - launch context
//      images - batch of images (4D tensor - [batch, width, height, pixels])
//      boxes - 2D tensor with boxes for crop
//      indices - 2D int tensor with indices of boxes to crop
//      cropSize - 2D int tensor with crop box sizes
//      method - (one of 0 - bilinear, 1 - nearest)
//      extrapolationVal - double value of extrapolation
//      crops - output (4D tensor - [batch, outWidth, outHeight, pixels])
//
    template <typename T, typename Z, typename I>
    static void cropAndResizeFunctor_(nd4j::LaunchContext* context, NDArray const *images, NDArray const *boxes, NDArray const *indices,
                                      NDArray const *cropSize, int method, double extrapolationVal, NDArray *crops) {
        const int batchSize = images->sizeAt(0);
        const int imageHeight = images->sizeAt(1);
        const int imageWidth = images->sizeAt(2);

        const int numBoxes = crops->sizeAt(0);
        const int cropHeight = crops->sizeAt(1);
        const int cropWidth = crops->sizeAt(2);
        const int depth = crops->sizeAt(3);
        auto stream = context->getCudaStream();
        T const* imagesBuf = reinterpret_cast<T const*>(images->getSpecialBuffer());
        Z const* boxesBuf = reinterpret_cast<Z const*>(boxes->getSpecialBuffer());
        I const* indexBuf = reinterpret_cast<I const*>(indices->getSpecialBuffer());
        I const* cropSizes = reinterpret_cast<I const*>(cropSize->getSpecialBuffer());
        Z* outBuf = reinterpret_cast<Z*>(crops->specialBuffer());

        NDArray::prepareSpecialUse({crops}, {images, boxes, indices, cropSize});
        cropAndResizeKernel<T,Z,I><<<batchSize, math::nd4j_max(imageHeight * imageWidth, cropHeight * cropWidth), 512, *stream>>>(imagesBuf, images->getSpecialShapeInfo(), boxesBuf, boxes->getSpecialShapeInfo(), indexBuf, indices->getSpecialShapeInfo(),
                cropSizes, cropSize->getSpecialShapeInfo(), method, extrapolationVal, outBuf, crops->specialShapeInfo(), numBoxes, cropHeight, cropWidth, batchSize, imageHeight, imageWidth, depth);
        NDArray::registerSpecialUse({crops}, {images, boxes, indices, cropSize});
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void cropAndResizeFunctor(nd4j::LaunchContext * context, NDArray const *images, NDArray const *boxes, NDArray const *indices, NDArray const *cropSize, int method, double extrapolationVal, NDArray *crops) {
        BUILD_TRIPLE_SELECTOR(images->dataType(), boxes->dataType(), indices->dataType(), cropAndResizeFunctor_,
                              (context, images, boxes, indices, cropSize, method, extrapolationVal, crops), NUMERIC_TYPES, FLOAT_TYPES, INTEGER_TYPES);
        //
    }
    BUILD_TRIPLE_TEMPLATE(template void cropAndResizeFunctor_,
                          (nd4j::LaunchContext * context, NDArray const* images, NDArray const* boxes, NDArray const* indices, NDArray const* cropSize, int method, double extrapolationVal, NDArray* crops),
                          NUMERIC_TYPES, FLOAT_TYPES, INTEGER_TYPES);
}
}
}