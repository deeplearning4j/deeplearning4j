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

        for (auto batch = blockIdx.x; batch < batchSize; batch += gridDim.x ) { // blockIdx.x as batch index
            auto pX = input + batch * inBatchNumValues;
            for (Nd4jLong y = threadIdx.x; y < outHeight; y += blockDim.x) {
                const T *ys_input_lower_ptr = pX + ys_[y].bottomIndex * inRowSize;
                const T *ys_input_upper_ptr = pX + ys_[y].topIndex * inRowSize;
                double yVal = ys_[y].interpolarValue;
                auto pZ = outputYptr + (batch * outHeight + y) * outRowSize;
                for (Nd4jLong x = threadIdx.y; x < outWidth; x += blockDim.y) {
                    auto xsBottom = xs_[x].bottomIndex;
                    auto xsTop = xs_[x].topIndex;
                    auto xVal = xs_[x].interpolarValue;
                    // process interpolation for all channels
                    for (int c = threadIdx.z; c < channels; c += blockDim.z) {
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
        dim3 batchSizeBlock(batchSize, 1, 1);
        dim3 pictureBlock(outHeight, outWidth, channels);
        resizeImageKernel<T><<<256, pictureBlock, 256, *stream>>>(input_b_ptr, images->getSpecialShapeInfo(), output_y_ptr, output->specialShapeInfo(), batchSize,
                outWidth, outHeight, channels, inRowSize, outRowSize, inBatchNumValues, xs_, ys_);

        auto err = cudaStreamSynchronize(*stream);
        if (err != 0) {
            throw cuda_exception::build("helpers::resizeImage_: Cannot synchronize kernel execution", err);
        }
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
// Bicubic interpolation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Utility functions and classes

    // calculateResizeScale determines the float scaling factor.
    inline float calculateResizeScale(Nd4jLong inSize, Nd4jLong outSize,
                                      bool alignCorners) {
        return (alignCorners && outSize > 1)
               ? (inSize - 1) / static_cast<float>(outSize - 1)
               : inSize / static_cast<float>(outSize);
    }

    struct ImageResizerState {
        explicit ImageResizerState(bool alignCorners, bool halfPixelCenters)
                : _alignCorners(alignCorners),
                  _halfPixelCenters(halfPixelCenters) {}

        // ValidateAndCalculateOutputSize checks the bounds on the input tensors
        // and requested size, sets up some of the resizing state such as the
        // heightScale and widthScale, and calculates the output size.
        // If any of these operations fails, it sets an error status in
        // the context, which the caller must check.
        int validateAndCalculateOutputSize(NDArray const* input, int const width, int const height) {
            //
            batchSize = input->sizeAt(0);//.dim_size(0);
            outHeight = height;
            outWidth = width; //internal::SubtleMustCopy(Svec(1));
            inHeight = static_cast<int32_t>(input->sizeAt(1));
            inWidth = static_cast<int32_t>(input->sizeAt(2));
            channels = input->sizeAt(3); //.dim_size(3);
            heightScale = calculateResizeScale(inHeight, outHeight, _alignCorners);
            widthScale = calculateResizeScale(inWidth, outWidth, _alignCorners);

            // Guard against overflows
            if (ceilf((outHeight - 1) * heightScale) > static_cast<float>(DataTypeUtils::max<int>())) {
                nd4j_printf("resize_bicubic: Upper overflow occurs for resize height (%f)\n", ceilf((outHeight - 1) * heightScale));
                return Status::CODE(ND4J_STATUS_BAD_INPUT, "resize_bicubic: Upper overflow occurs for resize height");
            }
            if (ceilf((outWidth - 1) * heightScale) > static_cast<float>(DataTypeUtils::max<int>())) {
                nd4j_printf("resize_bicubic: Upper overflow occurs for resize height (%f)\n", ceilf((outHeight - 1) * heightScale));
                return Status::CODE(ND4J_STATUS_BAD_INPUT, "resize_bicubic: Upper overflow occurs for resize width");
            }

            return Status::OK();
        }

        // Calculates all the required variables, and allocates the output.
        int validateAndCreateOutput(NDArray const* input, int const width, int const height) {
            return validateAndCalculateOutputSize(input, width, height);
        }

        Nd4jLong batchSize;
        Nd4jLong outHeight;
        Nd4jLong outWidth;
        Nd4jLong inHeight;
        Nd4jLong inWidth;
        Nd4jLong channels;
        float heightScale;
        float widthScale;
        NDArray* output = nullptr;
        cudaStream_t* stream;
    private:
        bool _alignCorners;
        bool _halfPixelCenters;
    };

    // Half pixel scaler scales assuming that the pixel centers are at 0.5, i.e. the
// floating point coordinates of the top,left pixel is 0.5,0.5.
    struct HalfPixelScaler {
        _CUDA_HD HalfPixelScaler(){};
        inline _CUDA_HD float operator()(const int x, const float scale) const {
            // Note that we subtract 0.5 from the return value, as the existing bilinear
            // sampling code etc assumes pixels are in the old coordinate system.
            return (static_cast<float>(x) + 0.5f) * scale - 0.5f;
        }
    };

    struct WeightsAndIndices {
        float _weight0;
        float _weight1;
        float _weight2;
        float _weight3;
        Nd4jLong _index0;
        Nd4jLong _index1;
        Nd4jLong _index2;
        Nd4jLong _index3;

        int _advance;  // advance value.
    };

    class CachedInterpolationCalculator {
    public:
        _CUDA_HD CachedInterpolationCalculator() : _indexes{-1, -1, -1, -1} {}

        // Advances iteration. Returns the number of values that should be copied from
        // the current point to the next point. The copying should always be done by
        // copying the last <retval> values from the old point to the first <retval>
        // values of the new point.
        inline _CUDA_HD int Advance(const Nd4jLong x0, const Nd4jLong x1, const Nd4jLong x2,
                           const Nd4jLong x3) {
            // We use 2 hands and walk through, copying from one to another where
            // we already have values.
            // Invariant, new_indicies_hand <= cached_values_hand
            const Nd4jLong new_x_indices[4] = {x0, x1, x2, x3};
            int cachedValuesHand = 0;
            int newIndiciesHand = 0;
            while (cachedValuesHand < 4) {
                if (_indexes[cachedValuesHand] == new_x_indices[newIndiciesHand]) {
                    if (newIndiciesHand < cachedValuesHand) {
                        _indexes[newIndiciesHand] = _indexes[cachedValuesHand];
                    }
                    newIndiciesHand++;
                }
                cachedValuesHand++;
            }
            switch (newIndiciesHand) {
                case 0:
                    _indexes[0] = x0;
                case 1:
                    _indexes[1] = x1;
                case 2:
                    _indexes[2] = x2;
                case 3:
                    _indexes[3] = x3;
                    break;
            }
            return newIndiciesHand;
        }

    private:
        Nd4jLong _indexes[4];
    };


    static __global__ void initCoefTableKernel(const double a, float* table, Nd4jLong tableSize) {
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;
        for (int i = start; i <= tableSize; i += step) {
            float x = i * 1.0 / tableSize;
            table[i * 2] = ((a + 2) * x - (a + 3)) * x * x + 1;
            x += 1.0;
            table[i * 2 + 1] = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
        }
    }

    static const Nd4jLong kTableSize = (1 << 10);
    float* initCoeffsTable(const double a, cudaStream_t* stream) {
        // Allocate and initialize coefficients table using Bicubic
        // convolution algorithm.
        // https://en.wikipedia.org/wiki/Bicubic_interpolation
        float* coeffs_table; // = new float[(kTableSize + 1) * 2];
        auto err = cudaMalloc(&coeffs_table, sizeof(float) * ((kTableSize + 1) * 2));
        if (err != 0) {
            throw cuda_exception::build("helpers::initCoeffsTable: Cannot allocate memory for vertical parts rectangulars", err);
        }


        initCoefTableKernel<<<128,128,128, *stream>>>(a, coeffs_table, kTableSize);
        err = cudaStreamSynchronize(*stream);
        if (err != 0) {
            throw cuda_exception::build("helpers::initCoeffsTable: Cannot syncronize kernel", err);
        }

            return coeffs_table;
    }
//    _CUDA_HD const  float* getCoeffsTable(const bool use_keys_cubic) {
//            // Static so that we initialize it on first use
//            if (use_keys_cubic) {
//                // http://ieeexplore.ieee.org/document/1163711/
//                // R. G. Keys. Cubic convolution interpolation for digital image
//                // processing. IEEE Transactions on Acoustics, Speech, and Signal
//                // Processing, 29(6):1153–1160, 1981.
//                //static const float* coeffs_table = initCoeffsTable(-0.5f, stream);
//                return sCoeffsTableHalf;
//            } else {
//                //static const float* coeffs_table = initCoeffsTable(-0.75f, stream);
//                return sCoeffsTableThreeFourth;
//            }
//        }

    inline _CUDA_HD Nd4jLong bound(Nd4jLong val, Nd4jLong limit) {
        return math::nd4j_min(limit - 1ll, math::nd4j_max(Nd4jLong{0}, val));
    }


    template <typename T>
    inline _CUDA_HD float interpolate1D(const float weight0, const float weight1, const float weight2, const float weight3,
                               const T value0, const T value1, const T value2, const T value3) {
        return static_cast<float>(value0) * weight0 +
               static_cast<float>(value1) * weight1 +
               static_cast<float>(value2) * weight2 +
               static_cast<float>(value3) * weight3;
    }

// Compute the 1D interpolation for a given X index using the y_weights
    static _CUDA_HD float compute(float values[4], const float xW0, const float xW1, const float xW2, const float xW3) {
        return interpolate1D(xW0, xW1, xW2, xW3, values[0], values[1],values[2], values[3]);
    }



    template <typename Scaler, bool use_keys_cubic>
    inline _CUDA_HD void getWeightsAndIndices(float const* coeffs_table, const float scale, const Nd4jLong out_loc, const Nd4jLong limit, WeightsAndIndices* out) {
        const Scaler scaler;
        const float in_loc_f = scaler(out_loc, scale);
        const Nd4jLong in_loc = math::nd4j_floor<float, Nd4jLong>(in_loc_f);
        const float delta = in_loc_f - in_loc;
        const Nd4jLong offset = math::nd4j_round<float, Nd4jLong>(delta * kTableSize);
        //const float* coeffs_table = getCoeffsTable(use_keys_cubic);
        if (use_keys_cubic) {
            // The legacy code placed more weight on the edge pixels, since bounding
            // the set of inputs to sample could cause an edge pixel to be repeated.
            // Here we change the behavior at borders to match that used by the
            // scale_and_translate_op, where sampling locations outside the image have
            // their weight set to 0, and the weights are renormalized so that their sum
            // is 1.0.
            out->_index0 = bound(in_loc - 1, limit);
            out->_weight0 =
                    (out->_index0 == in_loc - 1 ? coeffs_table[offset * 2 + 1] : 0.0f);
            out->_index1 = bound(in_loc, limit);
            out->_weight1 = (out->_index1 == in_loc ? coeffs_table[offset * 2] : 0.0f);
            out->_index2 = bound(in_loc + 1, limit);
            out->_weight2 =
                    (out->_index2 == in_loc + 1 ? coeffs_table[(kTableSize - offset) * 2]
                                                : 0.0f);
            out->_index3 = bound(in_loc + 2, limit);
            out->_weight3 = (out->_index3 == in_loc + 2
                             ? coeffs_table[(kTableSize - offset) * 2 + 1]
                             : 0.0f);

            const float weight_sum =
                    out->_weight0 + out->_weight1 + out->_weight2 + out->_weight3;
            if (math::nd4j_abs(weight_sum) >= 1000.0f * DataTypeUtils::min<float>()) {
                const float one_over_weight_sum = 1.0f / weight_sum;
                out->_weight0 *= one_over_weight_sum;
                out->_weight1 *= one_over_weight_sum;
                out->_weight2 *= one_over_weight_sum;
                out->_weight3 *= one_over_weight_sum;
            }
        } else {
            out->_weight0 = coeffs_table[offset * 2 + 1];
            out->_weight1 = coeffs_table[offset * 2];
            out->_weight2 = coeffs_table[(kTableSize - offset) * 2];
            out->_weight3 = coeffs_table[(kTableSize - offset) * 2 + 1];
            out->_index0 = bound(in_loc - 1, limit);
            out->_index1 = bound(in_loc, limit);
            out->_index2 = bound(in_loc + 1, limit);
            out->_index3 = bound(in_loc + 2, limit);
        }
    }

// Older incorrect scaling method that causes all resizes to have a slight
// translation leading to inconsistent results. For example, a flip then a
// resize gives different results then a resize then a flip.
    struct LegacyScaler {
        _CUDA_HD LegacyScaler(){};
        inline _CUDA_HD float operator()(const int x, const float scale) const {
            return static_cast<float>(x) * scale;
        }
    };

    static __global__ void accumulateChannelsKernel(WeightsAndIndices* pXWais, Nd4jLong outWidth, Nd4jLong channels) {
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (auto x = start; x < outWidth; x += step) {
            pXWais[x]._index0 *= channels;
            pXWais[x]._index1 *= channels;
            pXWais[x]._index2 *= channels;
            pXWais[x]._index3 *= channels;
        }
    }

    static __global__ void advaceWeightsAndIndicesKernel(float const* cacheTable, CachedInterpolationCalculator* calc, WeightsAndIndices* pXWais, Nd4jLong inWidth, float widthScale,
            Nd4jLong outWidth, Nd4jLong channels, bool halfPixelCenters) {
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (auto x = start; x < outWidth; x += step) {
            if (halfPixelCenters)
                getWeightsAndIndices<HalfPixelScaler, true>(cacheTable, widthScale, x, inWidth, &pXWais[x]);
            else
                getWeightsAndIndices<LegacyScaler, false>(cacheTable, widthScale, x, inWidth, &pXWais[x]);
            pXWais[x]._advance = calc->Advance(pXWais[x]._index0, pXWais[x]._index1, pXWais[x]._index2, pXWais[x]._index3);
        }
    }
    // resizerState and xWais are device allocated
    static void computeXWeightsAndIndices(float const* coeffsTable, const ImageResizerState& resizerState,
                                          const bool halfPixelCenters,
                                          WeightsAndIndices* pXWais) {

        auto stream = resizerState.stream;
        auto outWidth = resizerState.outWidth;
        CachedInterpolationCalculator calc; // = new CachedInterpolationCalculator;
        CachedInterpolationCalculator* pCalcD;
        auto err = cudaMalloc(&pCalcD, sizeof(CachedInterpolationCalculator));
        if (err != 0) {
            cuda_exception::build("helpers::computeXWeightsAndIndices: Cannot allocated device memory for interpolate calculator", err);
        }
        err = cudaMemcpy(pCalcD, &calc, sizeof(CachedInterpolationCalculator), cudaMemcpyHostToDevice);
        if (err != 0) {
            cuda_exception::build("helpers::computeXWeightsAndIndices: Cannot set up device memory for interpolate calculator", err);
        }

        advaceWeightsAndIndicesKernel<<<128, 128, 128, *stream>>>(coeffsTable, pCalcD, pXWais, resizerState.inWidth, resizerState.widthScale, outWidth, resizerState.channels, halfPixelCenters);
        err = cudaFree(pCalcD);
        if (err != 0) {
            cuda_exception::build("helpers::computeXWeightsAndIndices: Cannot deallocated device memory for interpolate calculator", err);
        }
        err = cudaStreamSynchronize(*stream);
        if (err != 0) {
            cuda_exception::build("helpers::computeXWeightsAndIndices: Cannot synchronize stream after advance weights and indicers", err);
        }
        // Scale the values so they can be used as offsets into buffers.
        accumulateChannelsKernel<<<128, 128, 512, *stream>>>(pXWais, outWidth, resizerState.channels);
        err = cudaStreamSynchronize(*stream);
        if (err != 0) {
            cuda_exception::build("helpers::computeXWeightsAndIndices: Cannot synchronize stream after accumulate channels", err);
        }

    }

    template <typename T>
    static _CUDA_HD FORCEINLINE float computeYInterpolation(
            int which, int channelNum, const WeightsAndIndices& yWai,
            const T* pY0, const T* pY1, const T* pY2, const T* pY3,
            const WeightsAndIndices& xWai) {
        int xIndex;
        switch (which) {
            case 0:
                xIndex = xWai._index0;
                break;
            case 1:
                xIndex = xWai._index1;
                break;
            case 2:
                xIndex = xWai._index2;
                break;
            default:
                xIndex = xWai._index3;
                break;
        }
        const Nd4jLong pt_index = xIndex + channelNum;
        return interpolate1D<T>(yWai._weight0, yWai._weight1, yWai._weight2,
                                yWai._weight3, pY0[pt_index], pY1[pt_index],
                                pY2[pt_index], pY3[pt_index]);
    }

    template <typename T>
    static __global__ void bicubicInterpolateWithCachingKernel(float const* cachedTable, float* cachedValue, T const* inputPtr, ImageResizerState* pResizerState, WeightsAndIndices* xWais, bool halfPixelCenters, Nd4jLong inBatchWidth, Nd4jLong inRowWidth, T* outputPtr) {
//        auto numChannels = pResizerState->channels;
        for (Nd4jLong b = blockIdx.x; b < pResizerState->batchSize; b += gridDim.x) {
            auto pInput = inputPtr + b * inBatchWidth;
            for (Nd4jLong y = threadIdx.x; y < pResizerState->outHeight; y += blockDim.x) {
                auto pos = (b * pResizerState->outHeight + y) * pResizerState->outWidth * pResizerState->channels;
                auto pOutput = &outputPtr[pos];
                struct WeightsAndIndices yWai;
                if (halfPixelCenters) {
                    getWeightsAndIndices<HalfPixelScaler, true>(cachedTable, pResizerState->heightScale, y, pResizerState->inHeight, &yWai);
                } else {
                    getWeightsAndIndices<LegacyScaler, false>(cachedTable, pResizerState->heightScale, y, pResizerState->inHeight, &yWai);
                }
                // Make pointers represent offsets of data in inputBPtr.
                const T* y_ptr_0 = pInput + yWai._index0 * inRowWidth;
                const T* y_ptr_1 = pInput + yWai._index1 * inRowWidth;
                const T* y_ptr_2 = pInput + yWai._index2 * inRowWidth;
                const T* y_ptr_3 = pInput + yWai._index3 * inRowWidth;

                if (pResizerState->channels == 3) {
                    // Manually unroll case of 3 channels.
                    float cached_value_0[4] = {0};
                    float cached_value_1[4] = {0};
                    float cached_value_2[4] = {0};
                    for (Nd4jLong x = 0; x < pResizerState->outWidth; ++x) {
                        const WeightsAndIndices& xWai = xWais[x];
                        // Shift values in cached_value_* to fill first '_advance' values.
                        switch (xWai._advance) {
                            case 3:
                                cached_value_0[0] = cached_value_0[1];
                                cached_value_0[1] = cached_value_0[2];
                                cached_value_0[2] = cached_value_0[3];
                                cached_value_1[0] = cached_value_1[1];
                                cached_value_1[1] = cached_value_1[2];
                                cached_value_1[2] = cached_value_1[3];
                                cached_value_2[0] = cached_value_2[1];
                                cached_value_2[1] = cached_value_2[2];
                                cached_value_2[2] = cached_value_2[3];
                                break;
                            case 2:
                                cached_value_0[0] = cached_value_0[2];
                                cached_value_0[1] = cached_value_0[3];
                                cached_value_1[0] = cached_value_1[2];
                                cached_value_1[1] = cached_value_1[3];
                                cached_value_2[0] = cached_value_2[2];
                                cached_value_2[1] = cached_value_2[3];
                                break;
                            case 1: {
                                cached_value_0[0] = cached_value_0[3];
                                cached_value_1[0] = cached_value_1[3];
                                cached_value_2[0] = cached_value_2[3];
                                break;
                            }
                        }

                        // Set the remaining '4-_advance' values by computing.
                        switch (xWai._advance) {
                            case 0:
                                cached_value_0[0] = computeYInterpolation(0, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                cached_value_1[0] = computeYInterpolation(0, 1, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                cached_value_2[0] = computeYInterpolation(0, 2, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                            case 1:
                                cached_value_0[1] = computeYInterpolation(1, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                cached_value_1[1] = computeYInterpolation(1, 1, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                cached_value_2[1] = computeYInterpolation(1, 2, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                            case 2:
                                cached_value_0[2] = computeYInterpolation(2, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                cached_value_1[2] = computeYInterpolation(2, 1, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                cached_value_2[2] = computeYInterpolation(2, 2, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                            case 3:
                                cached_value_0[3] = computeYInterpolation(3, 0, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                cached_value_1[3] = computeYInterpolation(3, 1, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                cached_value_2[3] = computeYInterpolation(3, 2, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                        //        break;
                        }
                        pOutput[x * pResizerState->channels + 0] = compute(cached_value_0, xWai._weight0, xWai._weight1,
                                        xWai._weight2, xWai._weight3);
                        pOutput[x * pResizerState->channels + 1] = compute(cached_value_1, xWai._weight0, xWai._weight1,
                                        xWai._weight2, xWai._weight3);
                        pOutput[x * pResizerState->channels + 2] = compute(cached_value_2, xWai._weight0, xWai._weight1,
                                        xWai._weight2, xWai._weight3);
                    }
                } else {
                    for (Nd4jLong x = 0; x < pResizerState->outWidth; ++x) {
                        const WeightsAndIndices& xWai = xWais[x];
                        // Shift values in cachedValue to fill first '_advance' values.
                        switch (xWai._advance) {
                            case 3:
                                for (Nd4jLong c = 0; c < pResizerState->channels; ++c) {
                                    cachedValue[4 * c + 0] = cachedValue[4 * c + 1];
                                    cachedValue[4 * c + 1] = cachedValue[4 * c + 2];
                                    cachedValue[4 * c + 2] = cachedValue[4 * c + 3];
                                }
                                break;
                            case 2:
                                for (Nd4jLong c = 0; c < pResizerState->channels; ++c) {
                                    cachedValue[4 * c + 0] = cachedValue[4 * c + 2];
                                    cachedValue[4 * c + 1] = cachedValue[4 * c + 3];
                                }
                                break;
                            case 1: {
                                for (Nd4jLong c = 0; c < pResizerState->channels; ++c) {
                                    cachedValue[4 * c + 0] = cachedValue[4 * c + 3];
                                }
                                break;
                            }
                        }

                        // Set the remaining '4-_advance' values by computing.
                        switch (xWai._advance) {
                            case 0:
                                for (Nd4jLong c = 0; c < pResizerState->channels; ++c) {
                                    cachedValue[4 * c + 0] = computeYInterpolation(0, c, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                }
                            case 1:
                                for (Nd4jLong c = 0; c < pResizerState->channels; ++c) {
                                    cachedValue[4 * c + 1] = computeYInterpolation(1, c, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                }
                            case 2:
                                for (Nd4jLong c = 0; c < pResizerState->channels; ++c) {
                                    cachedValue[4 * c + 2] = computeYInterpolation(2, c, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                }
                            case 3:
                                for (Nd4jLong c = 0; c < pResizerState->channels; ++c) {
                                    cachedValue[4 * c + 3] = computeYInterpolation(3, c, yWai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, xWai);
                                }
                               // break;
                        }
                        for (Nd4jLong c = 0; c < pResizerState->channels; ++c) {
                            pOutput[x * pResizerState->channels + c] = compute(&cachedValue[4 * c], xWai._weight0, xWai._weight1, xWai._weight2, xWai._weight3);
                        }
                    }
                }
            }
        }

    }


    template <typename T>
    static void
    bicubicInterpolateWithCaching(NDArray const* image, ImageResizerState const& resizerState, bool const halfPixelCenters, NDArray* output) {
        const auto numChannels = resizerState.channels;
        const Nd4jLong inRowWidth = resizerState.inWidth * numChannels;
        const Nd4jLong inBatchWidth = resizerState.inHeight * inRowWidth;

        auto stream = resizerState.stream; //output->getContext()->getCudaStream();
        ImageResizerState* resizerStateD;
        auto err = cudaMalloc(&resizerStateD, sizeof(ImageResizerState));
        if (err != 0) {
            throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Cannot allocate memory for resizerState", err);
        }
        err = cudaMemcpy(resizerStateD, &resizerState, sizeof(ImageResizerState), cudaMemcpyHostToDevice);
        if (err != 0) {
            throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Cannot set up memory for resizerState", err);
        }

        float* cachedValue = nullptr;
        size_t cachedSize = sizeof(float) * (numChannels == 3 ? 0 : 4 * numChannels);
        if (cachedSize) {
            err = cudaMalloc(reinterpret_cast<void**>(&cachedValue), cachedSize);
            if (err != 0) {
                throw cuda_exception::build(
                        "helpers::bicubicInterpolateWithCaching: Cannot allocate memory for cached values", err);
            }
            err = cudaMemset(cachedValue, 0, cachedSize);
            if (err != 0) {
                throw cuda_exception::build(
                        "helpers::bicubicInterpolateWithCaching: Cannot set up memory for cached values", err);
            }
        }

        WeightsAndIndices* xWais; //(resizerState.outWidth);
        err = cudaMalloc(&xWais, sizeof(WeightsAndIndices) * resizerState.outWidth);
        if (err != 0) {
            throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Cannot allocate memory for weights and indices", err);
        }

        auto coeffsTable = halfPixelCenters?initCoeffsTable(-0.5, stream): initCoeffsTable(-0.75, stream);
        if (err != 0) {
            throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: computeXWeigtsAndInidces finished with error", err);
        }
        computeXWeightsAndIndices(coeffsTable, resizerState, halfPixelCenters, xWais);
        err = cudaStreamQuery(*stream);
        if (err != 0) {
            throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: computeXWeigtsAndInidces finished with error", err);
        }
        const T* pInput = image->getDataBuffer()->specialAsT<T>();
        T* pOutput = output->dataBuffer()->specialAsT<T>(); //_data.data();
        bicubicInterpolateWithCachingKernel<T><<<128, 1, 512, *stream>>>(coeffsTable, cachedValue, pInput,
                resizerStateD, xWais, halfPixelCenters, inBatchWidth, inRowWidth, pOutput);
        err = cudaStreamSynchronize(*stream);
        if (err != 0) {
            throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Kernels finished with error", err);
        }

        err = cudaFree(resizerStateD);
        if (err != 0) {
            throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Cannot deallocate memory for resizerState", err);
        }
        if (cachedSize)
        err = cudaFree(cachedValue);
        if (err != 0) {
            throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Cannot deallocate memory for cached values", err);
        }

        err = cudaFree(xWais);
        if (err != 0) {
            throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Cannot deallocate memory for weights and indices", err);
        }

        err = cudaFree(coeffsTable);
        if (err != 0) {
            throw cuda_exception::build("helpers::bicubicInterpolateWithCaching: Cannot deallocate memory for coefficients table", err);
        }

    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    int resizeBicubicFunctor_(nd4j::LaunchContext * context, NDArray const* image, int width, int height,
                              bool preserveAspectRatio, bool antialias, NDArray* output) {
        return Status::OK();
    }

    int resizeBicubicFunctor(nd4j::LaunchContext * context, NDArray const* image, int width, int height,
                             bool preserveAspectRatio, bool antialias, NDArray* output) {
        BUILD_SINGLE_SELECTOR(image->dataType(), return resizeBicubicFunctor_, (context, image,
                width, height, preserveAspectRatio, antialias, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int resizeBicubicFunctor_, (nd4j::LaunchContext * context, NDArray const* image, int width, int height,
            bool preserveAspectRatio, bool antialias, NDArray* output), NUMERIC_TYPES);
// ------------------------------------------------------------------------------------------------------------------ //
// ------------------------------------------------------------------------------------------------------------------ //
// simplified bicubic resize without antialiasing
//
    template <typename T>
    int resizeBicubicFunctorA_(nd4j::LaunchContext * context, NDArray const* image, int width, int height,
                               bool const alignCorners, bool const halfPixelCenters, NDArray* output) {

            ImageResizerState st(alignCorners, halfPixelCenters); // align_corners, half_pixel_align
            st.stream = context->getCudaStream();
            NDArray::prepareSpecialUse({output}, {image});
            int res = st.validateAndCreateOutput(image, width, height);
            if (res == Status::OK())
                bicubicInterpolateWithCaching<T>(image, st, halfPixelCenters, output);
            NDArray::registerSpecialUse({output}, {image});
            return res;
    }

    int resizeBicubicFunctorA(nd4j::LaunchContext * context, NDArray const* image, int width, int height,
                              bool const alignCorners, bool const halfPixelCenters, NDArray* output) {
        BUILD_SINGLE_SELECTOR(image->dataType(), return resizeBicubicFunctorA_, (context,
                image, width, height, alignCorners, halfPixelCenters, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int resizeBicubicFunctorA_, (nd4j::LaunchContext * context,
            NDArray const* image, int width, int height, bool const alignCorners, bool const halfPixelCenters, NDArray* output), NUMERIC_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int resizeFunctor(nd4j::LaunchContext * context, NDArray const* image, int width, int height,
                      ImageResizeMethods method, bool preserveAspectRatio, bool antialias, NDArray* output) {
        switch (method) {
            case kResizeBilinear: return resizeBilinearFunctor(context, image, width, height, false, output); break;
            case kResizeNearest:  return resizeNeighborFunctor(context, image, width, height, true, output); break;
            case kResizeBicubic:  return resizeBicubicFunctor(context, image, width, height, preserveAspectRatio, antialias, output); break;
            case kResizeLanczos5:
            case kResizeGaussian:
            case kResizeArea:
            case kResizeMitchelcubic:
                 throw std::runtime_error("helper::resizeFunctor: Non implemented yet.");
        }
        return ND4J_STATUS_OK;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // --------------------------------------------------------------------------------------------------------------- //
    // Crop and Resize helper implementation
    // -------------------------------------------------------------------------------------------------------------- //
    // cropAndResize kernel   type of input(images) and output should be the same
    //
    template <typename T, typename Z, typename I>
    static __global__ void cropAndResizeKernel(T const *images, Nd4jLong* imagesShape, Z const* boxes, Nd4jLong* boxesShape,
            I const* indices, Nd4jLong* indexShape, I const* cropSize, Nd4jLong* cropShape, int method,
            double extrapolationVal, T* output, Nd4jLong* outputShape, int numBoxes, int cropHeight, int cropWidth,
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
        T* outBuf = reinterpret_cast<T*>(crops->specialBuffer());

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