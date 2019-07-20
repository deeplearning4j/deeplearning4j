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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/adjust_hue.h>
#include <helpers/ConstantTadHelper.h>
#include <PointersManager.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


///////////////////////////////////////////////////////////////////
template <typename T>
static void _CUDA_G adjustHueCuda(const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets,
                                        void* vz, const Nd4jLong *zShapeInfo, const Nd4jLong* zTadOffsets,
                                        const Nd4jLong numOfTads, const T delta, const int dimC) {

    const T* x = reinterpret_cast<const T*>(vx);
          T* z = reinterpret_cast<T*>(vz);

    __shared__ int rank;
    __shared__ Nd4jLong xDimCstride, zDimCstride;

    if (threadIdx.x == 0) {
        rank = shape::rank(xShapeInfo);
        xDimCstride = shape::stride(xShapeInfo)[dimC];
        zDimCstride = shape::stride(zShapeInfo)[dimC];
    }

    __syncthreads();

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < numOfTads; i += gridDim.x * blockDim.x) {

        const T* xTad = x + xTadOffsets[i];
              T* zTad = z + zTadOffsets[i];

        T h, s, v;

        rgbToHsv<T>(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], h, s, v);

        h += delta * 360;
        if(h > 360)
            h -= 360;
        else if(h < 0)
            h += 360;

        hsvToRgb<T>(h, s, v, zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
static _CUDA_H void adjustHueCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
                                          const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets,
                                                void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zTadOffsets,
                                          const Nd4jLong numOfTads, const NDArray* deltaScalarArr, const int dimC) {

    adjustHueCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, xTadOffsets, vz, zShapeInfo, zTadOffsets, numOfTads, deltaScalarArr->e<T>(0), dimC);
}
BUILD_SINGLE_TEMPLATE(template void adjustHueCudaLauncher, (const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream, const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets, void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zTadOffsets, const Nd4jLong numOfTads, const NDArray* deltaScalarArr, const int dimC), LIBND4J_TYPES);

////////////////////////////////////////////////////////////////////////
void adjustHue(nd4j::LaunchContext* context, const NDArray *input, const NDArray* deltaScalarArr, NDArray *output, const int dimC) {

    auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(),  {dimC});
    auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), {dimC});

    const Nd4jLong numOfTads = packX.numberOfTads();

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (numOfTads + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(context, "adjustHue");

    NDArray::prepareSpecialUse({output}, {input, deltaScalarArr});
    BUILD_SINGLE_SELECTOR(input->dataType(), adjustHueCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input->getSpecialBuffer(), input->getSpecialShapeInfo(), packX.platformOffsets(), output->specialBuffer(), output->specialShapeInfo(), packZ.platformOffsets(), numOfTads, deltaScalarArr, dimC), LIBND4J_TYPES);
    NDArray::registerSpecialUse({output}, {input, deltaScalarArr});

    manager.synchronize();
}


/*
template <typename T>
static void _CUDA_G adjustHueSingleNHWCKernel(void *xBuffer, Nd4jLong *xShapeInfo,  void *zBuffer, Nd4jLong *zShapeInfo, Nd4jLong tuples, float delta) {
    int numChannels = 3;
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;

    auto bIn = reinterpret_cast<T*>(xBuffer);
    auto bOut = reinterpret_cast<T*>(zBuffer);
    static const int kChannelRange = 6;

    for (Nd4jLong e = tid; e < tuples; e += blockDim.x * gridDim.x) {
        auto i = bIn + e * numChannels;
        auto o = bOut + e * numChannels;

        T h, v_min, v_max;
        helpers::rgb_to_hv(i[0], i[1], i[2], &h, &v_min, &v_max);

        h += delta * kChannelRange;
        while (h < (T) 0.)
            h += (T) kChannelRange;

        while (h >= (T) kChannelRange)
            h -= (T) kChannelRange;

        helpers::hv_to_rgb(h, v_min, v_max, o, o + 1, o + 2);
    }
}

template <typename T>
static void _CUDA_G adjustHueSingleNCHWKernel(void *xBuffer, Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, void *zBuffer, Nd4jLong *zTadShapeInfo, Nd4jLong *zOffsets, Nd4jLong tadLength, Nd4jLong tuples, float delta) {
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
        auto _ri = bufferR + shape::getIndexOffset(e, xTadShapeInfo, tadLength);;
        auto _gi = bufferG + shape::getIndexOffset(e, xTadShapeInfo, tadLength);;
        auto _bi = bufferB + shape::getIndexOffset(e, xTadShapeInfo, tadLength);;

        auto _ro = outputR + shape::getIndexOffset(e, xTadShapeInfo, tadLength);;
        auto _go = outputG + shape::getIndexOffset(e, xTadShapeInfo, tadLength);;
        auto _bo = outputB + shape::getIndexOffset(e, xTadShapeInfo, tadLength);;

        T h, v_min, v_max;
        helpers::rgb_to_hv(_ri[0], _gi[0], _bi[0], &h, &v_min, &v_max);

        h += delta * kChannelRange;
        while (h < (T) 0)
            h += (T) kChannelRange;

        while (h >= (T) kChannelRange)
            h -= (T) kChannelRange;

        helpers::hv_to_rgb(h, v_min, v_max, _ro, _go, _bo);
    }
}

template <typename T>
static void _adjust_hue_single(nd4j::LaunchContext * context, NDArray *array, NDArray *output, float delta, bool isNHWC) {
    // numChannels is always 3
    auto tuples = array->lengthOf() / 3;
    if (isNHWC) {
        adjustHueSingleNHWCKernel<T><<<256, 256, 1024, *context->getCudaStream()>>>(array->specialBuffer(), array->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(), tuples, delta);
    } else {
        // TODO: check this one
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(array->getShapeInfo(), {1, 2});
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), {1, 2});

        auto tadLength = shape::length(packX.primaryShapeInfo());

        adjustHueSingleNCHWKernel<T><<<256, 256, 1024, *context->getCudaStream()>>>(array->specialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), output->specialBuffer(), packZ.platformShapeInfo(), packZ.platformOffsets(), tadLength, tuples, delta);
    }
}


template <typename T>
static void _adjust_hue_batch(nd4j::LaunchContext * context, NDArray *array, NDArray *output, float delta, bool isNHWC) {
    auto xType = array->dataType();

    // numChannels is always 3
    auto tuples = array->lengthOf() / 3;

    if (isNHWC) {
        // in case of nhwc batch, we don't really care about examples: it's still bunch of RGB values
        BUILD_SINGLE_SELECTOR(xType, _adjust_hue_single, (context, array, output, delta, isNHWC);, FLOAT_TYPES);
    } else {
        // TODO: check this one
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(array->getShapeInfo(), {0, 2, 3});
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), {0, 2, 3});

        auto tadLength = shape::length(packX.primaryShapeInfo());

        adjustHueSingleNCHWKernel<T><<<256, 256, 1024, *context->getCudaStream()>>>(array->specialBuffer(), packX.platformShapeInfo(), packX.platformOffsets(), output->specialBuffer(), packZ.platformShapeInfo(), packZ.platformOffsets(), tadLength, tuples, delta);
    }
}

void _adjust_hue(nd4j::LaunchContext * context, NDArray *array, NDArray *output, NDArray* delta, bool isNHWC) {
    auto xType = array->dataType();

    float d = delta->e<float>(0);
    if (array->rankOf() == 4) {
        BUILD_SINGLE_SELECTOR(xType, _adjust_hue_batch, (context, array, output, d, isNHWC);, FLOAT_TYPES);
    } else {
        BUILD_SINGLE_SELECTOR(xType, _adjust_hue_single, (context, array, output, d, isNHWC);, FLOAT_TYPES);
    }
}

*/
}
}
}
