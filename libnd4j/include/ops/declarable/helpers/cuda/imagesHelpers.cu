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
// @author Yurii Shyrma (iuriish@yahoo.com)
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//

#include <system/op_boilerplate.h>
#include <ops/declarable/helpers/imagesHelpers.h>
#include <helpers/ConstantTadHelper.h>
#include <ops/declarable/helpers/adjust_hue.h>
#include <helpers/PointersManager.h>


namespace sd    {
namespace ops     {
namespace helpers {


///////////////////////////////////////////////////////////////////
template<typename T>
__global__ void rgbToYuvCuda(const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets, void* vz, const Nd4jLong *zShapeInfo, const Nd4jLong* zTadOffsets, const Nd4jLong numOfTads, const int dimC) {

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

        rgbYuv<T>(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);
    }

}

///////////////////////////////////////////////////////////////////
template<typename T>
linkage void rgbToYuvCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream, const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets, void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zTadOffsets, const Nd4jLong numOfTads, const int dimC) {

    rgbToYuvCuda<T> << <blocksPerGrid, threadsPerBlock, 256, * stream >> > (vx, xShapeInfo, xTadOffsets, vz, zShapeInfo, zTadOffsets, numOfTads, dimC);
}

///////////////////////////////////////////////////////////////////
void transformRgbYuv(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimC) {

    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), { dimC });
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output.shapeInfo(), { dimC });

    const Nd4jLong numOfTads = packX.numberOfTads();

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (numOfTads + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(context, "yuv_to_rgb");

    NDArray::prepareSpecialUse({ &output }, { &input });
    BUILD_SINGLE_SELECTOR(input.dataType(), rgbToYuvCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.specialBuffer(), input.specialShapeInfo(), packX.platformOffsets(), output.specialBuffer(), output.specialShapeInfo(), packZ.platformOffsets(), numOfTads, dimC), FLOAT_TYPES);
    NDArray::registerSpecialUse({ &output }, { &input });

    manager.synchronize();
}

///////////////////////////////////////////////////////////////////
template<typename T>
__global__ void yuvToRgbCuda(const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets, void* vz, const Nd4jLong *zShapeInfo, const Nd4jLong* zTadOffsets, const Nd4jLong numOfTads, const int dimC) {

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

        yuvRgb<T>(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);
    }

}

///////////////////////////////////////////////////////////////////
template<typename T>
linkage void yuvToRgbCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream, const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets, void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zTadOffsets, const Nd4jLong numOfTads, const int dimC) {

    yuvToRgbCuda<T> << <blocksPerGrid, threadsPerBlock, 256, * stream >> > (vx, xShapeInfo, xTadOffsets, vz, zShapeInfo, zTadOffsets, numOfTads, dimC);
}

///////////////////////////////////////////////////////////////////
void transformYuvRgb(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimC) {

    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), { dimC });
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output.shapeInfo(), { dimC });

    const Nd4jLong numOfTads = packX.numberOfTads();

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (numOfTads + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(context, "yuv_to_rgb");

    NDArray::prepareSpecialUse({ &output }, { &input });
    BUILD_SINGLE_SELECTOR(input.dataType(), yuvToRgbCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input.specialBuffer(), input.specialShapeInfo(), packX.platformOffsets(), output.specialBuffer(), output.specialShapeInfo(), packZ.platformOffsets(), numOfTads, dimC), FLOAT_TYPES);
    NDArray::registerSpecialUse({ &output }, { &input });

    manager.synchronize();
}

///////////////////////////////////////////////////////////////////
// for example xShapeInfo = {2,3,4}, zShapeInfo = {2,1,4}
template<typename T>
__global__ void rgbToGrsCuda(const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const int dimC) {

	const auto x = reinterpret_cast<const T*>(vx);
		  auto z = reinterpret_cast<T*>(vz);

	__shared__ Nd4jLong zLen;
	__shared__ int rank, *sharedMem;	// xRank == zRank

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<int*>(shmem);

		zLen = shape::length(zShapeInfo);
		rank = shape::rank(zShapeInfo);
	}
	__syncthreads();

	auto coords = sharedMem + threadIdx.x * rank;

	for (Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x; i < zLen; i +=  gridDim.x * blockDim.x) {

		if (dimC == (rank - 1) && 'c' == shape::order(xShapeInfo) && 1 == shape::elementWiseStride(xShapeInfo) && 'c' == shape::order(zShapeInfo) && 1 == shape::elementWiseStride(zShapeInfo)) {
			const auto xStep = i*3;
            z[i] = 0.2989f * x[xStep] + 0.5870f * x[xStep + 1] + 0.1140f * x[xStep + 2];
		}
		else {

	    	shape::index2coords(i, zShapeInfo, coords);

            const auto zOffset  = shape::getOffset(zShapeInfo, coords);
            const auto xOffset0 = shape::getOffset(xShapeInfo, coords);
            const auto xOffset1 = xOffset0 + shape::stride(xShapeInfo)[dimC];
            const auto xOffset2 = xOffset1 + shape::stride(xShapeInfo)[dimC];

            z[zOffset] = 0.2989f * x[xOffset0] + 0.5870f * x[xOffset1] + 0.1140f * x[xOffset2];
		}
	}
}

///////////////////////////////////////////////////////////////////
template<typename T>
linkage void rgbToGrsCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem, const cudaStream_t *stream, const void *vx, const Nd4jLong *xShapeInfo, void *vz, const Nd4jLong *zShapeInfo, const int dimC) {

	rgbToGrsCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(vx, xShapeInfo, vz, zShapeInfo, dimC);
}

///////////////////////////////////////////////////////////////////
void transformRgbGrs(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimC) {

	PointersManager manager(context, "rgbToGrs");

    const int threadsPerBlock = MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (input.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;
    const int sharedMem = input.rankOf() * sizeof(int) * threadsPerBlock + 128;

	NDArray::prepareSpecialUse({&output}, {&input});
	BUILD_SINGLE_SELECTOR(input.dataType(), rgbToGrsCudaLauncher, (blocksPerGrid, threadsPerBlock, sharedMem, context->getCudaStream(), input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), dimC), NUMERIC_TYPES);
	NDArray::registerSpecialUse({&output}, {&input});

	manager.synchronize();
}


///////////////////////////////////////////////////////////////////
template <typename T>
static void _CUDA_G rgbToHsvCuda(const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets,
                                  void* vz, const Nd4jLong *zShapeInfo, const Nd4jLong* zTadOffsets,
                                  const Nd4jLong numOfTads, const int dimC) {

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

        rgbToHsv<T>(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);
    }
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void _CUDA_G hsvToRgbCuda(const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets,
                                 void* vz, const Nd4jLong *zShapeInfo, const Nd4jLong* zTadOffsets,
                                 const Nd4jLong numOfTads, const int dimC) {

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

        hsvToRgb<T>(xTad[0], xTad[xDimCstride], xTad[2 * xDimCstride], zTad[0], zTad[zDimCstride], zTad[2 * zDimCstride]);
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
static _CUDA_H void hsvToRgbCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
                                          const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets,
                                          void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zTadOffsets,
                                          const Nd4jLong numOfTads, const int dimC) {

    hsvToRgbCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, xTadOffsets, vz, zShapeInfo, zTadOffsets, numOfTads, dimC);
}

template<typename T>
static _CUDA_H void rgbToHsvCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
                                         const void* vx, const Nd4jLong* xShapeInfo, const Nd4jLong* xTadOffsets,
                                         void* vz, const Nd4jLong* zShapeInfo, const Nd4jLong* zTadOffsets,
                                         const Nd4jLong numOfTads, const int dimC) {

    rgbToHsvCuda<T><<<blocksPerGrid, threadsPerBlock, 256, *stream>>>(vx, xShapeInfo, xTadOffsets, vz, zShapeInfo, zTadOffsets, numOfTads, dimC);
}

///////////////////////////////////////////////////////////////////
void transformHsvRgb(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {

    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(),  {dimC});
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), {dimC});

    const Nd4jLong numOfTads = packX.numberOfTads();

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (numOfTads + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(context, "hsv_to_rgb");

    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), hsvToRgbCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(), packX.platformOffsets(), output->specialBuffer(), output->specialShapeInfo(), packZ.platformOffsets(), numOfTads, dimC), FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});

    manager.synchronize();
}

///////////////////////////////////////////////////////////////////
void transformRgbHsv(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(),  {dimC});
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), {dimC});

    const Nd4jLong numOfTads = packX.numberOfTads();

    const int threadsPerBlock = MAX_NUM_THREADS / 2;
    const int blocksPerGrid = (numOfTads + threadsPerBlock - 1) / threadsPerBlock;

    PointersManager manager(context, "rgb_to_hsv");

    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), rgbToHsvCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input->specialBuffer(), input->specialShapeInfo(), packX.platformOffsets(), output->specialBuffer(), output->specialShapeInfo(), packZ.platformOffsets(), numOfTads, dimC), FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});

    manager.synchronize();
}

template<typename T>
__global__ void tripleTransformerCuda(const void *vx, const Nd4jLong *xShapeInfo, const Nd4jLong *xTadShapeInfo, const Nd4jLong *xOffsets, void *vz, const Nd4jLong *zShapeInfo, const Nd4jLong *zTadShapeInfo, const Nd4jLong *zOffsets, const int dimC, int mode, uint64_t numTads) {
    const auto x = reinterpret_cast<const T*>(vx);
    auto z = reinterpret_cast<T*>(vz);

    __shared__ Nd4jLong zLen, *sharedMem;
    __shared__ int rank;	// xRank == zRank

    float yiqarr[3][3] = {
            { 0.299f,  0.59590059f,  0.2115f },
            { 0.587f, -0.27455667f,  -0.52273617f },
            { 0.114f, -0.32134392f,  0.31119955f }
    };

    float rgbarr[3][3] = {
            { 1.f,  1.f,  1.f },
            { 0.95598634f, -0.27201283f, -1.10674021f },
            { 0.6208248f, -0.64720424f, 1.70423049f }
    };

    auto tr = mode == 1? yiqarr : rgbarr;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        sharedMem = reinterpret_cast<Nd4jLong*>(shmem);

        zLen = shape::length(zShapeInfo);
        rank = shape::rank(zShapeInfo);
    }
    __syncthreads();

    Nd4jLong* coords = sharedMem + threadIdx.x * rank;

    if (dimC == (rank - 1) && 'c' == shape::order(xShapeInfo) && 1 == shape::elementWiseStride(xShapeInfo) && 'c' == shape::order(zShapeInfo) && 1 == shape::elementWiseStride(zShapeInfo)) {
        for (uint64_t f = blockIdx.x * blockDim.x + threadIdx.x; f < zLen / 3; f +=  gridDim.x * blockDim.x) {
            auto i = f * 3;

            auto xi0 = x[i];
            auto xi1 = x[i+1];
            auto xi2 = x[i+2];

            for (int e = 0; e < 3; e++)
                z[i + e] = xi0 * tr[0][e] + xi1 * tr[1][e] + xi2 * tr[2][e];
        }
    } else {
        // TAD based case
        const Nd4jLong xDimCstride = shape::stride(xShapeInfo)[dimC];
        const Nd4jLong zDimCstride = shape::stride(zShapeInfo)[dimC];

        for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numTads; i += blockDim.x * gridDim.x) {
            const T* xTad = x + xOffsets[i];
            T* zTad = z + zOffsets[i];

            auto xi0 = xTad[0];
            auto xi1 = xTad[xDimCstride];
            auto xi2 = xTad[xDimCstride * 2];

            for (int e = 0; e < 3; e++)
                zTad[zDimCstride * e] = xi0 * tr[0][e] + xi1 * tr[1][e] + xi2 * tr[2][e];
        }
    }
}


template <typename T>
static void rgbYiq(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimC);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimC);

    NDArray::prepareSpecialUse({output}, {input});
    return tripleTransformerCuda<T><<<256, 256, 8192, *context->getCudaStream()>>>(input->specialBuffer(), input->specialShapeInfo(), packX.platformShapeInfo(), packX.platformOffsets(), output->specialBuffer(), output->specialShapeInfo(), packZ.platformShapeInfo(), packZ.platformOffsets(), dimC, 1, packZ.numberOfTads());
    NDArray::registerSpecialUse({output}, {input});
}

template <typename T>
FORCEINLINE static void yiqRgb(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
    auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), dimC);
    auto packZ = sd::ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), dimC);

    NDArray::prepareSpecialUse({output}, {input});
    return tripleTransformerCuda<T><<<256, 256, 8192, *context->getCudaStream()>>>(input->specialBuffer(), input->specialShapeInfo(), packX.platformShapeInfo(), packX.platformOffsets(), output->specialBuffer(), output->specialShapeInfo(), packZ.platformShapeInfo(), packZ.platformOffsets(), dimC, 2, packZ.numberOfTads());
    NDArray::registerSpecialUse({output}, {input});
}

void transformYiqRgb(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
    BUILD_SINGLE_SELECTOR(input->dataType(), yiqRgb, (context, input, output, dimC), FLOAT_TYPES);
}

void transformRgbYiq(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
    BUILD_SINGLE_SELECTOR(input->dataType(), rgbYiq, (context, input, output, dimC), FLOAT_TYPES);
}





}
}
}

