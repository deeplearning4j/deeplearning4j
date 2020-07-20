/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
// @author sgazeos@gmail.com
// @author raver119@gmail.com
//


#include <ops/declarable/helpers/transforms.h>
#include <helpers/ShapeUtils.h>
#include <helpers/PointersManager.h>
#include <helpers/ConstantTadHelper.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void clipByNormCuda(const void* vClipNorm, const void* vNorm, const Nd4jLong* normShapeInfo, void* vz, const Nd4jLong* zShapeInfo, const int* dimensions, const int dimsLen, const bool useAverage) {

    const T clipNorm = *reinterpret_cast<const T*>(vClipNorm);
    const T* norm    = reinterpret_cast<const T*>(vNorm);
          T* z       = reinterpret_cast<T*>(vz);

    __shared__ Nd4jLong zLen, tadLen, totalThreads;

    if (threadIdx.x == 0) {

        zLen   = shape::length(zShapeInfo);
        tadLen = zLen / shape::length(normShapeInfo);
        totalThreads = gridDim.x * blockDim.x;
    }

    __syncthreads();

    int zCoords[MAX_RANK], normCoords[MAX_RANK];

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

        shape::index2coords(i, zShapeInfo, zCoords);

        // deduce norm coords
        for (int j = 0; j < dimsLen; ++j)
            normCoords[j] = zCoords[dimensions[j]];

        const T actualNorm = useAverage ? norm[shape::getOffset(normShapeInfo, normCoords)] / tadLen : norm[shape::getOffset(normShapeInfo, normCoords)];

        if(actualNorm > clipNorm)
            z[shape::getOffset(zShapeInfo, zCoords)] *= clipNorm / actualNorm;
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
__host__ static void clipByNormCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t *stream,
                                            const void* vClipNorm, const void* vNorm, const Nd4jLong* normShapeInfo, void* vz, const Nd4jLong* zShapeInfo,
                                            const int* dimensions, const int dimsLen, const bool useAverage) {

    clipByNormCuda<T><<<blocksPerGrid, threadsPerBlock, 512, *stream>>>(vClipNorm, vNorm, normShapeInfo, vz, zShapeInfo, dimensions, dimsLen, useAverage);
}

//////////////////////////////////////////////////////////////////////////
void clipByNorm(sd::LaunchContext* context, NDArray& input, NDArray& output, const std::vector<int>& dims, const NDArray& clipNorm, const bool isInplace, const bool useAverage) {

    NDArray* z = nullptr;

    if(isInplace) {
        z = &input;
    }
    else {
        output.assign(input);
        z = &output;
    }

    if(dims.empty()) {

        const NDArray actualNorm = useAverage ? z->reduceAlongDimension(reduce::Norm2, {}) / z->lengthOf() : z->reduceAlongDimension(reduce::Norm2, {});

        if(actualNorm.e<float>(0) > clipNorm.e<float>(0))
            *z *= clipNorm / actualNorm;
    }
    else {

        const NDArray actualNorms = z->reduceAlongDimension(reduce::Norm2, dims);

        std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(z->rankOf(), dims);

        const int threadsPerBlock = MAX_NUM_THREADS / 2;
        const int blocksPerGrid = (z->lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

        PointersManager manager(context, "clipByNorm");

        const int* dimensions = reinterpret_cast<const int*>(manager.replicatePointer(dimsToExclude.data(), dimsToExclude.size() * sizeof(int)));

        NDArray::prepareSpecialUse({z}, {z, &actualNorms, &clipNorm});
        BUILD_SINGLE_SELECTOR(z->dataType(), clipByNormCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), clipNorm.specialBuffer(), actualNorms.specialBuffer(), actualNorms.specialShapeInfo(), z->specialBuffer(), z->specialShapeInfo(), dimensions, (int)dimsToExclude.size(), useAverage), FLOAT_TYPES);
        NDArray::registerSpecialUse({z}, {z, &actualNorms, &clipNorm});

        manager.synchronize();
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ static void clipByNormBpCuda(const void* vClipNorm,
                                        const void* vx, const Nd4jLong* xShapeInfo,         // input
                                        const void* vy, const Nd4jLong* yShapeInfo,         // gradO
                                        const void* vNorm, const Nd4jLong* normShapeInfo,
                                        const void* vSum, const Nd4jLong* sumShapeInfo,
                                        void* vz, const Nd4jLong* zShapeInfo,               // gradI
                                        const int* dimensions, const int dimsLen, const bool useAverage) {

    const T clipNorm = *reinterpret_cast<const T*>(vClipNorm);
    const T* norm    = reinterpret_cast<const T*>(vNorm);
    const T* sum     = reinterpret_cast<const T*>(vSum);
    const T* x       = reinterpret_cast<const T*>(vx);
    const T* y       = reinterpret_cast<const T*>(vy);
          T* z       = reinterpret_cast<T*>(vz);

    __shared__ Nd4jLong zLen, tadLen, totalThreads;
    __shared__ bool sameOffsets;

    if (threadIdx.x == 0) {

        zLen   = shape::length(zShapeInfo);
        tadLen = zLen / shape::length(normShapeInfo);
        totalThreads = gridDim.x * blockDim.x;

        sameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo, zShapeInfo);
    }

    __syncthreads();

    int zCoords[MAX_RANK], normCoords[MAX_RANK];

    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < zLen; i += totalThreads) {

        shape::index2coords(i, zShapeInfo, zCoords);

        const auto zOffset = shape::getOffset(zShapeInfo, zCoords);
        const auto yOffset = sameOffsets ? zOffset : shape::getOffset(yShapeInfo, zCoords);

        // deduce norm coords
        for (int j = 0; j < dimsLen; ++j)
            normCoords[j] = zCoords[dimensions[j]];

        const T actualNorm = useAverage ? norm[shape::getOffset(normShapeInfo, normCoords)] / tadLen : norm[shape::getOffset(normShapeInfo, normCoords)];

        if(actualNorm > clipNorm) {

            const T sumVal =  sum[shape::getOffset(sumShapeInfo, normCoords)];
            const auto xOffset = sameOffsets ? zOffset : shape::getOffset(xShapeInfo, zCoords);

            z[zOffset] = (clipNorm / actualNorm) * y[yOffset] * (static_cast<T>(1.f) - (x[xOffset] * sumVal) / (actualNorm * actualNorm));
        }
        else
            z[zOffset] = y[yOffset];
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void clipByNormBp_(sd::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const std::vector<int>& dims, const NDArray& clipNorm, const bool useAverage) {

    const int rank = input.rankOf();

    auto actualNorms = input.reduceAlongDimension(reduce::Norm2, dims);

    if(actualNorms.lengthOf() == 1) {

        const T norm = useAverage ? actualNorms.e<T>(0) / static_cast<T>(input.lengthOf()) : actualNorms.e<T>(0);

        auto clipVal = clipNorm.e<T>(0);

        if(norm > clipVal) {

            const T sum = input.reduceNumber(reduce::Sum).e<T>(0);    // reduce to scalar
            const T factor1 =  clipVal / norm;
            const T factor2 = static_cast<T>(1.f) / (norm * norm);                                            // 1 / (norm*norm*norm)

            auto lambda = LAMBDA_TT(x, y, sum, factor1, factor2) {
                return factor1 * y * (static_cast<T>(1.f) - factor2 * x * sum);
            };

            const_cast<NDArray&>(input).applyPairwiseLambda(const_cast<NDArray&>(gradO), lambda, gradI);
        }
        else
            gradI.assign(gradO);
    }
    else {

        const NDArray actualNorms = input.reduceAlongDimension(reduce::Norm2, dims);
        const NDArray sums        = input.reduceAlongDimension(reduce::Sum, dims);

        std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(gradI.rankOf(), dims);

        const int threadsPerBlock = MAX_NUM_THREADS / 2;
        const int blocksPerGrid = (gradI.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

        PointersManager manager(context, "clipByNormBp");

        const int* dimensions = reinterpret_cast<const int*>(manager.replicatePointer(dimsToExclude.data(), dimsToExclude.size() * sizeof(int)));

        NDArray::prepareSpecialUse({&gradI}, {&actualNorms, &sums, &clipNorm, &input, &gradO});
        clipByNormBpCuda<T><<<blocksPerGrid, threadsPerBlock, 512, *context->getCudaStream()>>>(clipNorm.specialBuffer(), input.specialBuffer(), input.specialShapeInfo(), gradO.specialBuffer(), gradO.specialShapeInfo(), actualNorms.specialBuffer(), actualNorms.specialShapeInfo(), sums.specialBuffer(), sums.specialShapeInfo(), gradI.specialBuffer(), gradI.specialShapeInfo(), dimensions, (int)dimsToExclude.size(), useAverage);
        NDArray::registerSpecialUse({&gradI}, {&actualNorms, &sums, &clipNorm, &input, &gradO});

        manager.synchronize();
    }
}
BUILD_SINGLE_TEMPLATE(template void clipByNormBp_, (sd::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool useAverage), FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
void clipByNormBp(sd::LaunchContext* context, const NDArray& input, const NDArray& gradO, NDArray& gradI, const std::vector<int>& dimensions, const NDArray& clipNorm, const bool useAverage) {

    const NDArray& castedInput = gradI.dataType() == input.dataType() ? input : input.cast(gradI.dataType());
    BUILD_SINGLE_SELECTOR(gradI.dataType(), clipByNormBp_, (context, castedInput, gradO, gradI, dimensions, clipNorm, useAverage), FLOAT_TYPES);
}






        template <typename T>
    void clipByGlobalNorm_(sd::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, sd::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        NDArray globalNorm = NDArrayFactory::create<T>(0, inputs[0]->getContext()); //sqrt(sum([l2norm(t)**2 for t in t_list]))

        for (auto i = 0; i < inputs.size(); i++) {
            auto input = inputs[i];
            auto l2norm = input->reduceNumber(reduce::Norm2);
            globalNorm += l2norm * l2norm;
        }

        globalNorm.applyTransform(transform::Sqrt, globalNorm);     // = sd::math::nd4j_sqrt(globalNorm);
        outputs[inputs.size()]->p(0, globalNorm);
        globalNorm.syncToHost();
        const T factor = static_cast<T>(clipNorm) / globalNorm.e<T>(0);

        for (size_t e = 0; e < inputs.size(); e++) {
            // all-reduce
            auto input = inputs[e];
            auto output = outputs[e];

            if (globalNorm.e<double>(0) <= clipNorm) {
                output->assign(input);
            }
            else {

                auto lambda = LAMBDA_T(_x, factor) { return _x * factor; };
                input->applyLambda(lambda, *output);
            }
        }
    }

    void clipByGlobalNorm(sd::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, sd::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
        BUILD_SINGLE_SELECTOR(outputs[0]->dataType(), clipByGlobalNorm_, (context, inputs, clipNorm, workspace, outputs, isInplace), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByGlobalNorm_, (sd::LaunchContext * context, std::vector<NDArray*> const& inputs, double clipNorm, sd::memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace), FLOAT_TYPES);


    template <typename T>
    static void __global__ clipByValueKernel(void* input, const Nd4jLong* inputShape, void* output, const Nd4jLong* outputShape, double leftBound, double rightBound) {
        __shared__ T* outputBuf;
        __shared__ T* inputBuf;
        __shared__ Nd4jLong length;
        __shared__ bool linearBuffers;
        if (threadIdx.x == 0) {
            outputBuf = reinterpret_cast<T *>(output);
            inputBuf = reinterpret_cast<T *>(input);
            length = shape::length(inputShape);
            linearBuffers = shape::elementWiseStride(inputShape) == shape::elementWiseStride(outputShape) && shape::elementWiseStride(inputShape) == 1;
        }
        __syncthreads();
        const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        const auto step = gridDim.x * blockDim.x;

        for (Nd4jLong e = tid; e < length; e += step) {
            if (linearBuffers) {
                if (inputBuf[e] > rightBound) outputBuf[e] = (T) rightBound;
                else if (inputBuf[e] < leftBound) outputBuf[e] = (T) leftBound;
                else outputBuf[e] = inputBuf[e];
            }
            else {
                auto inputOffset = shape::getIndexOffset(e, inputShape);
                auto outputOffset = shape::getIndexOffset(e, outputShape);
                if (inputBuf[inputOffset] > rightBound) outputBuf[outputOffset] = (T) rightBound;
                else if (inputBuf[inputOffset] < leftBound) outputBuf[outputOffset] = (T) leftBound;
                else outputBuf[outputOffset] = inputBuf[outputOffset];
            }
        }
    }

    template <typename T>
    static void clipByValue_(sd::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        auto stream = context->getCudaStream();
        if (!input.isActualOnDeviceSide())
            input.syncToDevice();
        NDArray::prepareSpecialUse({&output}, {&input});
        clipByValueKernel<T><<<256, 512, 8192, *stream>>>(input.specialBuffer(), input.specialShapeInfo(), output.specialBuffer(), output.specialShapeInfo(), leftBound, rightBound);
        NDArray::registerSpecialUse({&output}, {&input});
    }

    void clipByValue(sd::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output) {
        BUILD_SINGLE_SELECTOR(input.dataType(), clipByValue_, (context, input, leftBound, rightBound, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void clipByValue_, (sd::LaunchContext * context, NDArray& input, double leftBound, double rightBound, NDArray& output);, FLOAT_TYPES);

}
}
}

