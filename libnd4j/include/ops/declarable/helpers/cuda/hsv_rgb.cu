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

#include <ops/declarable/helpers/adjust_hue.h>
#include <ops/declarable/helpers/adjust_saturation.h>
#include <helpers/ConstantTadHelper.h>
#include <PointersManager.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

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


            void transform_hsv_rgb(nd4j::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
                auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(),  {dimC});
                auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), {dimC});

                const Nd4jLong numOfTads = packX.numberOfTads();

                const int threadsPerBlock = MAX_NUM_THREADS / 2;
                const int blocksPerGrid = (numOfTads + threadsPerBlock - 1) / threadsPerBlock;

                PointersManager manager(context, "hsv_to_rgb");

                NDArray::prepareSpecialUse({output}, {input});
                BUILD_SINGLE_SELECTOR(input->dataType(), hsvToRgbCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input->getSpecialBuffer(), input->getSpecialShapeInfo(), packX.platformOffsets(), output->specialBuffer(), output->specialShapeInfo(), packZ.platformOffsets(), numOfTads, dimC), FLOAT_TYPES);
                NDArray::registerSpecialUse({output}, {input});

                manager.synchronize();
            }

            void transform_rgb_hsv(nd4j::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC) {
                auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(),  {dimC});
                auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), {dimC});

                const Nd4jLong numOfTads = packX.numberOfTads();

                const int threadsPerBlock = MAX_NUM_THREADS / 2;
                const int blocksPerGrid = (numOfTads + threadsPerBlock - 1) / threadsPerBlock;

                PointersManager manager(context, "rgb_to_hsv");

                NDArray::prepareSpecialUse({output}, {input});
                BUILD_SINGLE_SELECTOR(input->dataType(), rgbToHsvCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), input->getSpecialBuffer(), input->getSpecialShapeInfo(), packX.platformOffsets(), output->specialBuffer(), output->specialShapeInfo(), packZ.platformOffsets(), numOfTads, dimC), FLOAT_TYPES);
                NDArray::registerSpecialUse({output}, {input});

                manager.synchronize();
            }
        }
    }
}