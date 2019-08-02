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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/top_k.h>
#include <MmulHelper.h>
#include <NDArrayFactory.h>
#include <Status.h>
#include <ConstantTadHelper.h>
#include <ShapeUtils.h>

#include <cusolverDn.h>
#include <cuda_exception.h>

namespace nd4j {
namespace ops {
namespace helpers {

//    template <typename T>
//    static __device__ void swapRows_(T* matrix, Nd4jLong* shape, int theFirst, int theSecond, Nd4jLong N) {
//        if (theFirst != theSecond) {
//            auto start = threadIdx.x + blockIdx.x * blockDim.x;
//            auto step = blockDim.x * gridDim.x;
//            for (auto i = start; i < N; i += step) {
//                Nd4jLong iCoord1[] = {theFirst, i};
//                Nd4jLong iCoord2[] = {theSecond, i};
//                auto iIndex1 = shape::getOffset(0, shape::shapeOf(shape), shape::stride(shape), iCoord1, 2);
//                auto iIndex2 = shape::getOffset(0, shape::shapeOf(shape), shape::stride(shape), iCoord2, 2);
//                //atomicExch(&matrix[iIndex1], matrix[iIndex2]);
//                T e0 = matrix[iIndex1];
//                T e1 = matrix[iIndex2];
//                matrix[iIndex1] = e0;
//                matrix[iIndex2] = e1;
//            }
//        }
//    }
//    BUILD_SINGLE_TEMPLATE(template void swapRows_, (NDArray* matrix, int theFirst, int theSecond), FLOAT_TYPES);
//
//    void swapRows(NDArray* matrix, int theFirst, int theSecond) {
//        BUILD_SINGLE_SELECTOR(matrix->dataType(), swapRows_, (matrix, theFirst, theSecond), FLOAT_TYPES);
//    }
    template <typename T>
    static __global__ void invertKernelLow(void* invertedBuf, Nd4jLong* invertedShape, void* inputBuf, Nd4jLong* inputShape, Nd4jLong n) {
        __shared__ T* inverted;
        __shared__ T* input;

        if (threadIdx.x == 0) {
            inverted = reinterpret_cast<T*>(invertedBuf);
            input = reinterpret_cast<T*>(inputBuf);
        }
        __syncthreads();

        auto start = threadIdx.x + blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;

        for (int i = start + 1; i < n; i += step) {
            Nd4jLong pos[] = {i, i - 1};
            Nd4jLong posX[] = {i, i};
            Nd4jLong posY[] = {i - 1, i - 1};
            auto xIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), pos, 2);
            auto dxIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), posX, 2);
            auto dyIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), posY, 2);
            auto zIndex = shape::getOffset(0, shape::shapeOf(invertedShape), shape::stride(invertedShape), pos, 2);
            inverted[zIndex] = -input[xIndex] / (input[dxIndex] * input[dyIndex]);
//            math::atomics::nd4j_atomicAdd(&inverted[zIndex], - input[xIndex] * inverted[iIndex] / input[dIndex]);
        }
    }

    template <typename T>
    static __global__ void upvertKernel(void* invertedBuf, Nd4jLong* invertedShape, void* inputBuf, Nd4jLong* inputShape, Nd4jLong n) {
        __shared__ T* inverted;
        __shared__ T* input;

        if (threadIdx.x == 0) {
            inverted = reinterpret_cast<T*>(invertedBuf);
            input = reinterpret_cast<T*>(inputBuf);
        }
        __syncthreads();

        auto start = threadIdx.x + blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;

        for (int i = start; i < n; i += step) {
            Nd4jLong pos[] = {i, i};
            auto xIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), pos, 2);
            auto zIndex = shape::getOffset(0, shape::shapeOf(invertedShape), shape::stride(invertedShape), pos, 2);
//            math::atomics::nd4j_atomicDiv(&inverted[zIndex], input[xIndex]);
            inverted[zIndex] /= input[xIndex];
        }
    }

    template <typename T>
    static __global__ void upvertKernelUp(void* invertedBuf, Nd4jLong* invertedShape, void* inputBuf, Nd4jLong* inputShape, Nd4jLong n) {
        __shared__ T* inverted;
        __shared__ T* input;

        if (threadIdx.x == 0) {
            inverted = reinterpret_cast<T*>(invertedBuf);
            input = reinterpret_cast<T*>(inputBuf);
        }
        __syncthreads();

        auto start = threadIdx.x + blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;

        for (int i = start; i < n - 1; i += step) {
            Nd4jLong pos[] = {i, i + 1};
            //Nd4jLong posY[] = {i, i};
            Nd4jLong posX[] = {i + 1, i + 1};
            auto xIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), pos, 2);
//            auto yIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), posY, 2);
//            auto yIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), pos, 2);
            auto iIndex = shape::getOffset(0, shape::shapeOf(invertedShape), shape::stride(invertedShape), posX, 2);
            auto zIndex = shape::getOffset(0, shape::shapeOf(invertedShape), shape::stride(invertedShape), pos, 2);
            math::atomics::nd4j_atomicAdd(&inverted[zIndex], - input[xIndex] * inverted[iIndex]); // / input[yIndex]);
            //inputMatrix->t<T>(i, i + 1) * invertedMatrix->t<T>(i + 1, i + 1) / inputMatrix->t<T>(i, i)
        }
    }

    template <typename T>
    static __global__ void invertLowKernel(void* invertedBuf, Nd4jLong* invertedShape, void* inputBuf, Nd4jLong* inputShape, Nd4jLong n) {
        __shared__ T* inverted;
        __shared__ T* input;

        if (threadIdx.x == 0) {
            inverted = reinterpret_cast<T*>(invertedBuf);
            input = reinterpret_cast<T*>(inputBuf);
        }
        __syncthreads();

//        auto start = threadIdx.x + blockIdx.x * blockDim.x;
//        auto step = blockDim.x * gridDim.x;

        for (int i = blockIdx.x + 2; i < n; i += gridDim.x) {
            for (int j = i - 2; j >= 0; --j)
                for (int k = threadIdx.x; k < i; k += blockDim.x) {
                    Nd4jLong posZ[] = {i, j};
                    Nd4jLong posY[] = {k, j};
                    Nd4jLong posX[] = {i, k};
                    Nd4jLong posD[] = {i, i};

                    auto xIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), posX, 2);
                    auto yIndex = shape::getOffset(0, shape::shapeOf(invertedShape), shape::stride(invertedShape), posY, 2);
                    auto dIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), posD, 2);
                    auto zIndex = shape::getOffset(0, shape::shapeOf(invertedShape), shape::stride(invertedShape), posZ, 2);
                    math::atomics::nd4j_atomicAdd(&inverted[zIndex], - inverted[yIndex] * input[xIndex] / input[dIndex]);
                }
        }
    }

    template <typename T>
    static __global__ void invertUpKernel(void* invertedBuf, Nd4jLong* invertedShape, void* inputBuf, Nd4jLong* inputShape, Nd4jLong n) {
        __shared__ T* inverted;
        __shared__ T* input;

        if (threadIdx.x == 0) {
            inverted = reinterpret_cast<T*>(invertedBuf);
            input = reinterpret_cast<T*>(inputBuf);
        }
        __syncthreads();

//        auto start = threadIdx.x + blockIdx.x * blockDim.x;
//        auto step = blockDim.x * gridDim.x;

        for (int i = n - blockIdx.x - 2; i >= 0; i -= gridDim.x) {
            for (int j = i + 2; j < n; j++)
                for (int k = i + threadIdx.x; k < n; k+= blockDim.x) {
                    Nd4jLong posZ[] = {i, j};
                    Nd4jLong posY[] = {k, j};
                    Nd4jLong posX[] = {i, k};
//                    Nd4jLong posD[] = {i, i};

                    auto xIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), posX, 2);
                    auto yIndex = shape::getOffset(0, shape::shapeOf(invertedShape), shape::stride(invertedShape), posY, 2);
  //                  auto dIndex = shape::getOffset(0, shape::shapeOf(inputShape), shape::stride(inputShape), posD, 2);
                    auto zIndex = shape::getOffset(0, shape::shapeOf(invertedShape), shape::stride(invertedShape), posZ, 2);
                    math::atomics::nd4j_atomicAdd(&inverted[zIndex], - inverted[yIndex] * input[xIndex]);// / input[dIndex]);
                }
        }
    }

    template <typename T>
    static void invertLowerMatrix_(NDArray* inputMatrix, NDArray* invertedMatrix) {
        int n = inputMatrix->rows();
        invertedMatrix->setIdentity();

        if (inputMatrix->isIdentityMatrix()) return;
        LaunchContext* context = inputMatrix->getContext();
        auto stream = context->getCudaStream();

        // invert main diagonal
        upvertKernel<T><<<1, n, 128, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(), inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
        // invert the second diagonal
        invertKernelLow<T><<<1, n, 128, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(), inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
//        invertKernelLow<T><<<1, n, 128, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(), inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
        invertLowKernel<T><<<n, n, 128, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(), inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
    }

    BUILD_SINGLE_TEMPLATE(template void invertLowerMatrix_, (NDArray* inputMatrix, NDArray* invertedMatrix);, FLOAT_NATIVE);

    void invertLowerMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
        BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), invertLowerMatrix_, (inputMatrix, invertedMatrix), FLOAT_NATIVE);
    }

    template <typename T>
    static void invertUpperMatrix_(NDArray* inputMatrix, NDArray* invertedMatrix) {
        int n = inputMatrix->rows();
        invertedMatrix->setIdentity();
        auto stream = inputMatrix->getContext()->getCudaStream();
        if (inputMatrix->isIdentityMatrix()) { // the inverse for I is I
            return;
        }

        //upvertKernel<T><<<1, n, 128, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(), inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
        upvertKernelUp<T><<<1, n, 128, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(), inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
        invertUpKernel<T><<<n, n, 256, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(), inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
    }

    BUILD_SINGLE_TEMPLATE(template void invertUpperMatrix_, (NDArray* inputMatrix, NDArray* invertedMatrix);, FLOAT_NATIVE);

    void invertUpperMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
        BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), invertUpperMatrix_, (inputMatrix, invertedMatrix), FLOAT_NATIVE);
    }

//    template <typename T>
//    static __global__ void lupKernel(T* compound, Nd4jLong* compoundShape, T* permutation, Nd4jLong* permutationShape, Nd4jLong rowNum) {
//        int swapCount = 0;
//        for(int i = blockIdx.x; i < rowNum; i += gridDim.x ) {
//            auto pivotValue = T(0.0);
//            auto pivot = -1;
//
//            for(int rowCounter = i; rowCounter < rowNum; rowCounter++ ) {
//                Nd4jLong rowCoord[] = {rowCounter, i};
//                auto rowPos = shape::getOffset(0, shape::shapeOf(compoundShape), shape::stride(compoundShape), rowCoord, 2);
//                if(nd4j::math::nd4j_abs(compound[rowPos]) > pivotValue ) {
//                    pivotValue = nd4j::math::nd4j_abs(compound[rowPos]);
//                    pivot = rowCounter;
//                }
//            }
//
//            if( pivotValue != T(0.0) ) {
//                swapRows_<T>(compound, compoundShape, pivot, i, rowNum);
//                swapRows_<T>(permutation, permutationShape, pivot, i, rowNum);
//                if (pivot != i)
//                    swapCount++;
//
//                for( int j = i + 1; j < rowNum; j++ ) {
//                    Nd4jLong posJIbuf[] = {j, i};
//                    Nd4jLong posIIbuf[] = {i, i};
//                    auto posJI = shape::getOffset(0, shape::shapeOf(compoundShape), shape::stride(compoundShape), posJIbuf, 2);
//                    auto posII = shape::getOffset(0, shape::shapeOf(compoundShape), shape::stride(compoundShape), posIIbuf, 2);
//
//                    compound[posJI] /= compound[posII];
//                    for( int k = i + 1; k < rowNum; k++ ) {
//                        Nd4jLong posJKbuf[] = {j, k};
//                        Nd4jLong posIKbuf[] = {i, k};
//                        auto posJK = shape::getOffset(0, shape::shapeOf(compoundShape), shape::stride(compoundShape), posJKbuf, 2);
//                        auto posIK = shape::getOffset(0, shape::shapeOf(compoundShape), shape::stride(compoundShape), posIKbuf, 2);
//                        T arg = compound[posJI] * compound[posIK];
//                        compound[posJK] -= arg;
//                    }
//                }
//            }
//        }
//    }

    template <typename T, typename F>
    static __global__ void determinantKernel(T* compound, T* result, Nd4jLong len) {
        __shared__ F tempRes;
        if (blockIdx.x == 0) {
            tempRes = (F)result[0];
        }
        __syncthreads();

        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;
        for (auto i = start; i < len; i += step) {
            auto pos = i * len + i; //shape::getOffset(0, shape::shapeOf(shape), shape::stride(shape), di, 2);
            math::atomics::nd4j_atomicMul<F>(&tempRes, (F)compound[pos]);
        }
        __syncthreads();

        if (blockIdx.x == 0) {
            result[0] = (T)tempRes;
        }
    }

        template <typename T, typename F>
        static __global__ void determinantLogKernel(T* compound, T* result, Nd4jLong len) {
            __shared__ F tempRes;
            if (blockIdx.x == 0) {
                tempRes = (F)result[0];
            }
            __syncthreads();

            auto start = blockIdx.x * blockDim.x + threadIdx.x;
            auto step = blockDim.x * gridDim.x;
            for (auto i = start; i < len; i += step) {
                auto pos = i * len + i; //shape::getOffset(0, shape::shapeOf(shape), shape::stride(shape), di, 2);
                math::atomics::nd4j_atomicMul<F>(&tempRes, (F)compound[pos]);
            }
            __syncthreads();

            if (blockIdx.x == 0) {
                result[0] = (T)math::nd4j_log<F,F>(math::nd4j_abs(tempRes));
            }
        }

    template <typename T, typename F>
    static __global__ void fillMatrix(void* output, Nd4jLong* outShape, void* input, Nd4jLong* inputShape, Nd4jLong pos, Nd4jLong rowLen) {
        __shared__ F* matrix;
        __shared__ T* inputBuf;
        __shared__ Nd4jLong inputLen;
        __shared__ Nd4jLong n2;

        if (threadIdx.x == 0) {
            matrix = reinterpret_cast<F*>(output);
            inputBuf = reinterpret_cast<T*>(input);
            inputLen = shape::length(inputShape);
            n2 = rowLen * rowLen;
        }
        __syncthreads();
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (int k = pos + start, j = start; j < n2; k += step, j += step) {
            auto xIndex = shape::getIndexOffset(k, inputShape, inputLen);
            matrix[j] = (F)inputBuf[xIndex];
        }
    }

    template <typename T, typename F>
    static __global__ void returnMatrix(void* output, Nd4jLong* outputShape, void* input, Nd4jLong* inputShape, Nd4jLong pos, Nd4jLong rowLen) {
        __shared__ F* matrix;
        __shared__ T* outputBuf;
        __shared__ Nd4jLong outputLen;
        __shared__ Nd4jLong n2;

        if (threadIdx.x == 0) {
            matrix = reinterpret_cast<F*>(input);
            outputBuf = reinterpret_cast<T*>(output);
            outputLen = shape::length(inputShape);
            n2 = rowLen * rowLen;
        }
        __syncthreads();
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (int k = pos + start, j = start; j < n2; k += step, j += step) {
            auto zIndex = shape::getIndexOffset(k, outputShape, outputLen);
            outputBuf[zIndex] = (T)matrix[j];
        }
    }

    template <typename F>
    static __global__ void fillUpPermutation(void* output, Nd4jLong* shape, int* source, int rowNum) {
        __shared__ F* permutation;

        if (threadIdx.x == 0) {
            permutation = reinterpret_cast<F*>(output);
        }
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;
        for (auto i = start; i < rowNum; i += step) {
            int val = source[i] - 1;
            Nd4jLong posF[] = {i, val};
            auto pos = shape::getOffset(0, shape::shapeOf(shape), shape::stride(shape), posF, 2);
            permutation[pos] = F(1.f);
        }
    }

    template <typename T>
    static void lup_(LaunchContext* context, NDArray* input, NDArray* compound, NDArray* permutation) {
        auto stream = context->getCudaStream();
        auto n = input->rows();
        cusolverDnHandle_t cusolverH = nullptr;
        cusolverStatus_t status = cusolverDnCreate(&cusolverH);
        if (CUSOLVER_STATUS_SUCCESS != status) {
            throw cuda_exception::build("Cannot create cuSolver handle", status);
        }
        status = cusolverDnSetStream(cusolverH, *stream);
        if (CUSOLVER_STATUS_SUCCESS != status) {
            throw cuda_exception::build("Cannot set up stream for cuda solver", status);
        }
        int lwork = 0;
        int *d_info = nullptr;

        auto err = cudaMalloc((void **) &d_info, sizeof(int));
        if (err) {
            throw cuda_exception::build("helpers::lup_: Cannot allocate memory for solver info buffer", err);
        }

        DataType dtype = input->dataType();
        switch(dtype) {

            case DataType::DOUBLE: {
                double *d_work = nullptr;
                err = cudaMalloc((void **) &d_work, sizeof(float) * lwork);
                if (err) {
                    throw cuda_exception::build("helpers::lup_: Cannot allocate memory for solver data buffer", err);
                }
                double *matrix = reinterpret_cast<double*>(input->specialBuffer());
                status = cusolverDnDgetrf_bufferSize(
                        cusolverH,
                        n,
                        n,
                        matrix,
                        n,
                        &lwork);
                if (CUSOLVER_STATUS_SUCCESS != status) {
                    throw cuda_exception::build("helpers::lup_: Cannot create cuSolver handle", status);
                }
                if (permutation == nullptr)
                    status = cusolverDnDgetrf(
                            cusolverH,
                            n,
                            n,
                            matrix,
                            n,
                            d_work,
                            nullptr,
                            d_info);
                else {
                    NDArray permutVector('c', {n}, nd4j::DataType::INT32, context);
                    int *permutationBuf = reinterpret_cast<int *>(permutVector.specialBuffer());
                    status = cusolverDnDgetrf(
                            cusolverH,
                            n,
                            n,
                            matrix,
                            n,
                            d_work,
                            permutationBuf,
                            d_info);
                    fillUpPermutation<double><<<n, n, 128, *stream>>>(permutation->specialBuffer(), permutation->specialShapeInfo(), permutationBuf, n);
                    permutation->tickWriteDevice();
                }
                err = cudaFree(d_work);
                if (err) {
                    throw cuda_exception::build("helpers::lup_: Cannot deallocate memory for solver data buffer", err);
                }
            }
                break;
            case DataType::FLOAT32: {
                float *matrix = reinterpret_cast<float*>(input->specialBuffer());
                float *d_work = nullptr;
                err = cudaMalloc((void **) &d_work, sizeof(float) * lwork);
                if (err) {
                    throw cuda_exception::build("helpers::lup_: Cannot allocate memory for solver data buffer", err);
                }

                status = cusolverDnSgetrf_bufferSize(
                        cusolverH,
                        n,
                        n,
                        matrix,
                        n,
                        &lwork);
                if (CUSOLVER_STATUS_SUCCESS != status) {
                    throw cuda_exception::build("helpers::lup_: Cannot create cuSolver handle", status);
                }

                if (permutation == nullptr)
                    status = cusolverDnSgetrf(
                            cusolverH,
                            n,
                            n,
                            matrix,
                            n,
                            d_work,
                            nullptr,
                            d_info);
                else {
                    NDArray permutVector('c', {n}, nd4j::DataType::INT32, context);
                    int *permutationBuf = reinterpret_cast<int *>(permutVector.specialBuffer());
                    status = cusolverDnSgetrf(
                            cusolverH,
                            n,
                            n,
                            matrix,
                            n,
                            d_work,
                            permutationBuf,
                            d_info);
                    fillUpPermutation<T><<<n, n, 128, *stream>>>(permutation->specialBuffer(), permutation->specialShapeInfo(), permutationBuf, n);
                    permutation->tickWriteDevice();
                }
                err = cudaFree(d_work);
                if (err) {
                    throw cuda_exception::build("helpers::lup_: Cannot deallocate memory for solver data buffer", err);
                }

            }
        }
        if (CUSOLVER_STATUS_SUCCESS != status) {
            throw cuda_exception::build("helpers::lup_: Cannot make LU decomposition", status);
        }
        err = cudaFree(d_info);
        if (err) {
            throw cuda_exception::build("helpers::lup_: Cannot deallocate memory for solver info buffer", err);
        }
        cusolverDnDestroy(cusolverH);
//        NDArray::registerSpecialUse({input}, {input});
        input->tickWriteDevice();
    }
    BUILD_SINGLE_TEMPLATE(template void lup_, (LaunchContext* context, NDArray* input, NDArray* output, NDArray* permutation), FLOAT_NATIVE);

    template <typename T>
    static int determinant_(nd4j::LaunchContext* context, NDArray* input, NDArray* output) {
        Nd4jLong n = input->sizeAt(-1);
        Nd4jLong n2 = n * n;
        std::vector<int> dims();
        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), {input->rankOf() - 2, input->rankOf() - 1});
        //auto packZ = ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), {output->rankOf() - 1});
        DataType dtype = input->dataType();
        if (dtype != DataType::DOUBLE)
            dtype = DataType::FLOAT32;

        auto matrix = NDArrayFactory::create(input->ordering(), {n, n}, dtype, input->getContext()); //, block.getWorkspace());
        auto det = NDArrayFactory::create<T>(1);
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {input});
        dim3 launchDims(256, 256, 1024);
        output->assign(1.f);
        for (int e = 0; e < output->lengthOf(); e++) {
            Nd4jLong pos = e * n2;
//            if (matrix.dataType() == input->dataType())
                fillMatrix<T, T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(matrix.specialBuffer(), matrix.specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), pos, n);
//            else
//                fillMatrix<T, float><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(matrix.specialBuffer(), matrix.specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), pos, n);

//            if (matrix.dataType() == input->dataType())
              lup_<T>(context, &matrix, nullptr, nullptr);
//            else
//                lup_<float>(context, &matrix, nullptr, nullptr);
            auto offset = shape::getIndexOffset(e, output->shapeInfo(), output->lengthOf());
            auto inputBuf = reinterpret_cast<T*>(matrix.specialBuffer());
            auto outputBuf = reinterpret_cast<T*>(output->specialBuffer()) + offset;
//            if (matrix.dataType() == input->dataType())
            determinantKernel<T, T><<<launchDims.x, launchDims.y, launchDims.z, *stream >>> (inputBuf, outputBuf, n);
//            else
//                determinantKernel<T, float><<<launchDims.x, launchDims.y, launchDims.z, *stream >>> (inputBuf, outputBuf, n);
        }
        NDArray::registerSpecialUse({output}, {input});

        return Status::OK();
    }

    BUILD_SINGLE_TEMPLATE(template int determinant_, (nd4j::LaunchContext* context, NDArray* input, NDArray* output), FLOAT_NATIVE);

    int determinant(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return determinant_, (context, input, output), FLOAT_NATIVE);
    }

    template <typename T>
    int logAbsDeterminant_(LaunchContext* context, NDArray* input, NDArray* output) {

        Nd4jLong n = input->sizeAt(-1);
        Nd4jLong n2 = n * n;
        std::vector<int> dims();
        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), {input->rankOf() - 2, input->rankOf() - 1});
        //auto packZ = ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), {output->rankOf() - 1});
        DataType dtype = input->dataType();
        if (dtype != DataType::DOUBLE)
            dtype = DataType::FLOAT32;

        auto matrix = NDArrayFactory::create(input->ordering(), {n, n}, dtype, input->getContext()); //, block.getWorkspace());
        auto det = NDArrayFactory::create<T>(1);
        auto stream = context->getCudaStream();
        NDArray::prepareSpecialUse({output}, {input});
        dim3 launchDims(256, 256, 1024);
        output->assign(1.f);
        for (int e = 0; e < output->lengthOf(); e++) {
            Nd4jLong pos = e * n2;
//            if (matrix.dataType() == input->dataType())
            fillMatrix<T, T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(matrix.specialBuffer(), matrix.specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), pos, n);
//            else
//                fillMatrix<T, float><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(matrix.specialBuffer(), matrix.specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), pos, n);

//            if (matrix.dataType() == input->dataType())
                lup_<T>(context, &matrix, nullptr, nullptr);
//            else
//                lup_<float>(context, &matrix, nullptr, nullptr);
            auto offset = shape::getIndexOffset(e, output->shapeInfo(), output->lengthOf());
            auto inputBuf = reinterpret_cast<T*>(matrix.specialBuffer());
            auto outputBuf = reinterpret_cast<T*>(output->specialBuffer()) + offset;
//            if (matrix.dataType() == input->dataType())
                determinantLogKernel<T, T><<<launchDims.x, launchDims.y, launchDims.z, *stream >>> (inputBuf, outputBuf, n);
//            else
//                determinantLogKernel<T, float><<<launchDims.x, launchDims.y, launchDims.z, *stream >>> (inputBuf, outputBuf, n);
        }
        NDArray::registerSpecialUse({output}, {input});

        return Status::OK();

        return ND4J_STATUS_OK;
    }

    BUILD_SINGLE_TEMPLATE(template int logAbsDeterminant_, (LaunchContext* context, NDArray* input, NDArray* output), FLOAT_NATIVE);

    int logAbsDeterminant(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return logAbsDeterminant_, (context, input, output), FLOAT_NATIVE);
    }

    template <typename T>
    static __global__ void fillLowerUpperKernel(void* lowerBuf, Nd4jLong* lowerShape, void* upperBuf, Nd4jLong* upperShape, void* matrixBuf, Nd4jLong* matrixShape, Nd4jLong n) {

        __shared__ Nd4jLong* xShapeOf;
        __shared__ Nd4jLong* yShapeOf;
        __shared__ Nd4jLong* zShapeOf;
        __shared__ Nd4jLong* xStrideOf;
        __shared__ Nd4jLong* yStrideOf;
        __shared__ Nd4jLong* zStrideOf;
        __shared__ T* lowerMatrix;
        __shared__ T* upperMatrix;
        __shared__ T* matrix;

        if (threadIdx.x == 0) {
            xShapeOf = shape::shapeOf(lowerShape);
            xStrideOf = shape::stride(lowerShape);

            yShapeOf = shape::shapeOf(upperShape);
            yStrideOf = shape::stride(upperShape);

            zShapeOf = shape::shapeOf(matrixShape);
            zStrideOf = shape::stride(matrixShape);
            lowerMatrix = reinterpret_cast<T*>(lowerBuf);
            upperMatrix = reinterpret_cast<T*>(upperBuf);
            matrix = reinterpret_cast<T*>(matrixBuf);
        }
        __syncthreads();

        for (int k = blockIdx.x; k < n; k += gridDim.x) {  // and then put all values under main diagonal on to it
            for (int j = threadIdx.x; j < n; j += blockDim.x) {
                Nd4jLong posX[] = {k, j};
                Nd4jLong posD[] = {j, j};
                auto xPos = shape::getOffset(0, xShapeOf, xStrideOf, posX, 2);
                auto yPos = shape::getOffset(0, yShapeOf, yStrideOf, posX, 2);
                auto iPos = shape::getOffset(0, zShapeOf, zStrideOf, posX, 2);
                auto dPos = shape::getOffset(0, zShapeOf, zStrideOf, posD, 2);
                if (k >= j)
                    lowerMatrix[xPos] = matrix[iPos];//(k, j);
                else
                    upperMatrix[yPos] = matrix[iPos]; //k, j);
            }
        }
    }

    template <typename T>
    static int inverse_(nd4j::LaunchContext* context, NDArray* input, NDArray* output) {
        auto n = input->sizeAt(-1);
        auto n2 = n * n;
        auto dtype = input->dataType();
        if (dtype != DataType::DOUBLE)
            dtype = DataType::FLOAT32;
        NDArray matrix = NDArrayFactory::create('c', {n, n}, dtype, input->getContext());
        NDArray upper = NDArrayFactory::create('c', {n, n}, dtype, input->getContext());
        NDArray lower = NDArrayFactory::create('c', {n, n}, dtype, input->getContext());
        NDArray compound = NDArrayFactory::create('c', {n, n}, dtype, input->getContext());
        NDArray permutation = NDArrayFactory::create('c', {n, n}, dtype, input->getContext());
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), {input->rankOf() - 2, input->rankOf() - 1});
        auto packZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(output->getShapeInfo(), {output->rankOf() - 2, output->rankOf() - 1});
        auto stream = context->getCudaStream();

//        PRAGMA_OMP_PARALLEL_FOR
        for (auto i = 0LL; i < packX.numberOfTads(); i++) {
            fillMatrix<T, T><<<1, n2, 128, *stream>>>(matrix.specialBuffer(), matrix.specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), i * n2, n);
            matrix.tickWriteDevice();
            compound.assign(matrix);
            lup_<T>(context, &compound, nullptr, nullptr);
            fillLowerUpperKernel<T><<<n, n, 128>>>(lower.specialBuffer(), lower.specialShapeInfo(), upper.specialBuffer(), upper.specialShapeInfo(), compound.specialBuffer(), compound.specialShapeInfo(), n);
            matrix.assign(0);
            invertUpperMatrix(&upper, &matrix); // U^{-1}
            compound.assign(0);
            invertLowerMatrix(&lower, &compound); // L{-1}

            nd4j::MmulHelper::mmul(&matrix, &compound, &upper, 1.0, 0.0);
            returnMatrix<T, T><<<1, n2, 128, *stream>>>(output->specialBuffer(), output->specialShapeInfo(), upper.specialBuffer(), upper.specialShapeInfo(), i * n2, n);
        }
        return Status::OK();
    }

    int inverse(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return inverse_, (context, input, output), FLOAT_NATIVE);
    }

    bool checkCholeskyInput(nd4j::LaunchContext * context, NDArray const* input) {
        return true;
    }

    template <typename F>
    __global__ void fillBatchKernel(F** dArrayBatch, F* buf, Nd4jLong* offsets, Nd4jLong batchSize) {
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (auto i = start; i < batchSize; i += step) {
            dArrayBatch[i] = buf + offsets[i];
        }
    }

    template <typename F>
    __global__ void adjustResultsKernel(F* dArray, Nd4jLong* shape, Nd4jLong* offsets, Nd4jLong batchSize, Nd4jLong n) {
        //auto i = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ Nd4jLong* shapeOf;
        __shared__ Nd4jLong* strideOf;
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            shapeOf = shape::shapeOf(shape);
            strideOf = shape::stride(shape);
        }
        __syncthreads();

        for (auto i = blockIdx.x; i < batchSize; i+= gridDim.x) {
            auto current = dArray + offsets[i];
            for (auto r = threadIdx.x; r < n; r += blockDim.x) {
                for (auto c = r + 1; c < n; c++) {
                    Nd4jLong posRC[] = {r, c};
                    auto pos = r * n + c; //shape::getOffset(0, shapeOf, strideOf, posRC, 2);
                    current[pos] = 0.;
                }
            }
        }
    }

    template <typename F>
    int cholesky__(LaunchContext* context, NDArray* input, NDArray* output, bool inplace) {
        if (!inplace)
            output->assign(input);
        std::unique_ptr<NDArray> tempOutput(output->dup());
        cusolverDnHandle_t handle = nullptr;
        auto n = input->sizeAt(-1);
        auto n2 = n * n;
        NDArray::prepareSpecialUse({output}, {input});
        auto status = cusolverDnCreate(&handle);
        if (CUSOLVER_STATUS_SUCCESS != status) {
            throw cuda_exception::build("helpers::cholesky_: Cannot create solver handle", status);
        }
        F** dArrayBatch = nullptr;
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(tempOutput->getShapeInfo(), {tempOutput->rankOf() - 2, tempOutput->rankOf() - 1});
        const Nd4jLong batchSize = packX.numberOfTads();
        int* dInfoArray = nullptr;
        auto err = cudaMalloc((void**)&dArrayBatch, sizeof(F*) * batchSize);
        if (err) {
            throw cuda_exception::build("helpers::cholesky_: Cannot allocate memory for solver batch data buffer", err);
        }
        err = cudaMalloc ((void**)&dInfoArray, sizeof(int) * batchSize);
        if (err) {
            throw cuda_exception::build("helpers::cholesky_: Cannot allocate memory for solver errors buffer", err);
        }
        auto stream = context->getCudaStream();
        fillBatchKernel<F><<<1, batchSize, 128, *stream>>>(dArrayBatch, reinterpret_cast<F*>(tempOutput->specialBuffer()), packX.specialOffsets(), batchSize);

        status = cusolverDnSetStream(handle, *stream);
        if (CUSOLVER_STATUS_SUCCESS != status) {
            throw cuda_exception::build("helpers::cholesky_: Cannot set stream to solver handle", status);
        }
        const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
        if (input->dataType() == DataType::DOUBLE)
        status = cusolverDnDpotrfBatched(
                handle,
                uplo,
                n,
                (double**)dArrayBatch,
                n,
                dInfoArray,
                batchSize);
        else
        status = cusolverDnSpotrfBatched(
                    handle,
                    uplo,
                    n,
                    (float**)dArrayBatch,
                    n,
                    dInfoArray,
                    batchSize);

        if (CUSOLVER_STATUS_SUCCESS != status) {
            throw cuda_exception::build("helpers::cholesky_: Cholesky factorization failed for batch", status);
        }
        adjustResultsKernel<F><<<batchSize, n2, 128, *stream>>>(reinterpret_cast<F*>(tempOutput->specialBuffer()), packX.specialShapeInfo(), packX.specialOffsets(), batchSize, n);

        err = cudaFree(dArrayBatch);
        if (err) {
            throw cuda_exception::build("helpers::cholesky_: Cannot deallocate memory for solver batch data buffer", err);
        }
        err = cudaFree(dInfoArray);
        if (err) {
            throw cuda_exception::build("helpers::cholesky_: Cannot allocate memory for solver errors buffer", err);
        }

        if(!inplace)
            output->assign(tempOutput.get());
        else
            input->assign(tempOutput.get());

        NDArray::registerSpecialUse({output}, {input});
        return Status::OK();
    }

//    template <typename T>
    int cholesky_(LaunchContext* context, NDArray* input, NDArray* output, bool inplace) {
        NDArray::prepareSpecialUse({output}, {input});
        if (input->dataType() == DataType::DOUBLE)
            cholesky__<double>(context, input, output, inplace);
        else if (input->dataType() == DataType::FLOAT32)
            cholesky__<float>(context, input, output, inplace);
        else {
            std::unique_ptr<NDArray> tempOutput(NDArrayFactory::create_('c', input->getShapeAsVector(), DataType::FLOAT32, input->getContext()));
            tempOutput->assign(input);
            cholesky__<float>(context, tempOutput.get(), tempOutput.get(), true);
            output->assign(tempOutput.get());
        }
        NDArray::registerSpecialUse({output}, {input});
        return Status::OK();
    }

    int cholesky(nd4j::LaunchContext* context, NDArray* input, NDArray* output, bool inplace) {
//        BUILD_SINGLE_SELECTOR(input->dataType(), return cholesky_, (context, input, output, inplace), FLOAT_TYPES);
        return cholesky_(context, input, output, inplace);
    }
//    BUILD_SINGLE_TEMPLATE(template int cholesky_, (LaunchContext* context, NDArray* input, NDArray* output, bool inplace), FLOAT_TYPES);
    BUILD_SINGLE_TEMPLATE(template int inverse_, (nd4j::LaunchContext* context, NDArray* input, NDArray* output), FLOAT_NATIVE);

    __global__ void logDetKernel(double* inputBuf, Nd4jLong* inputShape, Nd4jLong batchNum, Nd4jLong* tadShape, Nd4jLong* tadOffsets, double* outputBuf, Nd4jLong* outputShape) {

        __shared__ int n;
        if (threadIdx.x == 0) {
            n = shape::sizeAt(inputShape, -1); // * shape::sizeAt(inputShape, -1);
        }
        __syncthreads();

        double* output = outputBuf;
        double* input = inputBuf;

        for (auto i = blockIdx.x; i < batchNum; i += gridDim.x) {
            double* current = input + tadOffsets[i];
            Nd4jLong* shapeOf = shape::shapeOf(tadShape);
            Nd4jLong* strideOf = shape::stride(tadShape);
            auto zIndex = shape::getIndexOffset(i, outputShape, batchNum);
            for (auto e = threadIdx.x; e < n; e += blockDim.x) {
                Nd4jLong diag[] = {e, e};
                auto xIndex = shape::getOffset(0, shapeOf, strideOf, diag, 2);
                math::atomics::nd4j_atomicAdd(&output[zIndex], math::nd4j_log<double,double>(current[xIndex] * current[xIndex]));
            }
        }
    }

    int logdetFunctor(nd4j::LaunchContext* context, NDArray* input, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {input});
        auto n2 = input->sizeAt(-1) * input->sizeAt(-2);
        auto stream = context->getCudaStream();
        std::unique_ptr<NDArray> tempOutput(input->dup());
//        auto inputs = tempOutput->allTensorsAlongDimension({input->rankOf() - 2, input->rankOf() - 1});
//        for (Nd4jLong e = 0; e < packX.numberOfTads(); e++) {
//            auto subArray = inputs->at(e);
//            cholesky(context, subArray, subArray, true);
//        }
//        delete inputs;
        cholesky(context, input, tempOutput.get(), false);
        tempOutput->syncToHost();
        tempOutput->printIndexedBuffer("Cholesky res!!!");
        auto outputBuf = reinterpret_cast<double*>(output->specialBuffer()); // + e * n2; // + e * n2;
        auto inputBuf = reinterpret_cast<double*>(tempOutput->specialBuffer());
        output->assign(0);
        output->syncToDevice();
        auto packX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(tempOutput->getShapeInfo(), {input->rankOf() - 2, input->rankOf() - 1});
        logDetKernel<<<packX.numberOfTads(), n2, 128, *stream>>>(inputBuf, tempOutput->specialShapeInfo(), packX.numberOfTads(), packX.specialShapeInfo(), packX.specialOffsets(), outputBuf, output->specialShapeInfo());
//        }
        NDArray::registerSpecialUse({output}, {input});
        //delete tempOutput;
        return Status::OK();
    }
}
}
}
