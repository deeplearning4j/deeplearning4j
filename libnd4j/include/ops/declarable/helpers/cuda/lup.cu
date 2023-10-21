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
//  @author raver119@gmail.com
//
#include <array/NDArrayFactory.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/MmulHelper.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/helpers/top_k.h>
#include <cusolverDn.h>
#include <exceptions/cuda_exception.h>
#include <execution/cuda/LaunchDims.h>

namespace sd {
    namespace ops {
        namespace helpers {

// ------------------------------------------------------------------------------------------------------------------ //
//  invert the second diagonal for lower diagonal matrix
            template <typename T>
            static SD_KERNEL void invertKernelLow(void *invertedBuf, const sd::LongType *invertedShape, const void *inputBuf,
                                                  const sd::LongType *inputShape, sd::LongType n) {
                auto inverted = reinterpret_cast<T *>(invertedBuf);
                auto input = reinterpret_cast<const T *>(inputBuf);

                auto start = threadIdx.x + blockIdx.x * blockDim.x;
                auto step = blockDim.x * gridDim.x;

                for (int i = start + 1; i < n; i += step) {
                    sd::LongType pos[] = {i, i - 1};
                    sd::LongType posX[] = {i, i};
                    sd::LongType posY[] = {i - 1, i - 1};
                    auto xIndex = shape::getOffset(inputShape, pos);
                    auto dxIndex = shape::getOffset(inputShape, posX);
                    auto dyIndex = shape::getOffset(inputShape, posY);
                    auto zIndex = shape::getOffset(invertedShape, pos);
                    // invert lower triangular matrix
                    inverted[zIndex] = -input[xIndex] / (input[dxIndex] * input[dyIndex]);
                }
            }
// ------------------------------------------------------------------------------------------------------------------ //
// invert diagonal vals to upper diagonal matrix
            template <typename T>
            static SD_KERNEL void upvertKernel(void *invertedBuf, const sd::LongType *invertedShape, const void *inputBuf,
                                               const sd::LongType *inputShape, sd::LongType n) {
                auto inverted = reinterpret_cast<T *>(invertedBuf);
                auto input = reinterpret_cast<const T *>(inputBuf);

                auto start = threadIdx.x + blockIdx.x * blockDim.x;
                auto step = blockDim.x * gridDim.x;

                for (int i = start; i < n; i += step) {
                    sd::LongType pos[] = {i, i};
                    auto xIndex = shape::getOffset(inputShape, pos);
                    auto zIndex = shape::getOffset(invertedShape, pos);

                    // invert diagonal elements
                    inverted[zIndex] /= input[xIndex];
                }
            }

// ------------------------------------------------------------------------------------------------------------------ //
//  invert upper second diagonal
            template <typename T>
            static SD_KERNEL void upvertKernelUp(void *invertedBuf, const sd::LongType *invertedShape, const void *inputBuf,
                                                 const sd::LongType *inputShape, sd::LongType n) {
                __shared__ T *inverted;
                __shared__ const T *input;
                if (threadIdx.x == 0) {
                    inverted = reinterpret_cast<T *>(invertedBuf);
                    input = reinterpret_cast<const T *>(inputBuf);
                }
                __syncthreads();

                auto start = threadIdx.x + blockIdx.x * blockDim.x;
                auto step = blockDim.x * gridDim.x;

                for (int i = start; i < n - 1; i += step) {
                    sd::LongType pos[] = {i, i + 1};
                    sd::LongType posX[] = {i + 1, i + 1};
                    auto xIndex = shape::getOffset(inputShape, pos);
                    auto iIndex = shape::getOffset(invertedShape, posX);
                    auto zIndex = shape::getOffset(invertedShape, pos);
                    // invert upper matrix
                    math::atomics::sd_atomicAdd(&inverted[zIndex], -input[xIndex] * inverted[iIndex]);  // / input[yIndex]);
                }
            }

// ------------------------------------------------------------------------------------------------------------------ //
            template <typename T>
            static SD_KERNEL void invertLowKernel(void *invertedBuf, const sd::LongType *invertedShape, const void *inputBuf,
                                                  const sd::LongType *inputShape, sd::LongType n) {
                auto input = reinterpret_cast<const T *>(inputBuf);
                auto inverted = reinterpret_cast<T *>(invertedBuf);

                auto tid = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = gridDim.x * blockDim.x;

                for (int i = tid + 2; i < n; i += step) {
                    for (int j = i - 2; j >= 0; --j)
                        for (int k = 0; k < i; k++) {
                            sd::LongType posZ[] = {i, j};
                            sd::LongType posY[] = {k, j};
                            sd::LongType posX[] = {i, k};
                            sd::LongType posD[] = {i, i};

                            auto xIndex = shape::getOffset(inputShape, posX);
                            auto yIndex = shape::getOffset(invertedShape, posY);
                            auto dIndex = shape::getOffset(inputShape, posD);
                            auto zIndex = shape::getOffset(invertedShape, posZ);
                            // invert non-diagonal elements
                            math::atomics::sd_atomicAdd(&inverted[zIndex], -inverted[yIndex] * input[xIndex] / input[dIndex]);
                        }
                }
            }

// ------------------------------------------------------------------------------------------------------------------ //
// Invertion of upper triangular matrix non-diagonal elements when main and second diagonals already processed
            template <typename T>
            static SD_KERNEL void invertUpKernel(void *invertedBuf, const sd::LongType *invertedShape, const void *inputBuf,
                                                 const sd::LongType *inputShape, sd::LongType n) {
                auto inverted = reinterpret_cast<T *>(invertedBuf);
                auto input = reinterpret_cast<const T *>(inputBuf);

                auto tid = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;

                for (int i = (int)n - tid - 2; i >= 0; i -= step) {
                    for (int j = i + 2; j < (int)n; j++)
                        for (int k = i; k < (int)n; k++) {
                            sd::LongType posZ[] = {i, j};
                            sd::LongType posY[] = {k, j};
                            sd::LongType posX[] = {i, k};
                            // inversion with Joardan Gauss transformation
                            auto xIndex = shape::getOffset(inputShape, posX);
                            auto yIndex = shape::getOffset(invertedShape, posY);
                            auto zIndex = shape::getOffset(invertedShape, posZ);
                            // invert upper non-diagonal elements
                            math::atomics::sd_atomicAdd(&inverted[zIndex], -inverted[yIndex] * input[xIndex]);
                        }
                }
            }

// ------------------------------------------------------------------------------------------------------------------ //
// procedure to invert lower-triangular matrix.
// In current case lower triangular matrix has main diagonal with general values
//
            template <typename T>
            static void invertLowerMatrix_(LaunchContext *context, NDArray *inputMatrix, NDArray *invertedMatrix) {
                int n = inputMatrix->rows();
                invertedMatrix->setIdentity();

                if (inputMatrix->isIdentityMatrix()) return;

                auto stream = context->getCudaStream();

                dim3 lupLaunch = lupDims(n);
                dim3 lupLaunchLow = lupDimsLow(n);
                // invert lower matrix
                // invert main diagonal
                upvertKernel<T><<<lupLaunch.y,lupLaunch.x, lupLaunch.z, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(),
                                                                                   inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
                // invert the second diagonal
                invertKernelLow<T><<<lupLaunch.y, lupLaunch.x, lupLaunch.z, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(),
                                                                                       inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
                // invert non-diagonal elements
                invertLowKernel<T><<<lupLaunchLow.y, lupLaunchLow.x, lupLaunchLow.z, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(),
                                                                                                inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
            }

// ------------------------------------------------------------------------------------------------------------------ //
// caller for invert lower matrix routine
            void invertLowerMatrix(LaunchContext *context, NDArray *inputMatrix, NDArray *invertedMatrix) {
                NDArray::prepareSpecialUse({invertedMatrix}, {inputMatrix});
                BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), invertLowerMatrix_, (context, inputMatrix, invertedMatrix),
                                      SD_FLOAT_NATIVE);
                NDArray::registerSpecialUse({invertedMatrix}, {inputMatrix});
            }

// ------------------------------------------------------------------------------------------------------------------ //
// procedure to invert upper-triangular matrix.
// In current case upper triangular matrix has main diagonal with all ones on it.
            template <typename T>
            static void invertUpperMatrix_(LaunchContext *context, NDArray *inputMatrix, NDArray *invertedMatrix) {
                int n = inputMatrix->rows();
                invertedMatrix->setIdentity();
                auto stream = context->getCudaStream();
                if (inputMatrix->isIdentityMatrix()) {  // the inverse for I is I
                    return;
                }

                // invert upper matrix
                // invert the second diagonal
                upvertKernelUp<T><<<1, n, 512, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(),
                                                          inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);

                // invert other elements
                invertUpKernel<T><<<n, n, 512, *stream>>>(invertedMatrix->specialBuffer(), invertedMatrix->specialShapeInfo(),
                                                          inputMatrix->specialBuffer(), inputMatrix->specialShapeInfo(), n);
            }

// ------------------------------------------------------------------------------------------------------------------ //
//  invertion of upper triangular matrix - runner routine
            void invertUpperMatrix(LaunchContext *context, NDArray *inputMatrix, NDArray *invertedMatrix) {
                NDArray::prepareSpecialUse({invertedMatrix}, {inputMatrix});
                BUILD_SINGLE_SELECTOR(invertedMatrix->dataType(), invertUpperMatrix_, (context, inputMatrix, invertedMatrix),
                                      SD_FLOAT_NATIVE);
                NDArray::prepareSpecialUse({invertedMatrix}, {inputMatrix});
            }

// ------------------------------------------------------------------------------------------------------------------ //
// determinant kernel - accumulation product of all values on the main diagonal
            template <typename T>
            static SD_KERNEL void determinantKernel(T *compound, T *result, sd::LongType len) {
                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;
                for (auto i = start; i < len; i += step) {
                    auto pos = i * len + i;  // shape::getOffset(0, shape::shapeOf(shape), shape::stride(shape), di, 2);
                    // multiply all diagonal elements
                    math::atomics::sd_atomicMul(&result[0], compound[pos]);
                }
            }

// ------------------------------------------------------------------------------------------------------------------ //
// determinant logarithm - accumulation sum of all logarithm values on the main diagonal. All in logarithic values
// should be positive
            template <typename T>
            static SD_KERNEL void determinantLogKernel(T *compound, T *result, sd::LongType len) {
                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;
                for (auto i = start; i < len; i += step) {
                    auto pos = i * len + i;  // shape::getOffset(0, shape::shapeOf(shape), shape::stride(shape), di, 2);
                    // sum logs of all diagonal elements
                    math::atomics::sd_atomicAdd(result, math::sd_log<T, T>(math::sd_abs(compound[pos])));
                }
            }

// ------------------------------------------------------------------------------------------------------------------ //
// kernel to copy matrix with given shape to compound tensor with given pos
// output - a N-D tensor buffer with rank not less than 2, input - 2D square n x n matrix with n = rowLen
            template <typename T, typename F>
            static SD_KERNEL void fillMatrix(void *output, const sd::LongType *outShape, const void *input,
                                             const sd::LongType *inputShape, sd::LongType pos, sd::LongType rowLen) {
                __shared__ F *matrix;
                __shared__ const T *inputBuf;
                __shared__ sd::LongType inputLen;
                __shared__ sd::LongType n2;

                if (threadIdx.x == 0) {
                    matrix = reinterpret_cast<F *>(output);
                    inputBuf = reinterpret_cast<const T *>(input);
                    inputLen = shape::length(inputShape);
                    n2 = rowLen * rowLen;
                }
                __syncthreads();

                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;

                for (int k = pos + start, j = start; j < n2; k += step, j += step) {
                    auto xIndex = shape::getIndexOffset(k, inputShape);
                    matrix[j] = (F)inputBuf[xIndex];
                }
            }

// ------------------------------------------------------------------------------------------------------------------ //
// same as above, but without type conversion
            template <typename T>
            static SD_KERNEL void returnMatrix(void *output, const sd::LongType *outputShape, const void *input,
                                               const sd::LongType *inputShape, sd::LongType pos, sd::LongType rowLen) {
                __shared__ sd::LongType outputLen;
                __shared__ sd::LongType n2;
                auto matrix = reinterpret_cast<const T *>(input);
                auto outputBuf = reinterpret_cast<T *>(output);

                if (threadIdx.x == 0) {
                    outputLen = shape::length(inputShape);
                    n2 = rowLen * rowLen;
                }
                __syncthreads();
                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;

                for (int k = pos + start, j = start; j < n2; k += step, j += step) {
                    auto zIndex = shape::getIndexOffset(k, outputShape);
                    outputBuf[zIndex] = matrix[j];
                }
            }

// ------------------------------------------------------------------------------------------------------------------ //
// fill up permutaion matrix kernel. Permutation matrix filled with zeros and ones
            template <typename F>
            static SD_KERNEL void fillUpPermutation(void *output, const sd::LongType *shape, int *source, int rowNum) {
                F *permutation = reinterpret_cast<F *>(output);

                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;
                for (auto i = start; i < rowNum; i += step) {
                    int val = source[i] - 1;
                    sd::LongType posF[] = {i, val};
                    auto pos = shape::getOffset(shape, posF);
                    permutation[pos] = F(1.f);
                }
            }

// ------------------------------------------------------------------------------------------------------------------ //
// LUP decomposition runner - using CUBLAS SOLVER
// if permutation is given, then using LUP decomposition, LU decomposition otherwise
// L - lower triangular, U - upper triangular, P - permutation matrices
// PA = LU
//
// input - A matrix nxn
// compound - C matrix L + U - I, or main diagonal and lower - L matrix, from the 2nd diagonal - U matrix
            template <typename T, typename I>
            static void lup_(LaunchContext *context, NDArray *input, NDArray *compound, NDArray *permutation) {
                auto stream = context->getCudaStream();
                auto n = input->rows();
                std::lock_guard<std::mutex> lock(*LaunchContext::deviceMutex());

                cusolverDnHandle_t *cusolverH = (cusolverDnHandle_t *)context->getCusolverHandle();  // nullptr;
                // create solver handle
                cusolverStatus_t status;

                // set solver stream
                status = cusolverDnSetStream(*cusolverH, *stream);
                if (CUSOLVER_STATUS_SUCCESS != status) {
                    throw cuda_exception::build("Cannot set up stream for cuda solver", status);
                }
                int lwork = 0;
                int *d_info = nullptr;
                // allocate memory for permutation vector
                auto err = cudaMalloc((void **)&d_info, sizeof(sd::LongType));
                if (err) {
                    throw cuda_exception::build("helpers::lup_: Cannot allocate memory for solver info buffer", err);
                }

                DataType dtype = input->dataType();
                switch (dtype) {  // there are two implementations with cublas for LUP decomposition - double and float

                    case DataType::DOUBLE: {
                        double *d_work = nullptr;
                        // compute internal buffer size
                        double *matrix = reinterpret_cast<double *>(input->specialBuffer());
                        status = cusolverDnDgetrf_bufferSize(*cusolverH, n, n, matrix, n, &lwork);
                        if (CUSOLVER_STATUS_SUCCESS != status) {
                            throw cuda_exception::build("helpers::lup_: Cannot create cuSolver handle", status);
                        }

                        err = cudaMalloc((void **)&d_work, sizeof(float) * lwork);
                        if (err) {
                            throw cuda_exception::build("helpers::lup_: Cannot allocate memory for solver data buffer", err);
                        }

                        if (permutation == nullptr) {
                            status = cusolverDnDgetrf(*cusolverH, n, n, matrix, n, d_work, nullptr, d_info);

                            if (status != CUSOLVER_STATUS_SUCCESS) {
                                throw cuda_exception::build("helpers::lup_: LU factorization is failed due ", status);
                            }
                        } else {
                            NDArray permutVector('c', {n}, sd::DataType::INT32, context);
                            int *permutationBuf = permutVector.dataBuffer()->specialAsT<int>();
                            status = cusolverDnDgetrf(*cusolverH, n, n, matrix, n, d_work, permutationBuf, d_info);
                            if (status != CUSOLVER_STATUS_SUCCESS) {
                                throw cuda_exception::build("helpers::lup_: LU factorization is failed due ", status);
                            }

                            if (permutation->rankOf() == 2) {
                                fillUpPermutation<double><<<n, n, 1024, *stream>>>(permutation->specialBuffer(),
                                                                                   permutation->specialShapeInfo(), permutationBuf, n);
                            } else {
                                permutVector.tickWriteDevice();
                                input->tickWriteDevice();
                                compound->assign(input);
                                permutation->assign(permutVector);
                            }
                        }
                        err = cudaFree(d_work);
                        if (err) {
                            throw cuda_exception::build("helpers::lup_: Cannot deallocate memory for solver data buffer", err);
                        }
                    } break;
                    case DataType::FLOAT32: {
                        float *matrix = reinterpret_cast<float *>(input->specialBuffer());
                        float *d_work = nullptr;

                        status = cusolverDnSgetrf_bufferSize(*cusolverH, n, n, matrix, n, &lwork);
                        if (CUSOLVER_STATUS_SUCCESS != status) {
                            throw cuda_exception::build("helpers::lup_: Cannot create cuSolver handle", status);
                        }

                        err = cudaMalloc((void **)&d_work, sizeof(float) * lwork);
                        if (err) {
                            throw cuda_exception::build("helpers::lup_: Cannot allocate memory for solver data buffer", err);
                        }

                        if (permutation == nullptr)
                            status = cusolverDnSgetrf(*cusolverH, n, n, matrix, n, d_work, nullptr, d_info);
                        else {
                            NDArray permutVector('c', {n}, DataType::INT32, context);
                            int *permutationBuf = reinterpret_cast<int *>(permutVector.specialBuffer());
                            status = cusolverDnSgetrf(*cusolverH, n, n, matrix, n, d_work, permutationBuf, d_info);
                            if (permutation->rankOf() == 2) {
                                fillUpPermutation<I><<<n, n, 128, *stream>>>(permutation->specialBuffer(), permutation->specialShapeInfo(),
                                                                             permutationBuf, n);
                                permutation->tickWriteDevice();
                            } else {
                                input->tickWriteDevice();
                                compound->assign(input);
                                permutation->assign(permutVector);
                            }
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

                input->tickWriteDevice();
            }
// ------------------------------------------------------------------------------------------------------------------ //

            BUILD_DOUBLE_TEMPLATE(template void lup_,
                                  (LaunchContext * context, NDArray *input, NDArray *output, NDArray *permutation), SD_FLOAT_NATIVE,
                                  SD_INDEXING_TYPES);

            template <typename T>
            static SD_DEVICE void swapRows(T *matrix, const sd::LongType *shape, sd::LongType theFirst, sd::LongType theSecond, sd::LongType n) {
                if (theFirst != theSecond) {
                    for (auto i = 0; i < n; i++) {
                        sd::LongType theFirstPos[] = {theFirst, i};
                        sd::LongType theSecondPos[] = {theSecond, i};
                        auto theFirstIndex = shape::getOffset(shape, theFirstPos, 0);
                        auto theSecondIndex = shape::getOffset(shape, theSecondPos, 0);
                        math::sd_swap(matrix[theFirstIndex], matrix[theSecondIndex]);
                    }
                }

                __syncthreads();


            }



            template <typename T>
            static SD_DEVICE void processColumns(
                    sd::LongType currentRow,
                    sd::LongType rowNum,
                    T *compoundBuf,
                    const sd::LongType *compoundShape) {


                sd::LongType xDiag[] = {currentRow, currentRow};
                auto diagIndex = shape::getOffset(compoundShape, xDiag, 0);
                printf("Diagonal value before operation: %f\n", compoundBuf[diagIndex]);

                // Guard against zero division
                for (auto j = currentRow + 1; j < rowNum; j++) {
                    sd::LongType xRow[] = {j, currentRow};
                    auto rowIndex = shape::getOffset(compoundShape, xRow, 0);

                    compoundBuf[rowIndex] /= compoundBuf[diagIndex];

                    for (auto k = currentRow + 1; k < rowNum; k++) {
                        sd::LongType yRow[] = {j, k};
                        sd::LongType yCol[] = {currentRow, k};
                        auto rowIndexY = shape::getOffset(compoundShape, yRow, 0);
                        auto colIndex = shape::getOffset(compoundShape, yCol, 0);
                        compoundBuf[rowIndexY] -= compoundBuf[rowIndex] * compoundBuf[colIndex];
                    }
                }

            }


            template <typename T>
            SD_DEVICE sd::LongType argmaxCol(sd::LongType column, T *compoundBuffer, const sd::LongType *compoundShape) {
                auto rowNum = shape::sizeAt(compoundShape, 0);
                auto maxValue = T(0);
                auto result = -1LL;

                for (auto rowCounter = column; rowCounter < rowNum; rowCounter++) {
                    sd::LongType xPos[] = {rowCounter, column};
                    auto xIndex = shape::getOffset(compoundShape, xPos, 0);
                    if (sd::math::sd_abs(compoundBuffer[xIndex]) > maxValue) {
                        maxValue = sd::math::sd_max(maxValue, sd::math::sd_abs(compoundBuffer[xIndex]));
                        result = rowCounter;
                    }
                }

                return result;
            }


            template <typename T, typename I>
            static SD_KERNEL void luNN_(
                T *outputBuf,
                const sd::LongType *outputShape,
                I *permutations,
                const sd::LongType *permuShape,
                const sd::LongType *outputTadShape,
                const sd::LongType *outputTadOffsets,
                const sd::LongType *permuTadShape,
                const sd::LongType *permuTadOffsets,
                sd::LongType batchNum) {

              auto start = blockIdx.x * blockDim.x + threadIdx.x;
              auto step = blockDim.x * gridDim.x;

              for (auto b = start; b < batchNum; b += step) {
                T *matrix = outputBuf + outputTadOffsets[b];
                I *permutation = permutations + permuTadOffsets[b];
                for (auto i = 0; i < batchNum - 1; i++) {
                  auto pivotIndex = argmaxCol(i, matrix, outputTadShape);
                  if (pivotIndex < 0) {
                    continue;
                  }
                  printf("Before swapping rows: Permutation at i: %d, at pivotIndex: %d\n", permutation[i], permutation[pivotIndex]);
                  swapRows(matrix, outputTadShape,i, pivotIndex, batchNum);
                  printf("After swapping rows: Permutation at i: %d, at pivotIndex: %d\n", permutation[i], permutation[pivotIndex]);

                  printf("Before processColumns: matrix[%d] = %f\n", i, matrix[i]);
                  processColumns(i, batchNum, matrix, outputTadShape);
                  printf("After processColumns: matrix[%d] = %f\n", i, matrix[i]);
                }
              }
            }



            template <typename T, typename I>
            static void lu_(LaunchContext *context,
                            NDArray *compound,
                            NDArray *output,
                            NDArray *permutationVectors) {
                auto n = compound->sizeAt(-1);
                auto stream = context->getCudaStream();
                permutationVectors->linspace(0);
                permutationVectors->syncToDevice();
                output->assign(compound);
                std::vector<sd::LongType> dims = {-2, -1};
                std::vector<sd::LongType> lastDim = {-1};
                auto tads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(),&dims);
                auto permutationTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), &lastDim);
                auto batchNum = compound->sizeAt(-1);
                dim3 lupDims = getLupDims(batchNum);
                luNN_<T, I><<<lupDims.x, lupDims.y, lupDims.z, *stream>>>(
                        reinterpret_cast<T *>(output->platformBuffer()),
                        output->specialShapeInfo(),
                        reinterpret_cast<I *>(permutationVectors->platformBuffer()), permutationVectors->specialShapeInfo(),
                        tads->specialShapeInfo(),
                        tads->specialOffsets(),
                        permutationTads->specialShapeInfo(),
                        permutationTads->specialOffsets(), batchNum);



            }

            void lu(LaunchContext *context, NDArray *input, NDArray *output, NDArray *permutations) {
                NDArray::prepareSpecialUse({output}, {input, permutations});
                BUILD_DOUBLE_SELECTOR(input->dataType(), permutations->dataType(),
                                      lu_, (context, input, output, permutations),
                                      SD_FLOAT_NATIVE, SD_INDEXING_TYPES);
                NDArray::registerSpecialUse({output}, {input, permutations});
            }
// ------------------------------------------------------------------------------------------------------------------ //
            template <typename T>
            static sd::Status determinant_(sd::LaunchContext *context, NDArray *input, NDArray *output) {
                sd::LongType n = input->sizeAt(-1);
                sd::LongType n2 = n * n;
                std::vector<sd::LongType> dims();
                std::vector<sd::LongType> dims2 = {input->rankOf() - 2, input->rankOf() - 1};

                auto matrix =
                        NDArrayFactory::create(input->ordering(), {n, n}, DataTypeUtils::fromT<T>(), context);  //, block.getWorkspace());
                auto det = NDArrayFactory::create<T>(1, context);
                auto stream = context->getCudaStream();
                NDArray::prepareSpecialUse({output}, {input});
                dim3 launchDims = getLaunchDims("logAbsDeterminant");
                output->assign(1.f);
                for (int e = 0; e < output->lengthOf(); e++) {
                    sd::LongType pos = e * n2;
                    fillMatrix<T, T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
                            matrix.specialBuffer(), matrix.specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), pos, n);

                    lup_<T, int>(context, &matrix, nullptr, nullptr);
                    auto offset = shape::getIndexOffset(e, output->shapeInfo());
                    auto inputBuf = reinterpret_cast<T *>(matrix.specialBuffer());
                    auto outputBuf = reinterpret_cast<T *>(output->specialBuffer()) + offset;
                    determinantKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuf, outputBuf, n);
                }
                NDArray::registerSpecialUse({output}, {input});

                return sd::Status::OK;
            }

            sd::Status determinant(sd::LaunchContext *context, NDArray *input, NDArray *output) {
                NDArray::prepareSpecialUse({output}, {input});
                BUILD_SINGLE_SELECTOR(input->dataType(), return determinant_, (context, input, output), SD_FLOAT_NATIVE);
                NDArray::registerSpecialUse({output}, {input});
            }

            template <typename T>
            sd::Status logAbsDeterminant_(LaunchContext *context, NDArray *input, NDArray *output) {
                sd::LongType n = input->sizeAt(-1);
                sd::LongType n2 = n * n;
                std::vector<sd::LongType> dims();
                std::vector<sd::LongType> dims2 = {input->rankOf() - 2, input->rankOf() - 1};
                DataType dtype = input->dataType();
                if (dtype != DataType::DOUBLE) dtype = DataType::FLOAT32;

                auto matrix = NDArrayFactory::create(input->ordering(), {n, n}, dtype, context);
                auto det = NDArrayFactory::create<T>(1, context);
                auto stream = context->getCudaStream();
                NDArray::prepareSpecialUse({output}, {input});
                dim3 launchDims = getLaunchDims("logAbsDeterminant");
                output->assign(0.f);
                for (int e = 0; e < output->lengthOf(); e++) {
                    sd::LongType pos = e * n2;
                    fillMatrix<T, T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
                            matrix.specialBuffer(), matrix.specialShapeInfo(), input->specialBuffer(), input->specialShapeInfo(), pos, n);
                    lup_<T, int>(context, &matrix, nullptr, nullptr);
                    auto offset = shape::getIndexOffset(e, output->shapeInfo());
                    auto inputBuf = reinterpret_cast<T *>(matrix.specialBuffer());
                    auto outputBuf = reinterpret_cast<T *>(output->specialBuffer()) + offset;
                    determinantLogKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuf, outputBuf, n);
                }
                NDArray::registerSpecialUse({output}, {input});

                return sd::Status::OK;
            }

            sd::Status logAbsDeterminant(sd::LaunchContext *context, NDArray *input, NDArray *output) {
                NDArray::prepareSpecialUse({output}, {input});
                BUILD_SINGLE_SELECTOR(input->dataType(), return logAbsDeterminant_, (context, input, output), SD_FLOAT_NATIVE);
                NDArray::registerSpecialUse({output}, {input});
            }

            template <typename T>
            static SD_KERNEL void fillLowerUpperKernel(void *lowerBuf, const sd::LongType *lowerShape, void *upperBuf,
                                                       const sd::LongType *upperShape, void *matrixBuf,
                                                       const sd::LongType *matrixShape, sd::LongType n) {
                __shared__ T *lowerMatrix;
                __shared__ T *upperMatrix;
                __shared__ T *matrix;

                if (threadIdx.x == 0) {
                    lowerMatrix = reinterpret_cast<T *>(lowerBuf);
                    upperMatrix = reinterpret_cast<T *>(upperBuf);
                    matrix = reinterpret_cast<T *>(matrixBuf);
                }
                __syncthreads();

                for (int k = blockIdx.x; k < n; k += gridDim.x) {  // and then put all values under main diagonal on to it
                    for (int j = threadIdx.x; j < n; j += blockDim.x) {
                        sd::LongType posX[] = {k, j};
                        sd::LongType posD[] = {j, j};
                        auto xPos = shape::getOffset(lowerShape, posX);
                        auto yPos = shape::getOffset(upperShape, posX);
                        auto iPos = shape::getOffset(matrixShape, posX);
                        auto dPos = shape::getOffset(matrixShape, posD);
                        if (k >= j)
                            lowerMatrix[xPos] = matrix[iPos];  //(k, j);
                        else
                            upperMatrix[yPos] = matrix[iPos];  // k, j);
                    }
                }
            }

            template <typename T>
            static sd::Status inverse_(sd::LaunchContext *context, NDArray *input, NDArray *output) {
                auto n = input->sizeAt(-1);
                auto n2 = n * n;
                auto dtype = DataTypeUtils::fromT<T>();

                NDArray matrix = NDArrayFactory::create('c', {n, n}, dtype, context);
                NDArray upper = NDArrayFactory::create('c', {n, n}, dtype, context);
                NDArray lower = NDArrayFactory::create('c', {n, n}, dtype, context);
                NDArray compound = NDArrayFactory::create('c', {n, n}, dtype, context);
                NDArray permutation = NDArrayFactory::create('c', {n, n}, dtype, context);

                std::vector<sd::LongType> dims2 = {input->rankOf() - 2, input->rankOf() - 1};
                std::vector<sd::LongType> dims3 = {output->rankOf() - 2, output->rankOf() - 1};

                auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(),
                                                                                   &dims2);

                auto stream = context->getCudaStream();

                for (auto i = 0LL; i < packX->numberOfTads(); i++) {
                    fillMatrix<T, T><<<1, n2, 1024, *stream>>>(matrix.specialBuffer(), matrix.specialShapeInfo(),
                                                               input->specialBuffer(), input->specialShapeInfo(), i * n2, n);
                    matrix.tickWriteDevice();
                    lup_<T, int>(context, &matrix, nullptr, nullptr);
                    fillLowerUpperKernel<T><<<n, n, 1024, *stream>>>(lower.specialBuffer(), lower.specialShapeInfo(),
                                                                     upper.specialBuffer(), upper.specialShapeInfo(),
                                                                     matrix.specialBuffer(), matrix.specialShapeInfo(), n);
                    lower.tickWriteDevice();
                    upper.tickWriteDevice();

                    matrix.assign(0);
                    invertUpperMatrix(context, &upper, &matrix);  // U^{-1}
                    matrix.tickWriteDevice();
                    compound.assign(0);
                    invertLowerMatrix(context, &lower, &compound);  // L{-1}
                    compound.tickWriteDevice();

                    sd::MmulHelper::mmul(&matrix, &compound, &upper, 1.0, 0.0);
                    upper.tickWriteDevice();
                    returnMatrix<T><<<1, n2, 1024, *stream>>>(output->specialBuffer(), output->specialShapeInfo(),
                                                              upper.specialBuffer(), upper.specialShapeInfo(), i * n2, n);
                }
                return sd::Status::OK;
            }

            sd::Status inverse(sd::LaunchContext *context, NDArray *input, NDArray *output) {
                NDArray::prepareSpecialUse({output}, {input});
                BUILD_SINGLE_SELECTOR(input->dataType(), return inverse_, (context, input, output), SD_FLOAT_NATIVE);
                NDArray::registerSpecialUse({output}, {input});
            }

            bool checkCholeskyInput(sd::LaunchContext *context, NDArray const *input) { return true; }

            template <typename F>
            SD_KERNEL void fillBatchKernel(F **dArrayBatch, F *buf, const sd::LongType *offsets, sd::LongType batchSize) {
                auto start = blockIdx.x * blockDim.x + threadIdx.x;
                auto step = blockDim.x * gridDim.x;

                for (auto i = start; i < batchSize; i += step) {
                    dArrayBatch[i] = buf + offsets[i];
                }
            }

            template <typename F>
            SD_KERNEL void adjustResultsKernel(F *dArray, const sd::LongType *shape, const sd::LongType *offsets,
                                               sd::LongType batchSize, sd::LongType n) {
                // auto i = blockIdx.x * blockDim.x + threadIdx.x;
                sd::LongType *shapeOf = shape::shapeOf(shape);
                sd::LongType *strideOf = shape::stride(shape);

                for (auto i = blockIdx.x; i < batchSize; i += gridDim.x) {
                    auto current = dArray + offsets[i];
                    for (auto r = threadIdx.x; r < n; r += blockDim.x) {
                        for (auto c = r + 1; c < n; c++) {
                            sd::LongType posRC[] = {r, c};
                            auto pos = r * n + c;  // shape::getOffset(0, shapeOf, strideOf, posRC, 2);
                            current[pos] = 0.;
                        }
                    }
                }
            }

            template <typename F>
            sd::Status cholesky__(LaunchContext *context, NDArray *input, NDArray *output, bool inplace) {
                if (!inplace) output->assign(input);
                auto tempOutput = output->dup();
                cusolverDnHandle_t handle = nullptr;
                auto n = input->sizeAt(-1);
                auto n2 = n * n;
                NDArray::prepareSpecialUse({output}, {input});
                auto status = cusolverDnCreate(&handle);
                if (CUSOLVER_STATUS_SUCCESS != status) {
                    throw cuda_exception::build("helpers::cholesky_: Cannot create solver handle", status);
                }
                F **dArrayBatch = nullptr;
                std::vector<sd::LongType> dims = {tempOutput.rankOf() - 2, tempOutput.rankOf() - 1};
                auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(
                        tempOutput.shapeInfo(), &dims);
                const sd::LongType batchSize = packX->numberOfTads();
                int *dInfoArray = nullptr;
                auto err = cudaMalloc((void **)&dArrayBatch, sizeof(F *) * batchSize);
                if (err) {
                    throw cuda_exception::build("helpers::cholesky_: Cannot allocate memory for solver batch data buffer", err);
                }
                err = cudaMalloc((void **)&dInfoArray, sizeof(sd::LongType) * batchSize);
                if (err) {
                    throw cuda_exception::build("helpers::cholesky_: Cannot allocate memory for solver errors buffer", err);
                }
                auto stream = context->getCudaStream();
                fillBatchKernel<F><<<1, batchSize, 128, *stream>>>(dArrayBatch, reinterpret_cast<F *>(tempOutput.specialBuffer()),
                                                                   packX->specialOffsets(), batchSize);

                status = cusolverDnSetStream(handle, *stream);
                if (CUSOLVER_STATUS_SUCCESS != status) {
                    throw cuda_exception::build("helpers::cholesky_: Cannot set stream to solver handle", status);
                }
                const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
                if (input->dataType() == DataType::DOUBLE)
                    status = cusolverDnDpotrfBatched(handle, uplo, n, (double **)dArrayBatch, n, dInfoArray, batchSize);
                else
                    status = cusolverDnSpotrfBatched(handle, uplo, n, (float **)dArrayBatch, n, dInfoArray, batchSize);

                if (CUSOLVER_STATUS_SUCCESS != status) {
                    throw cuda_exception::build("helpers::cholesky_: Cholesky factorization failed for batch", status);
                }
                adjustResultsKernel<F><<<batchSize, n2, 128, *stream>>>(reinterpret_cast<F *>(tempOutput.specialBuffer()),
                                                                        packX->specialShapeInfo(), packX->specialOffsets(), batchSize,
                                                                        n);

                err = cudaFree(dArrayBatch);
                if (err) {
                    throw cuda_exception::build("helpers::cholesky_: Cannot deallocate memory for solver batch data buffer", err);
                }
                err = cudaFree(dInfoArray);
                if (err) {
                    throw cuda_exception::build("helpers::cholesky_: Cannot allocate memory for solver errors buffer", err);
                }

                if (!inplace)
                    output->assign(tempOutput);
                else
                    input->assign(tempOutput);

                NDArray::registerSpecialUse({output}, {input});
                return sd::Status::OK;
            }

//    template <typename T>
            sd::Status cholesky_(LaunchContext *context, NDArray *input, NDArray *output, bool inplace) {
                NDArray::prepareSpecialUse({output}, {input});
                if (input->dataType() == DataType::DOUBLE)
                    cholesky__<double>(context, input, output, inplace);
                else if (input->dataType() == DataType::FLOAT32)
                    cholesky__<float>(context, input, output, inplace);
                else {
                    std::unique_ptr<NDArray> tempOutput(
                            NDArrayFactory::create_('c', input->getShapeAsVector(), DataType::FLOAT32, context));
                    tempOutput->assign(input);
                    cholesky__<float>(context, tempOutput.get(), tempOutput.get(), true);
                    output->assign(tempOutput.get());
                }
                NDArray::registerSpecialUse({output}, {input});
                return sd::Status::OK;
            }

            sd::Status cholesky(sd::LaunchContext *context, NDArray *input, NDArray *output, bool inplace) {
                return cholesky_(context, input, output, inplace);
            }

            BUILD_SINGLE_TEMPLATE(template sd::Status inverse_, (sd::LaunchContext * context, NDArray *input, NDArray *output),
                                  SD_FLOAT_NATIVE);

            template <typename T>
            SD_KERNEL void logDetKernel(const T *inputBuf, const sd::LongType *inputShape, sd::LongType batchNum,
                                        const sd::LongType *tadShape, const sd::LongType *tadOffsets, T *outputBuf,
                                        const sd::LongType *outputShape) {
                __shared__ int n;
                if (threadIdx.x == 0) {
                    n = shape::sizeAt(inputShape, -1);
                }
                __syncthreads();

                auto output = outputBuf;
                auto input = inputBuf;

                for (auto i = blockIdx.x; i < batchNum; i += gridDim.x) {
                    auto current = input + tadOffsets[i];

                    auto zIndex = shape::getIndexOffset(i, outputShape);
                    for (auto e = threadIdx.x; e < n; e += blockDim.x) {
                        sd::LongType diag[] = {e, e};
                        auto xIndex = shape::getOffset(tadShape, diag);
                        math::atomics::sd_atomicAdd(&output[zIndex], math::sd_log<T, T>(current[xIndex] * current[xIndex]));
                    }
                }
            }

            template <typename T>
            sd::Status logdetFunctor_(sd::LaunchContext *context, NDArray *input, NDArray *output) {
                NDArray::prepareSpecialUse({output}, {input});
                auto n2 = input->sizeAt(-1) * input->sizeAt(-2);
                auto stream = context->getCudaStream();
                NDArray tempOutput(*input);

                cholesky(context, input, &tempOutput, false);

                auto outputBuf = output->dataBuffer()
                        ->specialAsT<T>();
                auto inputBuf = tempOutput.dataBuffer()->specialAsT<T>();
                output->nullify();

                std::vector<sd::LongType> dims = {tempOutput.rankOf() - 2, tempOutput.rankOf() - 1};
                auto packX = sd::ConstantTadHelper::getInstance().tadForDimensions(
                        tempOutput.shapeInfo(), &dims);
                logDetKernel<T><<<128, 512, 256, *stream>>>(inputBuf, tempOutput.specialShapeInfo(), packX->numberOfTads(),
                                                            packX->specialShapeInfo(), packX->specialOffsets(), outputBuf,
                                                            output->specialShapeInfo());
                output->tickWriteDevice();
                NDArray::registerSpecialUse({output}, {input});
                return sd::Status::OK;
            }

            sd::Status logdetFunctor(sd::LaunchContext *context, NDArray *input, NDArray *output) {
                BUILD_SINGLE_SELECTOR(output->dataType(), return logdetFunctor_, (context, input, output), SD_FLOAT_NATIVE);
            }

/*
 * lup - batched input, batched outputs
 * */
            sd::Status lup(LaunchContext *context, NDArray *input, NDArray *compound, NDArray *permutation) {
                BUILD_DOUBLE_SELECTOR(input->dataType(), permutation->dataType(), lup_, (context, input, compound, permutation),
                                      SD_FLOAT_NATIVE, SD_INDEXING_TYPES);
                return sd::Status::OK;
            }

        }  // namespace helpers
    }  // namespace ops
}  // namespace sd
