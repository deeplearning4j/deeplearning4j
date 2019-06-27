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

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T> 
    static __device__ void _swapRows(T* matrix, Nd4jLong* shape, int theFirst, int theSecond, Nd4jLong N) {
        if (theFirst != theSecond) {
            auto start = threadIdx.x + blockIdx.x * blockDim.x;
            auto step = blockDim.x * gridDim.x;
            for (auto i = start; i < N; i += step) {
                Nd4jLong iCoord1[] = {theFirst, i};
                Nd4jLong iCoord2[] = {theSecond, i};
                auto iIndex1 = shape::getOffset(0, shape::shapeOf(shape), shape::stride(shape), iCoord1, 2);
                auto iIndex2 = shape::getOffset(0, shape::shapeOf(shape), shape::stride(shape), iCoord2, 2);
                //atomicExch(&matrix[iIndex1], matrix[iIndex2]);
                T e0 = matrix[iIndex1];
                T e1 = matrix[iIndex2];
                matrix[iIndex1] = e0;
                matrix[iIndex2] = e1;
            }
        }
    }
//    BUILD_SINGLE_TEMPLATE(template void _swapRows, (NDArray* matrix, int theFirst, int theSecond), FLOAT_TYPES);
//
//    void swapRows(NDArray* matrix, int theFirst, int theSecond) {
//        BUILD_SINGLE_SELECTOR(matrix->dataType(), _swapRows, (matrix, theFirst, theSecond), FLOAT_TYPES);
//    }

    template <typename T>
    static void _invertLowerMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {

    }

    BUILD_SINGLE_TEMPLATE(template void _invertLowerMatrix, (NDArray* inputMatrix, NDArray* invertedMatrix);, FLOAT_TYPES);

    void invertLowerMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
        BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), _invertLowerMatrix, (inputMatrix, invertedMatrix), FLOAT_TYPES);
    }

    template <typename T>
    static void _invertUpperMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {

    }

    BUILD_SINGLE_TEMPLATE(template void _invertUpperMatrix, (NDArray* inputMatrix, NDArray* invertedMatrix);, FLOAT_TYPES);

    void invertUpperMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
        BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), _invertUpperMatrix, (inputMatrix, invertedMatrix), FLOAT_TYPES);
    }

    template <typename T>
    static __global__ void lupKernel(T* compound, Nd4jLong* compoundShape, T* permutation, Nd4jLong* permutationShape, Nd4jLong rowNum) {
        int swapCount = 0;
        for(int i = blockIdx.x; i < rowNum; i += gridDim.x ) {
            auto pivotValue = T(0.0);
            auto pivot = -1;

            for(int rowCounter = i; rowCounter < rowNum; rowCounter++ ) {
                Nd4jLong rowCoord[] = {rowCounter, i};
                auto rowPos = shape::getOffset(0, shape::shapeOf(compoundShape), shape::stride(compoundShape), rowCoord, 2);
                if(nd4j::math::nd4j_abs(compound[rowPos]) > pivotValue ) {
                    pivotValue = nd4j::math::nd4j_abs(compound[rowPos]);
                    pivot = rowCounter;
                }
            }

            if( pivotValue != T(0.0) ) {
                _swapRows<T>(compound, compoundShape, pivot, i, rowNum);
                _swapRows<T>(permutation, permutationShape, pivot, i, rowNum);
                if (pivot != i)
                    swapCount++;

                for( int j = i + 1; j < rowNum; j++ ) {
                    Nd4jLong posJIbuf[] = {j, i};
                    Nd4jLong posIIbuf[] = {i, i};
                    auto posJI = shape::getOffset(0, shape::shapeOf(compoundShape), shape::stride(compoundShape), posJIbuf, 2);
                    auto posII = shape::getOffset(0, shape::shapeOf(compoundShape), shape::stride(compoundShape), posIIbuf, 2);

                    compound[posJI] /= compound[posII];
                    for( int k = i + 1; k < rowNum; k++ ) {
                        Nd4jLong posJKbuf[] = {j, k};
                        Nd4jLong posIKbuf[] = {i, k};
                        auto posJK = shape::getOffset(0, shape::shapeOf(compoundShape), shape::stride(compoundShape), posJKbuf, 2);
                        auto posIK = shape::getOffset(0, shape::shapeOf(compoundShape), shape::stride(compoundShape), posIKbuf, 2);
                        T arg = compound[posJI] * compound[posIK];
                        compound[posJK] -= arg;
                    }
                }
            }
        }
    }
    template <typename T>
    static __global__ void determinantKernel(T* compound, Nd4jLong* shape, T* result) {
        __shared__ Nd4jLong len;

        if (threadIdx.x == 0) {
            len = shape::length(shape);
        }
        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;
        for (auto i = start; i < len; i += step) {
            Nd4jLong di[] = {i, i};
            auto pos = shape::getOffset(0, shape::shapeOf(shape), shape::stride(shape), di, 2);
            math::atomics::nd4j_atomicMul(result, compound[pos]);
        }
    }
    template <typename T>
    static __global__ void determinantFullKernel(T* input, Nd4jLong* inputShape, T* output, Nd4jLong* outputShape, Nd4jLong* tadShape, Nd4jLong* tadOffsets) {

    }

    template <typename T>
    static NDArray _lup(LaunchContext* context, NDArray* input, NDArray* compound, NDArray* permutation) {
        NDArray determinant = NDArrayFactory::create<T>(1.f);
        auto rowNum = input->rows();
        auto columnNum = input->columns();

        NDArray compoundMatrix = *input; // copy
        NDArray permutationMatrix(input, false, input->getContext()); // has same shape as input and contiguous strides
        permutationMatrix.setIdentity();

        T pivotValue; // = T(0.0);
        int pivot; // = -1;
        int swapCount = 0;
        T* compoundBuf = reinterpret_cast<T*>(compoundMatrix.specialBuffer());
        T* permutationBuf = reinterpret_cast<T*>(permutationMatrix.specialBuffer());
        auto stream = context->getCudaStream();
        lupKernel<T><<<256, 256, 1024, *stream>>>(compoundBuf, compoundMatrix.specialShapeInfo(), permutationBuf, permutationMatrix.specialShapeInfo(), rowNum);
        determinantKernel<T><<<256, 256, 1024, *stream>>>(compoundBuf, compoundMatrix.specialShapeInfo(), reinterpret_cast<T*>(determinant.specialBuffer()));
//        for (int e = 0; e < rowNum; e++) {
//            // nd4j_printf("Compound matrix diag %i %f.\n", e, (*compoundMatrix)(e, e));
//            determinant *= compoundMatrix.e<T>(e, e);
//        }
        if (swapCount % 2) determinant = -determinant;
        if (compound != nullptr)
            compound->assign(compoundMatrix);
        if (permutation != nullptr)
            permutation->assign(permutationMatrix);
        return determinant;
    }
    BUILD_SINGLE_TEMPLATE(template NDArray _lup, (LaunchContext* context, NDArray* input, NDArray* output, NDArray* permutation), FLOAT_TYPES);

    template <typename T>
    static int _determinant(nd4j::LaunchContext* context, NDArray* input, NDArray* output) {
        Nd4jLong n = input->sizeAt(-1);
        Nd4jLong n2 = n * n;
        std::vector<int> dims();
        auto packX = ConstantTadHelper::getInstance()->tadForDimensions(input->getShapeInfo(), {input->rankOf() - 2, input->rankOf() - 1});
        //auto packZ = ConstantTadHelper::getInstance()->tadForDimensions(output->shapeInfo(), {output->rankOf() - 1});

        //auto matrix = NDArrayFactory::create(input->ordering(), {n, n}, input->dataType(), input->getContext()); //, block.getWorkspace());
        auto stream = context->getCudaStream();
        auto inputBuf = reinterpret_cast<T*>(input->specialBuffer());
        auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());
        dim3 launchDims(256, 256, 1024);
        determinantFullKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuf, input->specialShapeInfo(), outputBuf, output->specialShapeInfo(), packX.specialShapeInfo(), packX.specialOffsets());
//        for (int e = 0; e < output->lengthOf(); e++) {
//            for (int k = e * n2, row = 0; k < (e + 1) * n2; ++k, ++row)
//                matrix.p(row, input->e<T>(k));
////            output->p(e, lup_<T>(&matrix, (NDArray*)nullptr, (NDArray*)nullptr));
//        }

        return Status::OK();
    }

    BUILD_SINGLE_TEMPLATE(template int _determinant, (nd4j::LaunchContext* context, NDArray* input, NDArray* output), FLOAT_TYPES);

    int determinant(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return _determinant, (context, input, output), FLOAT_TYPES);
    }

    template <typename T>
    int log_abs_determinant_(NDArray* input, NDArray* output) {
        return ND4J_STATUS_OK;
    }

    BUILD_SINGLE_TEMPLATE(template int log_abs_determinant_, (NDArray* input, NDArray* output), FLOAT_TYPES);

    int log_abs_determinant(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return log_abs_determinant_, (input, output), FLOAT_TYPES);
    }

    template <typename T>
    static int _inverse(NDArray* input, NDArray* output) {
        return Status::OK();
    }

    int inverse(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return _inverse, (input, output), FLOAT_TYPES);
    }

    bool checkCholeskyInput(nd4j::LaunchContext * context, NDArray const* input) {
        return false;
    }

    template <typename T>
    int cholesky_(NDArray* input, NDArray* output, bool inplace) {
        return Status::OK();
    }

    int cholesky(nd4j::LaunchContext * context, NDArray* input, NDArray* output, bool inplace) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return cholesky_, (input, output, inplace), FLOAT_TYPES);
    }    
    BUILD_SINGLE_TEMPLATE(template int cholesky_, (NDArray* input, NDArray* output, bool inplace), FLOAT_TYPES);
    BUILD_SINGLE_TEMPLATE(template int _inverse, (NDArray* input, NDArray* output), FLOAT_TYPES);


    int logdetFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        return 119;
    }
}
}
}
