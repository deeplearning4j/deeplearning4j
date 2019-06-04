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

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T> 
    static void _swapRows(NDArray* matrix, int theFirst, int theSecond) {

    }
    BUILD_SINGLE_TEMPLATE(template void _swapRows, (NDArray* matrix, int theFirst, int theSecond), FLOAT_TYPES);

    void swapRows(NDArray* matrix, int theFirst, int theSecond) {
        BUILD_SINGLE_SELECTOR(matrix->dataType(), _swapRows, (matrix, theFirst, theSecond), FLOAT_TYPES);
    }

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
    static NDArray _lup(NDArray* input, NDArray* compound, NDArray* permutation) {
        NDArray determinant = NDArrayFactory::create<T>(1.f);

        return determinant;
    }
    BUILD_SINGLE_TEMPLATE(template NDArray _lup, (NDArray* input, NDArray* output, NDArray* permutation), FLOAT_TYPES);

    template <typename T>
    static int _determinant(NDArray* input, NDArray* output) {
        return Status::OK();
    }

    BUILD_SINGLE_TEMPLATE(template int _determinant, (NDArray* input, NDArray* output), FLOAT_TYPES);

    int determinant(nd4j::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return _determinant, (input, output), FLOAT_TYPES);
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
