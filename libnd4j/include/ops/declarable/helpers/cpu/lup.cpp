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
#include <helpers/MmulHelper.h>
#include <array/NDArrayFactory.h>
#include <graph/Status.h>
#include <execution/Threads.h>
#include <execution/Threads.h>

namespace sd {
namespace ops {
namespace helpers {

    template <typename T>
    static void swapRows_(NDArray* matrix, int theFirst, int theSecond) {

        if (theFirst != theSecond)
            for (int i = 0; i < matrix->columns(); i++) {
                math::nd4j_swap(matrix->t<T>(theFirst, i), matrix->t<T>(theSecond, i));
            }
    }
    BUILD_SINGLE_TEMPLATE(template void swapRows_, (NDArray* matrix, int theFirst, int theSecond), FLOAT_TYPES);

    template <typename T>
    static void swapRows(T* matrixBuf, Nd4jLong* matrixShape, Nd4jLong theFirst, Nd4jLong theSecond) {
        if (theFirst != theSecond) {
            auto n = shape::sizeAt(matrixShape, -1);

            auto loop = PRAGMA_THREADS_FOR {
                for (auto i = start; i < stop; i++) {
                    Nd4jLong theFirstPos[] = {theFirst, i};
                    Nd4jLong theSecondPos[] = {theSecond, i};
                    auto theFirstIndex = shape::getOffset(matrixShape, theFirstPos, 0);
                    auto theSecondIndex = shape::getOffset(matrixShape, theSecondPos, 0);
                    math::nd4j_swap(matrixBuf[theFirstIndex], matrixBuf[theSecondIndex]);
                }
            };

            samediff::Threads::parallel_tad(loop, 0, n, 1);
        }
    }

    void swapRows(NDArray* matrix, int theFirst, int theSecond) {
        BUILD_SINGLE_SELECTOR(matrix->dataType(), swapRows_, (matrix, theFirst, theSecond), FLOAT_TYPES);
    }

    template <typename T>
    static void invertLowerMatrix_(NDArray* inputMatrix, NDArray* invertedMatrix) {
        int n = inputMatrix->rows();
        invertedMatrix->setIdentity();

        if (inputMatrix->isIdentityMatrix()) return;

        auto invertDiagonals = PRAGMA_THREADS_FOR {
            for (int i = start; i < stop; i += increment)
                invertedMatrix->t<T>(i, i) /= inputMatrix->t<T>(i, i);
        };

        auto invertSubDiagonals = PRAGMA_THREADS_FOR {
            for (int i = start; i < stop; i += increment)
                invertedMatrix->t<T>(i, i - 1) -= (inputMatrix->t<T>(i, i - 1) * invertedMatrix->t<T>(i - 1, i - 1) / inputMatrix->t<T>(i, i));
        };

        samediff::Threads::parallel_for(invertDiagonals, 0, n, 1);
        samediff::Threads::parallel_for(invertSubDiagonals, 1, n, 1);

//        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i - 1 ; j++)
                for (int k = 0; k < i; k++)
                    invertedMatrix->t<T>(i, j) -= ((invertedMatrix->t<T>(k, j) * inputMatrix->t<T>(i, k) / inputMatrix->t<T>(i, i)));
        }

    }

    BUILD_SINGLE_TEMPLATE(template void invertLowerMatrix_, (NDArray* inputMatrix, NDArray* invertedMatrix);, FLOAT_TYPES);

    void invertLowerMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
        BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), invertLowerMatrix_, (inputMatrix, invertedMatrix), FLOAT_TYPES);
    }

    template <typename T>
    static void _invertUpperMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
        int n = inputMatrix->rows();
        invertedMatrix->setIdentity();

        if (inputMatrix->isIdentityMatrix()) { // the inverse for I is I
            return;
        }

        auto invertDiagonals = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i += increment)
                invertedMatrix->t<T>(i, i) /= inputMatrix->t<T>(i, i);
        };

        //PRAGMA_OMP_PARALLEL_FOR_IF(n > Environment::getInstance()->elementwiseThreshold())
        auto invertUpDiagonals = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i += increment)
                invertedMatrix->t<T>(i, i + 1) -= (inputMatrix->t<T>(i, i + 1) * invertedMatrix->t<T>(i + 1, i + 1) /
                                                   inputMatrix->t<T>(i, i));
        };

        samediff::Threads::parallel_for(invertDiagonals, 0, n, 1);
        samediff::Threads::parallel_for(invertUpDiagonals, 0, n - 1, 1);

//        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (auto i = n - 2; i >= 0; i--) {
            for (auto j = i + 2; j < n; j++)
                for (auto k = i; k < n; k++)
                    invertedMatrix->t<T>(i, j) -= ((invertedMatrix->t<T>(k, j) * inputMatrix->t<T>(i, k) / inputMatrix->t<T>(i, i)));
        }
    }

    BUILD_SINGLE_TEMPLATE(template void _invertUpperMatrix, (NDArray* inputMatrix, NDArray* invertedMatrix);, FLOAT_TYPES);

    void invertUpperMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
        BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), _invertUpperMatrix, (inputMatrix, invertedMatrix), FLOAT_TYPES);
    }


    template <typename T, typename I>
    static NDArray lup_(LaunchContext *context, NDArray* input, NDArray* compound, NDArray* permutation) {

        const int rowNum = input->rows();
        const int columnNum = input->columns();

        NDArray determinant = NDArrayFactory::create<T>(1.f);
        NDArray compoundMatrix = *input; // copy
        NDArray permutationMatrix(input, false, context); // has same shape as input and contiguous strides
        permutationMatrix.setIdentity();

        T pivotValue; // = T(0.0);
        int pivot; // = -1;
        int swapCount = 0;

        for(int i = 0; i < rowNum; i++ ) {
            pivotValue = T(0.0);
            pivot = -1;
            //PRAGMA_OMP_PARALLEL_FOR //_ARGS(firstprivate(pivot,pivotValue))
            for(int rowCounter = i; rowCounter < rowNum; rowCounter++ ) {
                if (sd::math::nd4j_abs(compoundMatrix.t<T>(rowCounter, i)) > pivotValue) {
                    pivotValue = sd::math::nd4j_abs(compoundMatrix.t<T>(rowCounter, i));
                    pivot = rowCounter;
                }
            }

            if( pivotValue > DataTypeUtils::min<T>()) {
                swapRows(&compoundMatrix, pivot, i);
                swapRows(&permutationMatrix, pivot, i);
                if (pivot != i)
                    swapCount++;

                for( int j = i + 1; j < rowNum; j++ ) {
                    compoundMatrix.t<T>(j, i) /= compoundMatrix.t<T>(i, i);
                    //PRAGMA_OMP_PARALLEL_FOR
                    for( int k = i + 1; k < rowNum; k++ ) {
                        compoundMatrix.t<T>(j, k) -= compoundMatrix.t<T>(j, i) * compoundMatrix.t<T>(i, k);
                    }
                }
            }
        }

        for (int e = 0; e < rowNum; e++) {
            // nd4j_printf("Compound matrix diag %i %f.\n", e, (*compoundMatrix)(e, e));
            determinant *= compoundMatrix.e<T>(e, e);
        }
        if (swapCount % 2) determinant = -determinant;
        if (compound != nullptr)
            compound->assign(compoundMatrix);
        if (permutation != nullptr) {
            auto permutaionVector = NDArrayFactory::create('c', {rowNum}, DataTypeUtils::fromT<I>(), input->getContext());
            for (auto i = 0; i < rowNum; i++) {
                for (auto j = 0; j < columnNum; j++) {
                    if (permutationMatrix.t<T>(i, j) != 0) {
                        permutaionVector.template t<I>(i) = j;
                    }
                }
            }
            if (permutationMatrix.isSameShape(permutation))
                permutation->assign(permutationMatrix);
            else if (permutation->isSameShape(permutaionVector)) {
                permutation->assign(permutaionVector);
            }
        }
        return determinant;
    }

    BUILD_DOUBLE_TEMPLATE(template NDArray lup_, (LaunchContext *context, NDArray* input, NDArray* output, NDArray* permutation), FLOAT_TYPES, INDEXING_TYPES);
    /*
     * lu decomposition with naive algorithm with partial pivoting
     * */
    template <typename T, typename I>
    static I argmaxCol(I column, T* compoundBuffer, Nd4jLong* compoundShape) {
        auto rowNum = shape::sizeAt(compoundShape, 0);
        Nd4jLong xInitial[] = {column, column};
        auto xInitialIndex = shape::getOffset(compoundShape, xInitial, 0);
        auto maxValue = T(0); //sd::math::nd4j_abs(compoundBuffer[xInitialIndex]);
        auto result = -1;
        //auto loop = PRAGMA_THREADS_FOR {
            auto start = column, stop = rowNum, increment = 1;
            for (auto rowCounter = start; rowCounter < stop; rowCounter++) {
                Nd4jLong xPos[] = {rowCounter, column};
                auto xIndex = shape::getOffset(compoundShape, xPos, 0);
                if (sd::math::nd4j_abs(compoundBuffer[xIndex]) > maxValue) {
                    maxValue = sd::math::nd4j_max(maxValue, sd::math::nd4j_abs(compoundBuffer[xIndex]));
                    result = rowCounter;
                }
            }
        //};
        //samediff::Threads::parallel_for(loop, column, rowNum, 1);
        return result;
    }

    template <typename T>
    void processColumns(int currentRow, int rowNum, T* compoundBuf, Nd4jLong* compoundShape) {
        Nd4jLong xDiag[] = {currentRow, currentRow};
        auto diagIndex = shape::getOffset(compoundShape, xDiag, 0);
        auto loop = PRAGMA_THREADS_FOR {
            for (auto j = start; j < stop; j++) {
                Nd4jLong xRow[] = {j, currentRow};
                auto rowIndex = shape::getOffset(compoundShape, xRow, 0);
                compoundBuf[rowIndex] /= compoundBuf[diagIndex]; //output->t<T>(i, i);
                for (int k = currentRow + 1; k < rowNum; k++) {
                    Nd4jLong yRow[] = {j, k};
                    Nd4jLong yCol[] = {currentRow, k};
                    auto rowIndexY = shape::getOffset(compoundShape, yRow, 0);
                    auto colIndex = shape::getOffset(compoundShape, yCol, 0);
                    compoundBuf[rowIndexY] -= compoundBuf[rowIndex] * compoundBuf[colIndex];
                }
            }
        };
        samediff::Threads::parallel_tad(loop, currentRow + 1, rowNum, 1);
    }

    template <typename T>
    static void doolitleLU(LaunchContext* context, NDArray* compound, Nd4jLong rowNum) {
        auto input = compound->dup();
        compound->nullify();

        // Decomposing matrix into Upper and Lower
        // triangular matrix
        for (auto i = 0; i < rowNum; i++) {

            // Upper Triangular
            for (auto k = i; k < rowNum; k++) {

                // Summation of L(i, j) * U(j, k)
                int sum = 0;
                for (int j = 0; j < i; j++)
                    sum += compound->t<T>(i,j) * compound->t<T>(j,k);

                // Evaluating U(i, k)
                compound->t<T>(i, k) = input.t<T>(i, k) - sum;
            }

            // Lower Triangular
            for (int k = i + 1; k < rowNum; k++) {
                // Summation of L(k, j) * U(j, i)
                int sum = 0;
                for (int j = 0; j < i; j++)
                    sum += compound->t<T>(k,j) * compound->t<T>(j, i);

                // Evaluating L(k, i)
                compound->t<T>(k, i) = (input.t<T>(k, i) - sum) / compound->t<T>(i,i);
            }
        }
    }

    template <typename T, typename I>
    static void luNN_(LaunchContext *context, NDArray* compound, NDArray* permutation, Nd4jLong rowNum) {

        //const int rowNum = compound->rows();
//        const int columnNum = output->columns();
        if (permutation) { // LUP algorithm
            permutation->linspace(0);
            auto permutationBuf = permutation->bufferAsT<I>(); //dataBuffer()->primaryAsT<I>();
            auto compoundBuf = compound->bufferAsT<T>();
            auto compoundShape = compound->shapeInfo();
            auto permutationShape = permutation->shapeInfo();
            for (auto i = 0; i < rowNum - 1; i++) {
                auto pivotIndex = argmaxCol(i, compoundBuf, compoundShape);
                if (pivotIndex < 0) {
                    throw std::runtime_error("helpers::luNN_: input matrix is singular.");
                }
                math::nd4j_swap(permutationBuf[shape::getIndexOffset(i, permutationShape)],
                                permutationBuf[shape::getIndexOffset(pivotIndex, permutationShape)]);
                swapRows(compoundBuf, compoundShape, i, pivotIndex);

                processColumns(i, rowNum, compoundBuf, compoundShape);
            }
        }
        else { // Doolitle algorithm with LU decomposition
            doolitleLU<T>(context, compound, rowNum);
        }
    }

    template <typename T, typename I>
    static void lu_(LaunchContext * context, NDArray* input, NDArray* output, NDArray* permutationVectors) {
        auto n = input->sizeAt(-1);

        output->assign(input); // fill up output tensor with zeros
        ResultSet outputs = output->allTensorsAlongDimension({-2, -1});
        ResultSet permutations;
        if (permutationVectors)
            permutations = permutationVectors->allTensorsAlongDimension({-1});

        auto loop = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
                luNN_<T, I>(context, outputs.at(i), permutationVectors?permutations.at(i):nullptr, n);
            }
        };
        samediff::Threads::parallel_for(loop, 0, outputs.size(), 1);
    }

    void lu(LaunchContext *context, NDArray* input, NDArray* output, NDArray* permutation) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), permutation?permutation->dataType():DataType::INT32, lu_, (context, input, output, permutation), FLOAT_TYPES, INDEXING_TYPES);
    }

//    BUILD_DOUBLE_TEMPLATE(template NDArray lu_, (LaunchContext *context, NDArray* input, NDArray* output, NDArray* permutation), FLOAT_TYPES, INDEXING_TYPES);

    template <typename T>
    static int determinant_(LaunchContext *context, NDArray* input, NDArray* output) {

        Nd4jLong n = input->sizeAt(-1);
        Nd4jLong n2 = n * n;

        auto matrix = NDArrayFactory::create(input->ordering(), {n, n}, input->dataType(), context); //, block.getWorkspace());

        for (int e = 0; e < output->lengthOf(); e++) {
            for (int k = e * n2, row = 0; k < (e + 1) * n2; ++k, ++row)
                matrix.p(row, input->e<T>(k));
            output->p(e, lup_<T, int>(context, &matrix, (NDArray*)nullptr, (NDArray*)nullptr));
        }

        return Status::OK();
    }

    int determinant(sd::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return determinant_, (context, input, output), FLOAT_TYPES);
    }

template <typename T>
    int logAbsDeterminant_(LaunchContext *context, NDArray* input, NDArray* output) {

        Nd4jLong n = input->sizeAt(-1);
        Nd4jLong n2 = n * n;

        NDArray matrix = NDArrayFactory::create(input->ordering(), {n, n}, input->dataType(), context); //, block.getWorkspace());
        for (int e = 0; e < output->lengthOf(); e++) {
            for (int k = e * n2, row = 0; k < (e + 1) * n2; ++k, ++row) {
                matrix.p(row, input->e<T>(k));
            }
	    NDArray det = lup_<T, int>(context, &matrix, (NDArray*)nullptr, (NDArray*)nullptr);
	    if (det.e<T>(0) != 0.f)
             	output->p(e, sd::math::nd4j_log<T,T>(sd::math::nd4j_abs(det.t<T>(0))));
        }

        return ND4J_STATUS_OK;
    }

    int logAbsDeterminant(sd::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return logAbsDeterminant_, (context, input, output), FLOAT_TYPES);
    }

    template <typename T>
    static int inverse_(LaunchContext *context, NDArray* input, NDArray* output) {

        auto n = input->sizeAt(-1);
        auto n2 = n * n;
        auto totalCount = output->lengthOf() / n2;

        output->assign(0.f); // fill up output tensor with zeros
        auto matrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context); //, block.getWorkspace());
        auto compound = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context); //, block.getWorkspace());
        auto permutation = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
        auto lowerMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
        auto upperMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);

        for (int e = 0; e < totalCount; e++) {
            if (e)
                matrix.assign(0.f);

            for (int k = e * n2, row = 0; k < (e + 1) * n2; k++) {
                matrix.p(row++, input->e<T>(k));
            }
            T det = lup_<T, int>(context, &matrix, &compound, &permutation).template e<T>(0);

            // FIXME: and how this is going to work on float16?
            if (sd::math::nd4j_abs<T>(det) < T(0.000001)) {
                nd4j_printf("matrix_inverse: The matrix %i has no inverse due determinant is %lf. Quiting...\n", e, det);
                matrix.printIndexedBuffer("Wrong matrix");
                return ND4J_STATUS_VALIDATION;
            }
            lowerMatrix.setIdentity(); // set up U to identity matrix
            for (int k = 1; k < n; k++) {  // and then put all values under main diagonal on to it
                for (int j = 0; j < k; j++)
                    lowerMatrix.template t<T>(k, j) = compound.template t<T>(k, j);
            }
            upperMatrix.setIdentity(); // set up U to identity matrix
            for (int k = 0; k < n; k++) {  // and then put all values under main diagonal on to it
                for (int j = k; j < n; j++)
                    upperMatrix.template t<T>(k, j) = compound.template e<T>(k, j);
            }
            invertUpperMatrix(&upperMatrix, &matrix);

            invertLowerMatrix(&lowerMatrix, &upperMatrix);

            sd::MmulHelper::mmul(&matrix, &upperMatrix, &compound, 1.0, 0.0);
            sd::MmulHelper::mmul(&compound, &permutation, &matrix, 1.0, 0.0);
            for (int k = e * n2, row = 0; k < (e + 1) * n2; k++) {
                output->t<T>(k) = matrix.template t<T>(row++);
            }
        }

        return Status::OK();
    }

    template <typename T>
    static int lowerInverse_(LaunchContext *context, NDArray* input, NDArray* output) {

        auto n = input->sizeAt(-1);
        auto n2 = n * n;
        auto totalCount = output->lengthOf() / n2;

        output->assign(0.f); // fill up output tensor with zeros
        auto matrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context); //, block.getWorkspace());
        auto compound = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context); //, block.getWorkspace());
        auto permutation = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
        auto lowerMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
        auto upperMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);

//        auto batchLoop = PRAGMA_THREADS_FOR {
        for (int e = 0; e < totalCount; e++) {
            if (e)
                matrix.assign(0.f);

            for (int k = e * n2, row = 0; k < (e + 1) * n2; k++) {
                matrix.p(row++, input->e<T>(k));
            }
            T det = T(1.f);
            for (auto i = 0; i < n; i++) {
                det *= matrix. template t<T>(i, i);
            }

            // FIXME: and how this is going to work on float16?
            if (sd::math::nd4j_abs<T>(det) < T(0.000001)) {
                nd4j_printf("matrix_inverse: The matrix %i has no inverse due determinant is %lf. Quiting...\n", e, det);
                matrix.printIndexedBuffer("Wrong matrix");
                return ND4J_STATUS_VALIDATION;
            }
            lowerMatrix.nullify();
            invertLowerMatrix(&matrix, &lowerMatrix);

            for (int k = e * n2, row = 0; k < (e + 1) * n2; k++) {
                output->t<T>(k) = lowerMatrix.template t<T>(row++);
            }
        }

        return Status::OK();
    }

    template <typename T>
    static int upperInverse_(LaunchContext *context, NDArray* input, NDArray* output) {

        auto n = input->sizeAt(-1);
        auto n2 = n * n;

        output->nullify(); // fill up output tensor with zeros
//        auto matrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context); //, block.getWorkspace());
//        auto lowerMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
//        auto upperMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
        auto inputPart = input->allTensorsAlongDimension({-2, -1});
        auto outputPart = output->allTensorsAlongDimension({-2, -1});
        auto totalCount = outputPart.size(); //lengthOf() / n2;
        for (int e = 0; e < totalCount; e++) {
            invertUpperMatrix(inputPart.at(e), outputPart.at(e));
        }
        return Status::OK();
    }

    int inverse(sd::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return inverse_, (context, input, output), FLOAT_TYPES);
    }

    int lowerInverseFunctor(sd::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return lowerInverse_, (context, input, output), FLOAT_TYPES);
    }

    int upperInverseFunctor(sd::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return upperInverse_, (context, input, output), FLOAT_TYPES);
    }

    template <typename T>
    static bool checkCholeskyInput_(sd::LaunchContext * context, NDArray const* input) {
        //std::unique_ptr<NDArray> matrix(NDArrayFactory::create_('c', {n, n}, input->dataType())); //, block.getWorkspace());
        ResultSet lastMatrixList = input->allTensorsAlongDimension({input->rankOf() - 2, input->rankOf()-1});
        for (size_t i = 0; i < lastMatrixList.size(); i++) {
            auto thisMatrix = lastMatrixList.at(i);
            // check for symmetric
            for (Nd4jLong r = 0; r < thisMatrix->rows(); r++)
                for (Nd4jLong c = 0; c < thisMatrix->columns(); c++)
                    if (sd::math::nd4j_abs(thisMatrix->e<T>(r, c) - lastMatrixList.at(i)->e<T>(c,r)) > DataTypeUtils::min<T>()) return false;

            NDArray output = NDArrayFactory::create<T>(0., context);
            if (ND4J_STATUS_OK != determinant(context, thisMatrix, &output)) return false;
            if (output.e<T>(0) <= T(0)) return 0;
            NDArray reversedMatrix(*thisMatrix);
            if (ND4J_STATUS_OK != inverse(context, thisMatrix, &reversedMatrix)) return false;
            if (ND4J_STATUS_OK != determinant(context, &reversedMatrix, &output)) return false;
            if (output.e<T>(0) <= T(0)) return 0;

        }


        return true;
    }

    bool checkCholeskyInput(sd::LaunchContext * context, NDArray const* input) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return checkCholeskyInput_, (context, input), FLOAT_TYPES);
    }

    template <typename T>
    int cholesky_(LaunchContext *context, NDArray* input, NDArray* output, bool inplace) {

        auto n = input->sizeAt(-1);
        auto n2 = n * n;
        auto totalCount = output->lengthOf() / n2;
        if (!inplace)
             output->assign(0.f); // fill up output tensor with zeros only inplace=false

        std::unique_ptr<NDArray> matrix(NDArrayFactory::create_('c', {n, n}, input->dataType(), context)); //, block.getWorkspace());
        std::unique_ptr<NDArray> lowerMatrix(NDArrayFactory::create_('c',{n, n}, input->dataType(), context));

        for (int e = 0; e < totalCount; e++) {

            // fill up matrix
            for (int k = e * n2, l = 0; k < (e + 1) * n2; k++) {
                matrix->p(l++, input->e<T>(k));
            }
            //if (e) // from the second loop need to zero matrix
            lowerMatrix->assign(0.f);

            for (Nd4jLong col = 0; col < n; col++) {
                for (Nd4jLong row = 0; row < col; row++) {
                    T rowSum = 0;
                    for (Nd4jLong k = 0; k < row; ++k)
                        rowSum += (lowerMatrix->e<T>(col, k) * lowerMatrix->e<T>(row, k));
                    lowerMatrix->p(col, row, (matrix->e<T>(row, col) - rowSum) / lowerMatrix->e<double>(row, row));
                }
                T diagonalSum = 0;
                for (Nd4jLong k = 0; k < col;  ++k)
                    diagonalSum += lowerMatrix->e<T>(col, k) * lowerMatrix->e<T>(col, k);
                lowerMatrix->p(col, col, sd::math::nd4j_sqrt<T, T>(matrix->e<T>(col, col) - diagonalSum));
                //nd4j_printf("%i: ", col);
                //lowerMatrix->printIndexedBuffer("Lower matrix");
            }
            for (int k = e * n2, l = 0; k < (e + 1) * n2; k++) {
                output->p(k, lowerMatrix->e<T>(l++));
            }
        }

        return ND4J_STATUS_OK;
    }

    int cholesky(sd::LaunchContext * context, NDArray* input, NDArray* output, bool inplace) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return cholesky_, (context, input, output, inplace), FLOAT_TYPES);
    }

    template <typename T>
    int logdetFunctor_(LaunchContext *context, NDArray* input, NDArray* output) {
        auto tempOutput = input->dup();
        int res = cholesky_<T>(context, input, &tempOutput, false);
        if (res != ND4J_STATUS_OK)
            return res;
        auto n = input->sizeAt(-1);
        auto totalCount = output->lengthOf();
        std::vector<T> d(n);
        ResultSet matricies = tempOutput.allTensorsAlongDimension({input->rankOf()-2, input->rankOf() - 1});

        for (Nd4jLong e = 0; e < totalCount; e++) {
            for (size_t i = 0; i < n; ++i)
                output->t<T>(e) += sd::math::nd4j_log<T,T>(sd::math::nd4j_pow<T,T,T>(matricies.at(e)->t<T>(i, i), T(2)));
        }
        return ND4J_STATUS_OK;
    }

    int logdetFunctor(sd::LaunchContext * context, NDArray* input, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return logdetFunctor_, (context, input, output), FLOAT_TYPES);
    }

    int lup(sd::LaunchContext * context, NDArray* input, NDArray* compound, NDArray* permutation) {
        BUILD_DOUBLE_SELECTOR(input->dataType(), permutation->dataType(), lup_, (context, input, compound, permutation), FLOAT_NATIVE, INDEXING_TYPES);
        return Status::OK();
    }

}
}
}
