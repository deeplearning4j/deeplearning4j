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
#include <execution/Threads.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/helpers/top_k.h>
#if NOT_EXCLUDED(OP_lup)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void swapRows_(NDArray* matrix, sd::LongType theFirst, sd::LongType theSecond) {
  if (theFirst != theSecond)
    for (sd::LongType i = 0; i < matrix->columns(); i++) {
      math::sd_swap(matrix->r<T>(theFirst, i), matrix->r<T>(theSecond, i));
    }
}
BUILD_SINGLE_TEMPLATE(template void swapRows_, (NDArray * matrix, sd::LongType theFirst, sd::LongType theSecond), SD_FLOAT_TYPES);

template <typename T>
static void swapRows(T* matrixBuf, sd::LongType const* matrixShape, sd::LongType theFirst, sd::LongType theSecond) {
  if (theFirst != theSecond) {
    auto n = shape::sizeAt(matrixShape, static_cast<sd::LongType>(-1));

    auto loop = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        sd::LongType theFirstPos[] = {theFirst, i};
        sd::LongType theSecondPos[] = {theSecond, i};

        sd::LongType theFirstIndex;
        COORDS2INDEX(shape::rank(matrixShape), shape::stride(matrixShape), theFirstPos, theFirstIndex);

        sd::LongType theSecondIndex;
        COORDS2INDEX(shape::rank(matrixShape), shape::stride(matrixShape), theSecondPos, theSecondIndex);

        math::sd_swap(matrixBuf[theFirstIndex], matrixBuf[theSecondIndex]);
      }
    };

    samediff::Threads::parallel_tad(loop, 0, n, 1);
  }
}

void swapRows(NDArray* matrix, sd::LongType theFirst, sd::LongType theSecond) {
  BUILD_SINGLE_SELECTOR(matrix->dataType(), swapRows_, (matrix, theFirst, theSecond), SD_FLOAT_TYPES);
}

template <typename T>
static void invertLowerMatrix_(NDArray* inputMatrix, NDArray* invertedMatrix) {
  sd::LongType n = inputMatrix->rows();
  invertedMatrix->setIdentity();

  if (inputMatrix->isIdentityMatrix()) return;

  auto invertDiagonals = PRAGMA_THREADS_FOR {
    for (sd::LongType i = start; i < stop; i += increment) invertedMatrix->r<T>(i, i) /= inputMatrix->t<T>(i, i);
  };

  auto invertSubDiagonals = PRAGMA_THREADS_FOR {
    for (sd::LongType i = start; i < stop; i += increment)
      invertedMatrix->r<T>(i, i - 1) -=
          (inputMatrix->t<T>(i, i - 1) * invertedMatrix->t<T>(i - 1, i - 1) / inputMatrix->t<T>(i, i));
  };

  samediff::Threads::parallel_for(invertDiagonals, 0, n, 1);
  samediff::Threads::parallel_for(invertSubDiagonals, 1, n, 1);

  for (sd::LongType i = 1; i < n; i++) {
    for (sd::LongType j = 0; j < i - 1; j++)
      for (sd::LongType k = 0; k < i; k++)
        invertedMatrix->r<T>(i, j) -=
            ((invertedMatrix->t<T>(k, j) * inputMatrix->t<T>(i, k) / inputMatrix->t<T>(i, i)));
  }
}

BUILD_SINGLE_TEMPLATE(template void invertLowerMatrix_, (NDArray * inputMatrix, NDArray* invertedMatrix);
                      , SD_FLOAT_TYPES);

void invertLowerMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
  BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), invertLowerMatrix_, (inputMatrix, invertedMatrix), SD_FLOAT_TYPES);
}

template <typename T>
static void _invertUpperMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
  sd::LongType n = inputMatrix->rows();
  invertedMatrix->setIdentity();

  if (inputMatrix->isIdentityMatrix()) {  // the inverse for I is I
    return;
  }

  auto invertDiagonals = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i += increment) invertedMatrix->r<T>(i, i) /= inputMatrix->t<T>(i, i);
  };

  // PRAGMA_OMP_PARALLEL_FOR_IF(n > Environment::getInstance().elementwiseThreshold())
  auto invertUpDiagonals = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i += increment)
      invertedMatrix->r<T>(i, i + 1) -=
          (inputMatrix->t<T>(i, i + 1) * invertedMatrix->t<T>(i + 1, i + 1) / inputMatrix->t<T>(i, i));
  };

  samediff::Threads::parallel_for(invertDiagonals, 0, n, 1);
  samediff::Threads::parallel_for(invertUpDiagonals, 0, n - 1, 1);

  for (auto i = n - 2; i >= 0; i--) {
    for (auto j = i + 2; j < n; j++)
      for (auto k = i; k < n; k++)
        invertedMatrix->r<T>(i, j) -=
            ((invertedMatrix->t<T>(k, j) * inputMatrix->t<T>(i, k) / inputMatrix->t<T>(i, i)));
  }
}

BUILD_SINGLE_TEMPLATE(template void _invertUpperMatrix, (NDArray * inputMatrix, NDArray* invertedMatrix);
                      , SD_FLOAT_TYPES);

void invertUpperMatrix(NDArray* inputMatrix, NDArray* invertedMatrix) {
  BUILD_SINGLE_SELECTOR(inputMatrix->dataType(), _invertUpperMatrix, (inputMatrix, invertedMatrix), SD_FLOAT_TYPES);
}

template <typename T, typename I>
static NDArray lup_(LaunchContext* context, NDArray* input, NDArray* compound, NDArray* permutation) {
  const sd::LongType rowNum = input->rows();
  const sd::LongType columnNum = input->columns();

  NDArray determinant = NDArrayFactory::create<T>(1.f, context);
  NDArray compoundMatrix = *input;                   // copy
  NDArray permutationMatrix(input, false, context);  // has same shape as input and contiguous strides
  permutationMatrix.setIdentity();

  T pivotValue;  // = T(0.0);
  sd::LongType pivot;     // = -1;
  sd::LongType swapCount = 0;

  for (sd::LongType i = 0; i < rowNum; i++) {
    pivotValue = T(0.0);
    pivot = -1;
    for (sd::LongType rowCounter = i; rowCounter < rowNum; rowCounter++) {
      if (sd::math::sd_abs<T,T>(compoundMatrix.t<T>(rowCounter, i)) > pivotValue) {
        pivotValue = sd::math::sd_abs<T,T>(compoundMatrix.t<T>(rowCounter, i));
        pivot = rowCounter;
      }
    }

    if (pivotValue > DataTypeUtils::min_positive<T>()) {
      swapRows(&compoundMatrix, pivot, i);
      swapRows(&permutationMatrix, pivot, i);
      if (pivot != i) swapCount++;

      for (sd::LongType j = i + 1; j < rowNum; j++) {
        compoundMatrix.r<T>(j, i) /= compoundMatrix.t<T>(i, i);
        for (sd::LongType k = i + 1; k < rowNum; k++) {
          compoundMatrix.r<T>(j, k) -= compoundMatrix.t<T>(j, i) * compoundMatrix.t<T>(i, k);
        }
      }
    }
  }

  for (sd::LongType e = 0; e < rowNum; e++) {
    determinant *= compoundMatrix.e<T>(e, e);
  }
  if (swapCount % 2) determinant = -determinant;
  if (compound != nullptr) compound->assign(compoundMatrix);
  if (permutation != nullptr) {
    auto permutaionVector = NDArrayFactory::create('c', {rowNum}, DataTypeUtils::fromT<I>(), input->getContext());
    for (auto i = 0; i < rowNum; i++) {
      for (auto j = 0; j < columnNum; j++) {
        if (permutationMatrix.t<T>(i, j) != 0) {
          permutaionVector.template r<I>(i) = j;
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

BUILD_DOUBLE_TEMPLATE(template NDArray lup_,
                      (LaunchContext * context, NDArray* input, NDArray* output, NDArray* permutation), SD_FLOAT_TYPES,
                      SD_INDEXING_TYPES);
/*
 * lu decomposition with naive algorithm with partial pivoting
 * */
template <typename T, typename I>
static I argmaxCol(I column, T* compoundBuffer, sd::LongType const* compoundShape) {
  auto rowNum = shape::sizeAt(compoundShape, static_cast<sd::LongType>(0));
  sd::LongType xInitial[] = {column, column};
  sd::LongType xInitialIndex;
  COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), xInitial, xInitialIndex);
  auto maxValue = T(0);
  auto result = -1;
  auto start = column;
  auto stop = rowNum;
  auto increment = 1;
  for (auto rowCounter = start; rowCounter < stop; rowCounter++) {
    sd::LongType xPos[] = {rowCounter, column};
    sd::LongType xIndex;
    COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), xPos, xIndex);
    if (sd::math::sd_abs<T,T>(compoundBuffer[xIndex]) > maxValue) {
      maxValue = sd::math::sd_max(maxValue, sd::math::sd_abs<T,T>(compoundBuffer[xIndex]));
      result = rowCounter;
    }
  }

  return result;
}

template <typename T>
void processColumns(sd::LongType currentRow, sd::LongType rowNum, T* compoundBuf, sd::LongType const* compoundShape) {
  sd::LongType xDiag[] = {currentRow, currentRow};
  sd::LongType diagIndex;
  COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), xDiag, diagIndex);
  auto loop = PRAGMA_THREADS_FOR {
    for (auto j = start; j < stop; j++) {
      sd::LongType xRow[] = {j, currentRow};
      sd::LongType rowIndex;
      COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), xRow, rowIndex);
      compoundBuf[rowIndex] /= compoundBuf[diagIndex];  // output->t<T>(i, i);
      for (sd::LongType k = currentRow + 1; k < rowNum; k++) {
        sd::LongType yRow[] = {j, k};
        sd::LongType yCol[] = {currentRow, k};
        sd::LongType rowIndexY, colIndex;
        COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), yRow, rowIndexY);
        COORDS2INDEX(shape::rank(compoundShape), shape::stride(compoundShape), yCol, colIndex);
        compoundBuf[rowIndexY] -= compoundBuf[rowIndex] * compoundBuf[colIndex];
      }
    }
  };
  samediff::Threads::parallel_tad(loop, currentRow + 1, rowNum, 1);
}

template <typename T>
static void doolitleLU(LaunchContext* context, NDArray* compound, sd::LongType rowNum) {
  auto input = compound->dup();
  compound->nullify();

  // Decomposing matrix into Upper and Lower
  // triangular matrix
  for (auto i = 0; i < rowNum; i++) {
    // Upper Triangular
    for (auto k = i; k < rowNum; k++) {
      // Summation of L(i, j) * U(j, k)
      sd::LongType sum = 0;
      for (sd::LongType j = 0; j < i; j++) sum += compound->t<T>(i, j) * compound->t<T>(j, k);

      // Evaluating U(i, k)
      compound->r<T>(i, k) = input.t<T>(i, k) - sum;
    }

    // Lower Triangular
    for (sd::LongType k = i + 1; k < rowNum; k++) {
      // Summation of L(k, j) * U(j, i)
      sd::LongType sum = 0;
      for (sd::LongType j = 0; j < i; j++) sum += compound->t<T>(k, j) * compound->t<T>(j, i);

      // Evaluating L(k, i)
      compound->r<T>(k, i) = (input.t<T>(k, i) - sum) / compound->t<T>(i, i);
    }
  }
}

template <typename T, typename I>
static void luNN_(LaunchContext* context, NDArray* compound, NDArray* permutation, sd::LongType rowNum) {
  if (permutation) {  // LUP algorithm
    permutation->linspace(0);
    auto permutationBuf = permutation->bufferAsT<I>();
    auto compoundBuf = compound->bufferAsT<T>();
    auto compoundShape = compound->shapeInfo();
    auto permutationShape = permutation->shapeInfo();
    for (sd::LongType i = 0; i < rowNum - 1; i++) {
      auto pivotIndex = argmaxCol(i, compoundBuf, compoundShape);
      if (pivotIndex < 0) {
        THROW_EXCEPTION("helpers::luNN_: input matrix is singular.");
      }
      sd::LongType firstIndexCoords[SD_MAX_RANK];
      sd::LongType secondIndexCoords[SD_MAX_RANK];
      sd::LongType firstIndex;
      sd::LongType secondIndex;

      INDEX2COORDS(i, shape::rank(permutationShape), permutationShape, firstIndexCoords);
      COORDS2INDEX(shape::rank(permutationShape), shape::shapeOf(permutationShape), firstIndexCoords, firstIndex);
      INDEX2COORDS(pivotIndex, shape::rank(permutationShape), permutationShape, secondIndexCoords);
      COORDS2INDEX(shape::rank(permutationShape), shape::shapeOf(permutationShape), secondIndexCoords, secondIndex);

      math::sd_swap(permutationBuf[firstIndex], permutationBuf[secondIndex]);
      swapRows(compoundBuf, compoundShape, i, pivotIndex);

      processColumns(i, rowNum, compoundBuf, compoundShape);
    }
  } else {  // Doolitle algorithm with LU decomposition
    doolitleLU<T>(context, compound, rowNum);
  }
}

template <typename T, typename I>
static void lu_(LaunchContext* context, NDArray* input, NDArray* output, NDArray* permutationVectors) {
  auto n = input->sizeAt(-1);

  output->assign(*input);  // fill up output tensor with zeros
  ResultSet outputs = output->allTensorsAlongDimension({-2, -1});
  ResultSet permutations;
  if (permutationVectors) permutations = permutationVectors->allTensorsAlongDimension({-1});

  auto loop = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      luNN_<T, I>(context, outputs.at(i), permutationVectors ? permutations.at(i) : nullptr, n);
    }
  };
  samediff::Threads::parallel_for(loop, 0, outputs.size(), 1);
}

void lu(LaunchContext* context, NDArray* input, NDArray* output, NDArray* permutation) {
  BUILD_DOUBLE_SELECTOR(input->dataType(), permutation ? permutation->dataType() : DataType::INT32, lu_,
                        (context, input, output, permutation), SD_FLOAT_TYPES, SD_INDEXING_TYPES);
}



template <typename T>
static sd::Status determinant_(LaunchContext* context, NDArray* input, NDArray* output) {
  sd::LongType n = input->sizeAt(-1);
  sd::LongType n2 = n * n;

  auto matrix =
      NDArrayFactory::create(input->ordering(), {n, n}, input->dataType(), context);  //, block.getWorkspace());

  for (sd::LongType e = 0; e < output->lengthOf(); e++) {
    for (sd::LongType k = e * n2, row = 0; k < (e + 1) * n2; ++k, ++row) matrix.p(row, input->e<T>(k));
    output->p(e, lup_<T, sd::LongType>(context, &matrix, (NDArray*)nullptr, (NDArray*)nullptr));
  }

  return sd::Status::OK;
}

sd::Status determinant(sd::LaunchContext* context, NDArray* input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return determinant_, (context, input, output), SD_FLOAT_TYPES);
}

template <typename T>
sd::Status logAbsDeterminant_(LaunchContext* context, NDArray* input, NDArray* output) {
  sd::LongType n = input->sizeAt(-1);
  sd::LongType n2 = n * n;

  NDArray matrix =
      NDArrayFactory::create(input->ordering(), {n, n}, input->dataType(), context);  //, block.getWorkspace());
  for (sd::LongType e = 0; e < output->lengthOf(); e++) {
    for (sd::LongType k = e * n2, row = 0; k < (e + 1) * n2; ++k, ++row) {
      matrix.p(row, input->e<T>(k));
    }
    NDArray det = lup_<T, sd::LongType>(context, &matrix, (NDArray*)nullptr, (NDArray*)nullptr);
    if (det.e<T>(0) != 0.f) output->p(e, sd::math::sd_log<T, T>(sd::math::sd_abs<T,T>(det.t<T>(0))));
  }

  return sd::Status::OK;
}

sd::Status logAbsDeterminant(sd::LaunchContext* context, NDArray* input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return logAbsDeterminant_, (context, input, output), SD_FLOAT_TYPES);
}

template <typename T>
static sd::Status inverse_(LaunchContext* context, NDArray* input, NDArray* output) {
  auto n = input->sizeAt(-1);
  auto n2 = n * n;
  auto totalCount = output->lengthOf() / n2;
  float zerof = 0.f;
  output->assign(zerof);  // fill up output tensor with zeros
  auto matrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);    //, block.getWorkspace());
  auto compound = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);  //, block.getWorkspace());
  auto permutation = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
  auto lowerMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
  auto upperMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
  float zero = 0.f;
  for (sd::LongType e = 0; e < totalCount; e++) {
    if (e) matrix.assign(zero);

    for (sd::LongType k = e * n2, row = 0; k < (e + 1) * n2; k++) {
      matrix.p(row++, input->e<T>(k));
    }
    T det = lup_<T, sd::LongType>(context, &matrix, &compound, &permutation).template e<T>(0);

    // FIXME: and how this is going to work on float16?
    if (sd::math::sd_abs<T,T>(det) < T(0.000001)) {
      sd_printf("matrix_inverse: The matrix %i has no inverse due determinant is %lf. Quiting...\n", e, det);
      return sd::Status::VALIDATION;
    }
    lowerMatrix.setIdentity();     // set up U to identity matrix
    for (sd::LongType k = 1; k < n; k++) {  // and then put all values under main diagonal on to it
      for (sd::LongType j = 0; j < k; j++) lowerMatrix.template r<T>(k, j) = compound.template t<T>(k, j);
    }
    upperMatrix.setIdentity();     // set up U to identity matrix
    for (sd::LongType k = 0; k < n; k++) {  // and then put all values under main diagonal on to it
      for (sd::LongType j = k; j < n; j++) upperMatrix.template r<T>(k, j) = compound.template t<T>(k, j);
    }
    invertUpperMatrix(&upperMatrix, &matrix);

    invertLowerMatrix(&lowerMatrix, &upperMatrix);

    sd::MmulHelper::mmul(&matrix, &upperMatrix, &compound, 1.0, 0.0);
    sd::MmulHelper::mmul(&compound, &permutation, &matrix, 1.0, 0.0);
    for (sd::LongType k = e * n2, row = 0; k < (e + 1) * n2; k++) {
      output->r<T>(k) = matrix.template t<T>(row++);
    }
  }

  return sd::Status::OK;
}

template <typename T>
static sd::Status lowerInverse_(LaunchContext* context, NDArray* input, NDArray* output) {
  auto n = input->sizeAt(-1);
  auto n2 = n * n;
  auto totalCount = output->lengthOf() / n2;
  float zero = 0.f;
  output->assign(zero);  // fill up output tensor with zeros
  auto matrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
  auto compound = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
  auto permutation = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
  auto lowerMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
  auto upperMatrix = NDArrayFactory::create('c', {n, n}, DataTypeUtils::fromT<T>(), context);
  for (sd::LongType e = 0; e < totalCount; e++) {
    if (e) matrix.assign(zero);

    for (sd::LongType k = e * n2, row = 0; k < (e + 1) * n2; k++) {
      matrix.p(row++, input->e<T>(k));
    }
    T det = T(1.f);
    for (auto i = 0; i < n; i++) {
      det *= matrix.template t<T>(i, i);
    }

    // FIXME: and how this is going to work on float16?
    if (sd::math::sd_abs<T,T>(det) < T(0.000001)) {
      sd_printf("matrix_inverse: The matrix %i has no inverse due determinant is %lf. Quitting...\n", e, det);
      return sd::Status::VALIDATION;
    }
    lowerMatrix.nullify();
    invertLowerMatrix(&matrix, &lowerMatrix);

    for (sd::LongType k = e * n2, row = 0; k < (e + 1) * n2; k++) {
      output->r<T>(k) = lowerMatrix.template t<T>(row++);
    }
  }

  return sd::Status::OK;
}

template <typename T>
static sd::Status upperInverse_(LaunchContext* context, NDArray* input, NDArray* output) {
  auto n = input->sizeAt(-1);
  auto n2 = n * n;

  output->nullify();  // fill up output tensor with zeros
  auto inputPart = input->allTensorsAlongDimension({-2, -1});
  auto outputPart = output->allTensorsAlongDimension({-2, -1});
  auto totalCount = outputPart.size();
  for (sd::LongType e = 0; e < totalCount; e++) {
    invertUpperMatrix(inputPart.at(e), outputPart.at(e));
  }
  return sd::Status::OK;
}

sd::Status inverse(sd::LaunchContext* context, NDArray* input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return inverse_, (context, input, output), SD_FLOAT_TYPES);
}

sd::Status lowerInverseFunctor(sd::LaunchContext* context, NDArray* input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return lowerInverse_, (context, input, output), SD_FLOAT_TYPES);
}

sd::Status upperInverseFunctor(sd::LaunchContext* context, NDArray* input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return upperInverse_, (context, input, output), SD_FLOAT_TYPES);
}

template <typename T>
static bool checkCholeskyInput_(sd::LaunchContext* context, NDArray * input) {
  ResultSet lastMatrixList = input->allTensorsAlongDimension({input->rankOf() - 2, input->rankOf() - 1});
  for (sd::LongType i = 0; i < lastMatrixList.size(); i++) {
    auto thisMatrix = lastMatrixList.at(i);
    // check for symmetric
    for (sd::LongType r = 0; r < thisMatrix->rows(); r++)
      for (sd::LongType c = 0; c < thisMatrix->columns(); c++)
        if (sd::math::sd_abs<T,T>(thisMatrix->e<T>(r, c) - lastMatrixList.at(i)->e<T>(c, r)) >
            DataTypeUtils::min_positive<T>())
          return false;

    NDArray output = NDArrayFactory::create<T>(0., context);
    if (sd::Status::OK != determinant(context, thisMatrix, &output)) return false;
    if (output.e<T>(0) <= T(0)) return 0;
    NDArray reversedMatrix(*thisMatrix);
    if (sd::Status::OK != inverse(context, thisMatrix, &reversedMatrix)) return false;
    if (sd::Status::OK != determinant(context, &reversedMatrix, &output)) return false;
    if (output.e<T>(0) <= T(0)) return 0;
  }

  return true;
}

bool checkCholeskyInput(sd::LaunchContext* context, NDArray * input) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return checkCholeskyInput_, (context, input), SD_FLOAT_TYPES);
}

template <typename T>
sd::Status cholesky_(LaunchContext* context, NDArray* input, NDArray* output, bool inplace) {
  auto n = input->sizeAt(-1);
  auto n2 = n * n;
  auto totalCount = output->lengthOf() / n2;
  float zero = 0.f;
  if (!inplace) output->assign(zero);  // fill up output tensor with zeros only inplace=false

  std::vector<sd::LongType> shape = {n,n};
  std::unique_ptr<NDArray> matrix(
      NDArrayFactory::create_('c', shape, input->dataType(), context));  //, block.getWorkspace());
  std::unique_ptr<NDArray> lowerMatrix(NDArrayFactory::create_('c',shape, input->dataType(), context));

  for (sd::LongType e = 0; e < totalCount; e++) {
    // fill up matrix
    for (sd::LongType k = e * n2, l = 0; k < (e + 1) * n2; k++) {
      matrix->p(l++, input->e<T>(k));
    }
    float zero = 0.f;
    // if (e) // from the second loop need to zero matrix
    lowerMatrix->assign(zero);

    for (sd::LongType col = 0; col < n; col++) {
      for (sd::LongType row = 0; row < col; row++) {
        T rowSum = 0;
        for (sd::LongType k = 0; k < row; ++k) rowSum += (lowerMatrix->e<T>(col, k) * lowerMatrix->e<T>(row, k));
        lowerMatrix->p(col, row, (matrix->e<T>(row, col) - rowSum) / lowerMatrix->e<T>(row, row));
      }
      T diagonalSum = 0;
      for (sd::LongType k = 0; k < col; ++k) diagonalSum += lowerMatrix->e<T>(col, k) * lowerMatrix->e<T>(col, k);
      lowerMatrix->p(col, col, sd::math::sd_sqrt<T, T>(matrix->e<T>(col, col) - diagonalSum));
    }
    for (sd::LongType k = e * n2, l = 0; k < (e + 1) * n2; k++) {
      output->p(k, lowerMatrix->e<T>(l++));
    }
  }

  return sd::Status::OK;
}

sd::Status cholesky(sd::LaunchContext* context, NDArray* input, NDArray* output, bool inplace) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return cholesky_, (context, input, output, inplace), SD_FLOAT_TYPES);
}

template <typename T>
sd::Status logdetFunctor_(LaunchContext* context, NDArray* input, NDArray* output) {
  auto tempOutput = input->dup();
  auto res = cholesky_<T>(context, input, &tempOutput, false);
  if (res != sd::Status::OK) return res;
  auto n = input->sizeAt(-1);
  auto totalCount = output->lengthOf();
  std::vector<T> d(n);
  ResultSet matrices = tempOutput.allTensorsAlongDimension({input->rankOf() - 2, input->rankOf() - 1});

  for (sd::LongType e = 0; e < totalCount; e++) {
    for (size_t i = 0; i < n; ++i)
      output->r<T>(e) += sd::math::sd_log<T, T>(sd::math::sd_pow<T, T, T>(matrices.at(e)->t<T>(i, i), T(2)));
  }
  return sd::Status::OK;
}

sd::Status logdetFunctor(sd::LaunchContext* context, NDArray* input, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), return logdetFunctor_, (context, input, output), SD_FLOAT_TYPES);
}

sd::Status lup(sd::LaunchContext* context, NDArray* input, NDArray* compound, NDArray* permutation) {
  BUILD_DOUBLE_SELECTOR(input->dataType(), permutation->dataType(), lup_, (context, input, compound, permutation),
                        SD_FLOAT_NATIVE, SD_INDEXING_TYPES);
  return sd::Status::OK;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif