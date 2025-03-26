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
// @author George A. Shulinok <sgazeos@gmail.com>, created on 4/18/2019
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/BarnesHutTsne.h>

namespace sd {
namespace ops {
namespace helpers {

sd::LongType barnes_row_count(NDArray* rowP, NDArray* colP, sd::LongType N, NDArray& rowCounts) {
  int* pRowCounts = reinterpret_cast<int*>(rowCounts.buffer());
  int const* pRows = reinterpret_cast<int const*>(rowP->buffer());
  int const* pCols = reinterpret_cast<int const*>(colP->buffer());
  for (sd::LongType n = 0; n < N; n++) {
    int begin = pRows[n];    //->e<int>(n);
    int end = pRows[n + 1];  // rowP->e<int>(n + 1);
    for (int i = begin; i < end; i++) {
      bool present = false;
      for (int m = pRows[pCols[i]]; m < pRows[pCols[i] + 1]; m++)
        if (pCols[m] == n) {
          present = true;
          break;
        }

      ++pRowCounts[n];

      if (!present) ++pRowCounts[pCols[i]];
    }
  }
  NDArray numElementsArr = rowCounts.sumNumber();
  auto numElements = numElementsArr.e<sd::LongType>(0);
  return numElements;
}

template <typename T>
static void barnes_symmetrize_(NDArray* rowP, NDArray* colP, NDArray* valP, sd::LongType N,
                               NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts) {

  int const* pRows = reinterpret_cast<int const*>(rowP->buffer());
  int* symRowP = reinterpret_cast<int*>(outputRows->buffer());
  symRowP[0] = 0;
  for (sd::LongType n = 0; n < N; n++) symRowP[n + 1] = symRowP[n] + rowCounts->e<int>(n);

  int* symColP = reinterpret_cast<int*>(outputCols->buffer());
  int const* pCols = reinterpret_cast<int const*>(colP->buffer());
  T const* pVals = reinterpret_cast<T const*>(valP->buffer());
  T* pOutput = reinterpret_cast<T*>(outputVals->buffer());
  std::vector<int> offset(N);

  for (sd::LongType n = 0; n < N; n++) {
    int begin = pRows[n];
    int bound = pRows[n + 1];

    for (int i = begin; i < bound; i++) {
      bool present = false;
      int colPI = pCols[i];
      int start = pRows[colPI];
      int end = pRows[colPI + 1];

      for (int m = start; m < end; m++) {
        if (pCols[m] == n) {
          present = true;
          if (n <= colPI) {
            symColP[symRowP[n] + offset[n]] = colPI;
            symColP[symRowP[colPI] + offset[colPI]] = n;
            pOutput[symRowP[n] + offset[n]] = pVals[i] + pVals[m];
            pOutput[symRowP[colPI] + offset[colPI]] = pVals[i] + pVals[m];
          }
        }
      }

      if (!present) {
        symColP[symRowP[n] + offset[n]] = colPI;
        symColP[symRowP[pCols[i]] + offset[colPI]] = n;
        pOutput[symRowP[n] + offset[n]] = pVals[i];
        pOutput[symRowP[colPI] + offset[colPI]] = pVals[i];
        //}
      }
      // Update offsets
      if (!present || (present && n <= colPI)) {
        ++offset[n];

        if (colPI != n) ++offset[colPI];
      }
    }
  }
}
void barnes_symmetrize(NDArray* rowP, NDArray* colP, NDArray* valP, sd::LongType N,
                       NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts) {
  // Divide the result by two
  BUILD_SINGLE_SELECTOR(valP->dataType(), barnes_symmetrize_,
                        (rowP, colP, valP, N, outputRows, outputCols, outputVals, rowCounts), SD_NUMERIC_TYPES);

  *outputVals /= 2.0;
}
BUILD_SINGLE_TEMPLATE(template void barnes_symmetrize_,
                      (NDArray* rowP, NDArray* colP, NDArray* valP, sd::LongType N,
                       NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts),
                      SD_NUMERIC_TYPES);

template <typename T>
static void barnes_edge_forces_(NDArray* rowP, NDArray * colP, NDArray * valP, int N,
                                NDArray * data, NDArray* output) {
  T const* dataP = reinterpret_cast<T const*>(data->buffer());
  T const* vals = reinterpret_cast<T const*>(valP->buffer());
  T* outputP = reinterpret_cast<T*>(output->buffer());
  int colCount = data->columns();

  auto rowSize = sizeof(T) * colCount;

  auto func = PRAGMA_THREADS_FOR {
    for (auto n = start; n < stop; n++) {
      int s = rowP->e<int>(n);
      int end = rowP->e<int>(n + 1);
      int shift = n * colCount;
      for (int i = s; i < end; i++) {
        T const* thisSlice = dataP + colP->e<int>(i) * colCount;
        T res = 1;

        for (int k = 0; k < colCount; k++) {
          auto tempVal = dataP[shift + k] - thisSlice[k];  // thisSlice[k];
          res += tempVal * tempVal;
        }

        res = vals[i] / res;
        for (int k = 0; k < colCount; k++) outputP[shift + k] += ((dataP[shift + k] - thisSlice[k]) * res);
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, N);
}

void barnes_edge_forces(NDArray* rowP, NDArray * colP, NDArray * valP, int N, NDArray* output,
                        NDArray& data) {
  // Loop over all edges in the graph
  BUILD_SINGLE_SELECTOR(output->dataType(), barnes_edge_forces_, (rowP, colP, valP, N, &data, output), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void barnes_edge_forces_,
                      (NDArray* rowP, NDArray * colP, NDArray * valP, int N, NDArray * data,
                       NDArray* output),
                      SD_FLOAT_TYPES);

template <typename T>
static void barnes_gains_(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output) {
  auto gainsInternal = LAMBDA_TTT(x, grad, eps) {
    T res = sd::math::sd_sign<T, T>(grad) != sd::math::sd_sign<T, T>(eps) ? x + T(.2) : x * T(.8);
    if (res < .01) res = static_cast<T>(.01);
    return res;
  });

  input->applyTriplewiseLambda<T>(gradX, epsilon, gainsInternal, output);
}

void barnes_gains(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), barnes_gains_, (input, gradX, epsilon, output), SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void barnes_gains_, (NDArray * input, NDArray* gradX, NDArray* epsilon, NDArray* output),
                      SD_NUMERIC_TYPES);

bool cell_contains(NDArray* corner, NDArray* width, NDArray* point, sd::LongType dimension) {
  auto cornerMinusWidth = *corner - *width;
  auto cornerPlusWidth = *corner + *width;

  for (sd::LongType i = 0; i < dimension; i++) {
    if (cornerMinusWidth.e<double>(i) > point->e<double>(i)) return false;
    if (cornerPlusWidth.e<double>(i) < point->e<double>(i)) return false;
  }

  return true;
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
