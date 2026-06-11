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
//  @author George A. Shulinok <sgazeos@gmail.com>
//
#include <ops/declarable/helpers/matrix_band.h>
#if NOT_EXCLUDED(OP_matrix_band)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
void matrixBandPart_(NDArray* input, NDArray* output, sd::LongType lowerBand, sd::LongType upperBand) {
  const int rank = input->rankOf();
  // Last two dims are the matrix dimensions; all leading dims are batch.
  sd::LongType M = input->sizeAt(rank - 2);
  sd::LongType N = input->sizeAt(rank - 1);
  sd::LongType matrixSize = M * N;

  // Compute numMatrices from batch dimensions (everything except last two).
  sd::LongType numMatrices = 1;
  for (int i = 0; i < rank - 2; ++i) numMatrices *= input->sizeAt(i);

  PRAGMA_OMP_PARALLEL_FOR
  for (sd::LongType e = 0; e < numMatrices; ++e) {
    sd::LongType offset = e * matrixSize;

    // Copy this matrix from input to output first.
    if (output->buffer() != input->buffer()) {
      for (sd::LongType idx = 0; idx < matrixSize; ++idx)
        output->r<T>(offset + idx) = input->t<T>(offset + idx);
    }

    // Zero elements outside the band using flat 1-D indexing on the main arrays.
    for (sd::LongType row = 0; row < M; ++row) {
      for (sd::LongType col = 0; col < N; ++col) {
        bool inBand = true;
        if (lowerBand >= 0 && (row - col) > lowerBand) inBand = false;
        if (upperBand >= 0 && (col - row) > upperBand) inBand = false;
        if (!inBand) output->r<T>(offset + row * N + col) = T(0);
      }
    }
  }
}

void matrixBandPart(sd::LaunchContext* context, NDArray* input, NDArray* output, sd::LongType lowerBand,
                    sd::LongType upperBand) {
  BUILD_SINGLE_SELECTOR(input->dataType(), matrixBandPart_, (input, output, lowerBand, upperBand), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE( void matrixBandPart_,
                      (NDArray * input, NDArray* output, sd::LongType lowerBand, sd::LongType upperBand),
                      SD_FLOAT_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif