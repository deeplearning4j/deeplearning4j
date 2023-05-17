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
  // TO DO: retrieve all 2D submatricies with last dimensions and process them with given bands
  sd::LongType M = input->sizeAt(-2);
  sd::LongType N = input->sizeAt(-1);
  sd::LongType lastDim = input->rankOf() - 1;
  sd::LongType preLastDim = input->rankOf() - 2;
  ResultSet listOut = output->allTensorsAlongDimension({(int)preLastDim, (int)lastDim});
  ResultSet listDiag = input->allTensorsAlongDimension({(int)preLastDim, (int)lastDim});
  for (sd::LongType e = 0; e < static_cast<sd::LongType>(listOut.size()); ++e) {
    NDArray* inputMatrix = listDiag.at(e);
    NDArray* outputMatrix = listOut.at(e);
    if (outputMatrix != inputMatrix)  // if not inplace
      outputMatrix->assign(inputMatrix);
    if (lowerBand >= 0) {
      for (sd::LongType row = 0; row < inputMatrix->rows(); ++row) {
        for (sd::LongType col = 0; col < row; ++col) {
          if ((row - col) > lowerBand) outputMatrix->p(row, col, 0.);

        }

      }
    }
    if (upperBand >= 0) {
      for (sd::LongType col = 0; col < inputMatrix->columns(); ++col) {
        for (sd::LongType row = 0; row < col; ++row) {
          if ((col - row) > upperBand) outputMatrix->p(row, col, 0.);
        }
      }
    }
  }
}

void matrixBandPart(sd::LaunchContext* context, NDArray* input, NDArray* output, sd::LongType lowerBand,
                    sd::LongType upperBand) {
  BUILD_SINGLE_SELECTOR(input->dataType(), matrixBandPart_, (input, output, lowerBand, upperBand), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void matrixBandPart_,
                      (NDArray * input, NDArray* output, sd::LongType lowerBand, sd::LongType upperBand),
                      SD_FLOAT_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif