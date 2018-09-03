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
//  @author George A. Shulinok <sgazeos@gmail.com>
//
#include <ops/declarable/helpers/matrix_band.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void matrixBandPart(NDArray<T>* input, NDArray<T>* output, Nd4jLong lowerBand, Nd4jLong upperBand) {
        // TO DO: retrieve all 2D submatricies with last dimensions and process them with given bands
        Nd4jLong M = input->sizeAt(-2);
        Nd4jLong N = input->sizeAt(-1);
        Nd4jLong lastDim = input->rankOf() - 1;
        Nd4jLong preLastDim = input->rankOf() - 2;
        std::unique_ptr<ResultSet<T>> listOut(output->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
        std::unique_ptr<ResultSet<T>> listDiag(input->allTensorsAlongDimension({(int)preLastDim, (int)lastDim}));
        for (Nd4jLong e = 0; e < listOut->size(); ++e) {
            NDArray<T>* inputMatrix = listDiag->at(e);
            NDArray<T>* outputMatrix = listOut->at(e);
            if (outputMatrix != inputMatrix) // if not inplace
                outputMatrix->assign(inputMatrix);
            if (lowerBand >= 0) {
                for (Nd4jLong row = 0; row < inputMatrix->rows(); ++row) {
                    for (Nd4jLong col = 0; col < row; ++col) {
                        if ((row - col) > lowerBand)
                            (*outputMatrix)(row, col) = (T)0.;
//                        else
  //                          (*outputMatrix)(row, col) = (*inputMatrix)(row, col);
                    }
//                    in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) && (num_upper < 0 || (n-m) <= num_upper).
                }
            }
            if (upperBand >= 0) {
                for (Nd4jLong col = 0; col < inputMatrix->columns(); ++col) {
                    for (Nd4jLong row = 0; row < col; ++row) {
                        if ((col - row) > upperBand)
                            (*outputMatrix)(row, col) = (T)0.;
//                        else
  //                          (*outputMatrix)(row, col) = (*inputMatrix)(row, col);
                    }
//                    in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) && (num_upper < 0 || (n-m) <= num_upper).
                }

            }
        }
    }

    template void matrixBandPart(NDArray<float>* input, NDArray<float>* output, Nd4jLong lowerBand, Nd4jLong upperBand);
    template void matrixBandPart(NDArray<float16>* input, NDArray<float16>* output, Nd4jLong lowerBand, Nd4jLong upperBand);
    template void matrixBandPart(NDArray<double>* input, NDArray<double>* output, Nd4jLong lowerBand, Nd4jLong upperBand);

}
}
}

