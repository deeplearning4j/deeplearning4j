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
// @author George A. Shulinok <sgazeos@gmail.com>, created on 8/22/2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_matrix_band_part)
#include <ops/declarable/helpers/matrix_band.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(matrix_band_part, 1, 1, true, 0, 2) {

            NDArray<T>* input = INPUT_VARIABLE(0);

            NDArray<T>* output   = OUTPUT_VARIABLE(0);
            Nd4jLong minLower = INT_ARG(0);
            Nd4jLong maxUpper = INT_ARG(1);

            REQUIRE_TRUE(input->rankOf() >= 2, 0, "matrix_band_part: Input rank should be 2 or greater.");
            Nd4jLong N = input->sizeAt(-2);
            Nd4jLong M = input->sizeAt(-1);
            REQUIRE_TRUE(minLower > -N && minLower < N, 0, "matrix_band_part: lower diagonal count %i should be less than %i.",
                    minLower, N);
            REQUIRE_TRUE(maxUpper > -M && maxUpper < M, 0, "matrix_band_part: upper diagonal count %i should be less than %i.",
                    maxUpper, M);

            helpers::matrixBandPart(input, output, minLower, maxUpper);
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(band_part, matrix_band_part);
    }
}

#endif