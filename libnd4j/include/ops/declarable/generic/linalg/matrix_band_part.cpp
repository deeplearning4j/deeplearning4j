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
// @author GS <sgazeos@gmail.com>, created on 8/22/2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_matrix_band_part)
#include <ops/declarable/helpers/matrix_band.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
    namespace ops {
        CONFIGURABLE_OP_IMPL(matrix_band_part, 1, 1, true, 0, 0) {

            auto input = INPUT_VARIABLE(0);

            auto output   = OUTPUT_VARIABLE(0);

            Nd4jLong minLower(0LL);
            Nd4jLong maxUpper(0LL);
            if (block.width() == 1) {
                REQUIRE_TRUE(block.numI() == 2, 0, "matrix_band_part: min and max band numbers should be given before.");
                minLower = INT_ARG(0);
                maxUpper = INT_ARG(1);
            }
            else {
                REQUIRE_TRUE(block.width() == 3, 0, "matrix_band_part: min and max band numbers should be given as scalars before.");
                auto minLowerT = INPUT_VARIABLE(1);
                auto maxUpperT = INPUT_VARIABLE(2);
                REQUIRE_TRUE(minLowerT->isScalar() && maxUpperT->isScalar(), 0, "matrix_band_part: min and max should be scalars, but %i and %i ranks given", minLowerT->rankOf(), maxUpperT->rankOf());
                minLower = minLowerT->e<Nd4jLong>(0);
                maxUpper = maxUpperT->e<Nd4jLong>(0);
            }
            REQUIRE_TRUE(input->rankOf() >= 2, 0, "matrix_band_part: Input rank should be 2 or greater.");
            Nd4jLong N = input->sizeAt(-2);
            Nd4jLong M = input->sizeAt(-1);
            REQUIRE_TRUE(minLower > -N && minLower < N, 0, "matrix_band_part: lower diagonal count %i should be less than %i.",
                    minLower, N);
            REQUIRE_TRUE(maxUpper > -M && maxUpper < M, 0, "matrix_band_part: upper diagonal count %i should be less than %i.",
                    maxUpper, M);

            helpers::matrixBandPart(block.launchContext(), input, output, minLower, maxUpper);
            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(band_part, matrix_band_part);
    }

    DECLARE_TYPES(matrix_band_part) {
        getOpDescriptor()
            ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS})
            ->setAllowedInputTypes(1, {ALL_INTS})
            ->setAllowedInputTypes(2, {ALL_INTS})
            ->setAllowedInputTypes({ALL_INTS, ALL_FLOATS});
    }
}

#endif