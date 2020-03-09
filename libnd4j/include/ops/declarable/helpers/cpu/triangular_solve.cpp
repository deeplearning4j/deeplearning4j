/*******************************************************************************
 * Copyright (c) 2020 Konduit, K.K.
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
//  @author GS <sgazeos@gmail.com>
//
#include <system/op_boilerplate.h>
#include <array/NDArray.h>
#include <execution/Threads.h>
#include "../triangular_solve.h"

namespace sd {
namespace ops {
namespace helpers {
    /*
     * lower triangular process for system of linear equations
     * x_1 = b_1/a_1,1
     * x_2 = (b_2 - a_2,1 * x_1) / a_2,2
     * x_3 = (b_3 - a_3,1 * x_1 - a_3,2 * x_2) / a_3,3
     * ...
     * x_M = (b_M - a_M,1 * x_1 - ... a_M,M-1 * x_M-1)/ a_M,M
     *
     * output == x
     * a == leftInput
     * b == rightInput
     *
     * */
    template <typename T>
    static void lowerTriangularSolve(sd::LaunchContext * context, NDArray* leftInput, NDArray* rightInput, bool adjoint, NDArray* output) {
        auto rows = leftInput->rows();
        auto cols = rightInput->columns();
        //output->t<T>(0,0) = rightInput->t<T>(0,0) / leftInput->t<T>(0,0);
        for (Nd4jLong r = 0; r < rows; r++) {
            for (Nd4jLong j = 0; j < cols; j++) {
                auto sum = rightInput->t<T>(r, j);
                for (Nd4jLong c = 0; c < r; c++) {
                    sum -= leftInput->t<T>(r, c) * output->t<T>(c, j);
                }
                output->t<T>(r, j) = sum / leftInput->t<T>(r, r);
            }
        }
    }

    /*
     * upper triangular process for system of linear equations
     * x_M = b_M/a_M,M
     * x_M-1 = (b_M-1 - a_M-1,M-2 * x_M) / a_M-1,M-1
     * x_M-2 = (b_M-2 - a_M-2,M-3 * x_M-2 - a_M-2,M-1 * x_M) / a_3,3
     * ...
     * x_1 = (b_1 - a_1,2 * x_2 - ... a_1,M * x_M)/ a_1,1
     *
     * output == x
     * a == leftInput
     * b == rightInput
     *
     * */

    template <typename T>
    static void upperTriangularSolve(sd::LaunchContext * context, NDArray* leftInput, NDArray* rightInput, bool adjoint, NDArray* output) {
        auto rows = leftInput->rows();
        auto cols = rightInput->columns();
        for (Nd4jLong r = rows; r > 0; r--) {
            for (Nd4jLong j = 0; j < cols; j++) {
                auto sum = rightInput->t<T>(r - 1, j);
                for (Nd4jLong c = r; c < rows; c++) {
                    sum -= leftInput->t<T>(r - 1, c) * output->t<T>(c, j);
                }
                output->t<T>(r - 1, j) = sum / leftInput->t<T>(r - 1, r - 1);
            }
        }
    }

    template <typename T>
    static int triangularSolveFunctor_(sd::LaunchContext * context, NDArray* leftInput, NDArray* rightInput, bool lower, bool adjoint, NDArray* output) {
        auto leftPart = leftInput->allTensorsAlongDimension({-2, -1});
        auto rightPart = rightInput->allTensorsAlongDimension({-2, -1});
        auto outputPart = output->allTensorsAlongDimension({-2, -1});

        auto batchLoop = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {
                if (lower) {
                    lowerTriangularSolve<T>(context, leftPart[i], rightPart[i], adjoint, outputPart[i]);
                } else {
                    upperTriangularSolve<T>(context, leftPart[i], rightPart[i], adjoint, outputPart[i]);
                }
            }
        };

        sd::Threads::parallel_tad(batchLoop, 0, leftPart.size(), 1);

        return Status::OK();

    }
    template <typename T>
    static void adjointTriangularMatrix_(sd::LaunchContext* context, NDArray const* input, bool const lower, NDArray* output) {
        auto inputPart = input->allTensorsAlongDimension({-2, -1});
        auto outputPart = output->allTensorsAlongDimension({-2, -1});
        auto cols = input->sizeAt(-1);
        auto rows = input->sizeAt(-2);

        auto batchLoop = PRAGMA_THREADS_FOR {
            for (auto batch = start; batch < stop; batch++) {
                if (!lower) {
                    for (Nd4jLong r = 0; r < rows; r++) {
                        for (Nd4jLong c = 0; c <= r; c++) {
                            outputPart[batch]->t<T>(r, c) = inputPart[batch]->t<T>(c, r);
                        }
                    }
                } else {
                    for (Nd4jLong r = 0; r < rows; r++) {
                        for (Nd4jLong c = r; c < cols; c++) {
                            outputPart[batch]->t<T>(r, c) = inputPart[batch]->t<T>(c, r);
                        }
                    }
                }
            }
        };
        sd::Threads::parallel_tad(batchLoop, 0, inputPart.size(), 1);
    }

    int triangularSolveFunctor(sd::LaunchContext * context, NDArray* leftInput, NDArray* rightInput, bool lower, bool adjoint, NDArray* output) {
        BUILD_SINGLE_SELECTOR(leftInput->dataType(), return triangularSolveFunctor_, (context, leftInput, rightInput, lower, adjoint, output), FLOAT_NATIVE);
    }

    void adjointMatrix(sd::LaunchContext* context, NDArray const* input, bool const lower, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), adjointTriangularMatrix_, (context, input, lower, output), FLOAT_NATIVE);
    }
}
}
}
