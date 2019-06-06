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
// @author George A. Shulinok <sgazeos@gmail.com>, created on 4/18/2019
//

#include <ops/declarable/helpers/BarnesHutTsne.h>

namespace nd4j {
namespace ops {
namespace helpers {

    Nd4jLong barnes_row_count(const NDArray* rowP, const NDArray* colP, Nd4jLong N, NDArray& rowCounts) {

        int* pRowCounts = reinterpret_cast<int*>(rowCounts.buffer());
        int const* pRows = reinterpret_cast<int const*>(rowP->getBuffer());
        int const* pCols = reinterpret_cast<int const*>(colP->getBuffer());
        for (int n = 0; n < N; n++) {
            int begin = pRows[n];//->e<int>(n);
            int end = pRows[n + 1];//rowP->e<int>(n + 1);
            for (int i = begin; i < end; i++) {
                bool present = false;
                for (int m = pRows[pCols[i]]; m < pRows[pCols[i] + 1]; m++)
                    if (pCols[m] == n) {
                        present = true;
                        break;
                    }

                ++pRowCounts[n];

                if (!present)
                    ++pRowCounts[pCols[i]];
            }
        }
        NDArray numElementsArr = rowCounts.sumNumber(); //reduceAlongDimension(reduce::Sum, {});
        //rowCounts.printBuffer("Row counts");
        auto numElements = numElementsArr.e<Nd4jLong>(0);
        return numElements;
    }


    template <typename T>
    static void barnes_symmetrize_(const NDArray* rowP, const NDArray* colP, const NDArray* valP, Nd4jLong N, NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts) {

    }
    void barnes_symmetrize(const NDArray* rowP, const NDArray* colP, const NDArray* valP, Nd4jLong N, NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts) {
//
    }
    BUILD_SINGLE_TEMPLATE(template void barnes_symmetrize_, (const NDArray* rowP, const NDArray* colP, const NDArray* valP, Nd4jLong N, NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts), NUMERIC_TYPES);

    template <typename T>
    static void barnes_edge_forces_(const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray const* data, NDArray* output) {

    }

    void barnes_edge_forces(const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray* output, NDArray const& data) {

    }
    BUILD_SINGLE_TEMPLATE(template void barnes_edge_forces_, (const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray const* data, NDArray* output), FLOAT_TYPES);

    template <typename T>
    void barnes_gains_(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output) {
    }

    void barnes_gains(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output) {

    }
    BUILD_SINGLE_TEMPLATE(template void barnes_gains_, (NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output), NUMERIC_TYPES);

    bool cell_contains(NDArray* corner, NDArray* width, NDArray* point, Nd4jLong dimension) {
        auto  cornerMinusWidth = *corner - *width;
        auto cornerPlusWidth = *corner + *width;

        for (Nd4jLong i = 0; i < dimension; i++) {
            if (cornerMinusWidth.e<double>(i) > point->e<double>(i))
                return false;
            if (cornerPlusWidth.e<double>(i) < point->e<double>(i))
                return false;
        }

        return true;
    }
}
}
}

