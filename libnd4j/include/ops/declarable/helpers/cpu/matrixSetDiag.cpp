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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include "ResultSet.h"
#include <ops/declarable/helpers/matrixSetDiag.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template<typename T>
void matrixSetDiag_(const NDArray& input, const NDArray& diagonal, NDArray& output, const bool zeroPad) {

    // input and output are the same array (x == z) when zeroPad = true
    // xRank = zRank, xRank = yRank + 1
    // xLen = zLen

    const T* x = input.bufferAsT<T>();
    const T* y = diagonal.bufferAsT<T>();
          T* z = output.bufferAsT<T>();

    const Nd4jLong* xShapeInfo = input.getShapeInfo();
    const Nd4jLong* yShapeInfo = diagonal.getShapeInfo();
    const Nd4jLong* zShapeInfo = output.getShapeInfo();

    const bool areSameOffsets = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);    // shapes are definitely the same, but strides might not

    const int xRank = input.rankOf();
    const auto xLen = input.lengthOf();

    std::vector<Nd4jLong> coords(xRank);  // we use the same coordinates storage both for input and output since their ranks are the same

    PRAGMA_OMP_PARALLEL_FOR_ARGS(firstprivate(coords))
    for (Nd4jLong i = 0; i < xLen; ++i) {

        shape::index2coords(i, xShapeInfo, coords.data());

        const auto xOffset = shape::getOffset(xShapeInfo, coords.data());
        const auto zOffset = areSameOffsets ? xOffset : shape::getOffset(zShapeInfo, coords.data());

        // condition to be on diagonal of innermost matrix
        if(coords[xRank - 2] == coords[xRank - 1])
            z[zOffset] = y[shape::getOffset(yShapeInfo, coords.data())];
        else
            z[zOffset] = zeroPad ? static_cast<T>(0) : x[xOffset];
    }
}

//////////////////////////////////////////////////////////////////////////
void matrixSetDiag(nd4j::LaunchContext* context, const NDArray& input, const NDArray& diagonal, NDArray& output, const bool zeroPad) {
    BUILD_SINGLE_SELECTOR(input.dataType(), matrixSetDiag_, (input, diagonal, output, zeroPad), LIBND4J_TYPES);
}

}
}
}