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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/axis.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void _extractPatches(NDArray* images, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame){
        std::vector<int> restDims({1, 2, 3}); // the first and the last dims
        std::unique_ptr<ResultSet> listOfMatricies(images->allTensorsAlongDimension(restDims));
        std::unique_ptr<ResultSet> listOfOutputs(output->allTensorsAlongDimension(restDims));
        // 3D matricies - 2D matricies of vectors (if last dim is greater than 1)
        //int e = 0;
        int ksizeRowsEffective = sizeRow + (sizeRow - 1) * (rateRow - 1);
        int ksizeColsEffective = sizeCol + (sizeCol - 1) * (rateCol - 1);
        int ksize = ksizeRowsEffective * ksizeColsEffective;
        int batchCount = listOfMatricies->size(); //lengthOf() / ksize;
        Nd4jLong lastDim = images->sizeAt(3);
        Nd4jLong rowDim = images->sizeAt(1);
        Nd4jLong colDim = images->sizeAt(2);
        Nd4jLong outputLastDim = ksize * lastDim;
#pragma omp parallel for
        for (Nd4jLong e = 0; e < batchCount; ++e) {
            auto patch = listOfMatricies->at(e);
            auto outMatrix = listOfOutputs->at(e);
            auto patchBorder = patch->sizeAt(0);
            //int startRow = 0;
            //int startCol = 0;
            Nd4jLong pos = 0;
            for (int i = 0; i < rowDim; i += stradeRow)
            for (int j = 0; j < colDim; j += stradeCol)
                for (int l = 0; l < sizeRow; l++)
                for (int m = 0; m < sizeCol; m++) {
                    //for (Nd4jLong pos = 0; pos < outputLastDim; pos++)
                for (Nd4jLong k = 0; k < lastDim; ++k) {
                    if (l + i < rowDim  && m + j < colDim && i + rateRow * l < patchBorder) // && i + rateRow * l < sizeRow && j + m * rateCol < sizeCol
                        outMatrix->p<T>(pos, patch->e<T>(i + rateRow * l, j + m * rateCol, k));
                    pos++;
                    if (pos >= outMatrix->lengthOf()) {
                        k = lastDim; m = sizeCol; l = sizeRow; j = colDim; i = rowDim;
                    }
                }
            }
        }
    }


    void extractPatches(NDArray* images, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame){
        auto xType = images->dataType();

        BUILD_SINGLE_SELECTOR(xType, _extractPatches, (images, output, sizeRow, sizeCol, stradeRow, stradeCol, rateRow, rateCol, theSame), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _extractPatches, (NDArray* input, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame), LIBND4J_TYPES);

}
}
}