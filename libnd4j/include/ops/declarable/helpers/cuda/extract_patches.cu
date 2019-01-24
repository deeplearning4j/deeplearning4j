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

    }


    void extractPatches(NDArray* images, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame){
        auto xType = images->dataType();

        BUILD_SINGLE_SELECTOR(xType, _extractPatches, (images, output, sizeRow, sizeCol, stradeRow, stradeCol, rateRow, rateCol, theSame), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _extractPatches, (NDArray* input, NDArray* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame), LIBND4J_TYPES);

}
}
}