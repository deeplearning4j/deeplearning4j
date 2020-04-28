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
// @author Yurii Shyrma, created on 31.03.2018
//

#include <ops/declarable/CustomOperations.h>


namespace sd {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(tri, -2, 1, false, 0, 1) {

    auto output = OUTPUT_VARIABLE(0);

    const int diag = block.numI() > 2 ? INT_ARG(2) : 0;

    BUILD_SINGLE_SELECTOR(output->dataType(), output->fillAsTriangular, (1., diag + 1, 0,    *output, 'l'), LIBND4J_TYPES);  // fill with unities lower triangular block of matrix
    BUILD_SINGLE_SELECTOR(output->dataType(), output->fillAsTriangular, (0., 0,        diag, *output, 'u'), LIBND4J_TYPES);  // fill with zeros upper triangular block of matrix

    // output->setValueInDiagMatrix(1., diag,   'l');
    // output->setValueInDiagMatrix(0., diag+1, 'u');

    return Status::OK();
}

        DECLARE_TYPES(tri) {
            getOpDescriptor()
                    ->setAllowedOutputTypes(0, {ALL_FLOATS, ALL_INTS});
        }


DECLARE_SHAPE_FN(tri) {
	const int rows = INT_ARG(0);
    const int cols = block.numI() > 1 ? INT_ARG(1) : rows;

    auto dtype = block.numD() ? D_ARG(0) : DataType::FLOAT32;

    return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(dtype, 'c', {rows, cols}));
}




}
}