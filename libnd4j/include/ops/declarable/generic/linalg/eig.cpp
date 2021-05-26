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


#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_eig)

#include <ops/declarable/CustomOperations.h>
#include <helpers/EigenValsAndVecs.h>

namespace sd {
namespace ops  {

////////////////////////////////////////////////////////////////////////// 
CUSTOM_OP_IMPL(eig, 1, 2, false, 0, 0) {
    auto input  = INPUT_VARIABLE(0);
    auto eig_vals = OUTPUT_VARIABLE(0);
    auto eig_vectors = OUTPUT_VARIABLE(1);

    // input validation
    REQUIRE_TRUE(input->rankOf() == 2 , 0, "Eig: input is not a matrix. rank: %i == 2", input->rankOf());

    auto n1 = input->sizeAt(0);
    auto n2 = input->sizeAt(1);

    REQUIRE_TRUE(n1 == n2 , 0, "Eig: input is not a square matrix. rank: {%i, %i}", n1, n2);

    REQUIRE_TRUE(eig_vals->rankOf() == 2 && eig_vals->sizeAt(0) == n1 && eig_vals->sizeAt(1) == 2  , 0, "Eig: the shape of the eigenvalue results should be {%i, 2}", n1);
    REQUIRE_TRUE(eig_vectors->rankOf() == 3 && eig_vectors->sizeAt(0) == n1 && eig_vectors->sizeAt(1) == n1 && eig_vectors->sizeAt(2) == 2, 0, "Eig: the shape of the eigenvector results should be {%i, %i, 2}", n1);

    sd::ops::helpers::eig(*input, *eig_vals, *eig_vectors);

    return Status::OK();
}

DECLARE_TYPES(eig) {
        getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})
                     ->setSameMode(true);
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(eig) {
    auto inputShapeInfo =  inputShape->at(0);
    REQUIRE_TRUE(inputShapeInfo[0] ==2 , 0, "Eig: input is not a matrix. rank: %i == 2",
                 inputShapeInfo[0]);

    auto n1 = shape::shapeOf(inputShapeInfo)[0] ;
    auto n2 = shape::shapeOf(inputShapeInfo)[1] ;

    REQUIRE_TRUE(n1== n2 , 0, "Eig: input is not a square matrix. rank: {%i, %i}",
                 n1, n2);

    auto dtype_float = ArrayOptions::dataType(inputShapeInfo);
    auto ordering =  shape::order(inputShapeInfo);

    auto output0 = ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(dtype_float, ordering, {n1, 2}));
    auto output1 = ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(dtype_float, ordering, {n1, n1, 2}));
    return SHAPELIST(output0, output1);
}


}
}

#endif