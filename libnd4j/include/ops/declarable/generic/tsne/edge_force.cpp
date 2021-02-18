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
// @author George A. Shulinok <sgazeos@gmail.com>, created on 4/18/2019.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_barnes_edge_force)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/BarnesHutTsne.h>

namespace sd {
namespace ops  {
		
    CUSTOM_OP_IMPL(barnes_edge_forces, 4, 1, false, 0, 1) {
        auto rowP  = INPUT_VARIABLE(0);
        auto colP  = INPUT_VARIABLE(1);
        auto valP  = INPUT_VARIABLE(2);
        auto dataP  = INPUT_VARIABLE(3);
        auto N = INT_ARG(0);

        auto output = OUTPUT_NULLIFIED(0);

        REQUIRE_TRUE(rowP->isVector(), 0, "barnes_edge_force: row input must be a vector, but its rank is %i instead !", rowP->rankOf());
        REQUIRE_TRUE(colP->isVector(), 0, "barnes_edge_force: col input must be a vector, but its rank is %i instead !", colP->rankOf());
        REQUIRE_TRUE(dataP->dataType() == output->dataType() && dataP->dataType() == valP->dataType(), 0, "barnes_edge_force: data type of dataP, valP and output must be the same");

        helpers::barnes_edge_forces(rowP, colP, valP, N, output, *dataP);

        return Status::OK();
    }

    DECLARE_TYPES(barnes_edge_forces) {
        getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_INTS})
        ->setAllowedInputTypes(1, {ALL_INTS})
        ->setAllowedInputTypes(2, {ALL_FLOATS})
        ->setAllowedInputTypes(3, {ALL_FLOATS})
        ->setAllowedOutputTypes(0, {ALL_FLOATS})
        ->setSameMode(false);
    }

    DECLARE_SHAPE_FN(barnes_edge_forces) {
        Nd4jLong* bufShape;
        Nd4jLong* outShapeInfo;
        outShapeInfo = ShapeBuilders::copyShapeInfoAndType(inputShape->at(3), inputShape->at(3), false, block.getWorkspace());
        return SHAPELIST(CONSTANT(outShapeInfo));
    }


}
}

#endif