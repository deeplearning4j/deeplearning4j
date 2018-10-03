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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_listdiff)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/listdiff.h>

// this op will probably never become GPU-compatible
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(listdiff, 2, 2, false, 0, 0) {
            auto values = INPUT_VARIABLE(0);
            auto keep = INPUT_VARIABLE(1);
            auto output1 = OUTPUT_VARIABLE(0);
            auto output2 = OUTPUT_VARIABLE(1);

            REQUIRE_TRUE(values->rankOf() == 1, 0, "ListDiff: rank of values should be 1D, but got %iD instead", values->rankOf());
            REQUIRE_TRUE(keep->rankOf() == 1, 0, "ListDiff: rank of keep should be 1D, but got %iD instead", keep->rankOf());

            return helpers::listDiffFunctor(values, keep, output1, output2);
        };

        DECLARE_SHAPE_FN(listdiff) {
            auto values = INPUT_VARIABLE(0);
            auto keep = INPUT_VARIABLE(1);

            REQUIRE_TRUE(values->rankOf() == 1, 0, "ListDiff: rank of values should be 1D, but got %iD instead", values->rankOf());
            REQUIRE_TRUE(keep->rankOf() == 1, 0, "ListDiff: rank of keep should be 1D, but got %iD instead", keep->rankOf());

            int saved = helpers::listDiffCount(values, keep);

            REQUIRE_TRUE(saved > 0, 0, "ListDiff: no matches found");

            auto shapeX = ShapeBuilders::createVectorShapeInfo(values->dataType(), saved, block.workspace());
//            auto shapeY = ShapeBuilders::createVectorShapeInfo(keep->dataType(), saved, block.workspace());
            auto shapeY = ShapeBuilders::createVectorShapeInfo(DataType::INT64, saved, block.workspace());

            return SHAPELIST(shapeX, shapeY);
        }
    }
}

#endif