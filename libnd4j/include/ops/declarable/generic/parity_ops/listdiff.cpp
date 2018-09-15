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

// this op will probably never become GPU-compatible
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(listdiff, 2, 2, false, 0, 0) {
            auto values = INPUT_VARIABLE(0);
            auto keep = INPUT_VARIABLE(1);


            // FIXME: must be moved to helpers
            /**
            std::vector<double> saved;
            std::vector<Nd4jLong> indices;

            REQUIRE_TRUE(values->rankOf() == 1, 0, "ListDiff: rank of values should be 1D, but got %iD instead", values->rankOf());
            REQUIRE_TRUE(keep->rankOf() == 1, 0, "ListDiff: rank of keep should be 1D, but got %iD instead", keep->rankOf());

            for (Nd4jLong e = 0; e < values->lengthOf(); e++) {
                T v = values->getScalar(e);
                T extras[] = {v, (T) 0.0f, (T) 10.0f};
                auto idx = keep->indexReduceNumber(indexreduce::FirstIndex, extras);
                if (idx < 0) {
                    saved.emplace_back(v);
                    indices.emplace_back(e);
                }
            }


            // FIXME: we need 0-size NDArrays
            if (saved.size() == 0) {
                REQUIRE_TRUE(false, 0, "ListDiff: search returned no results");
            } else {
                auto z0 = OUTPUT_VARIABLE(0); //new NDArray<T>('c', {(int) saved.size()});
                auto z1 = OUTPUT_VARIABLE(1); //new NDArray<T>('c', {(int) saved.size()});


                REQUIRE_TRUE(z0->lengthOf() == saved.size(), 0, "ListDiff: output/actual size mismatch");
                REQUIRE_TRUE(z1->lengthOf() == saved.size(), 0, "ListDiff: output/actual size mismatch");

                memcpy(z0->buffer(), saved.data(), saved.size() * sizeof(T));
                memcpy(z1->buffer(), indices.data(), indices.size() * sizeof(T));

                //OVERWRITE_2_RESULTS(z0, z1);
                STORE_2_RESULTS(z0, z1);
            }
             */

            return Status::OK();
        };

        DECLARE_SHAPE_FN(listdiff) {
            auto values = INPUT_VARIABLE(0);
            auto keep = INPUT_VARIABLE(1);

            int saved = 0;

            REQUIRE_TRUE(values->rankOf() == 1, 0, "ListDiff: rank of values should be 1D, but got %iD instead", values->rankOf());
            REQUIRE_TRUE(keep->rankOf() == 1, 0, "ListDiff: rank of keep should be 1D, but got %iD instead", keep->rankOf());

            // FIXME: must be moved to helpers
            /*
            for (int e = 0; e < values->lengthOf(); e++) {
                T v = values->getScalar(e);
                T extras[] = {v, (T) 0.0f, (T) 10.0f};
                auto idx = keep->indexReduceNumber(indexreduce::FirstIndex, extras);
                if (idx < 0)
                    saved++;
            }
            */

            REQUIRE_TRUE(saved > 0, 0, "ListDiff: no matches found");

            auto shapeX = ShapeBuilders::createVectorShapeInfo(saved, block.workspace());
            auto shapeY = ShapeBuilders::createVectorShapeInfo(saved, block.workspace());

            return SHAPELIST(shapeX, shapeY);
        }
    }
}

#endif