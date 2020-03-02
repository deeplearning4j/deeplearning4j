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
//  modified by sgazeos@gmail.com with backprop implementation.
//
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_floormod)

#include <ops/declarable/generic/helpers/BroadcastHelper.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
    namespace ops {
        BROADCASTABLE_OP_IMPL(floormod, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            BROADCAST_CHECK_EMPTY(x,y,z);

            REQUIRE_TRUE(!y->isB(), 0, "FLOORMOD OP: you can't divide by bool array!");
            auto tZ = BroadcastHelper::broadcastApply(BROADCAST(FloorMod), x, y, z);
            if (tZ == nullptr)
                return ND4J_STATUS_KERNEL_FAILURE;
            else if (tZ != z) {
                OVERWRITE_RESULT(tZ);
            }

            return Status::OK();
        }

        DECLARE_TYPES(floormod) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, DataType::ANY)
                    ->setAllowedInputTypes(1, DataType::ANY)
                    ->setAllowedOutputTypes(0, DataType::INHERIT);
        }

        DECLARE_TYPES(floormod_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

        CUSTOM_OP_IMPL(floormod_bp, 3, 2, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto epsNext = INPUT_VARIABLE(2);

            auto gradX = OUTPUT_VARIABLE(0);
            auto gradY = OUTPUT_VARIABLE(1);
            gradX->assign(epsNext);
            sd::ops::floormod op;
            std::unique_ptr<ResultSet> tmpResult(op.evaluate({x, y}));

            if (gradY->rankOf() == gradX->rankOf())
                epsNext->applyPairwiseTransform(pairwise::Multiply, *tmpResult->at(0), *gradY);
            else // epsNext is greater than gradY
            {
                std::vector<Nd4jLong> dims(epsNext->rankOf() * 2);
                Nd4jLong gap = epsNext->rankOf() - gradY->rankOf();
                for (Nd4jLong d = 0; d < gap; d++) {
                    dims[d * 2 + 1] = 1;
                }
                auto tempIn((*tmpResult->at(0))(dims));
                (*epsNext)(dims).applyPairwiseTransform(pairwise::Multiply, tempIn, *gradY);
            }
            return Status::OK();
        }

        DECLARE_SHAPE_FN(floormod_bp) {
            auto x = inputShape->at(0);
            auto y = inputShape->at(1);
            auto e = inputShape->at(2);

            // eps always has shape of x
            // grad always has shape of y

            Nd4jLong *shapeE;
            Nd4jLong *shapeG;

            COPY_SHAPE(x, shapeE);
            COPY_SHAPE(y, shapeG);

            return SHAPELIST(CONSTANT(shapeE), CONSTANT(shapeG));
        }
    }
}

#endif