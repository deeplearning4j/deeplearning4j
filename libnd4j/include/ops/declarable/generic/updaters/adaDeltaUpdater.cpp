/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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
 // @author Oleh Semeniv (oleg.semeniv@gmail.com)
 //

#include <ops/declarable/headers/updaters.h>
#include <ops/declarable/CustomOperations.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>
#include <array/NDArray.h>

namespace sd {
    namespace ops {

        CONFIGURABLE_OP_IMPL(ada_delta_updater, 3, 3, true, 0, 0) {

            const auto gradient = INPUT_VARIABLE(0);
            const auto initStateMsg = INPUT_VARIABLE(1);
            const auto initStateMsdx = INPUT_VARIABLE(2);

            auto update = OUTPUT_VARIABLE(0);
            auto stateMsg = OUTPUT_VARIABLE(1);
            auto stateMsdx = OUTPUT_VARIABLE(2);

            if (gradient->isEmpty() || initStateMsg->isEmpty() || initStateMsdx->isEmpty())
                return Status::OK();

            REQUIRE_TRUE(gradient->isSameShape(initStateMsg), 0, "ADA_DELTA UPDATER OP: input state Msg must have the same shape as gradient,"
                "  expected shape %s, but got %s!", ShapeUtils::shapeAsString(gradient->getShapeInfo()).c_str(),
                ShapeUtils::shapeAsString(initStateMsg->getShapeInfo()).c_str());
            REQUIRE_TRUE(gradient->isSameShape(initStateMsdx), 0, "ADA_DELTA UPDATER OP: input state Msdx must have the same shape as gradient,"
                "  expected shape %s, but got %s!", ShapeUtils::shapeAsString(gradient->getShapeInfo()).c_str(),
                ShapeUtils::shapeAsString(initStateMsdx->getShapeInfo()).c_str());

            bool bParamsSupply = 5 == block.width() || 2 == block.getTArguments()->size();

            REQUIRE_TRUE(bParamsSupply, 0, "ADA_DELTA UPDATER OP: Rho and epsilon were not provided!");

            double dRho, dEpsilon;

            if (block.width() > 3) {
                const auto rho = INPUT_VARIABLE(3);
                const auto epsilon = INPUT_VARIABLE(4);

                REQUIRE_TRUE(rho->isScalar(), 0, "ADA_DELTA UPDATER OP: Rho has to be a scalar, but instead got rank %i!", rho->rankOf());
                REQUIRE_TRUE(epsilon->isScalar(), 0, "ADA_DELTA UPDATER OP: Epsilon has to be a scalar, but instead got rank %i!", epsilon->rankOf());

                dRho = rho->e<double>(0);
                dEpsilon = epsilon->e<double>(0);
            }
            else {
                dRho = T_ARG(0);
                dEpsilon = T_ARG(1);
            }

            helpers::updaterAdaDelta(block.launchContext(), *gradient, *initStateMsg, *initStateMsdx, *update, *stateMsg, *stateMsdx, dRho, dEpsilon);
            return Status::OK();
        }

        DECLARE_TYPES(ada_delta_updater) {
            getOpDescriptor()->setAllowedInputTypes({ ALL_FLOATS })
                ->setSameMode(true);
        }

    }
}
