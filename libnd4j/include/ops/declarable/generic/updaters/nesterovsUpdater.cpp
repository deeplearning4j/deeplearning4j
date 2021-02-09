/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

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

        CONFIGURABLE_OP_IMPL(nesterovs_updater, 2, 2, true, 0, 0) {

            const auto gradient = INPUT_VARIABLE(0);
            const auto initState = INPUT_VARIABLE(1);

            auto update = OUTPUT_VARIABLE(0);
            auto stateV = OUTPUT_VARIABLE(1);

            if (gradient->isEmpty() || initState->isEmpty())
                return Status::OK();

            REQUIRE_TRUE(gradient->isSameShape(initState), 0, "NESTEROVS UPDATER OP: input state Msg must have the same shape as gradient,"
                "  expected shape %s, but got %s!", ShapeUtils::shapeAsString(gradient->shapeInfo()).c_str(),
                ShapeUtils::shapeAsString(initState->shapeInfo()).c_str());

            bool bParamsSupply = 4 == block.width() || 2 == block.getTArguments()->size();

            REQUIRE_TRUE(bParamsSupply, 0, "NESTEROVS UPDATER OP: learning rate and momentum were not provided!");

            double dLr, dMomentum;

            if (block.width() > 2) {
                const auto lr = INPUT_VARIABLE(2);
                const auto momentum = INPUT_VARIABLE(3);

                REQUIRE_TRUE(lr->isScalar(), 0, "NESTEROVS UPDATER OP: Learning rate has to be a scalar, but instead got rank %i!", lr->rankOf());
                REQUIRE_TRUE(momentum->isScalar(), 0, "NESTEROVS UPDATER OP: Momentum has to be a scalar, but instead got rank %i!", momentum->rankOf());

                dLr = lr->e<double>(0);
                dMomentum = momentum->e<double>(0);
            }
            else {
                dLr = T_ARG(0);
                dMomentum = T_ARG(1);
            }
            helpers::updaterNesterovs(block.launchContext(), *gradient, *initState, *update, *stateV, dLr, dMomentum);
            return Status::OK();
        }

        DECLARE_TYPES(nesterovs_updater) {
            getOpDescriptor()->setAllowedInputTypes({ ALL_FLOATS })
                ->setSameMode(true);
        }

    }
}
