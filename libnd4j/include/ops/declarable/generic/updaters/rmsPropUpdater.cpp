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

        CONFIGURABLE_OP_IMPL(rms_prop_updater, 2, 2, true, 0, 0) {

            const auto gradient = INPUT_VARIABLE(0);
            const auto initState = INPUT_VARIABLE(1);

            auto update = OUTPUT_VARIABLE(0);
            auto stateG = OUTPUT_VARIABLE(1);

            if (gradient->isEmpty() || initState->isEmpty())
                return Status::OK();

            REQUIRE_TRUE(gradient->isSameShape(initState), 0, "RMS_PROB UPDATER OP: input state must have the same shape as gradient,"
                "  expected shape %s, but got %s!", ShapeUtils::shapeAsString(gradient->shapeInfo()).c_str(),
                ShapeUtils::shapeAsString(initState->shapeInfo()).c_str());

            bool bParamsSupply = 5 == block.width() || 3 == block.getTArguments()->size();

            REQUIRE_TRUE(bParamsSupply, 0, "RSM_PROB UPDATER OP: learning rate, rsm decay and epsilon were not provided!");

            double dLr, dRmsDecay, dEpsilon;

            if (block.width() > 2) {
                const auto lr = INPUT_VARIABLE(2);
                const auto rmsDecay = INPUT_VARIABLE(3);
                const auto epsilon = INPUT_VARIABLE(4);

                REQUIRE_TRUE(lr->isScalar(), 0, "RSM_PROB UPDATER OP: Learning rate has to be a scalar, but instead got rank %i!", lr->rankOf());
                REQUIRE_TRUE(rmsDecay->isScalar(), 0, "RSM_PROB UPDATER OP: Rms decay has to be a scalar, but instead got rank %i!", rmsDecay->rankOf());
                REQUIRE_TRUE(epsilon->isScalar(), 0, "RSM_PROB UPDATER OP: Epsilon has to be a scalar, but instead got rank %i!", epsilon->rankOf());

                dLr = lr->e<double>(0);
                dRmsDecay = rmsDecay->e<double>(0);
                dEpsilon = epsilon->e<double>(0);
            }
            else {
                dLr = T_ARG(0);
                dRmsDecay = T_ARG(1);
                dEpsilon = T_ARG(2);
            }

            helpers::updaterRmsProp(block.launchContext(), *gradient, *initState, *update, *stateG, dLr, dRmsDecay, dEpsilon);
            return Status::OK();
        }

        DECLARE_TYPES(rms_prop_updater) {
            getOpDescriptor()->setAllowedInputTypes({ ALL_FLOATS })
                ->setSameMode(true);
        }

    }
}
