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

        CONFIGURABLE_OP_IMPL(ams_grad_updater, 4, 4, true, 0, 0) {

            const auto gradient = INPUT_VARIABLE(0);
            const auto initStateV = INPUT_VARIABLE(1);
            const auto initStateM = INPUT_VARIABLE(2);
            const auto initStateH = INPUT_VARIABLE(3);

            auto update = OUTPUT_VARIABLE(0);
            auto stateV = OUTPUT_VARIABLE(1);
            auto stateM = OUTPUT_VARIABLE(2);
            auto stateH = OUTPUT_VARIABLE(3);

            // todo maybe we need an error like on Java side
            if (gradient->isEmpty() || initStateV->isEmpty() || initStateM->isEmpty() || initStateH->isEmpty())
                return Status::OK();

            REQUIRE_TRUE(gradient->isSameShape(initStateV), 0, "AMSGRAD UPDATER OP: input state Msg must have the same shape as gradient,"
                "  expected shape %s, but got %s!", ShapeUtils::shapeAsString(gradient->shapeInfo()).c_str(),
                ShapeUtils::shapeAsString(initStateV->shapeInfo()).c_str());
            REQUIRE_TRUE(gradient->isSameShape(initStateM), 0, "AMSGRAD UPDATER OP: input state Msdx must have the same shape as gradient,"
                "  expected shape %s, but got %s!", ShapeUtils::shapeAsString(gradient->shapeInfo()).c_str(),
                ShapeUtils::shapeAsString(initStateM->shapeInfo()).c_str());
            REQUIRE_TRUE(gradient->isSameShape(initStateH), 0, "AMSGRAD UPDATER OP: input state Msdx must have the same shape as gradient!,"
                "  expected shape %s, but got %s!", ShapeUtils::shapeAsString(gradient->shapeInfo()).c_str(),
                ShapeUtils::shapeAsString(initStateH->shapeInfo()).c_str());

            bool bParamsSupply = 8 == block.width() || 4 == block.getTArguments()->size();

            auto iteration = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

            REQUIRE_TRUE(bParamsSupply, 0, "AMSGRAD UPDATER OP: learning rate, beta 1, beta 2 and epsilon were not provided!");

            double dLr, dBeta1, dBeta2, dEpsilon;

            if (block.width() > 4) {
                const auto lr = INPUT_VARIABLE(4);
                const auto beta1 = INPUT_VARIABLE(5);
                const auto beta2 = INPUT_VARIABLE(6);
                const auto epsilon = INPUT_VARIABLE(7);

                REQUIRE_TRUE(lr->isScalar(), 0, "AMSGRAD UPDATER OP: Learning rate has to be a scalar, but instead got rank %i!", lr->rankOf());
                REQUIRE_TRUE(beta1->isScalar(), 0, "AMSGRAD UPDATER OP: beta 1 has to be a scalar, but instead got rank %i!", beta1->rankOf());
                REQUIRE_TRUE(beta2->isScalar(), 0, "AMSGRAD UPDATER OP: beta 2 has to be a scalar, but instead got rank %i!", beta2->rankOf());
                REQUIRE_TRUE(epsilon->isScalar(), 0, "AMSGRAD UPDATER OP: Epsilon has to be a scalar, but instead got rank %i!", epsilon->rankOf());

                dLr = lr->e<double>(0);
                dBeta1 = beta1->e<double>(0);
                dBeta2 = beta2->e<double>(0);
                dEpsilon = epsilon->e<double>(0);
            }
            else {
                dLr = T_ARG(0);
                dBeta1 = T_ARG(1);
                dBeta2 = T_ARG(2);
                dEpsilon = T_ARG(3);
            }

            helpers::updaterAmsGrad(block.launchContext(), *gradient, *initStateV, *initStateM, *initStateH,
                *update, *stateV, *stateM, *stateH, dLr, dBeta1, dBeta2, dEpsilon, iteration);
            return Status::OK();
        }

        DECLARE_TYPES(ams_grad_updater) {
            getOpDescriptor()->setAllowedInputTypes({ ALL_FLOATS })
                ->setSameMode(true);
        }

    }
}
