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

        CONFIGURABLE_OP_IMPL(sgd_updater, 1, 1, true, 0, 0) {

            const auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            if (input->isEmpty())
                return Status::OK();

            bool bLearningRate = 2 == block.width() || 1 == block.getTArguments()->size();

            REQUIRE_TRUE(bLearningRate, 0, "SGD UPDATER OP: Learning rate was not provided!");

            if (block.width() > 1) {
                const auto lr = INPUT_VARIABLE(1);
                REQUIRE_TRUE(lr->isScalar(), 0, "SGD UPDATER OP: Learning rate has to be a scalar, but instead got rank %i!", lr->rankOf());

                input->applyScalarArr(scalar::Multiply, *lr, *output);
            }
            else {
                input->applyScalar(scalar::Multiply, T_ARG(0), *output);
            }

            return Status::OK();
        }

        DECLARE_TYPES(sgd_updater) {
            getOpDescriptor()->setAllowedInputTypes({ ALL_FLOATS })
                ->setSameMode(true);
        }

    }
}
