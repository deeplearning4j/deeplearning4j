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
#if NOT_EXCLUDED(OP_adjust_hue)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/adjust_hue.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
    CONFIGURABLE_OP_IMPL(adjust_hue, 1, 1, true, -2, -2) {
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);

        REQUIRE_TRUE(input->rankOf() == 3 || input->rankOf() == 4, 0, "AdjustHue: op expects either 3D or 4D input, but got %i instead", input->rankOf());


        double delta = 0;
        if (block.numT() > 0)
            delta = T_ARG(0);
        else if (block.width() > 1) {
            auto _d = INPUT_VARIABLE(1);
            if (!_d->isScalar()) {
                auto str = ShapeUtils::shapeAsString(_d);
                REQUIRE_TRUE(_d->isScalar(), 0, "AdjustHue: delta should be scalar NDArray, but got %s instead", str.c_str());
            }
            delta = _d->getScalar<double>(0);
        }


        bool isNHWC = false;
        if (block.numI() > 0)
            isNHWC = INT_ARG(0) == 1;

        int numChannels = isNHWC ? input->sizeAt(-1) : input->sizeAt(-3);

        REQUIRE_TRUE(numChannels == 3, 0, "AdjustHue: this operation expects image with 3 channels (R, G, B), but got % instead", numChannels);

        auto ts = *(NDArrayFactory::scalar(delta));
        // FIXME: delta should be NDArray scalar
        helpers::_adjust_hue(input, output, &ts, isNHWC);

        return Status::OK();
    }
}
}

#endif