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

#include <ops/declarable/helpers/adjust_hue.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void _adjust_hue_single(NDArray *array, NDArray *output, float delta, bool isNHWC) {

    }

    void _adjust_hue(nd4j::LaunchContext * context, NDArray *array, NDArray *output, NDArray* delta, bool isNHWC) {
        auto xType = array->dataType();

        float d = delta->e<float>(0);
        if (array->rankOf() == 4) {
            auto tadsIn = array->allTensorsAlongDimension({0});
            auto tadsOut = output->allTensorsAlongDimension({0});

            // FIXME: template selector should be moved out of loop
PRAGMA_OMP_PARALLEL_FOR
            for (int e = 0; e < tadsIn->size(); e++) {
                BUILD_SINGLE_SELECTOR(xType, _adjust_hue_single, (tadsIn->at(e), tadsOut->at(e), d, isNHWC);, FLOAT_TYPES);
            }
            

            delete tadsIn;
            delete tadsOut;
        } else {
            BUILD_SINGLE_SELECTOR(xType, _adjust_hue_single, (array, output, d, isNHWC);, FLOAT_TYPES);
        }
    }

    BUILD_SINGLE_TEMPLATE(template void _adjust_hue_single, (NDArray *array, NDArray *output, float delta, bool isNHWC);, FLOAT_TYPES);

}
}
}
