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
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_histogram)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>
#include <ops/declarable/helpers/histogram.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(histogram, 1, 1, false, 0, 1) {
            auto input = INPUT_VARIABLE(0);
            auto numBins = INT_ARG(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(numBins == output->lengthOf(), 0, "Histogram: numBins must match output length")

            helpers::histogramHelper(block.launchContext(), *input, *output);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(histogram) {
            auto numBins = INT_ARG(0);

            return SHAPELIST(ConstantShapeHelper::getInstance()->vectorShapeInfo(numBins, nd4j::DataType::INT64));
        }


        DECLARE_TYPES(histogram) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_INTS});
        };
    }
}

#endif
