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
//  @author George A. Shulinok <sgazeos@gmail.com>
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_adjust_contrast)

#include <ops/declarable/headers/parity_ops.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {

CONFIGURABLE_OP_IMPL(adjust_contrast, 1, 1, true, 1, 0) {

    auto input  = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const double factor = T_ARG(0);

    REQUIRE_TRUE(input->rankOf() > 2, 0, "ADJUST_CONTRAST: op expects rank of input array to be >= 3, but got %i instead", input->rankOf());
    REQUIRE_TRUE(input->sizeAt(-1) == 3, 0, "ADJUST_CONTRAST: operation expects image with 3 channels (R, G, B), but got %i instead", input->sizeAt(-1));
    // compute mean before
    // fill up axes vector first
    std::vector<int> axes(input->rankOf() - 1);
    for (auto i = 0; i < axes.size(); ++i)
        axes[i] = i;
    // mean as reduction for last dimension set
    auto mean = input->reduceAlongDims(reduce::Mean, axes);

    NDArray factorT(output->dataType(), block.launchContext()); // = NDArrayFactory::create(factor, block.launchContext());
    factorT.p(0, factor);
    // this is contrast calculation
    *output = (*input - mean) * factorT + mean;

    return Status::OK();
}

DECLARE_TYPES(adjust_contrast) {
    getOpDescriptor()->setAllowedInputTypes(nd4j::DataType::ANY)
                     ->setAllowedOutputTypes({ALL_FLOATS})
                     ->setSameMode(true);
}


    CONFIGURABLE_OP_IMPL(adjust_contrast_v2, 1, 1, true, 1, 0) {

        auto input  = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);

        const double factor = T_ARG(0);

        REQUIRE_TRUE(input->rankOf() > 2, 0, "ADJUST_CONTRAST: op expects rank of input array to be >= 3, but got %i instead", input->rankOf());
        REQUIRE_TRUE(input->sizeAt(-1) == 3, 0, "ADJUST_CONTRAST: operation expects image with 3 channels (R, G, B), but got %i instead", input->sizeAt(-1));

        // compute mean before
        std::vector<int> axes(input->rankOf() - 1);
        for (auto i = 0; i < axes.size(); ++i)
            axes[i] = i;

        // mean as reduction for last dimension set
        auto mean = input->reduceAlongDims(reduce::Mean, axes);

        // result as (x - mean) * factor + mean
        std::unique_ptr<NDArray> temp(input->dup());
        input->applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &mean, temp.get());
        temp->applyScalar(scalar::Multiply, factor);
        temp->applyTrueBroadcast(BroadcastOpsTuple::Add(), &mean, output);

        return Status::OK();
    }

    DECLARE_TYPES(adjust_contrast_v2) {
        getOpDescriptor()->setAllowedInputTypes(nd4j::DataType::ANY)
                ->setAllowedOutputTypes({ALL_FLOATS})
                ->setSameMode(true);
    }

}
}

#endif