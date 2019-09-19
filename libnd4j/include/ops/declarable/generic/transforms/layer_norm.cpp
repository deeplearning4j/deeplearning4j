/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author Paul Dubs
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_layer_norm)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>
#include <ops/declarable/helpers/addBias.h>

namespace nd4j {
namespace ops  {

    CONFIGURABLE_OP_IMPL(layer_norm, 2, 1, false, 0, -1) {
        auto input = INPUT_VARIABLE(0);
        auto gain = INPUT_VARIABLE(1);
        auto output = OUTPUT_VARIABLE(0);

        auto axis = ArrayUtils::toIntVector(*block.getIArguments());

        const bool isNCHW = block.getBArguments()->size() > 0 ? B_ARG(0) : true;       // INT_ARG(9): 0-NCHW,  1-NHWC
        const int dimC = isNCHW ? 1 : input->rankOf() - 1;

        REQUIRE_TRUE(gain->rankOf() == 1 && gain->sizeAt(0) == input->sizeAt(dimC), 0, "LAYER_NORM OP: wrong shape of gain array, expected is {%i}, but got %s instead !", input->sizeAt(dimC), ShapeUtils::shapeAsString(gain).c_str());

        NDArray* bias = nullptr;
        if (block.width() > 2) {
            bias = INPUT_VARIABLE(2);
            REQUIRE_TRUE(bias->rankOf() == 1 && bias->sizeAt(0) == input->sizeAt(dimC), 0, "LAYER_NORM OP: wrong shape of bias array, expected is {%i}, but got %s instead !", input->sizeAt(dimC), ShapeUtils::shapeAsString(bias).c_str());
        }

        std::vector<Nd4jLong> longAxis = ArrayUtils::toLongVector(axis);

        nd4j::ops::standardize standardizeOp;
        std::vector<NDArray *> inputs = {input};
        std::vector<NDArray *> outputs = {output};
        std::vector<double> targs = {};
        std::vector<bool> bargs = {};
        standardizeOp.execute(inputs, outputs, targs, longAxis, bargs);

        // output->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), gain, output);
        output->applyBroadcast(nd4j::broadcast::Multiply, {dimC}, gain);
        if(bias != nullptr) {
            // output->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Add(), bias, output);
            // output->applyBroadcast(nd4j::broadcast::Add, {dimC}, bias);
            helpers::addBias(block, *output, *bias, *output, isNCHW);
        }

        return Status::OK();
    }


    DECLARE_TYPES(layer_norm) {
        getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
        getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    }

    CUSTOM_OP_IMPL(layer_norm_bp, 3, -1, false, 0, -1) {
        auto input = INPUT_VARIABLE(0);
        auto gain = INPUT_VARIABLE(1);
        auto bias = block.width() == 4 ? INPUT_VARIABLE(2) : nullptr;
        auto eps = block.width() == 4 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);

        auto dLdx = OUTPUT_VARIABLE(0);
        auto dLdg = OUTPUT_VARIABLE(1);
        auto dLdb = block.width() == 4 ? OUTPUT_VARIABLE(2) : nullptr;

        const bool isNCHW = block.getBArguments()->size() > 0 ? B_ARG(0) : true;       // INT_ARG(9): 0-NCHW,  1-NHWC
        const int dimC = isNCHW ? 1 : input->rankOf() - 1;

        REQUIRE_TRUE(gain->rankOf() == 1 && gain->sizeAt(0) == input->sizeAt(dimC), 0, "LAYER_NORM_BP OP: wrong shape of gain array, expected is {%i}, but got %s instead !", input->sizeAt(dimC), ShapeUtils::shapeAsString(gain).c_str());

        auto axis = ArrayUtils::toIntVector(*block.getIArguments());

        std::vector<Nd4jLong> longAxis = ArrayUtils::toLongVector(axis);

        if(bias != nullptr) {
            REQUIRE_TRUE(bias->rankOf() == 1 && bias->sizeAt(0) == input->sizeAt(dimC), 0, "LAYER_NORM_BP OP: wrong shape of bias array, expected is {%i}, but got %s instead !", input->sizeAt(dimC), ShapeUtils::shapeAsString(bias).c_str());
            // eps->reduceAlongDimension(nd4j::reduce::Sum, dLdb, {0}, true);
            eps->reduceAlongDimension(nd4j::reduce::Sum, dLdb, ShapeUtils::evalDimsToExclude(input->rankOf(), {dimC}));
        }

        NDArray standardized(input->shapeInfo(), false, block.launchContext());

        nd4j::ops::standardize standardizeOp;
        std::vector<NDArray *> inputs = {input};
        std::vector<NDArray *> outputs = {&standardized};
        std::vector<double> targs = {};
        std::vector<bool> bargs = {};

        standardizeOp.execute(inputs, outputs, targs, longAxis, bargs);
        standardized.applyPairwiseTransform(nd4j::pairwise::Multiply, eps, &standardized, nullptr);
        standardized.reduceAlongDimension(nd4j::reduce::Sum, dLdg, ShapeUtils::evalDimsToExclude(input->rankOf(), {dimC}));

        nd4j::ops::standardize_bp standardizeBp;
        // eps->applyTrueBroadcast(nd4j::BroadcastOpsTuple::Multiply(), gain, dLdx);
        eps->applyBroadcast(nd4j::broadcast::Multiply, {dimC}, gain, dLdx);

        auto dLdx_tmp = dLdx->dup();
        std::vector<NDArray *> standardizeBpArgs = {input, dLdx_tmp};
        std::vector<NDArray *> standardizeBpOut = {dLdx};
        standardizeBp.execute(standardizeBpArgs, standardizeBpOut, targs, longAxis, bargs);
        delete dLdx_tmp;

        return Status::OK();
    }

    DECLARE_TYPES(layer_norm_bp) {
        getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
        getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
    }

    DECLARE_SHAPE_FN(layer_norm_bp) {
        Nd4jLong *dLdx_shape;
        COPY_SHAPE(inputShape->at(0), dLdx_shape);
        Nd4jLong *dLdg_shape;
        COPY_SHAPE(inputShape->at(1), dLdg_shape);
        if(inputShape->size() > 3){
            Nd4jLong *dLdb_shape;
            COPY_SHAPE(inputShape->at(2), dLdb_shape);
            return SHAPELIST(CONSTANT(dLdx_shape), CONSTANT(dLdg_shape), CONSTANT(dLdb_shape));
        }
        return SHAPELIST(CONSTANT(dLdx_shape), CONSTANT(dLdg_shape));
    }

}
}

#endif