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
// @author Paul Dubs
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_layer_norm)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/reverse.h>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(layer_norm, 2, 1, false, 0, -1) {
  auto input = INPUT_VARIABLE(0);
  auto gain = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<sd::LongType> axis = *block.getIArguments();

  const bool isNCHW = block.getBArguments()->size() > 0 ? B_ARG(0) : true;  // 0-NCHW,  1-NHWC
  const int dimC = isNCHW ? 1 : input->rankOf() - 1;

  REQUIRE_TRUE(gain->rankOf() == 1 && gain->sizeAt(0) == input->sizeAt(dimC), 0,
               "LAYER_NORM OP: wrong shape of gain array, expected is {%i}, but got %s instead !", input->sizeAt(dimC),
               ShapeUtils::shapeAsString(gain).c_str());

  NDArray *bias = nullptr;
  if (block.width() > 2) {
    bias = INPUT_VARIABLE(2);
    REQUIRE_TRUE(bias->rankOf() == 1 && bias->sizeAt(0) == input->sizeAt(dimC), 0,
                 "LAYER_NORM OP: wrong shape of bias array, expected is {%i}, but got %s instead !",
                 input->sizeAt(dimC), ShapeUtils::shapeAsString(bias).c_str());
  }

  std::vector<sd::LongType> longAxis = ArrayUtils::toLongVector(axis);

  sd::ops::standardize standardizeOp;
  std::vector<NDArray *> inputs = {input};
  std::vector<NDArray *> outputs = {output};
  std::vector<double> targs = {};
  std::vector<bool> bargs = {};
  standardizeOp.execute(inputs, outputs, targs, longAxis, bargs);

  std::vector<sd::LongType> dimcVec = {dimC};
  output->applyBroadcast(sd::broadcast::Multiply, &dimcVec, gain, output);
  if (bias != nullptr) {
    helpers::addBias(block, *output, *bias, *output, isNCHW);
  }

  return sd::Status::OK;
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

  const bool isNCHW = block.getBArguments()->size() > 0 ? B_ARG(0) : true;  //  0-NCHW,  1-NHWC
  const int dimC = isNCHW ? 1 : input->rankOf() - 1;

  REQUIRE_TRUE(gain->rankOf() == 1 && gain->sizeAt(0) == input->sizeAt(dimC), 0,
               "LAYER_NORM_BP OP: wrong shape of gain array, expected is {%i}, but got %s instead !",
               input->sizeAt(dimC), ShapeUtils::shapeAsString(gain).c_str());

  std::vector<sd::LongType> axis = *block.getIArguments();

  std::vector<sd::LongType> longAxis = ArrayUtils::toLongVector(axis);

  if (bias != nullptr) {
    REQUIRE_TRUE(bias->rankOf() == 1 && bias->sizeAt(0) == input->sizeAt(dimC), 0,
                 "LAYER_NORM_BP OP: wrong shape of bias array, expected is {%i}, but got %s instead !",
                 input->sizeAt(dimC), ShapeUtils::shapeAsString(bias).c_str());
    std::vector<sd::LongType> dimCVector = {dimC};
    auto vec = ShapeUtils::evalDimsToExclude(input->rankOf(),1,dimCVector.data());
    eps->reduceAlongDimension(sd::reduce::Sum, dLdb, vec);
  }

  NDArray standardized(input->shapeInfo(), false, block.launchContext());

  sd::ops::standardize standardizeOp;
  std::vector<NDArray *> inputs = {input};
  std::vector<NDArray *> outputs = {&standardized};
  std::vector<double> targs = {};
  std::vector<bool> bargs = {};

  standardizeOp.execute(inputs, outputs, targs, longAxis, bargs);
  standardized.applyPairwiseTransform(sd::pairwise::Multiply, eps, &standardized);
  std::vector<sd::LongType> dimCVector = {dimC};
  auto vec = ShapeUtils::evalDimsToExclude(input->rankOf(),1,dimCVector.data());
  standardized.reduceAlongDimension(sd::reduce::Sum, dLdg, vec);

  sd::ops::standardize_bp standardizeBp;
  std::vector<sd::LongType> dimvC = {dimC};
  eps->applyBroadcast(sd::broadcast::Multiply, &dimvC, gain, dLdx);

  auto dLdx_tmp = dLdx->dup();
  std::vector<NDArray *> standardizeBpArgs = {input, &dLdx_tmp};
  std::vector<NDArray *> standardizeBpOut = {dLdx};
  standardizeBp.execute(standardizeBpArgs, standardizeBpOut, targs, longAxis, bargs);

  return sd::Status::OK;
}

DECLARE_TYPES(layer_norm_bp) {
  getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS});
  getOpDescriptor()->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(layer_norm_bp) {
  sd::LongType *dLdx_shape;
  COPY_SHAPE(inputShape->at(0), dLdx_shape);
  sd::LongType *dLdg_shape;
  COPY_SHAPE(inputShape->at(1), dLdg_shape);
  if (inputShape->size() > 3) {
    sd::LongType *dLdb_shape;
    COPY_SHAPE(inputShape->at(2), dLdb_shape);
    return SHAPELIST(CONSTANT(dLdx_shape), CONSTANT(dLdg_shape), CONSTANT(dLdb_shape));
  }
  return SHAPELIST(CONSTANT(dLdx_shape), CONSTANT(dLdg_shape));
}

}  // namespace ops
}  // namespace sd

#endif
