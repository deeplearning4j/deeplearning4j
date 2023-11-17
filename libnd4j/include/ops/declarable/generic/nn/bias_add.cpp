/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_biasadd)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/addBias.h>

namespace sd {
namespace ops {

////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(biasadd, 2, 1, true, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto bias = INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);

  const bool isNCHW = !block.getBArguments()->empty() ? B_ARG(0) : false;
  const LongType channelDim = isNCHW ? 1 : input->rankOf() - 1;  // second or last

  REQUIRE_TRUE(bias->rankOf() == 1, 0, "BIASADD CUSTOM_OP: bias array should have rank = 1, but got %i instead !",
               bias->rankOf());

  REQUIRE_TRUE(bias->sizeAt(0) == input->sizeAt(channelDim), 0,
               "BIASADD CUSTOM_OP: shapes of bias %s and input %s arrays are not suitable for broadcast operation "
               "along channel dimension %i !",
               ShapeUtils::shapeAsString(bias).c_str(), ShapeUtils::shapeAsString(input).c_str(), channelDim);

  REQUIRE_TRUE(output->isSameShape(input), 0,
               "BIASADD CUSTOM_OP: wrong shape of output array, expected is %s but got %s instead !",
               ShapeUtils::shapeAsString(input).c_str(), ShapeUtils::shapeAsString(output).c_str());

  helpers::addBias(block, *input, *bias, *output, isNCHW);

  return Status::OK;
}
DECLARE_SYN(bias_add, biasadd);

////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(biasadd) {
  auto xShape = inputShape->at(0);
  auto yShape = inputShape->at(1);

  auto dtype = ArrayOptions::dataType(yShape);
  auto desc = new ShapeDescriptor(xShape, dtype);
  auto ret =  SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
  delete desc;
  return ret;
}

DECLARE_TYPES(biasadd) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(biasadd_bp, 3, 2, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto bias = INPUT_VARIABLE(1);
  auto gradO = INPUT_VARIABLE(2);

  auto gradI = OUTPUT_VARIABLE(0);
  auto gradB = OUTPUT_VARIABLE(1);

  const bool isNCHW = !block.getBArguments()->empty() ? B_ARG(0) : false;
  const int channelDim = isNCHW ? 1 : input->rankOf() - 1;  // second or last

  gradI->assign(gradO);

  std::vector<LongType> channel;
  channel.push_back(channelDim);
  auto dims = ShapeUtils::evalDimsToExclude(gradO->rankOf(), 1,channel.data());
  gradO->reduceAlongDimension(reduce::Sum, *gradB, dims);
  delete dims;
  return Status::OK;
}
DECLARE_SYN(BiasAddGrad, biasadd_bp);

////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(biasadd_bp) {
  auto input = inputShape->at(0);
  auto bias = inputShape->at(1);

  LongType* epsShape;
  LongType* gradShape;

  COPY_SHAPE(input, epsShape);
  COPY_SHAPE(bias, gradShape);

  return SHAPELIST(CONSTANT(epsShape), CONSTANT(gradShape));
}

DECLARE_TYPES(biasadd_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif
