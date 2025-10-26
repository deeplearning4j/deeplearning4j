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
// Created by george@skymind.io on 6/1/2018.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>

#if NOT_EXCLUDED(OP_reduce_prod)
namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_prod, -1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  //numpy compat: default is 1 for 0 length arrays https://stackoverflow.com/questions/66746566/numpy-explanation-of-numpy-prod
  if(input->lengthOf() == 0) {
    int one = 1;
    output->assign(one);
    return sd::Status::OK;
  }
  std::vector<sd::LongType> dimensions;
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  } else if (block.getIArguments()->size())
    dimensions = *block.getIArguments();

  REQUIRE_TRUE(
      dimensions.size() <= static_cast<size_t>(input->rankOf()), 0,
      "REDUCE_PROD OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead",
      dimensions.size());

  for (const auto& item : dimensions)
  REQUIRE_TRUE(item >= -input->shapeInfo()[0] && item < input->shapeInfo()[0], 0,
               "REDUCE_PROD OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
               input->rankOf(), input->rankOf(), item);

  bool keepDims = false;
  if (block.getBArguments()->size())
    keepDims = B_ARG(0);
  else if (block.getTArguments()->size())
    keepDims = (bool)T_ARG(0);

  input->reduceAlongDimension(reduce::Prod, output, &dimensions, keepDims);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(reduce_prod) {
  bool keepDims = false;
  if (block.getBArguments()->size())
    keepDims = B_ARG(0);
  else if (block.getTArguments()->size())
    keepDims = (bool)T_ARG(0);

  std::vector<sd::LongType> dimensions;
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(INPUT_VARIABLE(0)->rankOf(), axesVector, dimensions);
  } else if (block.getIArguments()->size())
    dimensions = *block.getIArguments();

  REQUIRE_TRUE(
      dimensions.size() <= static_cast<size_t>(inputShape->at(0)[0]), 0,
      "REDUCE_PROD OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead",
      dimensions.size());

  for (const auto& item : dimensions)
  REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0,
               "REDUCE_PROD OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
               inputShape->at(0)[0], inputShape->at(0)[0], item);

  return SHAPELIST(ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), &dimensions, inputShape->at(0),
                                                   keepDims, false, block.getWorkspace()));
}

DECLARE_TYPES(reduce_prod) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_prod_bp, -1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  if (gradO->lengthOf() == 1) {
    NDArray assign = input->reduceNumber(sd::reduce::Prod);
    gradI->assign(&assign);
    *gradI /= *input;
    *gradI *= gradO->e(0);
  } else {
    bool keepDims = false;
    auto dimensions = *block.getIArguments();

    if (block.width() > 2) {
      auto axesVector = INPUT_VARIABLE(2);
      helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
    }

    if (block.getBArguments()->size())
      keepDims = B_ARG(0);
    else if (block.getTArguments()->size())
      keepDims = (bool)T_ARG(0);

    REQUIRE_TRUE(
        dimensions.size() <= static_cast<size_t>(input->rankOf()), 0,
        "REDUCE_NORM1_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead",
        dimensions.size());

    for (const auto& item : dimensions)
    REQUIRE_TRUE(
        item >= -input->rankOf() && item < input->rankOf(), 0,
        "REDUCE_NORM1_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
        input->rankOf(), input->rankOf(), item);

    // *** calculations *** //

    auto products = input->reduceAlongDimension(reduce::Prod, &dimensions, true);
    gradI->applyTrueBroadcast(sd::BroadcastOpsTuple::Assign(), &products, gradI);
    *gradI /= *input;

    if (!keepDims) {
      auto gradOShapeKeepDims =
          ShapeUtils::evalReduceShapeInfo(gradO->ordering(), &dimensions, *input, true, false, block.getWorkspace());
      std::vector<sd::LongType> shape =  ShapeUtils::pullShapeFromShapeInfo(
          gradOShapeKeepDims);
      auto reshaped = gradO->reshape(gradO->ordering(),
                                     shape);
      *gradI *= *reshaped;  // for example could be something like [a,b] -> [1,a,1,b]
      delete reshaped;
    } else
      *gradI *= *gradO;
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(reduce_prod_bp) {
  auto dimensions = *block.getIArguments();
  if (block.width() > 2) {
    auto axesVector = INPUT_VARIABLE(2);
    helpers::adjustAxis(INPUT_VARIABLE(0)->rankOf(), axesVector, dimensions);
  }

  REQUIRE_TRUE(
      dimensions.size() <= static_cast<size_t>(inputShape->at(0)[0]), 0,
      "REDUCE_PROD_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead",
      dimensions.size());

  for (const auto& item : dimensions)
  REQUIRE_TRUE(
      item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0,
      "REDUCE_PROD_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
      inputShape->at(0)[0], inputShape->at(0)[0], item);

  return SHAPELIST(CONSTANT(inputShape->at(0)));
}

DECLARE_TYPES(reduce_prod_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}


}  // namespace ops
}  // namespace sd

#endif
