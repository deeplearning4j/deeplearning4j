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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 04.06.2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/helpers/reductions.h>
#if NOT_EXCLUDED(OP_reduce_stdev)
namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_stdev, -1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  //numpy compat: default is 1 for 0 length arrays https://stackoverflow.com/questions/66746566/numpy-explanation-of-numpy-prod
  if(input->lengthOf() == 0) {
    int one = 1;
    output->assign(one);
    return sd::Status::OK;
  }
  bool biasCorrected = false;  // block.getTArguments()->size() > 1 ? (bool)T_ARG(1) : false;

  auto dimensions = *block.getIArguments();
  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  }

  if (block.getBArguments()->size()) {
    if (block.getBArguments()->size() > 1) biasCorrected = B_ARG(1);
  } else if (block.getTArguments()->size()) {
    if (block.getTArguments()->size() > 1) biasCorrected = (bool)T_ARG(1);
  }

  REQUIRE_TRUE(
      dimensions.size() <= static_cast<size_t>(input->rankOf()), 0,
      "REDUCE_STDEV OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead",
      dimensions.size());

  for (const auto& item : dimensions)
    REQUIRE_TRUE(
        item >= -input->rankOf() && item < input->rankOf(), 0,
        "REDUCE_STDEV OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
        input->rankOf(), input->rankOf(), item);

  sd::ops::helpers::standardDeviation(*input, *output, dimensions, biasCorrected);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(reduce_stdev) {
  auto in = inputShape->at(0);
  auto rank = shape::rank(in);
  bool keepDims = false;  // block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
  auto dimensions = *block.getIArguments();

  if (block.width() > 1) {
    auto axesVector = INPUT_VARIABLE(1);
    helpers::adjustAxis(rank, axesVector, dimensions);
  }

  if (block.getBArguments()->size()) {
    keepDims = B_ARG(0);
  } else if (block.getTArguments()->size()) {
    keepDims = (bool)T_ARG(0);
  }

  REQUIRE_TRUE(
      dimensions.size() <= static_cast<size_t>(rank), 0,
      "REDUCE_STDEV OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead",
      dimensions.size());

  for (const auto& item : dimensions)
    REQUIRE_TRUE(
        item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0,
        "REDUCE_STDEV OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
        inputShape->at(0)[0], inputShape->at(0)[0], item);

  auto outShapeInfo =
      ShapeUtils::evalReduceShapeInfo(shape::order(in), &dimensions, in, keepDims, false, block.getWorkspace());

  return SHAPELIST(outShapeInfo);
}

DECLARE_TYPES(reduce_stdev) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_stdev_bp, -1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);

  auto gradI = OUTPUT_VARIABLE(0);

  bool keepDims = false;       // block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
  bool biasCorrected = false;  // block.getTArguments()->size() > 1 ? (bool)T_ARG(1) : false;

  auto dimensions = *block.getIArguments();
  if (block.width() > 2) {
    auto axesVector = INPUT_VARIABLE(2);
    helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
  }

  if (block.getBArguments()->size()) {
    keepDims = B_ARG(0);
    if (block.getBArguments()->size() > 1) biasCorrected = B_ARG(1);
  } else if (block.getTArguments()->size()) {
    keepDims = (bool)T_ARG(0);
    if (block.getTArguments()->size() > 1) biasCorrected = (bool)T_ARG(1);
  }

  REQUIRE_TRUE(
      dimensions.size() <= static_cast<size_t>(input->rankOf()), 0,
      "REDUCE_STDEV_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead",
      dimensions.size());

  for (const auto& item : dimensions)
    REQUIRE_TRUE(
        item >= -input->rankOf() && item < input->rankOf(), 0,
        "REDUCE_STDEV_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
        input->rankOf(), input->rankOf(), item);

  auto gradOLen = gradO->lengthOf() < 1 ? 1 : gradO->lengthOf();
  const sd::LongType N = input->lengthOf() / gradOLen;
  const sd::LongType NminusOne = biasCorrected ? N - 1 : N;

  auto mean = input->reduceAlongDimension(reduce::Mean, &dimensions, true);

  NDArray variance(mean.shapeInfo(), true,
                   block.launchContext());  // create empty array with shape matching shape of mean array
  input->varianceAlongDimension(variance::SummaryStatsStandardDeviation, variance, biasCorrected, &dimensions);

  sd::ops::divide_no_nan divideNoNan;
  auto inputMinusMean  = (*input - mean);
  auto varianceTimesNMinusOne = variance * NminusOne;
  divideNoNan.execute({&inputMinusMean,&varianceTimesNMinusOne},{gradI});

  if (!keepDims) {
    auto gradOShapeKeepDims =
        ShapeUtils::evalReduceShapeInfo(gradO->ordering(), &dimensions, *input, true, false, block.getWorkspace());
    if (!gradO->isScalar()) {
      std::vector<sd::LongType> shape =  ShapeUtils::pullShapeFromShapeInfo(
          gradOShapeKeepDims);
      *gradI *= gradO->reshape(gradO->ordering(),
                               shape);  // for example could be something like [a,b] -> [1,a,1,b]
    } else {
      *gradI *= *gradO;  // for example could be something like [a,b] -> [1,a,1,b]
    }
  } else {
    *gradI *= *gradO;  // automatic broadcasting happens here
  }
  return sd::Status::OK;
}

DECLARE_SHAPE_FN(reduce_stdev_bp) {
  auto in = inputShape->at(0);
  auto rank = shape::rank(in);
  auto dimensions = *block.getIArguments();
  if (block.width() > 2) {
    auto axesVector = INPUT_VARIABLE(2);
    helpers::adjustAxis(rank, axesVector, dimensions);
  }

  REQUIRE_TRUE(
      dimensions.size() <= static_cast<size_t>(rank), 0,
      "REDUCE_STDEV_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead",
      dimensions.size());

  for (const auto& item : dimensions)
    REQUIRE_TRUE(
        item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0,
        "REDUCE_STDEV_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !",
        inputShape->at(0)[0], inputShape->at(0)[0], item);

  return SHAPELIST(CONSTANT(in));
}

DECLARE_TYPES(reduce_stdev_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd
#endif
