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
// Created by raver119 on 12.10.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_tear)

#include <helpers/ConstantTadHelper.h>

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(tear, 1, -1, false, 0, -1) {
  auto input = INPUT_VARIABLE(0);

  REQUIRE_TRUE(!block.getIArguments()->empty(), 0, "At least 1 dimension should be specified for Tear");

  std::vector<sd::LongType> dims(*block.getIArguments());

  for (auto &v : dims)
    REQUIRE_TRUE(v >= 0 && v < input->rankOf(), 0,
                 "Tear dimensions should be non-negative values, and lower then input rank. Got %i instead", v);

  auto tads = input->allTensorsAlongDimension(dims);
  for (sd::LongType e = 0; e < tads.size(); e++) {
    auto outE = OUTPUT_VARIABLE(e);
    outE->assign(tads.at(e));

    // just for debugging purposes
    this->storeResult(block, e, *outE);
  }

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(tear) {
  auto inShape = inputShape->at(0);

  std::vector<sd::LongType> dims(*block.getIArguments());

  if (dims.size() > 1) std::sort(dims.begin(), dims.end());

  auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(inShape, &dims);
  auto numTads = tadPack->numberOfTads();

  auto result = SHAPELIST();
  for (sd::LongType e = 0; e < numTads; e++) {
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(block.dataType(), shape::order(inShape),
                                                                       shape::rank(tadPack->primaryShapeInfo()),
                                                                       shape::shapeOf(tadPack->primaryShapeInfo()),0);
    result->push_back(newShape);
  }

  return result;
}

DECLARE_TYPES(tear) { getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setSameMode(true); }
}  // namespace ops
}  // namespace sd

#endif
