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

#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "vednnUtils.h"

#if defined(HAVE_VEDA)

namespace sd {
namespace ops {
namespace platforms {

PLATFORM_IMPL(pad, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto paddings = INPUT_VARIABLE(1);
  float padValue = (block.width() > 2) ? INPUT_VARIABLE(2)->e<float>(0) : T_ARG(0);
  auto output = OUTPUT_VARIABLE(0);
  auto zStrides = output->stridesOf();
  sd::LongType paddingOffsetCoords[SD_MAX_RANK] = {};
  sd::LongType* ptrPaddingCoords = (sd::LongType*)&paddingOffsetCoords;
  bool all_paddings_zero = true;
  for (int j = 0; j < input->rankOf(); j++) {
    auto p0 = paddings->e<sd::LongType>(j, 0);
    auto p1 = paddings->e<sd::LongType>(j, 1);
    paddingOffsetCoords[j] = p0;

    all_paddings_zero = all_paddings_zero && (p0 == 0) && (p1 == 0);
  }

  sd::LongType paddingOffset =
      all_paddings_zero ? 0L : sd::offset_from_coords(zStrides, ptrPaddingCoords, input->rankOf());
  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);
  VEDAdeviceptr vIn, vO;
  vIn = (VEDAdeviceptr)input->specialBuffer();
  vO = (VEDAdeviceptr)output->specialBuffer();

  auto func = handle.getFunctionByConstPtrName("vedaPadConstantRank4");
  VEDA_CALL_THROW(vedaLaunchKernel(
      func, 0, VEDAstack((void*)input->shapeInfo(), VEDA_ARGS_INTENT_IN, shape::shapeInfoByteLength(input->rankOf())),
      vIn, VEDAstack((void*)output->shapeInfo(), VEDA_ARGS_INTENT_IN, shape::shapeInfoByteLength(output->rankOf())), vO,
      paddingOffset, padValue));

  return sd::Status::OK;
}

PLATFORM_CHECK(pad, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto paddings = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("VEDNN Pad OP");
  req.expectEq(makeInfoVariable(INT_ARG(0), "Padding mode"), 0) &&
      req.expectEq(makeInfoVariable(input->ordering(), ORDERING_MSG_INPUT), 'c') &&
      req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT), 4) &&
      req.expectEq(makeInfoVariable(input->ews(), EWS_MSG_INPUT), 1) &&
      req.expectFalse(makeInfoVariable(input->isEmpty(), IS_EMPTY_MSG_INPUT)) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1) &&
      req.expectFalse(makeInfoVariable(output->isEmpty(), IS_EMPTY_MSG_OUTPUT)) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeShapeInfoVariable(paddings->getShapeAsVector(), SHAPE_MSG_INPUT0),
                   makeShapeInfoVariable(std::vector<sd::LongType>{input->rankOf(), 2}, NO_MSG));

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif
