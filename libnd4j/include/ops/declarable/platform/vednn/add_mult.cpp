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

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "vednnUtils.h"

#if defined(HAVE_VEDA)

namespace sd {
namespace ops {
namespace platforms {

bool lastDimensionsAreEqual(const sd::LongType* shapeInfo1, const sd::LongType* shapeInfo2) {
  auto rank1 = shape::rank(shapeInfo1);
  auto rank2 = shape::rank(shapeInfo2);
  int min_rank;
  auto skipLeading1s = [](int rank, const sd::LongType* shape) {
    // skip the leading 1s in the smaller shape [1,1,..,n,m] -> [n,m]
    int skip = 0;
    for (int i = 0; i < rank; i++) {
      if (shape[i] == 1)
        ++skip;
      else
        break;
    }
    return skip;
  };
  const sd::LongType *shapeA, *shapeB;
  if (rank1 > rank2) {
    shapeA = shapeInfo2 + 1;
    auto skip = skipLeading1s(rank2, shapeA);
    shapeA += skip;
    shapeB = shapeInfo1 + (rank1 - rank2) + skip + 1;
    min_rank = rank2 - skip;
  } else if (rank1 == rank2) {
    shapeA = shapeInfo1 + 1;
    shapeB = shapeInfo2 + 1;
    auto skip1 = skipLeading1s(rank1, shapeA);
    auto skip2 = skipLeading1s(rank2, shapeB);
    shapeA += skip1;
    shapeB += skip2;
    rank1 -= skip1;
    rank2 -= skip2;
    if (rank2 > rank1) {
      min_rank = rank1;
      shapeB += (rank2 - rank1);
    } else {
      min_rank = rank2;
      shapeA += (rank1 - rank2);
    }
  } else {
    shapeA = shapeInfo1 + 1;
    auto skip = skipLeading1s(rank1, shapeA);
    if (skip == rank2) return true;
    shapeA += skip;
    shapeB = shapeInfo2 + (rank2 - rank1) + skip + 1;
  }

  if (min_rank > 0) {
    for (int i = 0; i < min_rank; i++) {
      if (shapeA[i] != shapeB[i]) return false;
    }
  }
  return true;
}

PLATFORM_IMPL(add, ENGINE_CPU) {
  auto input0 = INPUT_VARIABLE(0);
  auto input1 = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);
  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);

  VEDAdeviceptr vIn0, vIn1, vO;
  vIn0 = (VEDAdeviceptr)input0->specialBuffer();
  vIn1 = (VEDAdeviceptr)input1->specialBuffer();
  vO = (VEDAdeviceptr)output->specialBuffer();

  auto length0 = input0->lengthOf();
  auto length1 = input1->lengthOf();
  // sd_printf("%s %d:  %d %d\n",__FILE__, __LINE__, (int)length0, (int)length1);
  auto func = handle.getFunctionByConstPtrName("vedaAdd_A");
  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, (uint64_t)length0, vIn0, (uint64_t)length1, vIn1, vO));

  return sd::Status::OK;
}

PLATFORM_CHECK(add, ENGINE_CPU) {
  auto input0 = INPUT_VARIABLE(0);
  auto input1 = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);
  Requirements req("VEDNN ADD OP");

  req.expectEq(makeInfoVariable(input0->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(input0->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectFalse(makeInfoVariable(input0->isEmpty(), IS_EMPTY_MSG_INPUT0)) &&
      req.expectEq(makeInfoVariable(input0->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input1->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(input1->ews(), EWS_MSG_INPUT1), 1) &&
      req.expectEq(makeInfoVariable(input1->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectFalse(makeInfoVariable(input1->isEmpty(), IS_EMPTY_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1) &&
      req.expectFalse(makeInfoVariable(output->isEmpty(), IS_EMPTY_MSG_OUTPUT)) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      // we will differentiate the following cases
      // one of input is scalar or has length 1
      // the rank of one of the inputs is smaller equal, and also has the same dimensions excluding leading 1s
      // generic broadcastable
      // for now we will not allow generic case
      req.expectTrue(makeInfoVariable((input0->lengthOf() == 1 || input1->lengthOf() == 1 ||
                                       lastDimensionsAreEqual(input0->shapeInfo(), input1->shapeInfo())),
                                      "Op is continously broadcastable"));

  req.logTheSuccess();

  return req;
}


PLATFORM_IMPL(multiply, ENGINE_CPU) {
  auto input0 = INPUT_VARIABLE(0);
  auto input1 = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);
  VEDA_HANDLE& handle = VEDA::getInstance().getVEDA_HANDLE(0);

  VEDAdeviceptr vIn0, vIn1, vO;
  vIn0 = (VEDAdeviceptr)input0->specialBuffer();
  vIn1 = (VEDAdeviceptr)input1->specialBuffer();
  vO = (VEDAdeviceptr)output->specialBuffer();

  auto length0 = input0->lengthOf();
  auto length1 = input1->lengthOf();
  // sd_printf("mult %s %d:  %d %d\n",__FILE__, __LINE__, (int)length0, (int)length1);
  auto func = handle.getFunctionByConstPtrName("vedaMult_A");
  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, (uint64_t)length0, vIn0, (uint64_t)length1, vIn1, vO));

  return sd::Status::OK;
}

PLATFORM_CHECK(multiply, ENGINE_CPU) {
  auto input0 = INPUT_VARIABLE(0);
  auto input1 = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);
  Requirements req("VEDNN MULT OP");
  req.expectEq(makeInfoVariable(input0->ordering(), ORDERING_MSG_INPUT0), 'c') &&
      req.expectEq(makeInfoVariable(input0->ews(), EWS_MSG_INPUT0), 1) &&
      req.expectFalse(makeInfoVariable(input0->isEmpty(), IS_EMPTY_MSG_INPUT0)) &&
      req.expectEq(makeInfoVariable(input0->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(input1->ordering(), ORDERING_MSG_INPUT1), 'c') &&
      req.expectEq(makeInfoVariable(input1->ews(), EWS_MSG_INPUT1), 1) &&
      req.expectEq(makeInfoVariable(input1->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectFalse(makeInfoVariable(input1->isEmpty(), IS_EMPTY_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1) &&
      req.expectFalse(makeInfoVariable(output->isEmpty(), IS_EMPTY_MSG_OUTPUT)) &&
      req.expectEq(makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      // we will differentiate the following cases
      // one of input is scalar or has length 1
      // the rank of one of the inputs is smaller equal, and also has the same dimensions excluding leading 1s
      // generic broadcastable
      // for now we will not allow generic case
      req.expectTrue(makeInfoVariable((input0->lengthOf() == 1 || input1->lengthOf() == 1 ||
                                       lastDimensionsAreEqual(input0->shapeInfo(), input1->shapeInfo())),
                                      "Op is continously broadcastable"));

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif
