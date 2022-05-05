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
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "vednnUtils.h"

#if defined(HAVE_VEDA)

namespace sd {
namespace ops {
namespace platforms {


PLATFORM_IMPL(concat, ENGINE_CPU) {
  auto output = OUTPUT_VARIABLE(0);

  std::vector<const NDArray *> nonEmptyArrs;
  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);
  const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();
  for (int i = 0; i < numOfInArrs; ++i) {
    auto input = INPUT_VARIABLE(i);
    if (!input->isEmpty()) nonEmptyArrs.push_back(input);
  }

  VEDA_HANDLE &handle = VEDA::getInstance().getVEDA_HANDLE(0);
  SCOPED_VEDA_CONTEXT scopedContext(handle.getDevice());

  auto func = handle.getFunctionByConstPtrName("vedaConcatUpTo32");

  VEDAdeviceptr vO;
  NDArray::prepareVedaUse({output}, nonEmptyArrs);
  
  std::vector<VEDAdeviceptr> inputList;
  for(auto input: nonEmptyArrs){
    
      inputList.push_back((VEDAdeviceptr)input->specialBuffer());
  }
  vO = (VEDAdeviceptr)output->specialBuffer();

  VEDA_CALL_THROW(vedaLaunchKernel(func, 0, (uint64_t)nonEmptyArrs.size(), VEDAstack(inputList.data(), VEDA_ARGS_INTENT_IN, inputList.size() * sizeof(VEDAdeviceptr)),  vO));

  NDArray::registerVedaUse({output}, nonEmptyArrs);

  // scopedContext.sync();

  return sd::Status::OK;
}

/**
 * @brief Checks if the shape of NDArray contains 1 before(order c) or after(order f) the specified axis
 *
 * @param input
 * @param axis
 * @return int
 */
SD_INLINE int isShapeExtendedWithOnes(const NDArray &input, int axis) {
  bool isAllOne = true;
  auto shapes = shape::shapeOf(input.shapeInfo());
  auto rank = input.rankOf();
  if (rank == 0 && axis == 0) return true;  // consider scalar as true
  if (rank > axis) {
    if (input.ordering() == 'c') {
      // check before the axis
      for (int i = 0; i < axis; i++) {
        isAllOne = isAllOne && (shapes[i] == 1);
      }
    } else {
      // check after the axis
      for (int i = axis + 1; i < rank; i++) {
        isAllOne = isAllOne && (shapes[i] == 1);
      }
    }
    return isAllOne;
  }

  return true;
}

PLATFORM_CHECK(concat, ENGINE_CPU) {
  auto output = OUTPUT_VARIABLE(0);
// sd::Environment::getInstance().setDebug(true);
// sd::Environment::getInstance().setVerbose(true);
  const bool isAxisInLastArr = block.getBArguments()->size() == 0 ? false : B_ARG(0);
  const int numOfInArrs = isAxisInLastArr ? block.width() - 1 : block.width();
  Requirements req("VEDNN CONCAT OP");
  req.expectLessEq(makeInfoVariable(numOfInArrs, "numOfinArrs"), 32) &&
      req.expectGreater(makeInfoVariable(numOfInArrs, "numOfinArrs"), 0) &&
      req.expectEq(makeInfoVariable(output->ordering(), ORDERING_MSG_OUTPUT), 'c') &&
      req.expectEq(makeInfoVariable(output->ews(), EWS_MSG_OUTPUT), 1) &&
      req.expectFalse(makeInfoVariable(output->isEmpty(), IS_EMPTY_MSG_OUTPUT));

  if (req) {
    int axis = isAxisInLastArr ? INPUT_VARIABLE(block.width() - 1)->e<int>(0) : INT_ARG(0);

    req.expectTrue(makeInfoVariable(
                       [&block, output, numOfInArrs, axis] {
                         bool allAreEmpty = true;
                         auto ax = axis;
                         for (int i = 0; i < numOfInArrs; ++i) {
                           auto input = INPUT_VARIABLE(i);
                           if (!input->isEmpty()) {
                             allAreEmpty = false;
                             if (ax < 0) {
                               ax += input->rankOf();
                             }
                             break;
                           }
                         }

                         if (allAreEmpty) return false;

                         bool matchesOutputOrdering = true;
                         bool shapeExtendedWithOnes = isShapeExtendedWithOnes(*output, ax);
                         bool followEws1 = true;
                         for (int i = 0; i < numOfInArrs; ++i) {
                           auto input = INPUT_VARIABLE(i);
                           if (!input->isEmpty()) {
                             shapeExtendedWithOnes = shapeExtendedWithOnes && isShapeExtendedWithOnes(*input, ax);
                             followEws1 = followEws1 && input->ews() == 1;
                             matchesOutputOrdering = matchesOutputOrdering && input->ordering() == output->ordering();
                           }
                         }

                         bool copyCaseEws1 = followEws1 & matchesOutputOrdering;
                         bool copyCase1 = numOfInArrs > 1 ? copyCaseEws1 & shapeExtendedWithOnes : copyCaseEws1;
                         return copyCase1;
                       },
                       NO_MSG),
                   NO_MSG);
  }
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif
