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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_roll)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/helpers/roll.h>

namespace sd {
namespace ops {

CONFIGURABLE_OP_IMPL(roll, -2, 1, true, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = block.isInplace() ? input : OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  std::vector<LongType> shifts;
  const auto inputCount = block.width();
  const bool hasTensorShift = inputCount > 1;
  const bool hasTensorAxes = inputCount == 3;

  REQUIRE_TRUE(inputCount >= 1 && inputCount <= 3, 0,
               "roll: expected input with an optional shift and axes, but received %i input arrays.", inputCount);

  if (hasTensorShift) {
    auto shiftsInput = INPUT_VARIABLE(1);
    if (!hasTensorAxes) {
      REQUIRE_TRUE(shiftsInput->lengthOf() == 1, 0,
                   "roll: an axis-free tensor shift must contain one value, but received %lld.",
                   static_cast<long long>(shiftsInput->lengthOf()));
      shifts.push_back(shiftsInput->e<LongType>(0));
    } else {
      auto axesInput = INPUT_VARIABLE(2);
      REQUIRE_TRUE(axesInput->rankOf() == shiftsInput->rankOf(), 0,
                   "roll: shifts and axes must have the same rank, but received %i and %i.",
                   static_cast<int>(shiftsInput->rankOf()), static_cast<int>(axesInput->rankOf()));
      REQUIRE_TRUE(axesInput->lengthOf() == shiftsInput->lengthOf(), 0,
                   "roll: shifts and axes must have the same length, but received %lld and %lld.",
                   static_cast<long long>(shiftsInput->lengthOf()), static_cast<long long>(axesInput->lengthOf()));

      helpers::adjustAxis(input->rankOf(), axesInput, axes);
      shifts.resize(static_cast<size_t>(shiftsInput->lengthOf()));
      for (LongType i = 0; i < shiftsInput->lengthOf(); ++i) {
        shifts[static_cast<size_t>(i)] = shiftsInput->e<LongType>(i);
        const auto axis = axes[static_cast<size_t>(i)];
        REQUIRE_TRUE(axis >= 0 && axis < input->rankOf(), 0,
                     "roll: axis must be in the range [-%i, %i), but received %lld.", input->rankOf(),
                     input->rankOf(), static_cast<long long>(axesInput->e<LongType>(i)));
      }
    }
  } else {
    REQUIRE_TRUE(block.getIArguments() != nullptr && !block.getIArguments()->empty(), 0,
                 "roll: a shift integer argument is required when shifts and axes tensors are absent.");
    const LongType shift = INT_ARG(0);
    axes.resize(block.getIArguments()->size() - 1);
    shifts.resize(axes.empty() ? 1 : axes.size(), shift);

    for (size_t i = 0; i < axes.size(); ++i) {
      const LongType suppliedAxis = INT_ARG(i + 1);
      REQUIRE_TRUE(suppliedAxis >= -input->rankOf() && suppliedAxis < input->rankOf(), 0,
                   "roll: axis must be in the range [-%i, %i), but received %lld.", input->rankOf(),
                   input->rankOf(), static_cast<long long>(suppliedAxis));
      axes[i] = suppliedAxis < 0 ? suppliedAxis + input->rankOf() : suppliedAxis;
    }
  }

  if (!hasTensorAxes && axes.empty()) {
    helpers::rollFunctorLinear(block.launchContext(), input, output, shifts[0], block.isInplace());
  } else {
    helpers::rollFunctorFull(block.launchContext(), input, output, shifts, axes, block.isInplace());
  }

  return Status::OK;
}

DECLARE_TYPES(roll) {
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {ALL_INDICES})
      ->setAllowedInputTypes(2, {ALL_INDICES})
      ->setAllowedOutputTypes(0, INHERIT)
      ->setSameMode(true);
}
}  // namespace ops
}  // namespace sd

#endif
