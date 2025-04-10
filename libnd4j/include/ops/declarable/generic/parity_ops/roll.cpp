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

CONFIGURABLE_OP_IMPL(roll, -2, 1, true, 0, 0) {
  auto output = OUTPUT_VARIABLE(0);
  auto input = INPUT_VARIABLE(0);
  int inputLen = input->lengthOf();

  bool shiftIsLinear = block.width() == 1;
  std::vector<LongType> axes;
  std::vector<LongType> shifts;
  if (block.width() > 1) {
    REQUIRE_TRUE(block.width() == 3, 0, "roll: 3 arguments required for roll - input, shifts and axes. But %i given.",
                 block.width());
    auto axesI = INPUT_VARIABLE(2);
    auto shiftsI = INPUT_VARIABLE(1);
    REQUIRE_TRUE(axesI->rankOf() == shiftsI->rankOf(), 0,
                 "roll: shifts and axes should be the same rank, but %i and %i given.", (int)shiftsI->rankOf(),
                 (int)axesI->rankOf());
    REQUIRE_TRUE(axesI->lengthOf() == shiftsI->lengthOf(), 0,
                 "roll: shifts and axes should be the same length, but %i and %i given.", (int)shiftsI->lengthOf(),
                 (int)axesI->lengthOf());
    helpers::adjustAxis(axesI->lengthOf(), axesI, axes);
    shifts.resize(shiftsI->lengthOf());
    for (LongType i = 0; i < shiftsI->lengthOf(); i++) {
      auto shift = shiftsI->e<int>(i);
      if (shift < 0) {
        shift -= input->sizeAt(i) * (shift / inputLen - 1);
      } else if (shift != 0) {
        shift %= input->sizeAt(i);
      }

      shifts[i] = shift;
    }

  } else {
    int shift = INT_ARG(0);
    if (shift < 0) {
      // convert shift to positive value between 1 and inputLen - 1
      shift -= inputLen * (shift / inputLen - 1);
    } else if (shift != 0)
      // cut shift to value between 1 and inputLen - 1
      shift %= inputLen;
    axes.resize(block.getIArguments()->size() - 1);
    if (axes.size())
      shifts.resize(axes.size());  // emplace_back(shift);
    else
      shifts.push_back(shift);

    for (auto& s : shifts) s = shift;

    for (unsigned e = 0; e < axes.size(); ++e) {
      int axis = INT_ARG(e + 1);
      REQUIRE_TRUE(axis < input->rankOf() && axis >= -input->rankOf(), 0,
                   "roll: axe value should be between -%i and %i, but %i was given.", input->rankOf(),
                   input->rankOf() - 1, axis);
      axes[e] = (axis < 0 ? (input->rankOf() + axis) : axis);
    }
  }

  if (block.isInplace()) output = input;

  shiftIsLinear = (axes.size() == 0) || (input->rankOf() == 1);
  sd_debug("Roll: Shift is linear %d Shift is %d, first dimension is %d\n", (int)shiftIsLinear, (int)shifts[0],
           axes.size() > 0 ? axes[0] : 0);
  bool shiftsSumZero = false;
  auto shiftSum = 0;
  for (auto& s : shifts) {
    shiftSum += s;
    sd_debug("Roll: Shift  is %d\n", s);
  }
  // all zeros is no op
  if (shiftSum < 1) {
    sd_debug("Roll: No shift needed. Shift total was %d\n", shiftSum);
    if (!block.isInplace()) {
      output->assign(input);
    }
    return Status::OK;
  }

  if (shiftIsLinear) {
    helpers::rollFunctorLinear(block.launchContext(), input, output, shifts[0], block.isInplace());
  } else {
    helpers::rollFunctorFull(block.launchContext(), input, output, shifts, axes, block.isInplace());
  }

  return Status::OK;
}

DECLARE_TYPES(roll) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, INT32)  // TODO: all ints in future
      ->setAllowedInputTypes(2, INT32)
      ->setAllowedOutputTypes(ANY)
      ->setSameMode(true);
}
}  // namespace ops
}  // namespace sd

#endif
