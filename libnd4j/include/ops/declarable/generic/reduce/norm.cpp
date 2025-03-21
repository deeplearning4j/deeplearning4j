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
#if NOT_EXCLUDED(OP_norm)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>

namespace sd {
namespace ops {
REDUCTION_OP_IMPL(norm, 1, 1, false, 1, -2) {
  auto input = INPUT_VARIABLE(0);
  NDArray *output = OUTPUT_VARIABLE(0);

  auto mode = (int)T_ARG(0);
  std::vector<sd::LongType> dims = *block.getIArguments();
  bool overwrite = false;

  if (block.width() == 1) {
    output = OUTPUT_VARIABLE(0);
  } else {
    auto axisVector = INPUT_VARIABLE(1);
    dims.resize(axisVector->lengthOf());
    helpers::adjustAxis(input->rankOf(), axisVector, dims);
    auto shape = ShapeUtils::evalReduceShapeInfo(input->ordering(), &dims, *input, false, false);
    if (!shape::equalsStrict(shape, output->shapeInfo())) {
      output = new NDArray(shape, false, block.launchContext());
      overwrite = true;
    }
  }
  switch (mode) {
    case 0: {
      REQUIRE_TRUE(dims.size() == 2 || (input->rankOf() == 2 && dims.size() == 0), 0,
                   "Norm: Frobenius is defined for 2D matrices or TADS only");
      // fro
      input->reduceAlongDimension(reduce::NormFrobenius, output, &dims, false, output->rankOf() == 2);
    } break;
    case 1: {
      // euclidean
      if ((input->rankOf() == 2 && dims.size() == 0) || dims.size() == 2) {
        input->reduceAlongDimension(reduce::NormFrobenius, output, &dims, false, output->rankOf() == 2);
      } else {
        input->reduceAlongDimension(reduce::Norm2, output, &dims, false, output->rankOf() == 2);
      }
    } break;
    case 2: {
      // 1
      input->reduceAlongDimension(reduce::Norm1, output, &dims, false, output->rankOf() == 2);
    } break;
    case 3: {
      // 2
      input->reduceAlongDimension(reduce::Norm2, output, &dims, false, output->rankOf() == 2);
    } break;
    case 4: {
      // inf-norm
      input->reduceAlongDimension(reduce::NormMax, output, &dims, false, output->rankOf() == 2);
    } break;
    default: {
      // p-norm
      REQUIRE_TRUE(block.getIArguments()->size() > 1, 0,
                   "P-Norm reductions requires 2 TArguments, but only 1 was provided");
      // FIXME: p is required here
      // T p = T_ARG(1);
      input->reduceAlongDimension(reduce::NormP, output, &dims, false, output->rankOf() == 2);
    }
  }

  if (overwrite) {
    OVERWRITE_RESULT(output);
  }

  return sd::Status::OK;
};

DECLARE_TYPES(norm) { getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS}); }
}  // namespace ops
}  // namespace sd

#endif
