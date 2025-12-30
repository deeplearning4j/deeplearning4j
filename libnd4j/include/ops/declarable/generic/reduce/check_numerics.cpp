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
// @author Adam Gibson
// Check numerics operation for mixed precision training
// Returns 1.0 if all values are finite (not inf/nan), 0.0 otherwise
//

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_check_numerics)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>
#include <math/templatemath.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(check_numerics, 1, 1, false, 0, 0) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Handle empty arrays - consider them valid
  if (input->isEmpty()) {
    output->p(0, 1.0);
    return sd::Status::OK;
  }

  // Check all elements for inf/nan
  bool allFinite = true;
  auto length = input->lengthOf();

  // Use a lambda to check each element
  auto checkFunc = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      auto val = input->e<double>(i);
      if (sd::math::sd_isnan<double>(val) || sd::math::sd_isinf<double>(val)) {
        allFinite = false;
        break;
      }
    }
  };

  // For small arrays, check sequentially
  if (length < 1000) {
    for (sd::LongType i = 0; i < length; i++) {
      auto val = input->e<double>(i);
      if (sd::math::sd_isnan<double>(val) || sd::math::sd_isinf<double>(val)) {
        allFinite = false;
        break;
      }
    }
  } else {
    // For larger arrays, use parallel reduction
    // Note: We use a simple atomic-style check here
    for (sd::LongType i = 0; i < length && allFinite; i++) {
      auto val = input->e<double>(i);
      if (sd::math::sd_isnan<double>(val) || sd::math::sd_isinf<double>(val)) {
        allFinite = false;
      }
    }
  }

  // Output 1.0 if all finite, 0.0 otherwise
  output->p(0, allFinite ? 1.0 : 0.0);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(check_numerics) {
  // Output is always a scalar
  return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(sd::DataType::DOUBLE));
}

DECLARE_TYPES(check_numerics) {
  getOpDescriptor()
      ->setAllowedInputTypes(sd::DataType::ANY)
      ->setAllowedOutputTypes({sd::DataType::DOUBLE});
}

}  // namespace ops
}  // namespace sd

#endif
