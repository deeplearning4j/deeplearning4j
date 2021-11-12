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
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_min_max_datatype)

#include <array/DataTypeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(min_max_datatype, -2, 1, false, 0, 2) {
  auto output = OUTPUT_VARIABLE(0);
  sd_debug("After min_max_datatype output\n",0);
  auto dataType = INT_ARG(0);
  sd_debug("After min_max_datatype output\n",0);
  DataType type = DataTypeUtils::fromInt(dataType);
  auto minOrMax = INT_ARG(1);
  sd_debug("After type output\n",0);
  if (minOrMax == 0) {
    switch (type) {
      case sd::DataType::UINT8:
        output->p(0, DataTypeUtils::min<uint8_t>());
        break;
      case sd::DataType::INT8:
        output->p(0, DataTypeUtils::min<int8_t>());
        break;
      case sd::DataType::BOOL:
        output->p(0, DataTypeUtils::min<bool>());
        break;
      case sd::DataType::BFLOAT16:
        output->p(0, DataTypeUtils::min<bfloat16>());
        break;
      case sd::DataType::HALF:
        output->p(0, DataTypeUtils::min<float16>());
        break;
      case sd::DataType::INT16:
        output->p(0, DataTypeUtils::min<int16_t>());
        break;
      case sd::DataType::UINT16:
        output->p(0, DataTypeUtils::min<uint16_t>());
        break;
      case sd::DataType::INT32:
        output->p(0, DataTypeUtils::min<int>());
        break;
      case sd::DataType::UINT32:
        output->p(0, DataTypeUtils::min<uint32_t>());
        break;
      case sd::DataType::FLOAT32:
        output->p(0, DataTypeUtils::min<float>());
        break;
      case sd::DataType::UINT64:
        output->p(0, DataTypeUtils::min<uint64_t>());
        break;
      case sd::DataType::INT64:
        output->p(0, DataTypeUtils::min<sd::LongType>());
        break;
      case sd::DataType::DOUBLE:
        output->p(0, DataTypeUtils::min<double>());
        break;
      default: {
        sd_printf("Unknown DataType used: [%i]\n", DataTypeUtils::asInt(type));
#ifndef __CUDA_ARCH__
        throw std::runtime_error("Unknown DataType requested");
#endif
      }
    }
  } else {
    switch (type) {
      case sd::DataType::UINT8:
        output->p(0, DataTypeUtils::max<uint8_t>());
        break;
      case sd::DataType::INT8:
        output->p(0, DataTypeUtils::max<int8_t>());
        break;
      case sd::DataType::BOOL:
        output->p(0, DataTypeUtils::max<bool>());
        break;
      case sd::DataType::BFLOAT16:
        output->p(0, DataTypeUtils::max<bfloat16>());
        break;
      case sd::DataType::HALF:
        output->p(0, DataTypeUtils::max<float16>());
        break;
      case sd::DataType::INT16:
        output->p(0, DataTypeUtils::max<int16_t>());
        break;
      case sd::DataType::UINT16:
        output->p(0, DataTypeUtils::max<uint16_t>());
        break;
      case sd::DataType::INT32:
        output->p(0, DataTypeUtils::max<int>());
        break;
      case sd::DataType::UINT32:
        output->p(0, DataTypeUtils::max<uint32_t>());
        break;
      case sd::DataType::FLOAT32:
        output->p(0, DataTypeUtils::max<float>());
        break;
      case sd::DataType::UINT64:
        output->p(0, DataTypeUtils::max<uint64_t>());
        break;
      case sd::DataType::INT64:
        output->p(0, DataTypeUtils::max<sd::LongType>());
        break;
      case sd::DataType::DOUBLE:
        output->p(0, DataTypeUtils::max<double>());
        break;
      default: {
        sd_printf("Unknown DataType used: [%i]\n", DataTypeUtils::asInt(type));
#ifndef __CUDA_ARCH__
        throw std::runtime_error("Unknown DataType requested");
#endif
      }
    }
    return sd::Status::OK;
  }
}

DECLARE_SHAPE_FN(min_max_datatype) {
  DataType newType = DataTypeUtils::fromInt(INT_ARG(0));

  return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(newType));
}

DECLARE_TYPES(min_max_datatype) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes(sd::DataType::ANY);
}
}  // namespace ops
}  // namespace sd

#endif
