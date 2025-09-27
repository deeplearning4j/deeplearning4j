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
  auto dataType = INT_ARG(0);
  DataType type = DataTypeUtils::fromInt(dataType);
  auto minOrMax = INT_ARG(1);
  if (minOrMax == 0) {
    switch (type) {
#ifdef HAS_UINT8
      case UINT8:
        output->p(0, DataTypeUtils::min<uint8_t>());
        break;
#endif
#ifdef HAS_INT8
      case INT8:
        output->p(0, DataTypeUtils::min<int8_t>());
        break;
#endif
#ifdef HAS_BOOL
      case BOOL:
        output->p(0, DataTypeUtils::min<bool>());
        break;
#endif
#ifdef HAS_BFLOAT16
      case BFLOAT16:
        output->p(0, DataTypeUtils::min<bfloat16>());
        break;
#endif
#ifdef HAS_FLOAT16
      case HALF:
        output->p(0, DataTypeUtils::min<float16>());
        break;
#endif
#ifdef HAS_INT16
      case INT16:
        output->p(0, DataTypeUtils::min<int16_t>());
        break;
#endif
#ifdef HAS_UINT16
      case UINT16:
        output->p(0, DataTypeUtils::min<uint16_t>());
        break;
#endif
#ifdef HAS_INT32
      case INT32:
        output->p(0, DataTypeUtils::min<int>());
        break;
#endif
#ifdef HAS_UINT32
      case UINT32:
        output->p(0, DataTypeUtils::min<uint32_t>());
        break;
#endif
#ifdef HAS_FLOAT32
      case FLOAT32:
        output->p(0, DataTypeUtils::min<float>());
        break;
#endif
#ifdef HAS_UNSIGNEDLONG
      case UINT64:
        output->p(0, DataTypeUtils::min<uint64_t>());
        break;
#endif
#ifdef HAS_LONG
      case INT64:
        output->p(0, DataTypeUtils::min<LongType>());
        break;
#endif
#ifdef HAS_DOUBLE
      case DOUBLE:
        output->p(0, DataTypeUtils::min<double>());
        break;
#endif
      default: {
        std::string errorMessage;
        errorMessage += "Min: Unknown type requested: " + DataTypeUtils::asString(type);
        THROW_EXCEPTION(errorMessage.c_str());
#ifndef __CUDA_ARCH__
        THROW_EXCEPTION("Unknown DataType requested");
#endif
      }
    }
  } else {
    switch (type) {
#ifdef HAS_UINT8
      case UINT8:
        output->p(0, DataTypeUtils::max<uint8_t>());
        break;
#endif
#ifdef HAS_INT8
      case INT8:
        output->p(0, DataTypeUtils::max<int8_t>());
        break;
#endif
#ifdef HAS_BOOL
      case BOOL:
        output->p(0, DataTypeUtils::max<bool>());
        break;
#endif
#ifdef HAS_BFLOAT16
      case BFLOAT16:
        output->p(0, DataTypeUtils::max<bfloat16>());
        break;
#endif
#ifdef HAS_FLOAT16
      case HALF:
        output->p(0, DataTypeUtils::max<float16>());
        break;
#endif
#ifdef HAS_INT16
      case INT16:
        output->p(0, DataTypeUtils::max<int16_t>());
        break;
#endif
#ifdef HAS_UINT16
      case UINT16:
        output->p(0, DataTypeUtils::max<uint16_t>());
        break;
#endif
#ifdef HAS_INT32
      case INT32:
        output->p(0, DataTypeUtils::max<int>());
        break;
#endif
#ifdef HAS_UINT32
      case UINT32:
        output->p(0, DataTypeUtils::max<uint32_t>());
        break;
#endif
#ifdef HAS_FLOAT32
      case FLOAT32:
        output->p(0, DataTypeUtils::max<float>());
        break;
#endif
#ifdef HAS_UNSIGNEDLONG
      case UINT64:
        output->p(0, DataTypeUtils::max<uint64_t>());
        break;
#endif
#ifdef HAS_LONG
      case INT64:
        output->p(0, DataTypeUtils::max<LongType>());
        break;
#endif
#ifdef HAS_DOUBLE
      case DOUBLE:
        output->p(0, DataTypeUtils::max<double>());
        break;
#endif
      default: {
        sd_printf("Unknown DataType used: [%i]\n", DataTypeUtils::asInt(type));
#ifndef __CUDA_ARCH__
        std::string errorMessage;
        errorMessage += "Unknown data type requested min max:";
        errorMessage += DataTypeUtils::asString(type);
        THROW_EXCEPTION(errorMessage.c_str());
#endif
      }
    }

  }

  return Status::OK;

}

DECLARE_SHAPE_FN(min_max_datatype) {
  DataType newType = DataTypeUtils::fromInt(INT_ARG(0));
  return SHAPELIST(ConstantShapeHelper::getInstance().scalarShapeInfo(newType));
}

DECLARE_TYPES(min_max_datatype) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes(ANY);
}
}  // namespace ops
}  // namespace sd

#endif