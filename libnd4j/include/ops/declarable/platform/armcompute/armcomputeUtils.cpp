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

// Created by Abdelrauf 2020

#include "armcomputeUtils.h"

#include <helpers/LoopsCoordsHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include <cstdint>

namespace sd {
namespace ops {
namespace platforms {

Arm_DataType getArmType(const DataType& dType) {
  Arm_DataType ret;
  switch (dType) {
    case HALF:
      ret = Arm_DataType::F16;
      break;
    case FLOAT32:
      ret = Arm_DataType::F32;
      break;
    case DOUBLE:
      ret = Arm_DataType::F64;
      break;
    case INT8:
      ret = Arm_DataType::S8;
      break;
    case INT16:
      ret = Arm_DataType::S16;
      break;
    case INT32:
      ret = Arm_DataType::S32;
      break;
    case INT64:
      ret = Arm_DataType::S64;
      break;
    case UINT8:
      ret = Arm_DataType::U8;
      break;
    case UINT16:
      ret = Arm_DataType::U16;
      break;
    case UINT32:
      ret = Arm_DataType::U32;
      break;
    case UINT64:
      ret = Arm_DataType::U64;
      break;
    case BFLOAT16:
      ret = Arm_DataType::BFLOAT16;
      break;
    default:
      ret = Arm_DataType::UNKNOWN;
  };

  return ret;
}

bool isArmcomputeFriendly(NDArray& arr) {
  auto dType = getArmType(arr.dataType());
  int rank = (int)(arr.rankOf());
  int ind = arr.ordering() == 'c' ? rank - 1 : 0;
  auto arrStrides = arr.stridesOf();
  return dType != Arm_DataType::UNKNOWN && rank <= arm_compute::MAX_DIMS && arr.ordering() == 'c' &&
         arrStrides[ind] == 1;
}

Arm_TensorInfo getArmTensorInfo(int rank, sd::LongType* bases, sd::DataType ndArrayType,
                                arm_compute::DataLayout layout) {
  constexpr int numChannels = 1;
  auto dType = getArmType(ndArrayType);

  Arm_TensorShape shape;
  shape.set_num_dimensions(rank);
  for (int i = 0, j = rank - 1; i < rank; i++, j--) {
    shape[i] = static_cast<uint32_t>(bases[j]);
  }
  // fill the rest unused with 1
  for (int i = rank; i < arm_compute::MAX_DIMS; i++) {
    shape[i] = 1;
  }

  return Arm_TensorInfo(shape, numChannels, dType, layout);
}

Arm_TensorInfo getArmTensorInfo(NDArray& arr, arm_compute::DataLayout layout) {
  auto dType = getArmType(arr.dataType());

  internal_print_nd_shape(arr, "shape");
  internal_print_nd_array(arr, "data");
  //
  constexpr int numChannels = 1;
  int rank = (int)(arr.rankOf());
  auto bases = arr.shapeOf();
  auto arrStrides = arr.stridesOf();

  // https://arm-software.github.io/ComputeLibrary/v20.05/_dimensions_8h_source.xhtml
  // note: underhood it is stored as std::array<T, num_SD_MAX_DIMENSIONs> _id;
  // TensorShape is derived from Dimensions<uint32_t>
  // as well as Strides : public Dimensions<uint32_t>
  Arm_TensorShape shape;
  Arm_Strides strides;
  shape.set_num_dimensions(rank);
  strides.set_num_dimensions(rank);
  size_t element_size = arr.sizeOfT();
  for (int i = 0, j = rank - 1; i < rank; i++, j--) {
    shape[i] = static_cast<uint32_t>(bases[j]);
    strides[i] = static_cast<uint32_t>(arrStrides[j] * element_size);
  }
  // fill the rest unused with 1
  for (int i = rank; i < arm_compute::MAX_DIMS; i++) {
    shape[i] = 1;
  }

  size_t total_size = arr.lengthOf() * element_size;
  size_t offset = 0;
  if (arr.hasPaddedBuffer()) {
    internal_printf("---has padded buffer %d\n", 0);
    total_size = arr.getDataBuffer()->getLenInBytes();
    offset = arr.offset() * element_size;
  }
  internal_printf(":: offset %d el size %d  arr.getDataBuffer()->getLenInBytes() %d lengthof %d \n",
                  (int)arr.offset(), (int)element_size, (int)arr.getDataBuffer()->getLenInBytes(),
                  (int)arr.lengthOf());
  Arm_TensorInfo info;
  info.init(shape, numChannels, dType, strides, offset, total_size);
  info.set_data_layout(layout);

  return info;
}

Arm_Tensor getArmTensor(NDArray& arr, arm_compute::DataLayout layout) {
  // - Ownership of the backing memory is not transferred to the tensor itself.
  // - The tensor mustn't be memory managed.
  // - Padding requirements should be accounted by the client code.
  // In other words, if padding is required by the tensor after the function
  // configuration step, then the imported backing memory should account for it.
  // Padding can be checked through the TensorInfo::padding() interface.

  // Import existing pointer as backing memory
  auto info = getArmTensorInfo(arr, layout);
  Arm_Tensor tensor;
  tensor.allocator()->init(info);
  // get without offset
  void* buff = arr.getDataBuffer()->primary();
  tensor.allocator()->import_memory(buff);
  return tensor;
}

void copyFromTensor(const Arm_Tensor& inTensor, sd::NDArray& output) {
  // only for C order
  if (output.ordering() != 'c') return;
  sd::LongType* shapeInfo = output.shapeInfo();
  sd::LongType* bases = &(shapeInfo[1]);
  sd::LongType rank = shapeInfo[0];
  sd::LongType* strides = output.stridesOf();
  int width = bases[rank - 1];
  uint8_t* outputBuffer = (uint8_t*)output.buffer();
  size_t offset = 0;
  arm_compute::Window window;
  arm_compute::Iterator tensor_it(&inTensor, window);

  int element_size = inTensor.info()->element_size();
  window.use_tensor_dimensions(inTensor.info()->tensor_shape(), /* first_dimension =*/arm_compute::Window::DimY);


    sd::LongType coords[SD_MAX_RANK] = {};
    auto copySize = width * element_size;
    arm_compute::execute_window_loop(
        window,
        [&](const arm_compute::Coordinates& id) {
          auto src = tensor_it.ptr();
          auto dest = outputBuffer + offset * element_size;
          memcpy(dest, src, copySize);
          offset = sd::inc_coords(bases, strides, coords, offset, rank, 1);
        },
        tensor_it);

}

void copyToTensor(sd::NDArray& input, Arm_Tensor& outTensor) {
  // only for C order
  if (input.ordering() != 'c') return;
  sd::LongType* shapeInfo = input.shapeInfo();
  sd::LongType* bases = &(shapeInfo[1]);
  sd::LongType rank = shapeInfo[0];
  sd::LongType* strides = input.stridesOf();
  uint8_t* inputBuffer = (uint8_t*)input.buffer();
  int width = bases[rank - 1];
  size_t offset = 0;
  arm_compute::Window window;
  arm_compute::Iterator tensor_it(&outTensor, window);
  int element_size = outTensor.info()->element_size();

  window.use_tensor_dimensions(outTensor.info()->tensor_shape(), /* first_dimension =*/arm_compute::Window::DimY);

    sd::LongType coords[SD_MAX_RANK] = {};
    auto copySize = width * element_size;
    arm_compute::execute_window_loop(
        window,
        [&](const arm_compute::Coordinates& id) {
          auto dest = tensor_it.ptr();
          auto src = inputBuffer + offset * element_size;
          memcpy(dest, src, copySize);
          offset = sd::inc_coords(bases, strides, coords, offset, rank, 1);
        },
        tensor_it);

}

// armcompute should be built with debug option
void print_tensor(Arm_ITensor& tensor, const char* msg) {
  auto info = tensor.info();
  auto padding = info->padding();
  std::cout << msg << "\ntotal: " << info->total_size() << "\n";

  for (int i = 0; i < arm_compute::MAX_DIMS; i++) {
    std::cout << info->dimension(i) << ",";
  }
  std::cout << std::endl;
  for (int i = 0; i < arm_compute::MAX_DIMS; i++) {
    std::cout << info->strides_in_bytes()[i] << ",";
  }
  std::cout << "\npadding: l " << padding.left << ", r " << padding.right << ", t " << padding.top << ", b "
            << padding.bottom << std::endl;

#ifdef ARM_COMPUTE_ASSERTS_ENABLED
  // note it did not print correctly fro NHWC
  std::cout << msg << ":\n";
  tensor.print(std::cout);
  std::cout << std::endl;
#endif
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
