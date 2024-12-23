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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author GS <sgazeos@gmail.com>, created on 21.01.2019
//
#include <array/NDArray.h>
#include <loops/special_kernels.h>
#include <execution/cuda/LaunchDims.h>

    namespace sd {

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // set up a given value above (or on) a specified diagonal in a 2D matrix
  template <typename T>
  static SD_KERNEL void setDiagValueUpperKernel(
      void* buffer,
      LongType* shape,
      T value,
      int diagonal,
      LongType rows,
      LongType cols) {

    __shared__ LongType rank;
    __shared__ const sd::LongType* stridePtr;
    __shared__ T* arr;

    if (threadIdx.x == 0) {
      rank      = shape::rank(shape);
      stridePtr = shape::stride(shape);
      arr       = reinterpret_cast<T*>(buffer);
    }
    __syncthreads();

    for (LongType r = blockIdx.x; r < rows; r += gridDim.x) {
      for (LongType c = threadIdx.x; c < cols; c += blockDim.x) {
        sd::LongType coords[2] = {r, c};
        sd::LongType offset;

        COORDS2INDEX(rank, stridePtr, coords, offset);

        // If c >= r + diagonal
        //   means c - r >= diagonal
        //   i.e. we are on/above the diagonal
        if (r + diagonal <= c) {
          arr[offset] = value;
        }
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // set up a given value below (or on) a specified diagonal in a 2D matrix
  template <typename T>
  static SD_KERNEL void setDiagValueLowerKernel(
      void* buffer,
      LongType* shape,
      T value,
      int diagonal,
      LongType rows,
      LongType cols) {

    __shared__ LongType rank;
    __shared__ const sd::LongType* stridePtr;
    __shared__ T* arr;

    if (threadIdx.x == 0) {
      rank      = shape::rank(shape);
      stridePtr = shape::stride(shape);
      arr       = reinterpret_cast<T*>(buffer);
    }
    __syncthreads();

    for (LongType r = blockIdx.x; r < rows; r += gridDim.x) {
      for (LongType c = threadIdx.x; c < cols; c += blockDim.x) {
        sd::LongType coords[2] = {r, c};
        sd::LongType offset;

        COORDS2INDEX(rank, stridePtr, coords, offset);

        // If c <= r + diagonal
        //   means c - r <= diagonal
        //   i.e. we are on/below the diagonal
        if (r + diagonal >= c) {
          arr[offset] = value;
        }
      }
    }
  }

  // Below are template instantiations for various data types
  template SD_KERNEL void setDiagValueUpperKernel<double>(
      void* buffer, LongType* shape, double value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueLowerKernel<double>(
      void* buffer, LongType* shape, double value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueUpperKernel<float>(
      void* buffer, LongType* shape, float value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueLowerKernel<float>(
      void* buffer, LongType* shape, float value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueUpperKernel<int>(
      void* buffer, LongType* shape, int value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueLowerKernel<int>(
      void* buffer, LongType* shape, int value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueUpperKernel<float16>(
      void* buffer, LongType* shape, float16 value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueLowerKernel<float16>(
      void* buffer, LongType* shape, float16 value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueUpperKernel<bfloat16>(
      void* buffer, LongType* shape, bfloat16 value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueLowerKernel<bfloat16>(
      void* buffer, LongType* shape, bfloat16 value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueUpperKernel<LongType>(
      void* buffer, LongType* shape, LongType value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueLowerKernel<LongType>(
      void* buffer, LongType* shape, LongType value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueUpperKernel<int16_t>(
      void* buffer, LongType* shape, int16_t value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueLowerKernel<int16_t>(
      void* buffer, LongType* shape, int16_t value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueUpperKernel<uint8_t>(
      void* buffer, LongType* shape, uint8_t value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueLowerKernel<uint8_t>(
      void* buffer, LongType* shape, uint8_t value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueUpperKernel<int8_t>(
      void* buffer, LongType* shape, int8_t value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueLowerKernel<int8_t>(
      void* buffer, LongType* shape, int8_t value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueUpperKernel<bool>(
      void* buffer, LongType* shape, bool value,
      int diagonal, LongType rows, LongType cols);

  template SD_KERNEL void setDiagValueLowerKernel<bool>(
      void* buffer, LongType* shape, bool value,
      int diagonal, LongType rows, LongType cols);

}  // namespace sd
