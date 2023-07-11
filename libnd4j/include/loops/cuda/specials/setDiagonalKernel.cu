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
// @author GS <sgazeos@gmail.com>, created on 21.01.2019
//
#include <array/NDArray.h>
#include <loops/special_kernels.h>

#include <execution/cuda/LaunchDims.h>
namespace sd {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// set up given value to upper diagonal given
// buffer - input buffer
// shape - input shape
// value - given value
// diagonal - given upper diagonal (acceptable negative values also, 0 - the main diagonal)
// row, cols - height and width of given matrix (MxN, rows = M, cols = N)
//
template <typename T>
static SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, T value, int diagonal,
                                              sd::LongType rows, sd::LongType cols) {
  __shared__ sd::LongType rank;
  __shared__ T* array;

  if (0 == threadIdx.x) {
    rank = shape::rank(shape);
    array = reinterpret_cast<T*>(buffer);
  }
  __syncthreads();

  for (sd::LongType i = blockIdx.x; i < rows; i += gridDim.x) {
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
      sd::LongType coords[2] = {i, j};
      sd::LongType xOffset = shape::getOffset(shape, coords);
      if (i + diagonal <= j) array[xOffset] = value;
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// set up given value to lower given diagonal
// buffer - input buffer
// shape - input shape
// value - given value
// diagonal - given lower diagonal (acceptable negative values also, 0 - the main diagonal)
// row, cols - height and width of given matrix (MxN, rows = M, cols = N)
//

template <typename T>
static SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, T value, int diagonal,
                                              sd::LongType rows, sd::LongType cols) {
  sd::LongType rank = shape::rank(shape);
  int totalThreads = blockDim.x;
  for (sd::LongType i = blockIdx.x; i < rows; i += gridDim.x) {
    for (int j = threadIdx.x; j < cols; j += totalThreads) {
      sd::LongType coords[2] = {i, j};
      auto xOffset = shape::getOffset(shape, coords);
      if (i + diagonal >= j) *(reinterpret_cast<T*>(buffer) + xOffset) = value;
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, double value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, double value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, float value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, float value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, int value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, int value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, float16 value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, float16 value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, bfloat16 value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, bfloat16 value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, sd::LongType value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, sd::LongType value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, int16_t value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, int16_t value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, uint8_t value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, uint8_t value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, int8_t value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, int8_t value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, sd::LongType* shape, bool value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, sd::LongType* shape, bool value, int diagonal,
                                                sd::LongType rows, sd::LongType cols);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
static void setDiagonalValueUpper(void* buffer, sd::LongType* shape, NDArray const& value, int diagonal,
                                  sd::LongType rows, sd::LongType cols, cudaStream_t& stream) {
  dim3 launchDims = getLaunchDims("diag");
  setDiagValueUpperKernel<T>
      <<<launchDims.x, launchDims.y, launchDims.z, stream>>>(buffer, shape, value.e<T>(0), diagonal, rows, cols);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
static void setDiagonalValueLower(void* buffer, sd::LongType* shape, NDArray const& value, int diagonal,
                                  sd::LongType rows, sd::LongType cols, cudaStream_t& stream) {
  dim3 launchDims = getLaunchDims("diag");
  setDiagValueLowerKernel<T>
      <<<launchDims.x, launchDims.y, launchDims.z, stream>>>(buffer, shape, value.e<T>(0), diagonal, rows, cols);
}

BUILD_SINGLE_TEMPLATE(template void setDiagonalValueUpper,
                      (void* buffer, sd::LongType* shape, NDArray const& value, int diagonal, sd::LongType rows,
                       sd::LongType cols, cudaStream_t& stream),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(template void setDiagonalValueLower,
                      (void* buffer, sd::LongType* shape, NDArray const& value, int diagonal, sd::LongType rows,
                       sd::LongType cols, cudaStream_t& stream),
                      SD_COMMON_TYPES);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace sd
