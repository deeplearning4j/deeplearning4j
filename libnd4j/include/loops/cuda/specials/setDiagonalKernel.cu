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
static SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, T value, int diagonal, LongType rows,
                                              LongType cols) {
  __shared__ LongType rank;
  __shared__ T* array;

  if (0 == threadIdx.x) {
    rank = shape::rank(shape);
    array = reinterpret_cast<T*>(buffer);
  }
  __syncthreads();

  for (LongType i = blockIdx.x; i < rows; i += gridDim.x) {
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
      LongType coords[2] = {i, j};
      LongType xOffset;
      COORDS2INDEX(rank, shape::stride(shape), coords, xOffset);
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
static SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, T value, int diagonal, LongType rows,
                                              LongType cols) {
  LongType rank = shape::rank(shape);
  int totalThreads = blockDim.x;
  for (LongType i = blockIdx.x; i < rows; i += gridDim.x) {
    for (int j = threadIdx.x; j < cols; j += totalThreads) {
      LongType coords[2] = {i, j};
      LongType xOffset;
      COORDS2INDEX(rank, shape::stride(shape), coords, xOffset);
      if (i + diagonal >= j) *(reinterpret_cast<T*>(buffer) + xOffset) = value;
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, double value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, double value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, float value, int diagonal, LongType rows,
                                                LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, float value, int diagonal, LongType rows,
                                                LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, int value, int diagonal, LongType rows,
                                                LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, int value, int diagonal, LongType rows,
                                                LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, float16 value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, float16 value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, bfloat16 value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, bfloat16 value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, LongType value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, LongType value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, int16_t value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, int16_t value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, uint8_t value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, uint8_t value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, int8_t value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, int8_t value, int diagonal,
                                                LongType rows, LongType cols);
template SD_KERNEL void setDiagValueLowerKernel(void* buffer, LongType* shape, bool value, int diagonal, LongType rows,
                                                LongType cols);
template SD_KERNEL void setDiagValueUpperKernel(void* buffer, LongType* shape, bool value, int diagonal, LongType rows,
                                                LongType cols);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace sd
