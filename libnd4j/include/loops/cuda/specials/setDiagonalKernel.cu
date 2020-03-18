/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <loops/special_kernels.h>
#include <array/NDArray.h>
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
    static __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, T value, int diagonal, Nd4jLong rows,
            Nd4jLong cols) {

        __shared__ Nd4jLong  rank;
        __shared__ T* array;

        if (0 == threadIdx.x) {
            rank = shape::rank(shape);
            array = reinterpret_cast<T *>(buffer);
        }
        __syncthreads();

        for (Nd4jLong i = blockIdx.x; i < rows; i += gridDim.x) {
            for (int j = threadIdx.x; j < cols; j += blockDim.x) {
                Nd4jLong coords[2] = {i, j};
                Nd4jLong xOffset = shape::getOffset(shape, coords);
                if (i + diagonal <= j)
                    array[xOffset] = value;
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
    static __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, T value, int diagonal, Nd4jLong rows, Nd4jLong cols) {
        Nd4jLong  rank = shape::rank(shape);
        int totalThreads = blockDim.x;
        for (Nd4jLong i = blockIdx.x; i < rows; i += gridDim.x) {
            for (int j = threadIdx.x; j < cols; j += totalThreads) {
                Nd4jLong coords[2] = {i, j};
                auto xOffset = shape::getOffset(shape, coords);
                if (i + diagonal >= j)
                    *(reinterpret_cast<T*>(buffer) + xOffset) = value;
            }
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, double value,   int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, double value,   int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, float value,    int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, float value,    int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, int value,      int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, int value,      int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, float16 value,  int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, float16 value,  int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, bfloat16 value, int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, bfloat16 value, int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, Nd4jLong value, int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, Nd4jLong value, int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, int16_t value,  int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, int16_t value,  int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, uint8_t value,  int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, uint8_t value,  int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, int8_t value,   int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, int8_t value,   int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueLowerKernel(void* buffer, Nd4jLong* shape, bool value,     int diagonal, Nd4jLong rows, Nd4jLong cols);
    template __global__ void setDiagValueUpperKernel(void* buffer, Nd4jLong* shape, bool value,     int diagonal, Nd4jLong rows, Nd4jLong cols);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    static void setDiagonalValueUpper(void* buffer, Nd4jLong* shape, NDArray const& value, int diagonal, Nd4jLong rows, Nd4jLong cols, cudaStream_t& stream) {
        dim3 launchDims(256, 512, 8192);
        setDiagValueUpperKernel<T><<<launchDims.x, launchDims.y, launchDims.z, stream>>>(buffer, shape, value.e<T>(0), diagonal, rows, cols);
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <typename T>
    static void setDiagonalValueLower(void* buffer, Nd4jLong* shape, NDArray const& value, int diagonal, Nd4jLong rows, Nd4jLong cols, cudaStream_t& stream) {
        dim3 launchDims(256, 512, 8192);
        setDiagValueLowerKernel<T><<<launchDims.x, launchDims.y, launchDims.z, stream>>>(buffer, shape, value.e<T>(0), diagonal, rows, cols);
    }

    BUILD_SINGLE_TEMPLATE(template void setDiagonalValueUpper, (void* buffer, Nd4jLong* shape, NDArray const& value,
            int diagonal, Nd4jLong rows, Nd4jLong cols, cudaStream_t& stream), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void setDiagonalValueLower, (void* buffer, Nd4jLong* shape, NDArray const& value,
            int diagonal, Nd4jLong rows, Nd4jLong cols, cudaStream_t& stream), LIBND4J_TYPES);
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}