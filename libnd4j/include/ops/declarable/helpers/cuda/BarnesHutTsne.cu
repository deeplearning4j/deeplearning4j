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
// @author George A. Shulinok <sgazeos@gmail.com>, created on 4/18/2019
//
#include <array/NDArrayFactory.h>
#include <execution/cuda/LaunchDims.h>
#include <helpers/DebugHelper.h>
#include <helpers/shape.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/BarnesHutTsne.h>


namespace sd {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// count rows kernel - count input pRows and pCols and put result onto pRowCounts
// pRowCounts - array of ints, with length N
// pRows - array of ints with length N, vals from 0 to N-1
// pCols - array of ints with length < N and vals between 0 and max(pRows)
//
static SD_KERNEL void countRowsKernel(int* pRowCounts, int const* pRows, int const* pCols, LongType N) {
  auto start = blockIdx.x * blockDim.x;
  auto step = blockDim.x * gridDim.x;
  for (int n = threadIdx.x + start; n < N; n += step) {
    int begin = pRows[n];    //->e<int>(n);
    int end = pRows[n + 1];  // rowP->e<int>(n + 1);
    for (int i = begin; i < end; i++) {
      bool present = false;
      // loop between near pRows
      for (int m = pRows[pCols[i]]; m < pRows[pCols[i] + 1]; m++)
        if (pCols[m] == n) {  // mark index as existed with columns array
          present = true;
          break;
        }

      atomicAdd(&pRowCounts[n], 1);

      if (!present)  // increment row counter for given index
        atomicAdd(&pRowCounts[pCols[i]], 1);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// row counter caller
LongType barnes_row_count(NDArray* rowP, NDArray* colP, LongType N, NDArray& rowCounts) {
  int* pRowCounts = reinterpret_cast<int*>(rowCounts.specialBuffer());
  int const* pRows = reinterpret_cast<int const*>(rowP->specialBuffer());
  int const* pCols = reinterpret_cast<int const*>(colP->specialBuffer());
  auto stream = rowCounts.getContext()->getCudaStream();
  countRowsKernel<<<1, 1, 128, *stream>>>(pRowCounts, pRows, pCols, N);
  sd::DebugHelper::checkErrorCode(stream, "countRows  failed");

  NDArray numElementsArr = rowCounts.sumNumber();
  auto numElements = numElementsArr.e<LongType>(0);
  return numElements;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// extend symRowP with pRowCounts array vals
//  pRowCounts - int array with length N
//  symRowP - int array with length N+1
//  N - given array length
//
static SD_KERNEL void fillUpsymRow(int const* pRowCounts, int* symRowP, int N) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int n = start; n < N + 1; n += step) {  // to avoid race condition use shift only for given index
    symRowP[n] = 0;
    for (int i = 0; i < n; i++) atomicAdd(&symRowP[n], pRowCounts[i]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  symmetrize routine kernel
// pRows - rows buffer (ints)
// pCols - column buffer (ints) with vals between 0 and max(pRows)
// pVals - values vector (floats)
// symRowP - ints, shifted pRows
// symColP - ints, shifted pCols,
// offset - ints, shitfs
// pOutput - result matrix (floats)
// N - pRows length
//
template <typename T>
static SD_KERNEL void symmetrizeKernel(int const* pRows, int const* pCols, T const* pVals, int* symRowP, int* symColP,
                                       int* offset, T* pOutput, int N) {
  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int n = start; n < N; n += step) {
    int begin = pRows[n];
    int bound = pRows[n + 1];

    for (int i = begin; i < bound; i++) {
      bool present = false;
      int colPI = pCols[i];
      int start = pRows[colPI];
      int end = pRows[colPI + 1];

      for (int m = start; m < end; m++) {
        if (pCols[m] == n) {
          present = true;
          if (n <= colPI) {
            symColP[symRowP[n] + offset[n]] = colPI;
            symColP[symRowP[colPI] + offset[colPI]] = n;
            pOutput[symRowP[n] + offset[n]] = pVals[i] + pVals[m];
            pOutput[symRowP[colPI] + offset[colPI]] = pVals[i] + pVals[m];
          }
        }
      }

      // If (colP[i], n) is not present, there is no addition involved
      if (!present) {
        symColP[symRowP[n] + offset[n]] = colPI;
        symColP[symRowP[pCols[i]] + offset[colPI]] = n;
        pOutput[symRowP[n] + offset[n]] = pVals[i];
        pOutput[symRowP[colPI] + offset[colPI]] = pVals[i];
      }
      // Update offsets
      if (!present || (present && n <= colPI)) {
        atomicAdd(&offset[n], 1);

        if (colPI != n) atomicAdd(&offset[colPI], 1);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// symmetrize algorithm itself
//
template <typename T>
static void barnes_symmetrize_(NDArray* rowP, NDArray* colP, NDArray* valP, LongType N,
                               NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts) {
  int const* pRows = reinterpret_cast<int const*>(rowP->specialBuffer());
  int* symRowP = reinterpret_cast<int*>(outputRows->specialBuffer());
  int* pRowCounts = reinterpret_cast<int*>(rowCounts->specialBuffer());
  auto stream = outputCols->getContext()->getCudaStream();
  // fill up syRowP array
  fillUpsymRow<<<1, N, 128, *stream>>>(pRowCounts, symRowP, N);
  sd::DebugHelper::checkErrorCode(stream, "fillUpsymRow  failed");

  outputRows->syncToHost();
  int* symColP = reinterpret_cast<int*>(outputCols->specialBuffer());
  int const* pCols = reinterpret_cast<int const*>(colP->specialBuffer());
  T const* pVals = reinterpret_cast<T const*>(valP->specialBuffer());
  T* pOutput = reinterpret_cast<T*>(outputVals->specialBuffer());
  auto offsetArr = NDArrayFactory::create<int>('c', {N});
  int* offset = reinterpret_cast<int*>(offsetArr->specialBuffer());
  // symmetrize itself
  symmetrizeKernel<T><<<1, 1, 1024, *stream>>>(pRows, pCols, pVals, symRowP, symColP, offset, pOutput, N);
  sd::DebugHelper::checkErrorCode(stream, "symmetrizeKernel  failed");

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// symmetrize caller and adoption
//
void barnes_symmetrize(NDArray* rowP, NDArray* colP, NDArray* valP, LongType N,
                       NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts) {
  BUILD_SINGLE_SELECTOR(valP->dataType(), barnes_symmetrize_,
                        (rowP, colP, valP, N, outputRows, outputCols, outputVals, rowCounts), SD_NUMERIC_TYPES);

  *outputVals /= 2.0;
}
BUILD_SINGLE_TEMPLATE( void barnes_symmetrize_,
                      (NDArray* rowP, NDArray* colP, NDArray* valP, sd::LongType N,
                       NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts),
                      SD_NUMERIC_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// edge forces implementation
//
template <typename T>
static SD_KERNEL void edgeForcesKernel(int const* pRows, int const* pCols, T const* dataP, T const* vals, T* outputP,
                                       int N, int colCount, int rowSize) {
  //        std::vector<T> buffer(colCount);

  auto start = blockIdx.x * blockDim.x + threadIdx.x;
  auto step = blockDim.x * gridDim.x;

  for (int n = start; n < N; n += step) {
    int start = pRows[n];
    int end = pRows[n + 1];
    int shift = n * colCount;
    for (int i = start; i < end; i++) {
      T const* thisSlice = dataP + pCols[i] * colCount;
      T res = static_cast<T>(1);

      for (int k = 0; k < colCount; k++) {
        auto valTemp = dataP[shift + k] - thisSlice[k];  // thisSlice[k];
        res += valTemp * valTemp;  // (dataP[shift + k] * dataP[shift + k] - 2 * dataP[shift + k] * thisSlice[k] +
                                   // thisSlice[k] * thisSlice[k])
      }
      res = vals[i] / res;
      for (int k = 0; k < colCount; k++)
        math::atomics::sd_atomicAdd(&outputP[shift + k], T((dataP[shift + k] - thisSlice[k]) * res));
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// edge forces algorithm
//

template <typename T>
static void barnes_edge_forces_(NDArray* rowP, NDArray * colP, NDArray * valP, int N,
                                NDArray * data, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {data, rowP, colP, valP, valP});
  T const* dataP = reinterpret_cast<T const*>(data->specialBuffer());
  T const* vals = reinterpret_cast<T const*>(valP->specialBuffer());
  T* outputP = reinterpret_cast<T*>(output->specialBuffer());
  int const* pRows = reinterpret_cast<int const*>(rowP->specialBuffer());
  int const* pCols = reinterpret_cast<int const*>(colP->specialBuffer());
  int colCount = data->columns();
  // auto shift = 0;
  auto rowSize = sizeof(T) * colCount;
  auto stream = output->getContext()->getCudaStream();
  edgeForcesKernel<T><<<1, 128, 1024, *stream>>>(pRows, pCols, dataP, vals, outputP, N, colCount, rowSize);
  sd::DebugHelper::checkErrorCode(stream, "edgeForces  failed");

  NDArray::registerSpecialUse({output}, {rowP, colP, valP, data});
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// edge forces caller
//
void barnes_edge_forces(NDArray* rowP, NDArray * colP, NDArray * valP, int N, NDArray* output,
                        NDArray& data) {
  // Loop over all edges in the graph
  BUILD_SINGLE_SELECTOR(output->dataType(), barnes_edge_forces_, (rowP, colP, valP, N, &data, output), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE( void barnes_edge_forces_,
                      (NDArray* rowP, NDArray * colP, NDArray * valP, int N, NDArray * data,
                       NDArray* output),
                      SD_FLOAT_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// barnes_gains kernel: per-element triplewise update
//   inputs:  x (gains), grad (gradient), eps (previous step / epsilon)
//   formula: res = (sign(grad) != sign(eps)) ? x + 0.2 : x * 0.8
//            res = max(res, 0.01)
//
template <typename T>
static SD_KERNEL void barnesGainsCuda(const void* vx,  const LongType* xShapeInfo,
                                      const void* vg,  const LongType* gShapeInfo,
                                      const void* ve,  const LongType* eShapeInfo,
                                      void* vz,        const LongType* zShapeInfo) {
  const T* x = reinterpret_cast<const T*>(vx);
  const T* g = reinterpret_cast<const T*>(vg);
  const T* e = reinterpret_cast<const T*>(ve);
  T*       z = reinterpret_cast<T*>(vz);

  __shared__ LongType len, totalThreads;
  __shared__ int rank;
  __shared__ const LongType *xShape, *gShape, *eShape, *zShape;
  __shared__ const LongType *xStride, *gStride, *eStride, *zStride;

  if (threadIdx.x == 0) {
    len          = shape::length(zShapeInfo);
    totalThreads = gridDim.x * blockDim.x;
    rank         = shape::rank(zShapeInfo);
    xShape  = shape::shapeOf(xShapeInfo);
    gShape  = shape::shapeOf(gShapeInfo);
    eShape  = shape::shapeOf(eShapeInfo);
    zShape  = shape::shapeOf(zShapeInfo);
    xStride = shape::stride(xShapeInfo);
    gStride = shape::stride(gShapeInfo);
    eStride = shape::stride(eShapeInfo);
    zStride = shape::stride(zShapeInfo);
  }
  __syncthreads();

  const LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

  LongType xCoords[SD_MAX_RANK], gCoords[SD_MAX_RANK], eCoords[SD_MAX_RANK], zCoords[SD_MAX_RANK];

  for (LongType i = tid; i < len; i += totalThreads) {
    INDEX2COORDS(i, rank, zShape, zCoords);
    INDEX2COORDS(i, rank, xShape, xCoords);
    INDEX2COORDS(i, rank, gShape, gCoords);
    INDEX2COORDS(i, rank, eShape, eCoords);

    LongType xIdx, gIdx, eIdx, zIdx;
    COORDS2INDEX(rank, xStride, xCoords, xIdx);
    COORDS2INDEX(rank, gStride, gCoords, gIdx);
    COORDS2INDEX(rank, eStride, eCoords, eIdx);
    COORDS2INDEX(rank, zStride, zCoords, zIdx);

    T xVal   = x[xIdx];
    T gVal   = g[gIdx];
    T eVal   = e[eIdx];

    T res = (math::sd_sign<T, T>(gVal) != math::sd_sign<T, T>(eVal))
                ? xVal + static_cast<T>(0.2)
                : xVal * static_cast<T>(0.8);

    if (res < static_cast<T>(0.01)) res = static_cast<T>(0.01);

    z[zIdx] = res;
  }
}

template <typename T>
static SD_HOST void barnesGainsCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const int sharedMem,
                                            const cudaStream_t* stream,
                                            const void* vx, const LongType* xShapeInfo,
                                            const void* vg, const LongType* gShapeInfo,
                                            const void* ve, const LongType* eShapeInfo,
                                            void* vz,       const LongType* zShapeInfo) {
  barnesGainsCuda<T><<<blocksPerGrid, threadsPerBlock, sharedMem, *stream>>>(
      vx, xShapeInfo, vg, gShapeInfo, ve, eShapeInfo, vz, zShapeInfo);
  DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "barnesGainsCuda failed");
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gains - run a function T((x + 2.) * sd::math::sd_sign<T,T>(grad) != sd::math::sd_sign<T,T>(eps)) + T(x * 0.8 *
// sd::math::sd_sign<T,T>(grad) != sd::math::sd_sign<T,T>(eps)); for all members in input and put all in output
//
template <typename T>
void barnes_gains_(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input, gradX, epsilon});

  dim3 launchDims = getLaunchDims("barnesGains");
  auto stream = output->getContext()->getCudaStream();

  BUILD_SINGLE_SELECTOR(input->dataType(), barnesGainsCudaLauncher,
                        (launchDims.y, launchDims.x, launchDims.z, stream,
                         input->specialBuffer(),   input->specialShapeInfo(),
                         gradX->specialBuffer(),   gradX->specialShapeInfo(),
                         epsilon->specialBuffer(), epsilon->specialShapeInfo(),
                         output->specialBuffer(),  output->specialShapeInfo()),
                        SD_NUMERIC_TYPES);

  NDArray::registerSpecialUse({output}, {input, gradX, epsilon});
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gains caller
void barnes_gains(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), barnes_gains_, (input, gradX, epsilon, output), SD_NUMERIC_TYPES);
}
BUILD_SINGLE_TEMPLATE( void barnes_gains_, (NDArray * input, NDArray* gradX, NDArray* epsilon, NDArray* output),
                      SD_NUMERIC_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cell contains - check cells for given point
//
bool cell_contains(NDArray* corner, NDArray* width, NDArray* point, LongType dimension) {
  auto cornerMinusWidth = *corner - *width;
  auto cornerPlusWidth = *corner + *width;
  // executes on host side, so sync all to host memory
  cornerMinusWidth->syncToHost();
  cornerPlusWidth->syncToHost();
  for (LongType i = 0; i < dimension; i++) {
    if (cornerMinusWidth->e<double>(i) > point->e<double>(i)) {
      delete cornerMinusWidth;
      delete cornerPlusWidth;
      return false;
    }
    if (cornerPlusWidth->e<double>(i) < point->e<double>(i)) {
      delete cornerMinusWidth;
      delete cornerPlusWidth;
      return false;
    }
  }

  delete cornerMinusWidth;
  delete cornerPlusWidth;
  return true;
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
