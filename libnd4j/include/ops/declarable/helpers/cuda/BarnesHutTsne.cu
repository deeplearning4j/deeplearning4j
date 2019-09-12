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
// @author George A. Shulinok <sgazeos@gmail.com>, created on 4/18/2019
//

#include <ops/declarable/helpers/BarnesHutTsne.h>

namespace nd4j {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// count rows kernel - count input pRows and pCols and put result onto pRowCounts
// pRowCounts - array of ints, with length N
// pRows - array of ints with length N, vals from 0 to N-1
// pCols - array of ints with length < N and vals between 0 and max(pRows)
//
    static __global__ void countRowsKernel(int* pRowCounts, int const* pRows, int const* pCols, Nd4jLong N) {
        auto start = blockIdx.x * blockDim.x;
        auto step = blockDim.x * gridDim.x;
        for (int n = threadIdx.x + start; n < N; n += step) {
            int begin = pRows[n];//->e<int>(n);
            int end = pRows[n + 1];//rowP->e<int>(n + 1);
            for (int i = begin; i < end; i++) {
                bool present = false;
                // loop between near pRows
                for (int m = pRows[pCols[i]]; m < pRows[pCols[i] + 1]; m++)
                    if (pCols[m] == n) { // mark index as existed with columns array
                        present = true;
                        break;
                    }

                atomicAdd(&pRowCounts[n], 1);

                if (!present) // increment row counter for given index
                    atomicAdd(&pRowCounts[pCols[i]], 1);
            }
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// row counter caller
    Nd4jLong barnes_row_count(const NDArray* rowP, const NDArray* colP, Nd4jLong N, NDArray& rowCounts) {

        int* pRowCounts = reinterpret_cast<int*>(rowCounts.specialBuffer());
        int const* pRows = reinterpret_cast<int const*>(rowP->getSpecialBuffer());
        int const* pCols = reinterpret_cast<int const*>(colP->getSpecialBuffer());
        auto stream = rowCounts.getContext()->getCudaStream();
        countRowsKernel<<<1, 1, 128, *stream>>>(pRowCounts, pRows, pCols, N);
        NDArray numElementsArr = rowCounts.sumNumber(); //reduceAlongDimension(reduce::Sum, {});
        //rowCounts.printBuffer("Row counts");
        auto numElements = numElementsArr.e<Nd4jLong>(0);
        return numElements;
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// extend symRowP with pRowCounts array vals
//  pRowCounts - int array with length N
//  symRowP - int array with length N+1
//  N - given array length
//
    static __global__ void fillUpsymRow(int const* pRowCounts, int* symRowP, int N) {

        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (int n = start; n < N + 1; n += step) { // to avoid race condition use shift only for given index
            symRowP[n] = 0;
            for (int i = 0; i < n; i++)
                atomicAdd(&symRowP[n], pRowCounts[i]);
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
    static __global__ void symmetrizeKernel(int const* pRows, int const* pCols, T const* pVals, int* symRowP, int* symColP, int* offset, T* pOutput, int N) {
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
                            symColP[symRowP[n] + offset[n]]        = colPI;
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

                    if (colPI != n)
                        atomicAdd(&offset[colPI], 1);
                }
            }
        }
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// symmetrize algorithm itself
//
    template <typename T>
    static void barnes_symmetrize_(const NDArray* rowP, const NDArray* colP, const NDArray* valP, Nd4jLong N, NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts) {
        int const* pRows = reinterpret_cast<int const*>(rowP->getSpecialBuffer());
        int* symRowP = reinterpret_cast<int*>(outputRows->specialBuffer());
        int* pRowCounts = reinterpret_cast<int*>(rowCounts->specialBuffer());
        auto stream = outputCols->getContext()->getCudaStream();
        // fill up syRowP array
        fillUpsymRow<<<1, N, 128, *stream>>>(pRowCounts, symRowP, N);
        outputRows->syncToHost();
//        outputRows->printBuffer("output rows");
        int* symColP = reinterpret_cast<int*>(outputCols->specialBuffer());
//        outputRows->printBuffer("SymRows are");
        int const* pCols = reinterpret_cast<int const*>(colP->getSpecialBuffer());
        T const* pVals = reinterpret_cast<T const*>(valP->getSpecialBuffer());
        T* pOutput = reinterpret_cast<T*>(outputVals->specialBuffer());
        //std::vector<int> rowCountsV = rowCounts->getBufferAsVector<int>();
        auto offsetArr = NDArrayFactory::create<int>('c', {N});
        int* offset = reinterpret_cast<int*>(offsetArr.specialBuffer());
        // symmetrize itself
        symmetrizeKernel<T><<<1, 1, 1024, *stream>>>(pRows, pCols, pVals, symRowP, symColP, offset, pOutput, N);
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// symmetrize caller and adoption
//
    void barnes_symmetrize(const NDArray* rowP, const NDArray* colP, const NDArray* valP, Nd4jLong N, NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts) {
        BUILD_SINGLE_SELECTOR(valP->dataType(), barnes_symmetrize_, (rowP, colP, valP, N, outputRows, outputCols, outputVals, rowCounts), NUMERIC_TYPES);

        *outputVals /= 2.0;
    }
    BUILD_SINGLE_TEMPLATE(template void barnes_symmetrize_, (const NDArray* rowP, const NDArray* colP, const NDArray* valP, Nd4jLong N, NDArray* outputRows, NDArray* outputCols, NDArray* outputVals, NDArray* rowCounts), NUMERIC_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// edge forces implementation
//
    template <typename T>
    static __global__ void edgeForcesKernel(int const* pRows, int const* pCols, T const* dataP, T const* vals, T* outputP, int N, int colCount, int rowSize) {
//        std::vector<T> buffer(colCount);

        auto start = blockIdx.x * blockDim.x + threadIdx.x;
        auto step = blockDim.x * gridDim.x;

        for (int n = start; n < N; n += step) {
            int start = pRows[n];
            int end = pRows[n + 1];
            int shift = n * colCount;
            for (int i = start; i < end; i++) {
                T const* thisSlice = dataP + pCols[i] * colCount;
                T res = 1;

                for (int k = 0; k < colCount; k++) {
                    auto valTemp = dataP[shift + k] - thisSlice[k];//thisSlice[k];
                    res += valTemp * valTemp; // (dataP[shift + k] * dataP[shift + k] - 2 * dataP[shift + k] * thisSlice[k] + thisSlice[k] * thisSlice[k])
                }
                res = vals[i] / res;
                for (int k = 0; k < colCount; k++)
                    math::atomics::nd4j_atomicAdd(&outputP[shift + k], T((dataP[shift + k] - thisSlice[k]) * res));
            }
        }
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// edge forces algorithm
//

    template <typename T>
    static void barnes_edge_forces_(const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray const* data, NDArray* output) {
        NDArray::prepareSpecialUse({output}, {data, rowP, colP, valP, valP});
        T const* dataP = reinterpret_cast<T const*>(data->getSpecialBuffer());
        T const* vals  = reinterpret_cast<T const*>(valP->getSpecialBuffer());
        T* outputP = reinterpret_cast<T*>(output->specialBuffer());
        int const* pRows = reinterpret_cast<int const*>(rowP->getSpecialBuffer());
        int const* pCols = reinterpret_cast<int const*>(colP->getSpecialBuffer());
        int colCount = data->columns();
        //auto shift = 0;
        auto rowSize = sizeof(T) * colCount;
        auto stream = output->getContext()->getCudaStream();
        edgeForcesKernel<T><<<1, 128, 1024, *stream>>>(pRows, pCols, dataP, vals, outputP, N, colCount, rowSize);
        NDArray::registerSpecialUse({output}, {rowP, colP, valP, data});
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// edge forces caller
//
    void barnes_edge_forces(const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray* output, NDArray const& data) {
        // Loop over all edges in the graph
        BUILD_SINGLE_SELECTOR(output->dataType(), barnes_edge_forces_, (rowP, colP, valP, N, &data, output), FLOAT_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void barnes_edge_forces_, (const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray const* data, NDArray* output), FLOAT_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gains - run a function T((x + 2.) * nd4j::math::nd4j_sign<T,T>(grad) != nd4j::math::nd4j_sign<T,T>(eps)) + T(x * 0.8 * nd4j::math::nd4j_sign<T,T>(grad) != nd4j::math::nd4j_sign<T,T>(eps));
// for all members in input and put all in output
//
    template <typename T>
    void barnes_gains_(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output) {
        auto gainsInternal = LAMBDA_TTT(x, grad, eps) {
            T res = nd4j::math::nd4j_sign<T,T>(grad) != nd4j::math::nd4j_sign<T,T>(eps) ? x + T(.2) : x * T(.8);
            if(res < .01) res = .01;
            return res;
        };

        input->applyTriplewiseLambda(gradX, epsilon, gainsInternal, output);
    }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gains caller
    void barnes_gains(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), barnes_gains_, (input, gradX, epsilon, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void barnes_gains_, (NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output), NUMERIC_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cell contains - check cells for given point
//
    bool cell_contains(NDArray* corner, NDArray* width, NDArray* point, Nd4jLong dimension) {
        auto  cornerMinusWidth = *corner - *width;
        auto cornerPlusWidth = *corner + *width;
        // executes on host side, so sync all to host memory
        cornerMinusWidth.syncToHost();
        cornerPlusWidth.syncToHost();
        for (Nd4jLong i = 0; i < dimension; i++) {
            if (cornerMinusWidth.e<double>(i) > point->e<double>(i))
                return false;
            if (cornerPlusWidth.e<double>(i) < point->e<double>(i))
                return false;
        }

        return true;
    }
}
}
}

