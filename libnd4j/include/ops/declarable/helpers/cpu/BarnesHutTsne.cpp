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

    Nd4jLong barnes_row_count(const NDArray* rowP, const NDArray* colP, NDArray& rowCounts) {
        auto N = rowP->lengthOf() / 2;

//        PRAGMA_OMP_PARALLEL_FOR
        for (int n = 0; n < N; n++) {
            int begin = rowP->e<int>(n);
            int end = rowP->e<int>(n + 1);
            for (int i = begin; i < end; i++) {
                bool present = false;
                for (int m = rowP->e<int>(colP->e<int>(i)); m < rowP->e<int>(colP->e<int>(i) + 1); m++)
                    if (colP->e<int>(m) == n) {
                        present = true;
                    }


                if (present)
                    rowCounts.p(n, rowCounts.e(n) + 1);

                else {
                    rowCounts.p(n, rowCounts.e(n) + 1);
                    rowCounts.p(colP->e<int>(i), rowCounts.e(colP->e<int>(i)) + 1);
                }
            }
        }

        NDArray* numElementsArr = rowCounts.reduceAlongDimension(reduce::Sum, {});
        if (numElementsArr == nullptr) throw std::runtime_error("helpers::barnes_symmertize: Cannot calculate num of Elements");
        auto numElements = numElementsArr->e<int>(0);
        delete numElementsArr;
        return numElements;
    }
//    static
//    void printVector(std::vector<int> const& v) {
//        for (auto x: v) {
//            printf("%d ", x);
//        }
//        printf("\n");
//        fflush(stdout);
//    }

    template <typename T>
    static void barnes_symmetrize_(const NDArray* rowP, const NDArray* colP, const NDArray* valP, NDArray* output, NDArray* rowCounts) {
        auto N = rowP->lengthOf() / 2 + rowP->lengthOf() % 2;
        //auto numElements = output->lengthOf();
        std::vector<int> symRowP = rowCounts->asVectorT<int>();//NDArrayFactory::create<int>('c', {numElements});
        //NDArray symValP = NDArrayFactory::create<double>('c', {numElements});
        symRowP.insert(symRowP.begin(),0);
        //symRowP(1, {0}) = *rowCounts;
//        for (int n = 0; n < N; n++)
//            symRowP.p(n + 1, symRowP.e(n) + rowCounts.e(n))
        int const* pRows = reinterpret_cast<int const*>(rowP->getBuffer());
        int const* pCols = reinterpret_cast<int const*>(colP->getBuffer());
        T const* pVals = reinterpret_cast<T const*>(valP->getBuffer());
        T* pOutput = reinterpret_cast<T*>(output->buffer());
        //std::vector<int> rowCountsV = rowCounts->getBufferAsVector<int>();
        std::vector<int> offset(N);// = NDArrayFactory::create<int>('c', {N});

//PRAGMA_OMP_PARALLEL_FOR_ARGS(schedule(guided) firstprivate(offset))
        for (int n = 0; n < N; n++) {
            int begin = pRows[n];
            int bound = pRows[n + 1];
//            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (int i = begin; i < bound; i++) {
                bool present = false;
                int start = pRows[pCols[i]];
                int end = pRows[pCols[i]] + 1;
                for (int m = start; m < end; m++) {
                    if (pCols[m] == n) {
                        present = true;
                        if (n < pCols[i]) {
                            pOutput[symRowP[n] + offset[n]] = pVals[i] + pVals[m];
                            pOutput[symRowP[pCols[i]] + offset[pCols[i]]] = pVals[i] + pVals[m];
                        }
                    }
                }

                // If (colP[i], n) is not present, there is no addition involved
                if (!present) {
                    int colPI = pCols[i];
                    if (n < colPI) {
                        pOutput[symRowP[n] + offset[n]] = T(colPI);
                        pOutput[symRowP[pCols[i]] + offset[colPI]] = T(n);
                        pOutput[symRowP[n] + offset[n]] = pVals[i];
                        pOutput[symRowP[colPI] + offset[colPI]] = pVals[i];
                    }

                }

                // Update offsets
                if (!present || (present && n < pCols[i])) {
                    ++offset[n];
                    int colPI = pCols[i];
                    if (colPI != n)
                        ++offset[colPI];
                }
//                printVector(offset);
            }
        }
    }
    void barnes_symmetrize(const NDArray* rowP, const NDArray* colP, const NDArray* valP, NDArray* output, NDArray* rowCounts) {

        // Divide the result by two
        BUILD_SINGLE_SELECTOR(valP->dataType(), barnes_symmetrize_, (rowP, colP, valP, output, rowCounts), NUMERIC_TYPES);

        *output /= 2.0;
        //output->assign(symValP);
    }
    BUILD_SINGLE_TEMPLATE(template void barnes_symmetrize_, (const NDArray* rowP, const NDArray* colP, const NDArray* valP, NDArray* output, NDArray* rowCounts), NUMERIC_TYPES);

    template <typename T>
    static void barnes_edge_forces_(const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray const* data, NDArray* output) {
        T const* dataP = reinterpret_cast<T const*>(data->getBuffer());
        T const* vals  = reinterpret_cast<T const*>(valP->getBuffer());
        T* outputP = reinterpret_cast<T*>(output->buffer());
        int colCount = data->columns();

        std::vector<T> slice(colCount);
        PRAGMA_OMP_PARALLEL_FOR_SIMD_ARGS(firstprivate(slice))
        for (int n = 0; n < N; n++) {
            T* currentSlice = &slice[0];
            memcpy(currentSlice, dataP + n * colCount, sizeof(T) * colCount);
            int start = rowP->e<int>(n);
            int end = rowP->e<int>(n+1);

            for (int i = start; i < end; i++) {
                T const* thisSlice = dataP + colP->e<int>(i) * colCount;
                T res = 1.e-12;
                for (int k = 0; k < colCount; k++) {
                    currentSlice[k] -= thisSlice[k];
                    res += currentSlice[k] * currentSlice[k];
                }

                for (int k = 0; k < colCount; k++)
                    outputP[n * colCount + k] += (currentSlice[k] * vals[i] / res);
            }
        }
    }

    void barnes_edge_forces(const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray* output, NDArray const& data) {
        // Loop over all edges in the graph
        BUILD_SINGLE_SELECTOR(data.dataType(), barnes_edge_forces_, (rowP, colP, valP, N, &data, output), NUMERIC_TYPES);
//       PRAGMA_OMP_PARALLEL_FOR
//        for (int n = 0; n < N; n++) {
//            NDArray slice = data(n, {0});
//            for (int i = rowP->e<int>(n); i < rowP->e<int>(n + 1); i++) {
//                // Compute pairwise distance and Q-value
//                slice -= data(colP->e<int>(i), {0});
//                //buf -= ;
//                auto res = slice.reduceAlongDimension(reduce::SquaredNorm, {});
//                // Sum positive force
//                (*output)(n, {0}) += (slice * valP->e(i) / (1.e-12 + *res));
//                delete res;
//            }
//        }
    }
    BUILD_SINGLE_TEMPLATE(template void barnes_edge_forces_, (const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray const* data, NDArray* output), NUMERIC_TYPES);

    template <typename T>
    static void barnes_gains_(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output) {
        //        gains = gains.add(.2).muli(sign(yGrads)).neq(sign(yIncs)).castTo(Nd4j.defaultFloatingPointType())
        //                .addi(gains.mul(0.8).muli(sign(yGrads)).neq(sign(yIncs)));
        auto gainsInternal = LAMBDA_TTT(x, grad, eps) {
            return T((x + 2.) * nd4j::math::nd4j_sign<T,T>(grad) != nd4j::math::nd4j_sign<T,T>(eps)) + T(x * 0.8 * nd4j::math::nd4j_sign<T,T>(grad) != nd4j::math::nd4j_sign<T,T>(eps));
        };

        input->applyTriplewiseLambda<T>(gradX, epsilon, gainsInternal, output);
    }

    void barnes_gains(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output) {
        //        gains = gains.add(.2).muli(sign(yGrads)).neq(sign(yIncs)).castTo(Nd4j.defaultFloatingPointType())
        //                .addi(gains.mul(0.8).muli(sign(yGrads)).neq(sign(yIncs)));
        BUILD_SINGLE_SELECTOR(input->dataType(), barnes_gains_, (input, gradX, epsilon, output), NUMERIC_TYPES);
//        auto signGradX = *gradX;
//        auto signEpsilon = *epsilon;
//        gradX->applyTransform(transform::Sign, &signGradX, nullptr);
//        epsilon->applyTransform(transform::Sign, &signEpsilon, nullptr);
//        auto leftPart = (*input + 2.) * signGradX;
//        auto leftPartBool = NDArrayFactory::create<bool>(leftPart.ordering(), leftPart.getShapeAsVector());
//
//        leftPart.applyPairwiseTransform(pairwise::NotEqualTo, &signEpsilon, &leftPartBool, nullptr);
//        auto rightPart = *input * 0.8 * signGradX;
//        auto rightPartBool = NDArrayFactory::create<bool>(rightPart.ordering(), rightPart.getShapeAsVector());
//        rightPart.applyPairwiseTransform(pairwise::NotEqualTo, &signEpsilon, &rightPartBool, nullptr);
//        leftPart.assign(leftPartBool);
//        rightPart.assign(rightPartBool);
//        leftPart.applyPairwiseTransform(pairwise::Add, &rightPart, output, nullptr);

    }
    BUILD_SINGLE_TEMPLATE(template void barnes_gains_, (NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output), NUMERIC_TYPES);
}
}
}

