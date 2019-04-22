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

void barnes_symmetrize(const NDArray* rowP, const NDArray* colP, const NDArray* valP, NDArray* output, NDArray* rowCounts) {
        auto N = rowP->lengthOf() / 2;
        auto numElements = output->lengthOf();
        NDArray offset = NDArrayFactory::create<int>('c', {N});
//        NDArray symRowP = NDArrayFactory::create<int>('c', {N + 1});
        std::vector<int> symRowP = rowCounts->asVectorT<int>();//NDArrayFactory::create<int>('c', {numElements});
        //NDArray symValP = NDArrayFactory::create<double>('c', {numElements});
        symRowP.insert(symRowP.begin(),0);
        //symRowP(1, {0}) = *rowCounts;
//        for (int n = 0; n < N; n++)
//            symRowP.p(n + 1, symRowP.e(n) + rowCounts.e(n));


        for (int n = 0; n < N; n++) {
            for (int i = rowP->e<int>(n); i < rowP->e<int>(n + 1); i++) {
                bool present = false;
                for (int m = rowP->e<int>(colP->e<int>(i)); m < rowP->e<int>(colP->e<int>(i)) + 1; m++) {
                    if (colP->e<int>(m) == n) {
                        present = true;
                        if (n < colP->e<int>(i)) {
                            // make sure we do not add elements twice
//                            symColP[symRowP.e<int>(n) + offset.e<int>(n)] = colP->e<int>(i);
//                            symColP[symRowP.e<int>(colP->e<int>(i)) + offset.e<int>(colP->e<int>(i))] = n;
                            output->p(symRowP[n] + offset.e<int>(n),
                                              valP->e<double>(i) + valP->e<double>(m));
                            output->p(symRowP[colP->e<int>(i)] + offset.e<int>(colP->e<int>(i)),
                                              valP->e<double>(i) + valP->e<double>(m));
                        }
                    }
                }

                // If (colP[i], n) is not present, there is no addition involved
                if (!present) {
                    int colPI = colP->e<int>(i);
                    if (n < colPI) {
                        output->p(symRowP[n] + offset.e<int>(n), colPI);
                        output->p(symRowP[colP->e<int>(i)] + offset.e<int>(colPI), n);
                        output->p(symRowP[n] + offset.e<int>(n), valP->e<double>(i));
                        output->p(symRowP[colPI] + offset.e<int>(colPI), valP->e<double>(i));
                    }

                }

                // Update offsets
                if (!present || (present && n < colP->e<int>(i))) {
                    offset.p(n, offset.e<int>(n) + 1);
                    int colPI = colP->e<int>(i);
                    if (colPI != n)
                        offset.p(colPI, offset.e<int>(colPI) + 1);
                }
            }
        }

        // Divide the result by two
        *output /= 2.0;
        //output->assign(symValP);
    }

    void barnes_edge_forces(const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray* output, NDArray const& data) {
        // Loop over all edges in the graph

        PRAGMA_OMP_PARALLEL_FOR
        for (int n = 0; n < N; n++) {
            NDArray slice = data(n, {0});
            //NDArray buff(slice);
            for (int i = rowP->e<int>(n); i < rowP->e<int>(n + 1); i++) {
                // Compute pairwise distance and Q-value
                slice -= data(colP->e<int>(i), {0});
                //buf -= ;
                auto res = slice.applyReduce3(reduce3::Dot, &slice, nullptr);
                *res += 1.e-12;// + buf * buf; //Nd4j.getBlasWrapper().dot(buf, buf);
                *res = valP->e<double>(i) / *res;

                // Sum positive force
                (*output)(n, {0}) += (slice * *res);
            }
        }
    }

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

