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

    void barnes_symmetrize(const NDArray* rowP, const NDArray* colP, const NDArray* valP, NDArray* output) {
        NDArray rowCounts; //() = Nd4j.create(N);
        int N = rowP->lengthOf() / 2;
#pragma omp parallel for
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
        
        NDArray offset = NDArrayFactory::create<int>({N});
        NDArray symRowP = NDArrayFactory::create<int>({N + 1});
        NDArray symColP = NDArrayFactory::create<int>({numElements});
        NDArray symValP = NDArrayFactory::create<double>('c', {numElements});

        for (int n = 0; n < N; n++)
            symRowP.p(n + 1, symRowP.e(n) + rowCounts.e(n));


        for (int n = 0; n < N; n++) {
            for (int i = rowP->e<int>(n); i < rowP->e<int>(n + 1); i++) {
                bool present = false;
                for (int m = rowP->e<int>(colP->e<int>(i)); m < rowP->e<int>(colP->e<int>(i)) + 1; m++) {
                    if (colP->e<int>(m) == n) {
                        present = true;
                        if (n < colP->e<int>(i)) {
                            // make sure we do not add elements twice
                            symColP.p(symRowP.e<int>(n) + offset.e<int>(n), colP->e<int>(i));
                            symColP.p(symRowP.e<int>(colP->e<int>(i)) + offset.e<int>(colP->e<int>(i)), n);
                            symValP.p(symRowP.e<int>(n) + offset.e<int>(n),
                                              valP->e<double>(i) + valP->e<double>(m));
                            symValP.p(symRowP.e<int>(colP->e<int>(i)) + offset.e<int>(colP->e<int>(i)),
                                              valP->e<double>(i) + valP->e<double>(m));
                        }
                    }
                }

                // If (colP[i], n) is not present, there is no addition involved
                if (!present) {
                    int colPI = colP->e<int>(i);
                    if (n < colPI) {
                        symColP.p(symRowP.e<int>(n) + offset.e<int>(n), colPI);
                        symColP.p(symRowP.e<int>(colP->e<int>(i)) + offset.e<int>(colPI), n);
                        symValP.p(symRowP.e<int>(n) + offset.e<int>(n), valP->e<double>(i));
                        symValP.p(symRowP.e<int>(colPI) + offset.e<int>(colPI), valP->e<double>(i));
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
        symValP /= 2.0;
        output->assign(symValP);
    }

    void barnes_edge_forces(const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray* output, NDArray& data, NDArray& buf) {
        // Loop over all edges in the graph
#pragma omp parallel for schedule(guided)
        for (int n = 0; n < N; n++) {
            NDArray slice = data(n, {0});

            for (int i = rowP->e<int>(n); i < rowP->e<int>(n + 1); i++) {
                // Compute pairwise distance and Q-value
                buf.assign(slice);
                buf -= data(colP->e<int>(i), {0});
                auto res = buf.applyReduce3(reduce3::Dot, &buf, nullptr);
                *res += 1.e-12;// + buf * buf; //Nd4j.getBlasWrapper().dot(buf, buf);
                *res = valP->e<double>(i) / *res;

                // Sum positive force
                (*output)(n, {0}) += (buf * *res);
            }
        }
    }
}
}
}

