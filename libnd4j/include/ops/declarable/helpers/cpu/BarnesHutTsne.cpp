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

    }

    void barnes_edge_forces(const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray* output, NDArray& data, NDArray& buf) {
        // Loop over all edges in the graph
#pragma omp parallel for
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

