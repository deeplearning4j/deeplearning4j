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

#ifndef LIBND4J_BARNES_HUT_TSNE_HELPERS_H
#define LIBND4J_BARNES_HUT_TSNE_HELPERS_H

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {

    Nd4jLong barnes_row_count(const NDArray* rowP, const NDArray* colP, Nd4jLong N, NDArray& rowCounts);
    void barnes_symmetrize(const NDArray* rowP, const NDArray* colP, const NDArray* valP, Nd4jLong N, NDArray* output, NDArray* rowCounts = nullptr);
    void barnes_edge_forces(const NDArray* rowP, NDArray const* colP, NDArray const* valP, int N, NDArray* output, NDArray const& data);
    void barnes_gains(NDArray* input, NDArray* gradX, NDArray* epsilon, NDArray* output);

}
}
}


#endif //LIBND4J_ACTIVATIONS_H
