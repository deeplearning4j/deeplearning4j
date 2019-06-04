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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/lrn.h>
#include <Status.h>

namespace nd4j {
namespace ops {
namespace helpers {

    // FIXME: double
    int lrnFunctor(nd4j::graph::Context& block, NDArray* input, NDArray* output, int depth, double bias, double alpha, double beta) {
        return Status::OK();
    }

    int lrnFunctorEx(nd4j::graph::Context& block, NDArray* input, NDArray* output, NDArray* unitScale, NDArray* scale, int depth, double bias, double alpha, double beta) {

        return Status::OK();
    }

    int lrnFunctorEx(nd4j::graph::Context& block, NDArray* input, NDArray* output, NDArray* scale, int depth, double bias, double alpha, double beta) {
        return Status::OK();
    }

    void lrnBP(const NDArray& input, const NDArray& gradO, NDArray& gradI, const int depth, const float bias, const float alpha, const float beta) {
        //
    }
}
}
}
