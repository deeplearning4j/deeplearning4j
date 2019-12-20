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

#include <ops/declarable/helpers/helpers.h>
#include <execution/Threads.h>

namespace nd4j {
namespace ops {
namespace helpers {

void crossBatched(nd4j::LaunchContext * context, NDArray *a, NDArray *b, NDArray *o);

void FORCEINLINE cross(nd4j::LaunchContext * context, NDArray *a, NDArray *b, NDArray *o) {

    if (a->isR()) {
        auto a0 = a->e<double>(0);
        auto a1 = a->e<double>(1);
        auto a2 = a->e<double>(2);

        auto b0 = b->e<double>(0);
        auto b1 = b->e<double>(1);
        auto b2 = b->e<double>(2);

        o->p(Nd4jLong(0L), a1 * b2 - a2 * b1);
        o->p(1L, a2 * b0 - a0 * b2);
        o->p(2L, a0 * b1 - a1 * b0);
    } else {
        auto a0 = a->e<Nd4jLong>(0);
        auto a1 = a->e<Nd4jLong>(1);
        auto a2 = a->e<Nd4jLong>(2);

        auto b0 = b->e<Nd4jLong>(0);
        auto b1 = b->e<Nd4jLong>(1);
        auto b2 = b->e<Nd4jLong>(2);

        o->p(Nd4jLong(0L), a1 * b2 - a2 * b1);
        o->p(1L, a2 * b0 - a0 * b2);
        o->p(2L, a0 * b1 - a1 * b0);
    }
}

    void FORCEINLINE _crossBatched(nd4j::LaunchContext * context, NDArray *a, NDArray *b, NDArray *o) {
        auto a_ = a->reshape(a->ordering(), {-1, 3});
        auto b_ = b->reshape(b->ordering(), {-1, 3});
        auto o_ = o->reshape(o->ordering(), {-1, 3});

        auto tadsA = a_.allTensorsAlongDimension({1});
        auto tadsB = b_.allTensorsAlongDimension({1});
        auto tadsO = o_.allTensorsAlongDimension({1});

        int tads = tadsA.size();

        auto func = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e += increment) {
                auto a_ = tadsA.at(e);
                auto b_ = tadsB.at(e);
                auto o_ = tadsO.at(e);

                helpers::cross(context, a_, b_, o_);
            }
        };

        samediff::Threads::parallel_tad(func,  0, tads);
    }

    void weightedCrossEntropyWithLogitsFunctor(nd4j::LaunchContext * context, NDArray const* targets, NDArray const* input, NDArray const* weights, NDArray* output);
}
}
}