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

namespace nd4j {
namespace ops {
namespace helpers {
    void FORCEINLINE _cross(NDArray *a, NDArray *b, NDArray *o) {
        if (a->isR()) {
            auto a0 = a->e<double>(0);
            auto a1 = a->e<double>(1);
            auto a2 = a->e<double>(2);

            auto b0 = b->e<double>(0);
            auto b1 = b->e<double>(1);
            auto b2 = b->e<double>(2);

            Nd4jLong idx = 0L;
            o->putScalar(Nd4jLong(0L), a1 * b2 - a2 * b1);
            o->putScalar(1L, a2 * b0 - a0 * b2);
            o->putScalar(2L, a0 * b1 - a1 * b0);
        } else {
            auto a0 = a->e<Nd4jLong>(0);
            auto a1 = a->e<Nd4jLong>(1);
            auto a2 = a->e<Nd4jLong>(2);

            auto b0 = b->e<Nd4jLong>(0);
            auto b1 = b->e<Nd4jLong>(1);
            auto b2 = b->e<Nd4jLong>(2);

            Nd4jLong idx = 0L;
            o->putScalar(Nd4jLong(0L), a1 * b2 - a2 * b1);
            o->putScalar(1L, a2 * b0 - a0 * b2);
            o->putScalar(2L, a0 * b1 - a1 * b0);
        }
    }

    void FORCEINLINE _crossBatched(NDArray *a, NDArray *b, NDArray *o) {
        auto _a = a->reshape(a->ordering(), {-1, 3});
        auto _b = b->reshape(b->ordering(), {-1, 3});
        auto _o = o->reshape(o->ordering(), {-1, 3});

        auto tadsA = _a->allTensorsAlongDimension({1});
        auto tadsB = _b->allTensorsAlongDimension({1});
        auto tadsO = _o->allTensorsAlongDimension({1});

        int tads = tadsA->size();

#pragma omp parallel for simd schedule(static)
        for (int e = 0; e < tads; e++) {
            auto a_ = tadsA->at(e);
            auto b_ = tadsB->at(e);
            auto o_ = tadsO->at(e);

            helpers::_cross(a_, b_, o_);
        }

        delete tadsA;
        delete tadsB;
        delete tadsO;
        delete _a;
        delete _b;
        delete _o;
    }
}
}
}