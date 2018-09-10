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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 12.12.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_zeta)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/zeta.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(zeta, 2, 1, false, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto q = INPUT_VARIABLE(1);

            auto output   = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->isSameShape(q), 0, "ZETA op: two input arrays must have the same shapes, bot got x=%s and q=%s !", ShapeUtils<T>::shapeAsString(x).c_str(), ShapeUtils<T>::shapeAsString(q).c_str());

            int arrLen = x->lengthOf();

            for(int i = 0; i < arrLen; ++i ) {

                REQUIRE_TRUE((*x)(i) > (T)1., 0, "ZETA op: all elements of x array must be > 1 !");

                REQUIRE_TRUE((*q)(i) > (T)0., 0, "ZETA op: all elements of q array must be > 0 !");
            }

            *output = helpers::zeta<T>(*x, *q);

            return Status::OK();
        }
        DECLARE_SYN(Zeta, zeta);
    }
}

#endif