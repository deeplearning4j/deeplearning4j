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
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_testop2i2o)

#include <ops/declarable/headers/tests.h>

namespace nd4j {
    namespace ops {
        //////////////////////////////////////////////////////////////////////////
        // test op, non-divergent
        OP_IMPL(testop2i2o, 2, 2, true) {
            //nd4j_printf("CPU op used!\n","");
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);

            auto xO = OUTPUT_VARIABLE(0);
            auto yO = OUTPUT_VARIABLE(1);

            x->template applyScalar<simdOps::Add<T>>(1.0, xO, nullptr);
            y->template applyScalar<simdOps::Add<T>>(2.0, yO, nullptr);

            STORE_2_RESULTS(*xO, *yO);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TestOp2i2o, testop2i2o);
    }
}

#endif
