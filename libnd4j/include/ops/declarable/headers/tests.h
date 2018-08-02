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
#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        #if NOT_EXCLUDED(OP_test_output_reshape)
        DECLARE_OP(test_output_reshape, 1, 1, true);
        #endif

        #if NOT_EXCLUDED(OP_test_scalar)
        DECLARE_CUSTOM_OP(test_scalar, 1, 1, false, 0, 0);
        #endif

        #if NOT_EXCLUDED(OP_testreduction)
        DECLARE_REDUCTION_OP(testreduction, 1, 1, false, 0, -1);
        #endif

        #if NOT_EXCLUDED(OP_noop)
        DECLARE_OP(noop, -1, -1, true);
        #endif

        #if NOT_EXCLUDED(OP_testop2i2o)
        DECLARE_OP(testop2i2o, 2, 2, true);
        #endif

        #if NOT_EXCLUDED(OP_testcustom)
        DECLARE_CUSTOM_OP(testcustom, 1, 1, false, 0, -1);
        #endif
    }
}