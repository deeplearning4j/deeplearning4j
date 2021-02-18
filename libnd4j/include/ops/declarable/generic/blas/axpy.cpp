/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_axpy)

#include <ops/declarable/CustomOperations.h>

namespace sd {
    namespace ops {
        CONFIGURABLE_OP_IMPL(axpy, 2, 1, false, -2, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->isSameShape(y),0, "Axpy: both arguments should have the same shape");
            REQUIRE_TRUE(x->dataType() == y->dataType() && x->dataType() == z->dataType(), 0, "Axpy: all arguments must have the same data type");

            double a = 1.0;

            if (block.width() > 2) {
                auto alpha = INPUT_VARIABLE(2);
                REQUIRE_TRUE(alpha->isScalar(), 0, "Axpy: alpha argument should be scalar or TArg");
            } else if (block.getTArguments()->size() > 0) {
                a = T_ARG(0);
            }

            ExtraArguments arguments({a});

            y->applyPairwiseTransform(pairwise::Axpy, *x, *z, &arguments);

            return ND4J_STATUS_OK;
        }

        DECLARE_TYPES(axpy) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_FLOATS})
                    ->setAllowedOutputTypes(0, {ALL_FLOATS});
        }
    }
}

#endif