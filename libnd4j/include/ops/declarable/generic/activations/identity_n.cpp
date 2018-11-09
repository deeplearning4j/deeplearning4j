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
//  @author sgazeos@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_identity_n)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(identity_n, 1, 1, true, 0, 0) {

            // just for lulz
            for (Nd4jLong i = 0; i < block.width(); ++i) {
                NDArray<T>* x = INPUT_VARIABLE(i);
                NDArray<T>* z = OUTPUT_VARIABLE(i);

                x->template applyTransform<simdOps::Identity<T>>(z, nullptr);
            }

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(identity_n) {
            auto shapes = SHAPELIST();
            for (size_t i = 0; i < inputShape->size(); ++i) {
                Nd4jLong* shape;
                COPY_SHAPE_EX(inputShape->at(i), shape, block.getWorkspace());
                shapes->push_back(shape);
            }
            return shapes;
        }
    }
}

#endif