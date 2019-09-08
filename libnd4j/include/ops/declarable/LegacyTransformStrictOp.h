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
// Created by raver119 on 16.10.2017.
// Modified by GS <sgazeos@gmail.com> on 19.10.2018
// Modified by GS <sgazeos@gmail.com> on 19.10.2018
//
#ifndef LIBND4J__LEGACY_TRANSFORM_STRICT_OP__H
#define LIBND4J__LEGACY_TRANSFORM_STRICT_OP__H


#include <ops/declarable/LegacyOp.h>

namespace nd4j {
    namespace ops {
        /**
        *   This class provides wrapper for Transform operations (i.e. Pow or OneMinus)
        */
        class ND4J_EXPORT LegacyTransformStrictOp : public LegacyOp {
        protected:
            Nd4jStatus validateAndExecute(Context &block) override;
        public:
            LegacyTransformStrictOp();
            LegacyTransformStrictOp(int opNum);

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context &block) override;
            LegacyOp* clone() override;
        };
    }
}


#endif //LIBND4J__LEGACY_TRANSFORM_SAME_OP__H
