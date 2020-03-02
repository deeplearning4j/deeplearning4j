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
//

#ifndef LIBND4J__LEGACY_TRANSFORM_OP__H
#define LIBND4J__LEGACY_TRANSFORM_OP__H


//#include <ops/declarable/LegacyOp.h>
#ifdef ONLY_SAME_TRANSFORM
namespace sd {
    namespace ops {
        /**
        *   This class provides wrapper for Transform operations (i.e. Pow or OneMinus)
        */
        class ND4J_EXPORT LegacyTransformOp : public LegacyOp {
        protected:
            Nd4jStatus validateAndExecute(Context &block);
        public:
            LegacyTransformOp();
            LegacyTransformOp(int opNum);

            ShapeList* calculateOutputShape(ShapeList* inputShape, sd::graph::Context &block);
            virtual LegacyOp* clone();
        };
    }
}
#endif

#include <ops/declarable/LegacyTransformFloatOp.h>
#include <ops/declarable/LegacyTransformSameOp.h>
#include <ops/declarable/LegacyTransformBoolOp.h>
#include <ops/declarable/LegacyTransformStrictOp.h>


#endif //LIBND4J__LEGACY_TRANSFORM_OP__H
