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
// Created by raver119 on 16.10.2017.
//

#ifndef LIBND4J_LEGACYOP_H
#define LIBND4J_LEGACYOP_H

#include <ops/declarable/DeclarableOp.h>
#include <helpers/PointersManager.h>

namespace sd {
    namespace ops {

        /**
        * This class is root abstraction for legacy XYZ ops wrappers.
        * All wrappers for specific op groups (i.e. LegacyTransformOp for Transform ops) are inheriting this class.
        *
        *
        */
        class ND4J_EXPORT LegacyOp : public DeclarableOp {
        protected:
            // this field is mainly for debugging
            // it defines, which legacy op should be invoked on a given data
            int _opNum = -1;
            int _numInputs = 0;

            // All Op classes provide own specific implementation for this method
            Nd4jStatus validateAndExecute(Context& block) override = 0;
        public:
            LegacyOp(int numInputs);
            LegacyOp(int numInputs, int opNum);
            ~LegacyOp() = default;

            // All Op classes provide own specific implementation for this method
            ShapeList* calculateOutputShape(ShapeList* inputShape, sd::graph::Context& block) override = 0;
            virtual LegacyOp* clone() = 0;
        };
    }
}


#endif //LIBND4J_LEGACYOP_H
