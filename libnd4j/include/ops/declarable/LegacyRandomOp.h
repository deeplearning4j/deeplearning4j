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

#ifndef LIBND4J_LEGACYRANDOMOP_H
#define LIBND4J_LEGACYRANDOMOP_H


#include <helpers/helper_random.h>
#include <ops/declarable/LegacyOp.h>

namespace nd4j {
    namespace ops {
        /**
        *   This class provides wrapper for Random operations (i.e. linspace or Uniform)
        */
        template <typename T>
        class ND4J_EXPORT LegacyRandomOp : public LegacyOp<T> {
        protected:
            Nd4jStatus validateAndExecute(Context<T>& block);
        public:
            LegacyRandomOp();
            LegacyRandomOp(int opNum);
            ~LegacyRandomOp() = default;

            nd4j::ResultSet<T>*  execute(nd4j::random::RandomBuffer* rng, std::initializer_list<NDArray<T>*> inputs, std::initializer_list<T> tArgs, std::initializer_list<int> iArgs, bool isInplace = false);
            nd4j::ResultSet<T>*  execute(nd4j::random::RandomBuffer* rng, std::vector<NDArray<T>*>& inputs, std::vector<T>& tArgs, std::vector<int>& iArgs, bool isInplace = false);
            Nd4jStatus execute(Context<T>* block);

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context<T>& block);
            virtual LegacyOp<T>* clone();
        };
    }
}


#endif //LIBND4J_LEGACYTRANSFORMOP_H
