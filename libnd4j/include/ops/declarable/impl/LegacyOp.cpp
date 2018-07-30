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

#include <ops/declarable/LegacyOp.h>


namespace nd4j {
    namespace ops {

        template <typename T>
        LegacyOp<T>::LegacyOp(int numInputs) : DeclarableOp<T>::DeclarableOp(numInputs , 1, "LegacyOp", true) {
            _numInputs = numInputs;
        }

        template <typename T>
        LegacyOp<T>::LegacyOp(int numInputs, int opNum) : DeclarableOp<T>::DeclarableOp(numInputs , 1, "LegacyOp", true) {
            _opNum = opNum;
            _numInputs = numInputs;
        }


        template class ND4J_EXPORT LegacyOp<float>;
        template class ND4J_EXPORT LegacyOp<float16>;
        template class ND4J_EXPORT LegacyOp<double>;
    }
}