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
// Created by raver119 on 11/06/18.
//

#ifndef LIBND4J_RESULTWRAPPER_H
#define LIBND4J_RESULTWRAPPER_H

#include <op_boilerplate.h>
#include <pointercast.h>
#include <dll.h>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT ResultWrapper {
        private:
            Nd4jLong _size = 0L;
            Nd4jPointer _pointer = nullptr;

        public:
            ResultWrapper(Nd4jLong size, Nd4jPointer ptr);
            ~ResultWrapper();

            Nd4jLong size();

            Nd4jPointer pointer();
        };
    }
}


#endif //LIBND4J_RESULTWRAPPER_H
