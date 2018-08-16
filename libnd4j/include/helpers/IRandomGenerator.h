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

#ifndef LIBND4J_IRANDOMGENERATOR_H
#define LIBND4J_IRANDOMGENERATOR_H

#include <op_boilerplate.h>
#include <pointercast.h>
#include <dll.h>

namespace nd4j {
    class ND4J_EXPORT IRandomGenerator {

    public:
        virtual _CUDA_D uint32_t relativeUint32(Nd4jLong index) = 0;
        virtual _CUDA_D uint64_t relativeUint64(Nd4jLong index) = 0;
        virtual _CUDA_H void rewindH(Nd4jLong steps) = 0;

        /**
         * This method returns T value between 0 and MAX_T
         */
        template <typename T>
        T relativeT(Nd4jLong index);

        /**
         * This method returns T value between from and to
         */
        template <typename T>
        T relativeT(Nd4jLong index, T from, T to);

        /**
         * This method returns int value between 0 and MAX_INT
         * @param index
         * @return
         */
        int relativeInt(Nd4jLong index);

        /**
         * This method returns int value between 0 and MAX_LONG
         * @param index
         * @return
         */
        Nd4jLong relativeLong(Nd4jLong index);
    };
}

#endif //DEV_TESTS_IRANDOMGENERATOR_H
