/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

#ifndef DEV_TESTS_TADPACK_H
#define DEV_TESTS_TADPACK_H

#include "DataBuffer.h"

namespace nd4j {
    class ND4J_EXPORT TadPack {
    private:
        DataBuffer _tadShape;
        DataBuffer _tadOffsets;
        Nd4jLong _numTads;
    public:
        explicit TadPack(DataBuffer &shapes, DataBuffer &offets, Nd4jLong numTads);
        TadPack() = default;
        ~TadPack() = default;

        Nd4jLong* primaryShapeInfo();
        Nd4jLong* primaryOffsets();

        Nd4jLong* specialShapeInfo();
        Nd4jLong* specialOffsets();

        Nd4jLong numberOfTads();
    };
}


#endif //DEV_TESTS_TADPACK_H
