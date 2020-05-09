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
//  @author raver119@gmail.com
//

#ifndef DEV_TESTS_TADPACK_H
#define DEV_TESTS_TADPACK_H

#include "ConstantDataBuffer.h"

namespace sd {
    class ND4J_EXPORT TadPack {
    private:
        ConstantDataBuffer _tadShape;
        ConstantDataBuffer _tadOffsets;
        Nd4jLong _numTads = 0 ;
        int _shapeInfoLength = 0;
    public:
        explicit TadPack(ConstantDataBuffer &shapes, ConstantDataBuffer &offets, Nd4jLong numTads);
        TadPack() = default;
        ~TadPack() = default;

        const Nd4jLong* primaryShapeInfo() const;
        const Nd4jLong* primaryOffsets() const;

        const Nd4jLong* specialShapeInfo() const;
        const Nd4jLong* specialOffsets() const;

        Nd4jLong numberOfTads() const;
        int shapeInfoLength() const;

        /**
         * These methods return either primary or special pointers depending on platform binaries were compiled for
         * @return
         */
        const Nd4jLong *platformShapeInfo() const;
        const Nd4jLong *platformOffsets() const;
    };
}


#endif //DEV_TESTS_TADPACK_H
