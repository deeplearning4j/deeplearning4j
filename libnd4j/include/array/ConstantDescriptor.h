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

#ifndef DEV_TESTS_CONSTANTDESCRIPTOR_H
#define DEV_TESTS_CONSTANTDESCRIPTOR_H

#include <array/DataType.h>
#include <map>
#include <vector>
#include <pointercast.h>
#include <dll.h>
#include <array/ConstantDataBuffer.h>

namespace nd4j {
    class ND4J_EXPORT ConstantDescriptor {
    private:
        std::vector<Nd4jLong> _integerValues;
        std::vector<double> _floatValues;
    public:
        ConstantDescriptor(std::initializer_list<Nd4jLong> &values);
        ConstantDescriptor(std::initializer_list<double> &values);

        ConstantDescriptor(std::vector<Nd4jLong> &values);
        ConstantDescriptor(std::vector<double> &values);

        ~ConstantDescriptor() = default;

        // equal to operator
        bool operator==(const ConstantDescriptor &other) const;

        // less than operator
        bool operator<(const ConstantDescriptor &other) const;

        bool isInteger();
        bool isFloat();

        Nd4jLong length();

        std::vector<Nd4jLong>& integerValues();
        std::vector<double>& floatValues();
    };
}


#endif //DEV_TESTS_CONSTANTDESCRIPTOR_H
