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
// Created by yurii@skymind.io on 24.10.2017.
//

#ifndef LIBND4J_INTERVALS_H
#define LIBND4J_INTERVALS_H

#include <pointercast.h>
#include <vector>
#include <initializer_list>
#include <dll.h>

namespace  nd4j {

    class ND4J_EXPORT Intervals {
    
    private:
        std::vector<std::vector<Nd4jLong>> _content;

    public:

        // default constructor
        Intervals();
        
        // constructor
        Intervals(const std::initializer_list<std::vector<Nd4jLong>>& content );
        Intervals(const std::vector<std::vector<Nd4jLong>>& content );
        
        // accessing operator
        std::vector<Nd4jLong> operator[](const Nd4jLong i) const;

        // returns size of _content
        int size() const;

    };


}

#endif //LIBND4J_INTERVALS_H
