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
// Created by raver119 on 11.10.2017.
//

#ifndef LIBND4J_OPTUPLE_H
#define LIBND4J_OPTUPLE_H

#include <vector>
#include <initializer_list>
#include <NDArray.h>

namespace nd4j {
    namespace ops {
        class OpTuple {
        public:
            const char * _opName;
            std::vector<nd4j::NDArray*> _inputs;
            std::vector<nd4j::NDArray*> _outputs;
            std::vector<double> _tArgs;
            std::vector<Nd4jLong> _iArgs;

            OpTuple(const char *opName);
            OpTuple(const char *opName, std::initializer_list<nd4j::NDArray *>&& inputs, std::initializer_list<double>&& tArgs, std::initializer_list<Nd4jLong>&& iArgs);
            ~OpTuple();

            OpTuple* addInput(nd4j::NDArray *array);
            OpTuple* addOutput(nd4j::NDArray *array);
            OpTuple* setTArgs(std::initializer_list<double> tArgs);
            OpTuple* setIArgs(std::initializer_list<Nd4jLong> iArgs);
        };
    }
}


#endif //LIBND4J_OPTUPLE_H
