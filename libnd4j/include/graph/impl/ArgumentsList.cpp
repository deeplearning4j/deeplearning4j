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
// Created by raver119 on 24.01.18.
//

#include <graph/ArgumentsList.h>

namespace nd4j {
namespace graph {
    ArgumentsList::ArgumentsList(std::initializer_list<Pair> arguments) {
        _arguments = arguments;
    }

    ArgumentsList::ArgumentsList(std::initializer_list<int> arguments) {
        std::vector<int> args(arguments);
        for (int e = 0; e < args.size(); e++) {
            Pair pair(args[e]);
            _arguments.emplace_back(pair);
        }

    }

    int ArgumentsList::size() {
        return (int) _arguments.size();
    }

    Pair&  ArgumentsList::at(int index) {
        return _arguments.at(index);
    }
}
}
