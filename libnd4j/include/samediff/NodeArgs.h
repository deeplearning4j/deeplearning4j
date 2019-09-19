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

#ifndef SAMEDIFF_NODEARGS_H
#define SAMEDIFF_NODEARGS_H

#include <dll.h>
#include <pointercast.h>
#include <vector>

namespace samediff {
    class ND4J_EXPORT NodeArgs {
    private:
        std::vector<Nd4jLong> _iArgs;
        std::vector<double> _tArgs;
        std::vector<bool> _bArgs;

        std::vector<std::pair<int, int>> _inputs;
    public:
        NodeArgs() = default;
        ~NodeArgs() = default;

        void addInput(int position, int nodeId, int index);

        void addIArg(int position, Nd4jLong value);
        void addTArg(int position, double value);
        void addBArg(int position, bool value);


        std::vector<std::pair<int, int>>& inputs();
        std::vector<Nd4jLong>& iargs();
        std::vector<double>& targs();
        std::vector<bool>& bargs();
    };
}


#endif //SAMEDIFF_NODEARGS_H
