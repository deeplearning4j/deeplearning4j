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
// Created by raver119 on 16/11/17.
//

#ifndef LIBND4J_FLOWPATH_H
#define LIBND4J_FLOWPATH_H

#include <map>
#include <pointercast.h>
#include <graph/NodeState.h>
#include <graph/FrameState.h>
#include <graph/profiling/GraphProfile.h>
#include <dll.h>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT FlowPath {
        private:
            std::map<int, NodeState> _states;
            std::map<Nd4jLong, FrameState> _frames;

            void ensureNode(int nodeId);
            void ensureFrame(int nodeId);

            GraphProfile _profile;
        public:
            FlowPath() = default;
            ~FlowPath() = default;

            void setInnerTime(int nodeId, Nd4jLong time);
            void setOuterTime(int nodeId, Nd4jLong time);

            Nd4jLong innerTime(int nodeId);
            Nd4jLong outerTime(int nodeId);

            bool isNodeActive(int nodeId);
            void markNodeActive(int nodeId, bool isActive);

            bool wasExecuted(int nodeId);
            void markExecuted(int nodeId, bool wasExecuted);

            int branch(int nodeId);
            void markBranch(int nodeId, int index);

            // Frame-related methods

            void registerFrame(Nd4jLong frameId);
            void forgetFrame(Nd4jLong frameId);

            bool isFrameActive(Nd4jLong frameId);
            void markFrameActive(Nd4jLong frameId, bool isActive);

            bool isRewindPlanned(Nd4jLong frameId);
            void planRewind(Nd4jLong frameId, bool reallyRewind);

            int getRewindPosition(Nd4jLong frameId);
            void setRewindPosition(Nd4jLong frameId, int position);
            void setRewindPositionOnce(Nd4jLong frameId, int position);

            void incrementNumberOfCycles(Nd4jLong frameId);
            Nd4jLong getNumberOfCycles(Nd4jLong frameId);

            GraphProfile* profile();
        };
    }
}


#endif //LIBND4J_FLOWPATH_H
