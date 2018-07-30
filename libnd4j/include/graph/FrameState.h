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
// Created by raver119 on 06.02.2018.
//

#ifndef LIBND4J_FRAMESTATE_H
#define LIBND4J_FRAMESTATE_H

#include <string>
#include <pointercast.h>

namespace nd4j {
    namespace graph {
        class FrameState {
        private:
            std::string _name;
            Nd4jLong _id = 0;
            int _numberOfCycles = 0;
            bool _activated = false;

            bool _rewindPlanned = false;
            int _rewindPosition = -1;
        public:
             FrameState(Nd4jLong id = 0);
            ~FrameState() = default;

            /**
             * This method returns number of cycles passed for this Frame
             *
             * @return
             */
            int getNumberOfCycles();

            /**
             * This method increments number of cycles by 1 for this Frame
             */
            void incrementNumberOfCycles();

            /**
             * This method returns TRUE is frame was activated at LoopCond
             * @return
             */
            bool wasActivated();

            /**
             * This method allows to toggle activated state of this Frame
             * @param reallyActivated
             */
            void markActivated(bool reallyActivated);

            /**
             * This method returns of this Frame (if it's set)
             * @return
             */
            std::string& getFrameName();

            /**
             * This method returns TRUE if reset is planned for this Frame
             * @return
             */
            bool isRewindPlanned();

            /**
             * This method allows you to toggle flag for planned rewind
             * @param reallyPlanning
             */
            void planRewind(bool reallyPlanning);

            /**
             * This method returns planned reset position for given Frame
             * @return
             */
            int getRewindPosition();

            /**
             * This method allows to set rewind position for this Frame
             * @param pos
             */
            void setRewindPosition(int pos);

            /**
             * This method allows to set rewind position for this Frame, but only if it wasn't set earlier
             * @param pos
             */
            void setRewindPositionOnce(int pos);
        };
    }
}


#endif //LIBND4J_FRAMESTATE_H
