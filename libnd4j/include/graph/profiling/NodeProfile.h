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

#ifndef LIBND4J_NODE_PROFILE_H
#define LIBND4J_NODE_PROFILE_H

#include <pointercast.h>
#include <dll.h>
#include <string>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT NodeProfile {
        private:
            int _id;
            std::string _name;

            Nd4jLong _merges = 1L;

            // time spent during deserialization
            Nd4jLong _buildTime = 0L;
            
            // time spent before op execution
            Nd4jLong _preparationTime = 0L;

            // time spent for op execution
            Nd4jLong _executionTime = 0L;

            // total time spent during node execution
            Nd4jLong _totalTime = 0L;

            // time spent for output shape creation
            Nd4jLong _shapeTime = 0L;

            // time spent for output arrays creation
            Nd4jLong _arrayTime = 0L;

            Nd4jLong _inputTime = 0L;

            // amount of memory used for outputs
            Nd4jLong _memoryActivations = 0L;

            // amount of memory used internally for temporary arrays
            Nd4jLong _memoryTemporary = 0L;

            // amount of memory used internally for objects
            Nd4jLong _memoryObjects = 0L;

            // total amount of memory used during execution
            Nd4jLong _memoryTotal = 0L;
        public:
            NodeProfile() = default;
            ~NodeProfile() = default;

            explicit NodeProfile(int id, const char *name);

            void setBuildTime(Nd4jLong time);
            void setPreparationTime(Nd4jLong time);
            void setExecutionTime(Nd4jLong time);
            void setTotalTime(Nd4jLong time);
            void setShapeFunctionTime(Nd4jLong time);
            void setArrayTime(Nd4jLong time);
            void setInputTime(Nd4jLong time);

            void setActivationsSize(Nd4jLong bytes);
            void setTemporarySize(Nd4jLong bytes);
            void setObjectsSize(Nd4jLong bytes);
            void setTotalSize(Nd4jLong bytes);

            Nd4jLong getActivationsSize();
            Nd4jLong getTemporarySize();
            Nd4jLong getObjectsSize();
            Nd4jLong getTotalSize();

            std::string& name();

            void merge(NodeProfile *other);
            void assign(NodeProfile *other);

            void printOut();
        };
    }
}

#endif