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

#ifndef LIBND4J_BLOCK_H
#define LIBND4J_BLOCK_H

#include <vector>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/ContextPrototype.h>
#include <memory/Workspace.h>


// CUDA-specific includes
#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#endif

namespace nd4j {
    namespace graph {
        /**
         * This class defines input desired for any given node/operation within graph
         */
        class Context : public nd4j::graph::ContextPrototype {
        protected:
            nd4j::memory::Workspace* _workspace = nullptr;
            nd4j::graph::VariableSpace* _variableSpace = nullptr;
            std::pair<Nd4jLong, Nd4jLong> _executionTime;
            nd4j::random::RandomBuffer* _rng = nullptr;

            nd4j::DataType _dataType;
            // branch for divergent_op
            int _branch = 0;
        public:
            // TODO: maybe override new here as well?

            // CUDA-specific fields
#ifdef __CUDACC__
            cudaStream_t* _stream;
#endif

            Context(ContextPrototype *prototype, VariableSpace *variableSpace);

            explicit Context(int nodeId, VariableSpace *variableSpace = nullptr);
            Context(int nodeId, VariableSpace *variableSpace, bool isInplace);

            // default destructor
            ~Context();

            // these methods are for execution timing
            void setOuterTime(Nd4jLong time);
            void setInnerTime(Nd4jLong time);
            Nd4jLong getOuterTime();
            Nd4jLong getInnerTime();

            nd4j::DataType dataType();

            // these methods are related to Workspace abstraction
            bool hasWorkspaceProvided();
            void attachWorkspace(nd4j::memory::Workspace* workspace);
            void forgetWorkspace();

            // these methods return full-time workspace
            nd4j::memory::Workspace* getWorkspace();
            nd4j::memory::Workspace* workspace();
            nd4j::memory::Workspace* fWorkspace();

            // this method returns workspace for temporary allocations
            nd4j::memory::Workspace* tWorkspace();

            // this method returns workspace for object allocations
            nd4j::memory::Workspace* oWorkspace();


            void setVariableSpace(VariableSpace *variableSpace);

            nd4j::random::RandomBuffer* getRNG();
            void setRNG(nd4j::random::RandomBuffer* rng);

            VariableSpace *getVariableSpace();

            // these fields define, if we can execute specific node in-place, without generating new array


            // these variables are only for Divergent Nodes
            int getBranch();
            void setBranch(int branch);

            /**
             *
             * @return
             */
            Stash* getStash();

            /**
             *
             */
            void trackList(NDArrayList *list);


            /**
             * This method returns variable for a given input index for this block
             * @param idx
             * @return
             */
            Variable* getVariable(int idx);
            Variable* variable(int idx);


            /**
             * This method fetches variable from Workspace DIRECTLY
             * @param p
             * @return
             */
            Variable* variable(int node, int index);
            Variable* variable(std::pair<int,int>& p);
            Variable* variable(std::initializer_list<int> p);


            void pushNDArrayToVariableSpace(int nodeId, int index, NDArray* array, bool removable = true);
            void pushNDArrayToVariableSpace(std::pair<int, int>& pair, NDArray* array, bool removable = true);

            void pushNDArrayListToVariableSpace(int nodeId, int index, NDArrayList* list, bool track = true);
            void pushNDArrayListToVariableSpace(std::pair<int, int>& pair, NDArrayList* list, bool track = true);

            bool isValueAvailable(int idx = 0);

            Variable* ensureVariable(int idx = 0);
        };
    }
}


#endif //LIBND4J_BLOCK_H
