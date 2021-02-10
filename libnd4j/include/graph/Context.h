/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019-2020 Konduit K.K.
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

#ifndef LIBND4J_CONTEXT_H
#define LIBND4J_CONTEXT_H

#include <vector>
#include <array/NDArray.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/ContextPrototype.h>
#include <memory/Workspace.h>
#include <execution/Engine.h>

// CUDA-specific includes
#ifdef __CUDACC__

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#endif

namespace sd {
    namespace graph {
        /**
         * This class defines input desired for any given node/operation within graph
         */
        class ND4J_EXPORT Context : public sd::graph::ContextPrototype {
        protected:
            sd::memory::Workspace* _workspace = nullptr;
            sd::graph::VariableSpace* _variableSpace = nullptr;
            std::pair<Nd4jLong, Nd4jLong> _executionTime;
            sd::random::RandomBuffer* _rng = nullptr;

            sd::DataType _dataType = sd::DataType::FLOAT32;
            // branch for divergent_op
            int _branch = 0;

            // temporary context for standalone ops execution
            LaunchContext* _context = nullptr;

            std::vector<sd::DataType> _dataTypes;

            // fields for fast execution (out-of-graph ops use)
            std::vector<NDArray*> _fastpath_in;
            std::vector<NDArray*> _fastpath_out;
            std::vector<NDArray*> _handles;

            bool _helpersAllowed = true;

            // in some cases we might be able to skip shape function for validation purposes
            bool _shapeFunctionOverride = false;

            // special flag used during conversion from Graph exec to FastPath exec
            bool _forbidFastPath = false;
        public:
            Context(ContextPrototype* prototype, VariableSpace* variableSpace);

            explicit Context(int nodeId, VariableSpace *variableSpace = nullptr);
            Context(int nodeId, VariableSpace *variableSpace, bool isInplace);

            // default destructor
            ~Context();

            // these methods are for execution timing
            void setOuterTime(Nd4jLong time);
            void setInnerTime(Nd4jLong time);
            Nd4jLong getOuterTime();
            Nd4jLong getInnerTime();

            sd::DataType dataType() override;

            sd::DataType dataType(int index) override;
            void setDataType(int index, sd::DataType type) override;
            // these methods are related to Workspace abstraction
            bool hasWorkspaceProvided();
            void attachWorkspace(sd::memory::Workspace* workspace);
            void forgetWorkspace();

            // these methods return full-time workspace
            sd::memory::Workspace* getWorkspace();
            sd::memory::Workspace* workspace();
            sd::memory::Workspace* fWorkspace();

            // this method returns workspace for temporary allocations
            sd::memory::Workspace* tWorkspace();

            // this method returns workspace for object allocations
            sd::memory::Workspace* oWorkspace();

            void setVariableSpace(VariableSpace* variableSpace);

            sd::random::RandomBuffer* getRNG();
            void setRNG(sd::random::RandomBuffer* rng);

            void setTargetEngine(samediff::Engine engine);

            VariableSpace *getVariableSpace();

            LaunchContext* launchContext();

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
            void trackList(NDArrayList* list);


            /**
             * This method returns variable for a given input index for this block
             * @param idx
             * @return
             */
            Variable* getVariable(int idx);
            Variable* variable(int idx);

            /**
             * This method is shortcut to getVariable(int idx);
             *
             * + it check fastpath for array availability (preferred)
             * @return
             */
            NDArray* getNDArray(int idx);
            NDArray* array(int idx);


            /**
             * This method fetches variable from VariableSpace DIRECTLY
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

            unsigned long width() override;

            // methods used in java interop
            /**
             * This method checks if Context uses fastpath variable access
             * @return
             */
            bool isFastPath();

            /**
             * Method allows to forbid FastPath execution
             * @param reallyForbid
             */
            void forbidFastPath(bool reallyForbid);

#ifndef __JAVACPP_HACK__
            std::vector<NDArray*>& fastpath_in();
            std::vector<NDArray*>& fastpath_out();
#endif

            void setInputArray(int index, NDArray *array, bool removable = false);
            void setInputArray(int index, void *buffer, void const* shapeInfo, void *specialBuffer, void const* specialShapeInfo);
            void setInputArray(int index, void *buffer, void * shapeInfo, void *specialBuffer, void * specialShapeInfo);
            void setInputArray(int index, void *databuffer, void const* shapeInfo, void const* specialShapeInfo);

            void setOutputArray(int index, NDArray *array, bool removable = false);
            void setOutputArray(int index, void *buffer, const void * shapeInfo, void *specialBuffer, const void * specialShapeInfo);
            void setOutputArray(int index, void *buffer, void * shapeInfo, void *specialBuffer, void * specialShapeInfo);
            void setOutputArray(int index, void *databuffer, void const* shapeInfo, void const* specialShapeInfo);

            void setTArguments(double *arguments, int numberOfArguments);
            void setIArguments(Nd4jLong *arguments, int numberOfArguments);
            void setBArguments(bool *arguments, int numberOfArguments);
            void setDArguments(sd::DataType *arguments, int numberOfArguments);

            void setTArguments(const std::vector<double> &tArgs);
            void setIArguments(const std::vector<Nd4jLong> &tArgs);
            void setBArguments(const std::vector<bool> &tArgs);
            void setDArguments(const std::vector<sd::DataType> &dArgs);

            /**
             * This method purges fastpath in/out contents and releases all the handles.
             *
             * PLEASE NOTE: I/T/B/D args will stay intact
             */
            void clearFastPath();

            void setCudaContext(Nd4jPointer cudaStream, Nd4jPointer reductionPointer, Nd4jPointer allocationPointer);

            void allowHelpers(bool reallyAllow);
            bool helpersAllowed();

            void setShapeFunctionOverride(bool reallyOverride);
            bool shapeFunctionOverride();

            samediff::ExecutionMode executionMode();
            void setExecutionMode(samediff::ExecutionMode executionMode);

            bool isTraining();
            bool isInference();
        };
    }
}


#endif //LIBND4J_BLOCK_H
