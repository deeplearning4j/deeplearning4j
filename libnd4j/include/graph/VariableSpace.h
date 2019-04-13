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

#ifndef LIBND4J_VARIABLESPACE_H
#define LIBND4J_VARIABLESPACE_H

#include <helpers/logger.h>
#include <helpers/helper_random.h>
#include <string>
#include <vector>
#include <list>
#include <map>
#include <mutex>
#include <NDArray.h>
#include <array/NDArrayList.h>
#include <graph/Variable.h>
#include <memory/Workspace.h>
#include <graph/Stash.h>
#include <graph/FlowPath.h>


namespace nd4j {
    namespace graph {
        class ND4J_EXPORT VariableSpace {
        protected:

            nd4j::memory::Workspace _workspace;
            nd4j::random::RandomBuffer* _rng = nullptr;

            // stash is NOT cloned
            nd4j::graph::Stash _stash;

            std::map<std::pair<int, int>, Variable*> _paired;
            std::map<std::string, Variable*> _symbolic;
            std::map<int, Variable*> _variables;
            std::vector<Variable*> _external;
            std::vector<Variable*> _internal;

            std::vector<nd4j::NDArrayList*> _lists;

            std::vector<nd4j::graph::Variable*> _placeholders;

            void silentPutVariable(std::pair<int,int>& pair, Variable *variable);

            int _auto_counter = -1;

            std::mutex _varmap;

            std::map<int, nd4j::graph::Variable*> _temporary;

            std::vector<nd4j::graph::Variable*> *_handles;

            FlowPath* _flow = nullptr;

        public:
            VariableSpace();
            virtual ~VariableSpace();

            virtual VariableSpace& operator=(const VariableSpace& other);

            virtual int numberOfPlaceholders();
            virtual std::vector<Variable*>* getPlaceholders();
            virtual nd4j::random::RandomBuffer* getRNG();
            virtual void setRNG(nd4j::random::RandomBuffer* rng);
            virtual void setWorkspace(nd4j::memory::Workspace *workspace);
            
            virtual nd4j::memory::Workspace *workspace();

            virtual bool hasExternalVariable(int it);
            virtual bool hasExternalVariable(std::pair<int,int>& pair);
            virtual bool hasExternalVariable(std::string *symbol);

            virtual bool hasVariable(int id);
            virtual bool hasVariable(int id, int idx);
            virtual bool hasVariable(std::pair<int,int>& pair);
            virtual bool hasVariable(std::string *symbol);

            virtual nd4j::graph::Variable* getVariable(int id);
            virtual nd4j::graph::Variable* getVariable(int id, int idx);
            virtual nd4j::graph::Variable* getVariable(std::pair<int,int>& pair);
            virtual nd4j::graph::Variable* getVariable(std::string *symbol);

            virtual std::vector<Variable*> getVariables();

            virtual void putVariable(std::pair<int,int>& pair, NDArray *array);
            virtual void putVariable(std::pair<int,int>& pair, Variable *variable);
            virtual void putVariable(int id, Variable *variable);
            virtual void putVariable(int id, NDArray *array);
            virtual void putVariable(int id, int idx, NDArray *array);
            virtual void putVariable(int id, int idx, Variable *array);

            virtual void dropVariable(std::pair<int,int> &pair);
            virtual void dropVariable(int id, int idx);

            virtual void trackList(nd4j::NDArrayList *list);

            virtual void putOutputVariable(Variable *variable);

            virtual void replaceVariable(Variable *variable);

            // memory-related statistics
            virtual Nd4jLong externalMemory();
            virtual Nd4jLong internalMemory();
            virtual Nd4jLong totalMemory();

            virtual int externalEntries();
            virtual int internalEntries();
            virtual int totalEntries();

            virtual nd4j::graph::VariableSpace* clone();

            std::vector<Variable*> *handles();


            nd4j::graph::VariableSpace* asT();
            void injectVariable(std::pair<int, int> &pair, Variable* variable);

            virtual nd4j::graph::Stash* getStash();

            virtual std::vector<nd4j::graph::Variable*> * getExternalVariables();

            virtual void setFlowPath(FlowPath* timers);
            virtual FlowPath* flowPath();
        };
    }
}


#endif //LIBND4J_VARIABLESPACE_H
