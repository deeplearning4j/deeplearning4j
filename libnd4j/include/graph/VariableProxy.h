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
//  @author raver119@gmail.com
//

#include <graph/VariableSpace.h>

namespace nd4j {
    namespace graph {
        class VariableProxy: public VariableSpace {
        protected:
            VariableSpace* _backed = nullptr;
            VariableSpace* _current = nullptr;
        public:
            explicit VariableProxy(VariableSpace* reference);
            ~VariableProxy();

            virtual VariableSpace& operator=(const VariableSpace<T>& other);

            virtual int numberOfPlaceholders();
            virtual std::vector<Variable*>* getPlaceholders();
            virtual nd4j::random::RandomBuffer* getRNG();
            virtual void setRNG(nd4j::random::RandomBuffer* rng);

            virtual nd4j::memory::Workspace *workspace();

            virtual bool hasExternalVariable(int it);
            virtual bool hasExternalVariable(std::pair<int,int>& pair);
            virtual bool hasExternalVariable(std::string *symbol);

            virtual bool hasVariable(int id);
            virtual bool hasVariable(int id, int idx);
            virtual bool hasVariable(std::pair<int,int>& pair);
            virtual bool hasVariable(std::string *symbol);

            virtual nd4j::graph::Variable *getVariable(int id);
            virtual nd4j::graph::Variable *getVariable(int id, int idx);
            virtual nd4j::graph::Variable *getVariable(std::pair<int,int>& pair);
            virtual nd4j::graph::Variable *getVariable(std::string *symbol);

            virtual std::vector<Variable<T>*> getVariables();

            virtual void putVariable(std::pair<int,int>& pair, NDArray *array);
            virtual void putVariable(std::pair<int,int>& pair, Variable *variable);
            virtual void putVariable(int id, Variable *variable);
            virtual void putVariable(int id, NDArray *array);
            virtual void putVariable(int id, int idx, NDArray *array);
            virtual void putVariable(int id, int idx, Variable *array);

            virtual void replaceVariable(Variable *variable);

            virtual void dropVariable(std::pair<int,int> &pair);
            virtual void dropVariable(int id, int idx);

            virtual void putOutputVariable(Variable *variable);

            virtual void trackList(nd4j::NDArrayList *list);

            // memory-related statistics
            virtual Nd4jLong externalMemory();
            virtual Nd4jLong internalMemory();
            virtual Nd4jLong totalMemory();

            virtual int externalEntries();
            virtual int internalEntries();
            virtual int totalEntries();

            virtual nd4j::graph::VariableSpace *clone();

            virtual nd4j::graph::Stash* getStash();
            virtual void setFlowPath(FlowPath* timers);
            virtual FlowPath* flowPath();
        };
    }
}