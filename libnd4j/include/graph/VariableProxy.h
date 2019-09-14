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
        class ND4J_EXPORT VariableProxy: public VariableSpace {
        protected:
            VariableSpace* _backed = nullptr;
            VariableSpace* _current = nullptr;
        public:
            explicit VariableProxy(VariableSpace* reference);
            ~VariableProxy();

            VariableSpace& operator=(const VariableSpace& other) override;

            int numberOfPlaceholders() override;
            std::vector<Variable*>* getPlaceholders() override;

            virtual nd4j::memory::Workspace *workspace();

            bool hasExternalVariable(int it) override;
            bool hasExternalVariable(const std::pair<int,int>& pair) override;
            bool hasExternalVariable(const std::string &symbol) override;

            bool hasVariable(int id) override;
            bool hasVariable(int id, int idx) override;
            bool hasVariable(const std::pair<int,int>& pair) override;
            bool hasVariable(const std::string &symbol) override;

            nd4j::graph::Variable *getVariable(int id) override;
            nd4j::graph::Variable *getVariable(int id, int idx) override;
            nd4j::graph::Variable *getVariable(const std::pair<int,int>& pair) override;
            nd4j::graph::Variable *getVariable(const std::string &symbol) override;

            std::vector<Variable*> getVariables() override;

            void putVariable(const std::pair<int,int>& pair, NDArray *array) override;
            void putVariable(const std::pair<int,int>& pair, Variable *variable) override;
            void putVariable(const std::string &name, Variable *variable) override;
            void putVariable(int id, Variable *variable) override;
            void putVariable(int id, NDArray *array) override;
            void putVariable(int id, int idx, NDArray *array) override;
            void putVariable(int id, int idx, Variable *array) override;

            void replaceVariable(Variable *variable) override;

            void dropVariable(std::pair<int,int> &pair) override;
            void dropVariable(int id, int idx) override;

            void putOutputVariable(Variable *variable) override;

            void trackList(nd4j::NDArrayList *list) override;

            // memory-related statistics
            Nd4jLong externalMemory() override;
            Nd4jLong internalMemory() override;
            Nd4jLong totalMemory() override;

            int externalEntries() override;
            int internalEntries() override;
            int totalEntries() override;

            nd4j::graph::VariableSpace *clone() override;

            nd4j::graph::Stash* getStash() override;
            void setFlowPath(FlowPath* timers) override;
            FlowPath* flowPath() override;
        };
    }
}