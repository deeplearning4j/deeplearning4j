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

#include <helpers/logger.h>
#include <pointercast.h>
#include <map>
#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        class GraphHolder {
        private:
            static GraphHolder *_INSTANCE;
            std::map<Nd4jLong, Graph<float>*> _graphF;
            std::map<Nd4jLong, Graph<double>*> _graphD;
            std::map<Nd4jLong, Graph<float16>*> _graphH;
            std::map<Nd4jLong, Graph<int>*> _graphI;
            std::map<Nd4jLong, Graph<Nd4jLong>*> _graphL;

            GraphHolder() = default;
            ~GraphHolder() = default;

            template <typename T>
            std::map<Nd4jLong, Graph<T>*>& getGraphMap();

        public:
            static GraphHolder* getInstance();

            template <typename T>
            void registerGraph(Nd4jLong graphId, Graph<T>* graph);
            
            template <typename T>
            Graph<T>* cloneGraph(Nd4jLong graphId);

            template <typename T>
            Graph<T>* pullGraph(Nd4jLong graphId);

            

            template <typename T>
            void forgetGraph(Nd4jLong graphId);

            template <typename T>
            void dropGraph(Nd4jLong graphId);


            void dropGraphAny(Nd4jLong graphId);

            template <typename T>
            bool hasGraph(Nd4jLong graphId);
        };
    }
}