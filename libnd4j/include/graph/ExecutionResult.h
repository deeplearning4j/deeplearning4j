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

#ifndef LIBND4J_EXECUTION_RESULT
#define LIBND4J_EXECUTION_RESULT

#include <vector>
#include <initializer_list>
#include <map>
#include <string>
#include <flatbuffers/flatbuffers.h>
#include <graph/Variable.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        class ExecutionResult {
        private:
            std::vector<Variable<T> *> _variables;
            std::map<std::string, Variable<T>*> _stringIdMap;
            std::map<std::pair<int, int>, Variable<T>*> _pairIdMap;

            // this flag is used to optionally release variables
            bool _releasable = false;
        public:
            ExecutionResult(const FlatResult* flatResult);
            ExecutionResult(std::initializer_list<Variable<T> *> variables);
            ExecutionResult() = default;
            ~ExecutionResult();

            /**
             * This method adds variable pointer to result
             */
            void emplace_back(Variable<T> *variable);

            /**
             * This method returns Variable by its position in output
             */
            Variable<T>* at(int position);

            /**
             * This method returns Variable by its string id
             */
            Variable<T>* byId(std::string &id);

            /**
             * This method returns Variable by its string id
             */
            Variable<T>* byId(const char *str);

            /**
             * This method returns Variable by its numeric id:index pair
             */
            Variable<T>* byId(std::pair<int, int> &id);

            /**
             * This method returns Variable by its numeric id with index 0
             */
            Variable<T>* byId(int id);

            /**
             * This method returns number of elements stored in this entity
             * @return
             */
            Nd4jLong size();

#ifndef __JAVACPP_HACK__
            /**
             * This method converts ExecutionResult entity to FlatResult
             */
            flatbuffers::Offset<FlatResult> asFlatResult(flatbuffers::FlatBufferBuilder &builder);
#endif
        };
    }
}

#endif