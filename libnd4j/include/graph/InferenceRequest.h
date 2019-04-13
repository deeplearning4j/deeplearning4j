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
#ifndef DEV_TESTS_INFERENCEREQUEST_H
#define DEV_TESTS_INFERENCEREQUEST_H

#include <op_boilerplate.h>
#include <pointercast.h>
#include <dll.h>
#include <graph/Variable.h>
#include "ExecutorConfiguration.h"

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT InferenceRequest {
        private:
            Nd4jLong _id;
            std::vector<Variable*> _variables;
            std::vector<Variable*> _deletables;

            ExecutorConfiguration *_configuration = nullptr;

            void insertVariable(Variable* variable);
        public:

            InferenceRequest(Nd4jLong graphId, ExecutorConfiguration *configuration = nullptr);
            ~InferenceRequest();

            void appendVariable(int id, NDArray *array);
            void appendVariable(int id, int index, NDArray *array);
            void appendVariable(std::string &name, NDArray *array);
            void appendVariable(std::string &name, int id, int index, NDArray *array);
            void appendVariable(Variable *variable);

#ifndef __JAVACPP_HACK__
            flatbuffers::Offset<FlatInferenceRequest> asFlatInferenceRequest(flatbuffers::FlatBufferBuilder &builder);
#endif
        };
    }
}



#endif //DEV_TESTS_INFERENCEREQUEST_H
