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

#include <graph/InferenceRequest.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        InferenceRequest<T>::InferenceRequest(Nd4jLong graphId, ExecutorConfiguration *configuration) {
            this->_id = graphId;
            this->_configuration = configuration;
        }

        template <typename T>
        InferenceRequest<T>::~InferenceRequest() {
            for (auto v : _deletables)
                delete v;
        }


        template <typename T>
        void InferenceRequest<T>::appendVariable(int id, NDArray<T> *array) {
            appendVariable(id, 0, array);
        }

        template <typename T>
        void InferenceRequest<T>::appendVariable(int id, int index, NDArray<T> *array) {
            auto v = new Variable<T>(array, nullptr, id, index);
            insertVariable(v);
        }

        template <typename T>
        void InferenceRequest<T>::appendVariable(std::string &id, NDArray<T> *array) {
            auto v = new Variable<T>(array, id.c_str());
            insertVariable(v);
        }

        template <typename T>
        void InferenceRequest<T>::appendVariable(std::string &name, int id, int index, NDArray<T> *array) {
            auto v = new Variable<T>(array, name.c_str(), id, index);
            insertVariable(v);
        }

        template <typename T>
        void InferenceRequest<T>::insertVariable(Variable<T> *variable) {
            variable->markRemovable(false);
            variable->markReadOnly(true);
            _variables.emplace_back(variable);
            _deletables.emplace_back(variable);
        }

        template <typename T>
        void InferenceRequest<T>::appendVariable(Variable<T> *variable) {
            _variables.emplace_back(variable);
        }

        template <typename T>
        flatbuffers::Offset<FlatInferenceRequest> InferenceRequest<T>::asFlatInferenceRequest(flatbuffers::FlatBufferBuilder &builder) {
            std::vector<flatbuffers::Offset<FlatVariable>> vec;
            for (Variable<T>* v : _variables) {
                vec.emplace_back(v->asFlatVariable(builder));
            }

            auto confOffset = _configuration != nullptr ? _configuration->asFlatConfiguration(builder) : 0;

            auto vecOffset = builder.CreateVector(vec);

            return CreateFlatInferenceRequest(builder, _id, vecOffset, confOffset);
        }


        template class ND4J_EXPORT InferenceRequest<float>;
        template class ND4J_EXPORT InferenceRequest<float16>;
        template class ND4J_EXPORT InferenceRequest<double>;
    }
}