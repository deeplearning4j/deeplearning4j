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


#include <grpc++/grpc++.h>
#include <NDArray.h>
#include <graph/Graph.h>
#include <ops/declarable/CustomOperations.h>

#include <graph/generated/graph.grpc.fb.h>

namespace nd4j {
    namespace graph {
        class GraphInferenceServerImpl final : public GraphInferenceServer::Service {
        private:
            flatbuffers::grpc::MessageBuilder mb_;
        public:
            virtual grpc::Status RegisterGraph( grpc::ServerContext *context, const flatbuffers::grpc::Message<FlatGraph> *request_msg, flatbuffers::grpc::Message<FlatResponse> *response_msg);

            virtual grpc::Status ForgetGraph( grpc::ServerContext *context, const flatbuffers::grpc::Message<FlatDropRequest> *request_msg, flatbuffers::grpc::Message<FlatResponse> *response_msg);

            virtual grpc::Status GraphInferenceServerImpl::ReplaceGraph( grpc::ServerContext *context, const flatbuffers::grpc::Message<FlatGraph> *request_msg, flatbuffers::grpc::Message<FlatResponse> *response_msg);

            virtual grpc::Status InferenceRequest( grpc::ServerContext *context, const flatbuffers::grpc::Message<FlatInferenceRequest> *request_msg, flatbuffers::grpc::Message<FlatResult> *response_msg);
        };
    }
}