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

#include <NDArray.h>
#include <graph/Graph.h>
#include <ops/declarable/CustomOperations.h>

#include <graph/generated/graph.grpc.fb.h>

namespace nd4j {
    namespace graph {
        class GraphInferenceServerImpl final : public GraphInferenceServer::Service {
            virtual grpc::Status RegisterGraph( grpc::ServerContext *context, const flatbuffers::grpc::Message<FlatGraph> *request_msg, flatbuffers::grpc::Message<FlatResponse> *response_msg) {
                return grpc::Status::OK;
            }

            virtual grpc::Status ForgetGraph( grpc::ServerContext *context, const flatbuffers::grpc::Message<FlatDropRequest> *request_msg, flatbuffers::grpc::Message<FlatResponse> *response_msg) {
                return grpc::Status::OK;
            }

            virtual grpc::Status InferenceRequest( grpc::ServerContext *context, const flatbuffers::grpc::Message<FlatInferenceRequest> *request_msg, flatbuffers::grpc::Message<FlatResponse> *response_msg) {
                return grpc::Status::OK;
            }
        };
    }
}

void RunServer() {
  std::string server_address("0.0.0.0:40123");
  nd4j::graph::GraphInferenceServerImpl service;

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cerr << "Server listening on " << server_address << std::endl;

  server->Wait();
}

int main(int argc, const char *argv[]) {
    /**
     * basically we only care about few things here:
     * 1) port number
     * 2) graphs and their IDs
     */
    RunServer();

    return 0;
}