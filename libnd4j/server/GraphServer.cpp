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

#include "GraphServer.h"
#include <graph/GraphHolder.h>
#include <GraphExecutioner.h>
#include <graph/generated/result_generated.h>
#include <helpers/StringUtils.h>
#include <algorithm>


namespace nd4j {
    namespace graph {
            grpc::Status GraphInferenceServerImpl::RegisterGraph( grpc::ServerContext *context, const flatbuffers::grpc::Message<FlatGraph> *request_msg, flatbuffers::grpc::Message<FlatResponse> *response_msg) {
                nd4j_printf("Starting %i\n", 0);

                auto flat_graph = request_msg->GetRoot();

                nd4j_printf("Starting %i\n", 1);

                // building our graph
                auto graph = new Graph<float>(flat_graph);

                nd4j_printf("Starting %i\n", 2);
                
                // single data type for now
                GraphHolder::getInstance()->registerGraph<float>(flat_graph->id(), graph);

                // sending out OK response
                auto response_offset = CreateFlatResponse(mb_, 0);
                mb_.Finish(response_offset);
                *response_msg = mb_.ReleaseMessage<FlatResponse>();
                assert(response_msg->Verify());

                return grpc::Status::OK;
            }

            grpc::Status GraphInferenceServerImpl::ForgetGraph( grpc::ServerContext *context, const flatbuffers::grpc::Message<FlatDropRequest> *request_msg, flatbuffers::grpc::Message<FlatResponse> *response_msg) {
                // getting drop request
                auto request = request_msg->GetRoot();

                // dropping out graph (any datatype)
                GraphHolder::getInstance()->dropGraphAny(request->id());

                // sending out OK response
                auto response_offset = CreateFlatResponse(mb_, 0);
                mb_.Finish(response_offset);
                *response_msg = mb_.ReleaseMessage<FlatResponse>();
                assert(response_msg->Verify());

                return grpc::Status::OK;
            }

            grpc::Status GraphInferenceServerImpl::InferenceRequest( grpc::ServerContext *context, const flatbuffers::grpc::Message<FlatInferenceRequest> *request_msg, flatbuffers::grpc::Message<FlatResult> *response_msg) {
                auto request = request_msg->GetRoot();

                // trying to get graph by id
                auto graph = GraphHolder::getInstance()->cloneGraph<float>(request->id());

                // provide results here
                auto response_offset = GraphExecutioner<float>::execute(graph, mb_, request);

                mb_.Finish(response_offset);
                *response_msg = mb_.ReleaseMessage<FlatResult>();
                assert(response_msg->Verify());

                return grpc::Status::OK;
            }
    }
}

void RunServer(int port) {
  assert(port > 0 && port < 65535);

  std::string server_address("0.0.0.0:");
  server_address += nd4j::StringUtils::valueToString<int>(port);

  nd4j::graph::GraphInferenceServerImpl service;
  auto registrator = nd4j::ops::OpRegistrator::getInstance();

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cerr << "Server listening on: [" << server_address << "]; Number of operations: [" <<  registrator->numberOfOperations()  << "]"<< std::endl;

  server->Wait();
}

char* getCmdOption(char **begin, char **end, const std::string & option) {
    auto itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
        return *itr;

    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

int main(int argc, char *argv[]) {
    /**
     * basically we only care about few things here:
     * 1) port number
     * 2) if we should use gprc, json, or both
     * 3) if there's any graph(s) provided at startup
     */
     int port = 40123;
     if(cmdOptionExists(argv, argv+argc, "-p")) {
        auto sPort = getCmdOption(argv, argv + argc, "-p");
        port = atoi(sPort);
     }

    if(cmdOptionExists(argv, argv+argc, "-f")) {
        auto file = getCmdOption(argv, argv + argc, "-f");
        auto graph = GraphExecutioner<float>::importFromFlatBuffers(file);
        nd4j::graph::GraphHolder::getInstance()->registerGraph<float>(0L, graph);
    }

    RunServer(port);

    return 0;
}