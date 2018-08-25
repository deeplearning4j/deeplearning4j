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

#ifndef LIBND4J_GRAPHEXECUTIONER_H
#define LIBND4J_GRAPHEXECUTIONER_H

#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/Node.h>
#include <graph/Graph.h>
#include <graph/ResultWrapper.h>
#include <sys/stat.h>
#include <graph/ExecutionResult.h>

#define TF_INPUT "Placeholder"
#define TF_CONST "Const"
#define TF_VAR "VariableV2"

namespace nd4j {
    namespace graph {

    template <typename T>
    class GraphExecutioner {
    protected:


    public:
        //static Nd4jStatus executeFlatNode(nd4j::graph::Graph *graph, nd4j::graph::Node *node, nd4j::graph::VariableSpace<float> *variableSpace);

        static Nd4jStatus executeFlatNode(Graph<T> *graph, Node<T> *node, VariableSpace<T> *variableSpace);

        /**
        * This method executes given Graph
        * @return
        */
        static Nd4jStatus execute(Graph<T> *graph, VariableSpace<T>* variableSpace = nullptr);


        /**
        * This method executes graph stored at given FlatBuffers pointer
        *
        * @param pointer Pointer to FlatBuffer
        * @return pointer to FlatBuffer with result
        */
        static nd4j::graph::ResultWrapper* executeFlatBuffer(Nd4jPointer pointer);

        static flatbuffers::Offset<FlatResult> execute(Graph<T> *graph, flatbuffers::FlatBufferBuilder &builder, FlatInferenceRequest* request);

        static Graph<T> *importFromTensorFlow(const char *fileName);


        static Graph<T> *importFromFlatBuffers(const char *filename);

        static Graph<T> *importFromFlatPointer(Nd4jPointer ptr);
    };

    long getFileSize(const char * filename);
    uint8_t* readFlatBuffers(const char * filename);

    }
}


#endif //LIBND4J_GRAPHEXECUTIONER_H
