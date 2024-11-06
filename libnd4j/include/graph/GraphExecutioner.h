/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <graph/ExecutionResult.h>
#include <graph/Graph.h>
#include <graph/Node.h>
#include <graph/ResultWrapper.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/scheme/graph_generated.h>
#include <graph/scheme/node_generated.h>
#include <sys/stat.h>

#define TF_INPUT "Placeholder"
#define TF_CONST "Const"
#define TF_VAR "VariableV2"

namespace sd {
namespace graph {

class SD_LIB_EXPORT GraphExecutioner {
 protected:
 public:


  static Status executeFlatNode(Graph *graph, Node *node, VariableSpace *variableSpace);

  /**
   * This method executes given Graph
   * @return
   */
  static Status execute(Graph *graph, VariableSpace *variableSpace = nullptr);

  /**
   * This method executes graph stored at given FlatBuffers pointer
   *
   * @param pointer Pointer to FlatBuffer
   * @return pointer to FlatBuffer with result
   */
  static ResultWrapper *executeFlatBuffer(Pointer pointer);

  static flatbuffers::Offset<FlatResult> execute(Graph *graph, flatbuffers::FlatBufferBuilder &builder,
                                                 const FlatInferenceRequest *request);

  static Graph *importFromTensorFlow(const char *fileName);

  static Graph *importFromFlatBuffers(const char *filename);

  static Graph *importFromFlatPointer(Pointer ptr);
};

long getFileSize(const char *filename);
uint8_t *readFlatBuffers(const char *filename);

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_GRAPHEXECUTIONER_H
