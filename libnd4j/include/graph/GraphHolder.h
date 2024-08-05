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
//  @author raver119@gmail.com
//
#include <exceptions/unknown_graph_exception.h>
#include <graph/Graph.h>
#include <helpers/SimpleReadWriteLock.h>
#include <helpers/logger.h>

#include <map>
#include <unordered_map>

namespace sd {
namespace graph {
class SD_LIB_EXPORT GraphHolder {
 private:
  SD_MAP_IMPL<LongType, Graph*> _graphF;

  SD_MAP_IMPL<LongType, SimpleReadWriteLock> _locks;

  GraphHolder() = default;
  ~GraphHolder() = default;

 public:
  static GraphHolder& getInstance();

  void registerGraph(LongType graphId, Graph* graph);

  Graph* cloneGraph(LongType graphId);

  Graph* pullGraph(LongType graphId);

  void forgetGraph(LongType graphId);

  void dropGraph(LongType graphId);

  void dropGraphAny(LongType graphId);

  bool hasGraph(LongType graphId);

  bool hasGraphAny(LongType graphId);

  flatbuffers::Offset<FlatResult> execute(LongType graphId, flatbuffers::FlatBufferBuilder& builder,
                                          const FlatInferenceRequest* request);

  void replaceGraph(LongType graphId, Graph* graph);

  /////////////////////////////

  SD_INLINE void lockWrite(LongType graphId) {
    if (_locks.count(graphId) == 0) return;

    _locks[graphId].lockWrite();
  }

  SD_INLINE void unlockWrite(LongType graphId) {
    if (_locks.count(graphId) == 0) return;

    _locks[graphId].unlockWrite();
  }

  SD_INLINE void lockRead(LongType graphId) {
    if (_locks.count(graphId) == 0) return;

    _locks[graphId].lockRead();
  }

  SD_INLINE void unlockRead(LongType graphId) {
    if (_locks.count(graphId) == 0) return;

    _locks[graphId].unlockRead();
  }
};
}  // namespace graph
}  // namespace sd
