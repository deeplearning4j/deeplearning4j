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

#ifndef LIBND4J_EXECUTORCONFIGURATION_H
#define LIBND4J_EXECUTORCONFIGURATION_H
#include <graph/scheme/config_generated.h>
#include <system/common.h>

namespace sd {
namespace graph {
class SD_LIB_EXPORT ExecutorConfiguration {
 public:
  sd::graph::ProfilingMode _profilingMode;
  sd::graph::ExecutionMode _executionMode;
  sd::graph::OutputMode _outputMode;
  bool _timestats;
  sd::LongType _footprintForward = 0L;
  sd::LongType _footprintBackward = 0L;
  Direction _direction = Direction_FORWARD_ONLY;

  explicit ExecutorConfiguration(const sd::graph::FlatConfiguration *conf = nullptr);
  ~ExecutorConfiguration() = default;

  ExecutorConfiguration *clone();

#ifndef __JAVACPP_HACK__
  flatbuffers::Offset<FlatConfiguration> asFlatConfiguration(flatbuffers::FlatBufferBuilder &builder);
#endif
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_EXECUTORCONFIGURATION_H
