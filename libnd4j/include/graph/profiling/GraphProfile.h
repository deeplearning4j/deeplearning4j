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

#ifndef ND4J_GRAPH_PROFILE_H
#define ND4J_GRAPH_PROFILE_H
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "NodeProfile.h"

namespace sd {
namespace graph {
class SD_LIB_EXPORT GraphProfile {
 private:
  // this variable
  LongType _merges = 1L;

  /**
   * This is global memory values
   */
  LongType _memoryTotal = 0L;
  LongType _memoryActivations = 0L;
  LongType _memoryTemporary = 0L;
  LongType _memoryObjects = 0L;

  // time spent for graph construction
  LongType _buildTime = 0L;

  // time spent for graph execution
  LongType _executionTime = 0L;

  // collection of pointers to profile results
  std::vector<NodeProfile *> _profiles;
  std::map<int, NodeProfile *> _profilesById;

  // collection of various timing reports
  std::map<std::string, LongType> _timings;
  std::chrono::time_point<std::chrono::system_clock> _last;

  std::map<std::string, std::chrono::time_point<std::chrono::system_clock>> _timers;

  void updateLast();

 public:
  GraphProfile();
  ~GraphProfile();

  /**
   * These methods just adding amount of bytes to various counters
   */
  void addToTotal(LongType bytes);
  void addToActivations(LongType bytes);
  void addToTemporary(LongType bytes);
  void addToObjects(LongType bytes);

  /**
   * This method allows to set graph construction (i.e. deserialization) time in nanoseconds
   */
  void setBuildTime(LongType nanos);

  /**
   * This method sets graph execution time in nanoseconds.
   */
  void setExecutionTime(LongType nanos);

  void startEvent(const char *name);
  void recordEvent(const char *name);
  void deleteEvent(const char *name);

  /**
   * This method saves time as delta from last saved time
   */
  void spotEvent(const char *name);

  /**
   * This method returns pointer to NodeProfile by ID
   * PLEASE NOTE: this method will create new NodeProfile if there's none
   */
  NodeProfile *nodeById(int id, const char *name = nullptr);
  bool nodeExists(int id);

  /**
   * This method merges values from other profile report
   * @param other
   */
  void merge(GraphProfile *other);
  void assign(GraphProfile *other);

  /**
   * These methods are just utility methods for time
   */
  static LongType currentTime();
  static LongType relativeTime(LongType time);

  void printOut();
};
}  // namespace graph
}  // namespace sd

#endif
