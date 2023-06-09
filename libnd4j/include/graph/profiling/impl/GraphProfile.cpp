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
#include <graph/profiling/GraphProfile.h>
#include <helpers/logger.h>
#include <math/templatemath.h>

#include <algorithm>
#include <chrono>

namespace sd {
namespace graph {
GraphProfile::GraphProfile() { updateLast(); }

GraphProfile::~GraphProfile() {
  // releasing NodeProfile pointers
  for (auto v : _profiles) delete v;

  _timings.clear();
}

void GraphProfile::addToTotal(sd::LongType bytes) { _memoryTotal += bytes; }

void GraphProfile::addToActivations(sd::LongType bytes) { _memoryActivations += bytes; }

void GraphProfile::addToTemporary(sd::LongType bytes) { _memoryTemporary += bytes; }

void GraphProfile::addToObjects(sd::LongType bytes) { _memoryObjects += bytes; }

void GraphProfile::setBuildTime(sd::LongType nanos) { _buildTime = nanos; }

void GraphProfile::setExecutionTime(sd::LongType nanos) { _executionTime = nanos; }

sd::LongType GraphProfile::currentTime() {
  auto t = std::chrono::system_clock::now();
  auto v = std::chrono::time_point_cast<std::chrono::nanoseconds>(t);
  auto epoch = v.time_since_epoch();
  return (sd::LongType)std::chrono::duration_cast<std::chrono::nanoseconds>(epoch).count();
}

sd::LongType GraphProfile::relativeTime(sd::LongType time) {
  auto t1 = currentTime();
  return t1 - time;
}

void GraphProfile::updateLast() { _last = std::chrono::system_clock::now(); }

void GraphProfile::startEvent(const char *name) {
  std::string k = name;
  _timers[k] = std::chrono::system_clock::now();
}

void GraphProfile::recordEvent(const char *name) {
  std::string k = name;
  if (_timers.count(k) == 0) {
    sd_printf("Can't find timer key: [%s]", name);
    THROW_EXCEPTION("Missing timer key");
  }
  auto t0 = _timers[k];
  auto t1 = std::chrono::system_clock::now();
  auto v = (sd::LongType)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

  _timings[k] = v;
  _timers.erase(k);
}

void GraphProfile::deleteEvent(const char *name) {
  std::string k = name;
  _timers.erase(k);
}

void GraphProfile::spotEvent(const char *name) {
  auto t = std::chrono::system_clock::now();
  auto d = (sd::LongType)std::chrono::duration_cast<std::chrono::nanoseconds>(t - _last).count();
  std::string k = name;
  _timings[k] = d;
  updateLast();
}

NodeProfile *GraphProfile::nodeById(int id, const char *name) {
  if (_profilesById.count(id) == 0) {
    auto node = new NodeProfile(id, name);
    _profiles.emplace_back(node);
    _profilesById[id] = node;
    return node;
  }

  return _profilesById[id];
}

void GraphProfile::merge(GraphProfile *other) {
  _merges += other->_merges;
  _memoryActivations += other->_memoryActivations;
  _memoryTemporary += other->_memoryTemporary;
  _memoryTotal += other->_memoryTotal;
  _memoryObjects += other->_memoryObjects;

  _executionTime += other->_executionTime;
  _buildTime += other->_buildTime;

  for (auto v : _profilesById) {
    if (!other->nodeExists(v.first)) continue;

    v.second->merge(other->nodeById(v.first));
  }
}

void GraphProfile::assign(GraphProfile *other) {
  _merges = other->_merges;
  _memoryActivations = other->_memoryActivations;
  _memoryTemporary = other->_memoryTemporary;
  _memoryTotal = other->_memoryTotal;
  _memoryObjects = other->_memoryObjects;

  _executionTime = other->_executionTime;
  _buildTime = other->_buildTime;

  for (auto v : other->_profilesById) {
    nodeById(v.first, v.second->name().c_str())->assign(v.second);
  }
}

bool GraphProfile::nodeExists(int id) { return _profilesById.count(id) > 0; }

void GraphProfile::printOut() {
  sd_printf("Graph profile: %i executions\n", _merges);
  sd_printf("\nMemory:\n", "");

  sd::LongType tmp = 0L;
  sd::LongType obj = 0L;
  sd::LongType act = 0L;
  sd::LongType ttl = 0L;
  for (auto v : _profiles) {
    tmp += v->getTemporarySize();
    obj += v->getObjectsSize();
    act += v->getActivationsSize();
    ttl += v->getTotalSize();
  }

  sd_printf("ACT: %lld; TMP: %lld; OBJ: %lld; TTL: %lld;\n", act / _merges, tmp / _merges, obj / _merges,
            ttl / _merges);

  sd_printf("\nTime:\n", "");
  sd_printf("Construction time: %lld ns;\n", _buildTime / _merges);
  sd_printf("Execution time: %lld ns;\n", _executionTime / _merges);

  sd_printf("\nPer-node reports:\n", "");
  if (_profiles.empty()) sd_printf("No nodes in graph\n", "");

  // printint out stuff
  std::vector<NodeProfile *> sorted;
  for (auto v : _profiles) {
    v->printOut();
    sorted.emplace_back(v);
  }

  if (_profiles.size() > 1) {
    // building hot spots
    std::sort(sorted.begin(), sorted.end(), [](const NodeProfile *a, const NodeProfile *b) -> bool {
      return a->getExecutionTime() > b->getExecutionTime();
    });

    sd_printf("\nTop 50 reports by EXEC:\n", "");
    auto limit = sd::math::sd_min<int>(50, sorted.size());
    for (int e = 0; e < limit; e++) {
      sorted[e]->printOut();
    }
  }

  sd_printf("\nSpecial timers:\n", "");
  if (_timings.empty()) sd_printf("No special timers were set\n", "");

  for (auto v : _timings) sd_printf("%s: %lld ns;\n", v.first.c_str(), v.second);
}
}  // namespace graph
}  // namespace sd
