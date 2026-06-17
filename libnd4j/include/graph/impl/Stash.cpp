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
#include <graph/Stash.h>
#include <array/ConstantShapeBuffer.h>
#include <unordered_set>

namespace std {
size_t hash<sd::graph::KeyPair>::operator()(const sd::graph::KeyPair &k) const {
  using std::hash;
  auto res = std::hash<std::string>()(k.name());
  res ^= std::hash<int>()(k.key()) + 0x9e3779b9 + (res << 6) + (res >> 2);
  return res;
}
}  // namespace std

namespace sd {
namespace graph {
KeyPair::KeyPair(int node, const char *name) {
  _node = node;
  _name = std::string(name);
}

bool KeyPair::operator<(const KeyPair &other) const {
  if (_node < other._node)
    return true;
  else if (_node > other._node)
    return false;
  else
    return _name < other._name;
}

Stash::Stash() {
  //
}

Stash::~Stash() {
  if (_handles.size() > 0) this->clear();
}

/*
bool sd::graph::Stash::checkStash(sd::graph::Block& block, const char *name) {
    return checkStash(block.getNodeId(), name);
}
 */

bool Stash::checkStash(int nodeId, const char *name) {
  KeyPair kp(nodeId, name);
  return _stash.count(kp) > 0;
}

/*
sd::NDArray* sd::graph::Stash::extractArray(sd::graph::Block& block, const char *name) {
    return extractArray(block.getNodeId(), name);
}
*/
NDArray *Stash::extractArray(int nodeId, const char *name) {
  KeyPair kp(nodeId, name);
  return _stash[kp];
}
/*
void sd::graph::Stash::storeArray(sd::graph::Block& block, const char *name, sd::NDArray *array) {
    storeArray(block.getNodeId(), name, array);
}
*/

void Stash::storeArray(int nodeId, const char *name, NDArray *array) {
  KeyPair kp(nodeId, name);
  _stash[kp] = array;

  // storing reference to delete it once it's not needed anymore
  _handles.push_back(array);
}

void Stash::clear() {
  // Use a set to track deleted pointers and prevent double-deletion
  std::unordered_set<NDArray*> deletedPtrs;
  for (auto v : _handles) {
    if (v == nullptr || deletedPtrs.find(v) != deletedPtrs.end()) {
      continue;
    }
    deletedPtrs.insert(v);

    // SAFETY CHECK: Validate the NDArray before deletion using ConstantShapeBuffer's magic number
    ConstantShapeBuffer* shapeBuffer = v->shapeInfoConstBuffer();
    if (shapeBuffer == nullptr) {
      sd_printf("Stash::clear: Skipping NDArray at %p with null shapeInfoConstBuffer\n", v);
      continue;
    }

    // Check magic number validity - this detects use-after-free and garbage pointers
    if (!shapeBuffer->isValid()) {
      sd_printf("Stash::clear: Skipping NDArray at %p with invalid ConstantShapeBuffer (magic check failed)\n", v);
      continue;
    }

    // Additional validation: check that shapeInfo values are reasonable
    LongType* shapeInfo = shapeBuffer->primary();
    if (shapeInfo != nullptr) {
      LongType rank = shapeInfo[0];
      if (rank < 0 || rank > SD_MAX_RANK) {
        sd_debug("Stash::clear: Skipping corrupted NDArray at %p with invalid rank %lld\n",
                  v, (long long)rank);
        continue;
      }
    }

    delete v;
  }

  _handles.clear();
  _stash.clear();
}
}  // namespace graph
}  // namespace sd
