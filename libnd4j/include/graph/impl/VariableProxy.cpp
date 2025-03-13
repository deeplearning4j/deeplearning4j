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

#include <graph/VariableProxy.h>

namespace sd {
namespace graph {

VariableProxy::VariableProxy(VariableSpace *ref) {
  if (ref == nullptr) _backed = new VariableSpace();

  _backed = ref;
  _current = new VariableSpace();
}

VariableProxy::~VariableProxy() {
  delete _current;
}

int VariableProxy::numberOfPlaceholders() { return _backed->numberOfPlaceholders(); }

std::vector<Variable *> *VariableProxy::getPlaceholders() { return _backed->getPlaceholders(); }

bool VariableProxy::hasExternalVariable(int it) { return _backed->hasExternalVariable(it); }

bool VariableProxy::hasExternalVariable(std::pair<int, int> &pair) { return _backed->hasExternalVariable(pair); }

bool VariableProxy::hasExternalVariable(std::string *symbol) { return _backed->hasExternalVariable(symbol); }

bool VariableProxy::hasVariable(int id) { return _current->hasVariable(id) || _backed->hasVariable(id); }

bool VariableProxy::hasVariable(int id, int idx) {
  return _current->hasVariable(id, idx) || _backed->hasVariable(id, idx);
}

bool VariableProxy::hasVariable(std::pair<int, int> &pair) {
  return _current->hasVariable(pair) || _backed->hasVariable(pair);
}

void VariableProxy::dropVariable(std::pair<int, int> &pair) { dropVariable(pair.first, pair.second); }

void VariableProxy::dropVariable(int id, int idx) {
  assert(_current->hasVariable(id, idx));

  _current->dropVariable(id, idx);
}

std::vector<Variable *> VariableProxy::getVariables() {
  std::vector<Variable *> result;

  auto b = _backed->getVariables();
  auto c = _current->getVariables();

  for (auto v : b) result.emplace_back(v);

  for (auto v : c) result.emplace_back(v);

  return result;
}

bool VariableProxy::hasVariable(std::string *symbol) {
  return _current->hasVariable(symbol) || _backed->hasVariable(symbol);
}

Variable *VariableProxy::getVariable(int id) {
  if (_current->hasVariable(id)) return _current->getVariable(id);

  if (_backed->hasVariable(id)) return _backed->getVariable(id);

  sd_printf("Unable to get Variable to proxy: [%i]\n", id);
  THROW_EXCEPTION("Bad arguments");
  return nullptr;
}

Variable *VariableProxy::getVariable(int id, int idx) {
  if (_current->hasVariable(id, idx)) return _current->getVariable(id, idx);

  if (_backed->hasVariable(id, idx)) return _backed->getVariable(id, idx);

  sd_printf("Unable to get Variable to proxy: [%i:%i]\n", id, idx);
  THROW_EXCEPTION("Bad arguments");
  return nullptr;
}

Variable *VariableProxy::getVariable(std::pair<int, int> &pair) {
  if (_current->hasVariable(pair)) return _current->getVariable(pair);

  if (_backed->hasVariable(pair)) return _backed->getVariable(pair);

  sd_printf("Unable to get Variable to proxy: [%i:%i]\n", pair.first, pair.second);
  THROW_EXCEPTION("Bad arguments");
  return nullptr;
}

Variable *VariableProxy::getVariable(std::string *symbol) {
  if (_current->hasVariable(symbol)) return _current->getVariable(symbol);

  if (_backed->hasVariable(symbol)) return _backed->getVariable(symbol);

  sd_printf("Unable to get Variable to proxy: [%s]\n", symbol->c_str());
  THROW_EXCEPTION("Bad arguments");
  return nullptr;
}

void VariableProxy::replaceVariable(Variable *variable) {
  if (variable->getName() != nullptr && !variable->getName()->empty()) {
    // if variable has name defined - we should resolve it via backing var space
    if (_backed->hasVariable(variable->getName())) {
      auto origVar = _backed->getVariable(variable->getName());
      variable->setId(origVar->id(), origVar->index());
      _current->replaceVariable(variable);
    } else
      _current->replaceVariable(variable);
  } else  // if proxy has variable - that's one story
    _current->replaceVariable(variable);
}

Variable *VariableProxy::putVariable(std::pair<int, int> &pair, NDArray *array) {
  return _current->putVariable(pair, array);
}

void VariableProxy::putVariable(std::pair<int, int> &pair, Variable *variable) {
  _current->putVariable(pair, variable);
}

void VariableProxy::putVariable(int id, Variable *variable) { _current->putVariable(id, variable); }

void VariableProxy::putVariable(int id, NDArray *array) { _current->putVariable(id, array); }

void VariableProxy::putVariable(int id, int idx, NDArray &array) { _current->putVariable(id, idx, array); }

Variable *VariableProxy::putVariable(int id, int idx, NDArray *array) { return _current->putVariable(id, idx, array); }

void VariableProxy::putVariable(int id, int idx, Variable *array) { _current->putVariable(id, idx, array); }

void VariableProxy::trackList(NDArrayList *list) { _current->trackList(list); }

Stash *VariableProxy::getStash() { return _current->getStash(); }

void VariableProxy::setFlowPath(FlowPath *timers) { _current->setFlowPath(timers); }

FlowPath *VariableProxy::flowPath() { return _current->flowPath(); }

void VariableProxy::putOutputVariable(Variable *variable) { _current->putOutputVariable(variable); }

LongType VariableProxy::externalMemory() { return _backed->externalMemory() + _current->externalMemory(); }

LongType VariableProxy::internalMemory() { return _backed->internalMemory() + _current->internalMemory(); }

LongType VariableProxy::totalMemory() { return _backed->totalMemory() + _current->totalMemory(); }

int VariableProxy::externalEntries() { return _backed->externalEntries() + _current->externalEntries(); }

int VariableProxy::internalEntries() { return _backed->internalEntries() + _current->internalEntries(); }

int VariableProxy::totalEntries() { return _backed->totalEntries() + _current->totalEntries(); }

VariableSpace *VariableProxy::clone() {
  auto clone = new VariableProxy(_backed);

  delete clone->_current;
  clone->_current = _current->clone();

  return clone;
}

VariableSpace &VariableProxy::operator=(const VariableSpace &other) {
  if (this == &other) return *this;

  sd_printf("VariableProxy = not implemented\n", "");

  return *this;
}

memory::Workspace *VariableProxy::workspace() { return _workspace; }
}  // namespace graph
}  // namespace sd
