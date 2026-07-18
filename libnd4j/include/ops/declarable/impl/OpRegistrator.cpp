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
// Created by raver119 on 07.10.2017.
//

#include <ops/declarable/OpRegistrator.h>

#include <mutex>
#include <sstream>

namespace sd {
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN

///////////////////////////////

template <typename OpName>
__registrator<OpName>::__registrator() {
  auto ptr = new OpName();
  OpRegistrator::getInstance().registerOperation(ptr);
}

template <typename OpName>
__registratorSynonym<OpName>::__registratorSynonym(const char* name, const char* oname) {
  auto ptr = reinterpret_cast<OpName*>(OpRegistrator::getInstance().getOperation(oname));
  if (ptr == nullptr) {
    std::string newName(name);
    std::string oldName(oname);

    OpRegistrator::getInstance().updateMSVC(HashHelper::getInstance().getLongHash(newName), oldName);
    return;
  }
  OpRegistrator::getInstance().registerOperation(name, ptr);
}

///////////////////////////////

OpRegistrator& OpRegistrator::getInstance() {
  static OpRegistrator* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new OpRegistrator();
  });
  return *instance;
}

void OpRegistrator::updateMSVC(LongType newHash, std::string& oldName) {
  std::pair<LongType, std::string> pair(newHash, oldName);
  _msvc.insert(pair);
}

template <typename T>
std::string OpRegistrator::local_to_string(T value) {
  // create an output string stream
  std::ostringstream os;

  // throw the value into the string stream
  os << value;

  // convert the string stream into a string and return
  return os.str();
}

template <>
std::string OpRegistrator::local_to_string(int value) {
  // create an output string stream
  std::ostringstream os;

  // throw the value into the string stream
  os << value;

  // convert the string stream into a string and return
  return os.str();
}

OpRegistrator::~OpRegistrator() {
  // Clean up OpExecTrace objects - must happen in both debug and release builds
  for (auto trace : opexecTrace) {
    delete trace;
  }
  opexecTrace.clear();

#ifndef _RELEASE
  _msvc.clear();

  for (auto x : _uniqueD) delete x;

  for (auto x : _uniqueH) delete x;

  _uniqueD.clear();

  _uniqueH.clear();

  _declarablesD.clear();

  _declarablesLD.clear();
#endif
}

const char* OpRegistrator::getAllCustomOperations() {
  _locker.lock();

  if (!isInit) {
    for (SD_MAP_IMPL<std::string, DeclarableOp*>::iterator it = _declarablesD.begin();
         it != _declarablesD.end(); ++it) {
      std::string op = it->first + ":" + local_to_string(it->second->getOpDescriptor()->getHash()) + ":" +
                       local_to_string(it->second->getOpDescriptor()->getNumberOfInputs()) + ":" +
                       local_to_string(it->second->getOpDescriptor()->getNumberOfOutputs()) + ":" +
                       local_to_string(it->second->getOpDescriptor()->allowsInplace()) + ":" +
                       local_to_string(it->second->getOpDescriptor()->getNumberOfTArgs()) + ":" +
                       local_to_string(it->second->getOpDescriptor()->getNumberOfOrdinaryIArgs()) + ":" + ";";
      _opsList += op;
    }

    isInit = true;
  }

  _locker.unlock();

  return _opsList.c_str();
}

bool OpRegistrator::registerOperation(const char* name, DeclarableOp* op) {
  op->initializeDescriptor();
  std::string str(name);
  std::pair<std::string, DeclarableOp*> pair(str, op);
  _declarablesD.insert(pair);

  auto hash = HashHelper::getInstance().getLongHash(str);
  std::pair<LongType, DeclarableOp*> pair2(hash, op);
  _declarablesLD.insert(pair2);
  return true;
}

void OpRegistrator::registerOpExec(OpExecTrace *opExecTrace) {
  this->opexecTrace.push_back(opExecTrace);
}

bool OpRegistrator::traceOps() {
  return this->isTrace;
}

void OpRegistrator::toggleTraceOps(bool traceOps) {
  this->isTrace = traceOps;
}

void OpRegistrator::purgeOpExecs() {
  for (auto trace : this->opexecTrace) {
    delete trace;
  }
  this->opexecTrace.clear();
}

std::vector<OpExecTrace *>  * OpRegistrator::execTrace() {
  return &(this->opexecTrace);
}

/**
 * This method registers operation
 *
 * @param op
 */
bool OpRegistrator::registerOperation(DeclarableOp* op) {
  _uniqueD.emplace_back(op);
  return registerOperation(op->getOpName()->c_str(), op);
}

void OpRegistrator::registerHelper(platforms::PlatformHelper* op) {
  std::pair<LongType, samediff::Engine> p = {op->hash(), op->engine()};
  if (_helpersLH.count(p) > 0) THROW_EXCEPTION("Tried to double register PlatformHelper");

  _uniqueH.emplace_back(op);

  sd_debug("Adding helper for op \"%s\": [%lld - %i]\n", op->name().c_str(), op->hash(), (int)op->engine());

  std::pair<std::pair<std::string, samediff::Engine>, platforms::PlatformHelper*> pair(
      {op->name(), op->engine()}, op);
  _helpersH.insert(pair);

  std::pair<std::pair<LongType, samediff::Engine>, platforms::PlatformHelper*> pair2(p, op);
  _helpersLH.insert(pair2);
}



DeclarableOp* OpRegistrator::getOperation(const char* name) {
  std::string str(name);
  return getOperation(str);
}

/**
 * This method returns registered Op by name
 *
 * @param name
 * @return
 */
DeclarableOp* OpRegistrator::getOperation(LongType hash) {
  // All accesses to _declarablesLD must be under the lock because concurrent
  // insert() can trigger a rehash, invalidating iterators/buckets that a
  // lockless count()/at() is traversing → heap corruption.
  _locker.lock();

  auto it = _declarablesLD.find(hash);
  if (it != _declarablesLD.end()) {
    auto* op = it->second;
    _locker.unlock();
    return op;
  }

  // Not in the fast-lookup map — check the string-keyed map
  if (!_msvc.count(hash)) {
    _locker.unlock();
    sd_printf("Unknown D operation requested by hash: [%lld]\n", hash);
    return nullptr;
  }

  auto str = _msvc.at(hash);
  auto op = _declarablesD.at(str);
  auto oHash = op->getOpDescriptor()->getHash();

  std::pair<LongType, DeclarableOp*> pair(oHash, op);
  _declarablesLD.insert(pair);

  _locker.unlock();
  return op;
}

DeclarableOp* OpRegistrator::getOperation(std::string& name) {
  if (!_declarablesD.count(name)) {
    sd_debug("Unknown operation requested: [%s]\n", name.c_str());
    return nullptr;
  }

  return _declarablesD.at(name);
}

platforms::PlatformHelper* OpRegistrator::getPlatformHelper(LongType hash, samediff::Engine engine) {
  std::pair<LongType, samediff::Engine> p = {hash, engine};
  if (_helpersLH.count(p) == 0) THROW_EXCEPTION("Requested helper can't be found");

  return _helpersLH[p];
}


bool OpRegistrator::hasHelper(LongType hash, samediff::Engine engine) {
  std::pair<LongType, samediff::Engine> p = {hash, engine};
  return _helpersLH.count(p) > 0;
}

bool OpRegistrator::hasAnyHelper(LongType hash) {
  for (const auto& entry : _helpersLH) {
    if (entry.first.first == hash) {
      return true;
    }
  }
  return false;
}

std::vector<platforms::PlatformHelper*> OpRegistrator::getAllHelpersForOp(LongType hash) {
  std::vector<platforms::PlatformHelper*> result;

  for (const auto& entry : _helpersLH) {
    if (entry.first.first == hash) {
      result.push_back(entry.second);
    }
  }

  return result;
}

std::vector<samediff::Engine> OpRegistrator::getAvailableEnginesForOp(LongType hash) {
  std::vector<samediff::Engine> result;

  for (const auto& entry : _helpersLH) {
    if (entry.first.first == hash) {
      result.push_back(entry.first.second);
    }
  }

  return result;
}

int OpRegistrator::numberOfOperations() { return (int)_declarablesLD.size(); }

std::vector<std::string> OpRegistrator::getAllRegisteredOpNames() {
  std::vector<std::string> result;

  for (const auto& entry : _declarablesD) {
    result.push_back(entry.first);
  }

  return result;
}

std::vector<LongType> OpRegistrator::getAllHashes() {
  std::vector<LongType> result;

  for (auto& v : _declarablesLD) {
    result.emplace_back(v.first);
  }

  return result;
}
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd

namespace std {
size_t hash<std::pair<sd::LongType, samediff::Engine>>::operator()(
    const std::pair<sd::LongType, samediff::Engine>& k) const {
  using std::hash;
  auto res = std::hash<sd::LongType>()(k.first);
  res ^= std::hash<sd::LongType>()((sd::LongType)k.second) + 0x9e3779b9 + (res << 6) + (res >> 2);
  return res;
}

size_t hash<std::pair<std::string, samediff::Engine>>::operator()(
    const std::pair<std::string, samediff::Engine>& k) const {
  using std::hash;
  auto res = std::hash<std::string>()(k.first);
  res ^= std::hash<sd::LongType>()((sd::LongType)k.second) + 0x9e3779b9 + (res << 6) + (res >> 2);
  return res;
}
}  // namespace std
