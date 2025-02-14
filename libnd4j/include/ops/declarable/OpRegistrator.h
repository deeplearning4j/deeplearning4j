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

#ifndef LIBND4J_OPREGISTRATOR_H
#define LIBND4J_OPREGISTRATOR_H

#include <execution/Engine.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/PlatformHelperLegacy.h>

#include <mutex>
#include <unordered_map>
#include <vector>

// handlers part
#include <csignal>
#include <cstdlib>

#ifndef __JAVACPP_HACK__

namespace std {

template <>
class hash<std::pair<sd::LongType, samediff::Engine>> {
 public:
  size_t operator()(const std::pair<sd::LongType, samediff::Engine>& k) const;
};

template <>
class hash<std::pair<std::string, samediff::Engine>> {
 public:
  size_t operator()(const std::pair<std::string, samediff::Engine>& k) const;
};
};  // namespace std

#endif

namespace sd {
namespace ops {
/**
 *   This class provides runtime ops lookup, based on opName or opHash.
 *   To build lookup directory we use *_OP_IMPL macro, which puts static structs at compile time in .cpp files,
 *   so once binary is executed, static objects are initialized automatically, and we get list of all ops
 *   available at runtime via this singleton.
 *
 */
class SD_LIB_EXPORT OpRegistrator {
 private:
  static OpRegistrator* _INSTANCE;
  OpRegistrator() {
    sd_debug("OpRegistrator started\n", "");
  };

  SD_MAP_IMPL<LongType, std::string> _msvc;

  // pointers to our operations
  SD_MAP_IMPL<LongType, DeclarableOp*> _declarablesLD;
  SD_MAP_IMPL<std::string, DeclarableOp*> _declarablesD;
  std::vector<DeclarableOp*> _uniqueD;

  // pointers to platform-specific helpers
  SD_MAP_IMPL<std::pair<LongType, samediff::Engine>, platforms::PlatformHelper*> _helpersLH;
  SD_MAP_IMPL<std::pair<std::string, samediff::Engine>, platforms::PlatformHelper*> _helpersH;
  std::vector<platforms::PlatformHelper*> _uniqueH;


  std::mutex _locker;
  std::string _opsList;
  std::vector<OpExecTrace *> opexecTrace;

  bool isInit = false;
  bool isTrace = false;
 public:
  ~OpRegistrator();

  void purgeOpExecs();
  void registerOpExec(OpExecTrace *opExecTrace);
  std::vector<OpExecTrace *> * execTrace();

  static OpRegistrator& getInstance();

  static void exitHandler();
  static void sigIntHandler(int sig);
  static void sigSegVHandler(int sig);

  void updateMSVC(LongType newHash, std::string& oldName);

  template <typename T>
  std::string local_to_string(T value);
  const char* getAllCustomOperations();

  /**
   * This method registers operation in our registry, so we can use them later
   *
   * @param op
   */
  bool registerOperation(const char* name, DeclarableOp* op);
  bool registerOperation(DeclarableOp* op);
  bool traceOps();
  void toggleTraceOps(bool traceOps);
  void registerHelper(platforms::PlatformHelper* op);


  bool hasHelper(LongType hash, samediff::Engine engine);

  DeclarableOp* getOperation(const char* name);
  DeclarableOp* getOperation(LongType hash);
  DeclarableOp* getOperation(std::string& name);

  platforms::PlatformHelper* getPlatformHelper(LongType hash, samediff::Engine engine);

  std::vector<LongType> getAllHashes();

  int numberOfOperations();
};

/*
 *  These structs are used to "register" our ops in OpRegistrator.
 */
template <typename OpName>
struct __registrator {
  __registrator();
};

template <typename OpName>
struct __registratorSynonym {
  __registratorSynonym(const char* name, const char* oname);
};

}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_OPREGISTRATOR_H
