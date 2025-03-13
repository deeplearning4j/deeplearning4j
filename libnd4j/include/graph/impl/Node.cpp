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
#include <array/NDArrayFactory.h>
#include <graph/FlatUtils.h>
#include <graph/Node.h>
#include <ops/declarable/LegacyBroadcastBoolOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyOp.h>
#include <ops/declarable/LegacyPairwiseTransformBoolOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/LegacyReduce3Op.h>
#include <ops/declarable/LegacyReduceBoolOp.h>
#include <ops/declarable/LegacyReduceFloatOp.h>
#include <ops/declarable/LegacyReduceLongOp.h>
#include <ops/declarable/LegacyReduceSameOp.h>
#include <ops/declarable/LegacyScalarBoolOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyStatsOp.h>
#include <ops/declarable/LegacyTransformBoolOp.h>
#include <ops/declarable/LegacyTransformFloatOp.h>
#include <ops/declarable/LegacyTransformSameOp.h>
#include <ops/declarable/LegacyTransformStrictOp.h>
#include <ops/declarable/OpRegistrator.h>

namespace sd {
namespace graph {
void sd::graph::Node::setOuterTime(sd::LongType time) {
  //            if (hasBlockAttached())
  //                _block->setOuterTime(time);
}

void sd::graph::Node::setInnerTime(sd::LongType time) {
  //            if (hasBlockAttached())
  //                _block->setInnerTime(time);
}

void sd::graph::Node::setGraph(sd::graph::Graph* graph) { _graph = graph; }

sd::graph::Graph* sd::graph::Node::getGraph() { return _graph; }

bool sd::graph::Node::hasGraphEmbedded() { return _graph != nullptr; }

void sd::graph::Node::markInplace(bool reallyInplace) {
  _isInplace = reallyInplace;
  if (_protoContext != nullptr) {
    _protoContext->markInplace(reallyInplace);
  }
}

::graph::OpClass sd::graph::Node::getOpClass() { return _opClass; }

bool sd::graph::Node::hasBlockAttached() { return _protoContext != nullptr; }

bool sd::graph::Node::isInplace() { return _isInplace; }

bool sd::graph::Node::isDivergencePoint() {
  if (hasCustomOp()) {
    return _customOp->getOpDescriptor()->isDivergent();
  } else if (opType() == ::graph::OpType_LOGIC && opNum() == 30)
    return true;
  else
    return false;
}

void sd::graph::Node::setActive(bool reallyActive) { _active = reallyActive; }

bool sd::graph::Node::isActive() { return _active; }

sd::LongType Node::getFrameId() { return _frameId; }

void Node::setFrameId(sd::LongType frameId) { _frameId = frameId; }

ContextPrototype* sd::graph::Node::getContextPrototype() {
  if (_protoContext == nullptr)
    _protoContext = new ContextPrototype(
        this->getCustomOp() != nullptr ? this->getCustomOp()->getOpDescriptor() : nullptr, this->id());
  if (_protoContext->inputs()->empty()) {
    for (size_t e = 0; e < this->input()->size(); e++) {
      _protoContext->inputs()->emplace_back(this->input()->at(e));
    }
  }
  return _protoContext;
}

void sd::graph::Node::setContextPrototype(ContextPrototype* block) {
  if (_protoContext != nullptr) THROW_EXCEPTION("Block already exists");

  _protoContext = block;
}

void sd::graph::Node::setId(int id) { _id = id; }

sd::ops::DeclarableOp* sd::graph::Node::getCustomOp() { return _customOp; }

void sd::graph::Node::setCustomOp(sd::ops::DeclarableOp* customOp) {
  _customOp = customOp;

  // divergent ops (Switch etc) are always inplace, they don't allocate anything
  if (_customOp != nullptr && customOp->getOpDescriptor()->isDivergent()) _isInplace = true;
}

bool sd::graph::Node::hasCustomOp() { return _customOp != nullptr; }

std::string* sd::graph::Node::name() { return this->getName(); }

std::string* sd::graph::Node::getName() { return &_name; }

void sd::graph::Node::setName(const std::string& name) { _name = name.c_str(); }

void sd::graph::Node::setName(std::string* name) { _name = *name; }

;

void sd::graph::Node::pickInput(std::pair<int, int>& pair) { _input.push_back(pair); }

void sd::graph::Node::pickInput(int inputId, int outputId) {
  std::pair<int, int> p(inputId, outputId);
  pickInput(p);
}

void sd::graph::Node::pickInput(int inputId) {
  pickInput(inputId, 0);

  if (inputId < 0)
    _hasExternalInputs = true;
  else
    _hasInternalInputs = true;
}

void sd::graph::Node::pickExternalOutput(int outputId) {
  std::pair<int, int> pair(outputId, 0);
  _output.push_back(pair);

  _hasExternalOutputs = true;
}

void sd::graph::Node::pickOutputOnce(int outputId) {
  std::pair<int, int> pair(outputId, 0);
  if (std::find(_output.begin(), _output.end(), pair) == _output.end()) pickOutput(outputId);
}

void sd::graph::Node::pickOutput(int nodeId, int outputId) {
  std::pair<int, int> pair(nodeId, outputId);
  _output.emplace_back(pair);
}

void sd::graph::Node::pickOutput(int outputId) {
  std::pair<int, int> pair(outputId, 0);
  _output.emplace_back(pair);

  if (outputId < 0)
    _hasExternalOutputs = true;
  else
    _hasInternalOutputs = true;
}

sd::LongType* sd::graph::Node::getDimensionsPtr() { return _dim; }

std::vector<sd::LongType>* sd::graph::Node::getDimensions() { return &_dimensions; }

int sd::graph::Node::getLayer() { return _layer; }

void sd::graph::Node::setLayer(int layer) { _layer = layer; }

bool sd::graph::Node::hasExternalOutputs() { return _hasExternalOutputs; }

bool sd::graph::Node::hasExternalInputs() { return _hasExternalInputs; }

bool sd::graph::Node::hasInternalOutputs() { return _hasInternalOutputs; }

bool sd::graph::Node::hasInternalInputs() { return _hasInternalInputs; }

bool sd::graph::Node::isMultiInput() { return _input.size() > 1; }

bool sd::graph::Node::isMultiOutput() { return _output.size() > 1; }

double* sd::graph::Node::extraParams() { return _extraParams; }

int Node::totalReferences() { return _referencedBy.size(); }

void Node::addReference(int nodeId) { _referencedBy.emplace_back(nodeId); }

::graph::OpType sd::graph::Node::opType() { return _opType; }

int sd::graph::Node::id() { return _id; }

sd::LongType sd::graph::Node::opNum() { return _opNum; }

std::vector<std::pair<int, int>>* sd::graph::Node::input() { return &_input; }

std::vector<std::pair<int, int>>* sd::graph::Node::output() { return &_output; }

bool Node::isScoped() { return _scope_id != 0; }

void Node::setScopeInfo(int id, const char* name) {
  _scope_id = id;

  if (name != nullptr) _scope_name = name;
}

int Node::scopeId() { return _scope_id; }

std::string* Node::scopeName() { return &_scope_name; }

template <typename T>
Node* Node::asT() {
  auto node = this->clone();
  node->_dataType = DataTypeUtils::fromT<T>();
  return node;
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT Node* Node::asT, (), SD_COMMON_TYPES);

sd::graph::Node::Node(sd::ops::DeclarableOp* customOp, int id, std::initializer_list<int> input,
                      std::initializer_list<int> output, std::initializer_list<int> dimensions, float scalar,
                      std::initializer_list<double> tArgs, std::initializer_list<int> iArgs) {
  this->_opType = ::graph::OpType_CUSTOM;
  this->_id = id;
  this->_opNum = customOp->getOpHash();
  this->_extraParams = nullptr;
  this->_dataType = sd::DataType::FLOAT32;  // float as default
  this->_dim = nullptr;
  this->_customOp = customOp;

  _hasExternalInputs = false;
  _hasExternalOutputs = false;
  _hasInternalInputs = false;
  _hasInternalOutputs = false;


  for (auto i : input) pickInput(i);

  for (auto o : output) pickOutput(o);

  if (dimensions.size() > 0) {
    _dim = new sd::LongType[dimensions.size()];
    int cnt = 0;
    for (auto d : dimensions) {
      _dimensions.push_back(d);
      _dim[cnt++] = d;
    }
  }

  auto block = new ContextPrototype(this->getCustomOp()->getOpDescriptor(), this->id(), false);

  for (auto v : dimensions) block->getAxis()->emplace_back(v);

  for (auto v : iArgs) block->getIArguments()->emplace_back(v);

  for (auto v : tArgs) block->getTArguments()->emplace_back(v);

  this->setContextPrototype(block);
}

void sd::graph::Node::setOpType(::graph::OpType opType) { this->_opType = opType; }

sd::graph::Node::Node(::graph::OpType opType, int opNum, int id, std::initializer_list<int> input,
                      std::initializer_list<int> output, std::initializer_list<int> dimensions, float scalar,
                      std::initializer_list<double> tArgs, std::initializer_list<int> iArgs) {
  this->_opType = opType;
  this->_id = id;
  this->_opNum = opNum;
  this->_extraParams = nullptr;
  this->_dataType = sd::DataType::FLOAT32;  // float as default
  this->_dim = nullptr;

  _hasExternalInputs = false;
  _hasExternalOutputs = false;
  _hasInternalInputs = false;
  _hasInternalOutputs = false;


  for (auto i : input) pickInput(i);

  for (auto o : output) pickOutput(o);

  if (dimensions.size() > 0) {
    _dim = new sd::LongType[dimensions.size()];
    int cnt = 0;
    for (auto d : dimensions) {
      _dimensions.push_back(d);
      _dim[cnt++] = d;
    }
  }

  // these ops allow in-place execution by design
  if (opType == ::graph::OpType_TRANSFORM_SAME || opType == ::graph::OpType_TRANSFORM_FLOAT || opType == ::graph::OpType_TRANSFORM_STRICT ||
      opType == ::graph::OpType_TRANSFORM_BOOL || opType == ::graph::OpType_SCALAR || opType == ::graph::OpType_BROADCAST) {
    if (_output.size() <= 1) {
      _isInplace = true;
    }
    _opClass = ::graph::OpClass_TRANSFORM;
  } else if (opType == ::graph::OpType_REDUCE_SAME || opType == ::graph::OpType_REDUCE_FLOAT || opType == ::graph::OpType_REDUCE_BOOL ||
             opType == ::graph::OpType_REDUCE_LONG || opType == ::graph::OpType_SUMMARYSTATS) {
    _opClass = ::graph::OpClass_REDUCTION;
  }

  if (opType == ::graph::OpType_BROADCAST || opType == ::graph::OpType_BROADCAST_BOOL || opType == ::graph::OpType_INDEX_REDUCE ||
      opType == ::graph::OpType_SUMMARYSTATS || opType == ::graph::OpType_REDUCE_BOOL || opType == ::graph::OpType_REDUCE_SAME ||
      opType == ::graph::OpType_REDUCE_FLOAT || opType == ::graph::OpType_REDUCE_3 || opType == ::graph::OpType_TRANSFORM_STRICT ||
      opType == ::graph::OpType_TRANSFORM_SAME || opType == ::graph::OpType_TRANSFORM_FLOAT || opType == ::graph::OpType_TRANSFORM_BOOL ||
      opType == ::graph::OpType_RANDOM || opType == ::graph::OpType_PAIRWISE || opType == ::graph::OpType_PAIRWISE_BOOL ||
      opType == ::graph::OpType_SCALAR_BOOL || opType == ::graph::OpType_SCALAR) {
    this->_isDeductable = true;

    auto block = new ContextPrototype(nullptr, this->id(), false);

    for (auto v : dimensions) block->getAxis()->emplace_back(v);

    for (auto v : iArgs) block->getIArguments()->emplace_back(v);

    for (auto v : tArgs) block->getTArguments()->emplace_back(v);
    NDArray _scalar = NDArrayFactory::create(0.0f);

    this->setContextPrototype(block);
    this->setCustomOp(Node::buildOpByType(opType, (int)input.size(), (int)block->getIArguments()->size(),
                                          (int)block->getTArguments()->size(), opNum, &_scalar));
    block->setOpDescriptor(this->getCustomOp()->getOpDescriptor());
  } else if (opType == ::graph::OpType_CUSTOM) {
    if (this->getCustomOp()) {
      auto block = new ContextPrototype(this->getCustomOp()->getOpDescriptor(), this->id(), false);

      for (auto v : dimensions) block->getAxis()->emplace_back(v);

      for (auto v : iArgs) block->getIArguments()->emplace_back(v);

      for (auto v : tArgs) block->getTArguments()->emplace_back(v);

      this->setContextPrototype(block);
    } else
      THROW_EXCEPTION("wrong custom operation given");
  }
};

sd::graph::Node::Node(const ::graph::FlatNode* node) {
  _hasExternalInputs = false;
  _hasExternalOutputs = false;
  _hasInternalInputs = false;
  _hasInternalOutputs = false;
  _extraParams = nullptr;
  _dim = nullptr;
  _dataType = sd::DataType::FLOAT32;  // float as default
  if (node->scope_id() != 0) this->_scope_id = node->scope_id();

  if (node->scope_name() != nullptr && node->scope_name()->size() > 0) this->_scope_name = node->scope_name()->str();



  if (node != nullptr) {
    this->_id = node->id();
    // this->_dataType = DataTypeUtils::fromFlatDataType(node->dataType());
    this->_opNum = node->opNum();
    this->_opType = node->opType();

    if (node->name() != nullptr && node->name()->c_str() != nullptr) {
      this->_name = node->name()->str();
    }

    if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
      for (int e = 0; e < (int)node->inputPaired()->size(); e++) {
        auto pair = node->inputPaired()->Get(e);
        pickInput(pair->first(), pair->second());
      }
    } else if (node->input() != nullptr && node->input()->size() > 0) {
      for (int e = 0; e < (int)node->input()->size(); e++) pickInput(node->input()->Get(e));
    } else {
      if (this->opType() != ::graph::OpType_LOGIC) {
        if (this->_name.size() > 0) {
          sd_debug("Node [%i:<%s>] has no inputs defined\n", this->_id, this->_name.c_str());
        } else {
          sd_debug("Node [%i:<noname>] has no inputs defined\n", this->_id);
        }
      }
    }


    if (node->extraParams() != nullptr && node->extraParams()->size() > 0) {
      _extraParams = new double[node->extraParams()->size()];
      for (int e = 0; e < (int)node->extraParams()->size(); e++) {
        _extraParams[e] = static_cast<double>(node->extraParams()->Get(e));
      }
    }

    if (node->dimensions() != nullptr && node->dimensions()->size() > 0) {
      _dim = new sd::LongType [node->dimensions()->size()];
      for (int e = 0; e < (int)node->dimensions()->size(); e++) {
        _dimensions.emplace_back(node->dimensions()->Get(e));
        _dim[e] = node->dimensions()->Get(e);
      }
    }

    if (this->opType() == ::graph::OpType_LOGIC && this->opNum() == 100L) {
      if (node->extraInteger()->size() < 1) {
        sd_printf("Node_%i is type of Enter, but has no FrameID defined\n", this->id());
        THROW_EXCEPTION("Enter node must have FrameID specified");
      }

      this->setFrameId(node->extraInteger()->Get(0));
    }

    // these ops allow in-place execution by design
    if (_opType == ::graph::OpType_BROADCAST || _opType == ::graph::OpType_BROADCAST_BOOL || _opType == ::graph::OpType_INDEX_REDUCE ||
        _opType == ::graph::OpType_SUMMARYSTATS || _opType == ::graph::OpType_REDUCE_BOOL || _opType == ::graph::OpType_REDUCE_SAME ||
        _opType == ::graph::OpType_REDUCE_FLOAT || _opType == ::graph::OpType_REDUCE_3 || _opType == ::graph::OpType_TRANSFORM_STRICT ||
        _opType == ::graph::OpType_TRANSFORM_SAME || _opType == ::graph::OpType_TRANSFORM_FLOAT || _opType == ::graph::OpType_TRANSFORM_BOOL ||
        _opType == ::graph::OpType_RANDOM || _opType == ::graph::OpType_PAIRWISE || _opType == ::graph::OpType_PAIRWISE_BOOL ||
        _opType == ::graph::OpType_SCALAR_BOOL || _opType == ::graph::OpType_SCALAR) {
      if (_output.size() <= 1) {
        _isInplace = true;
      }

      if (node->input() != nullptr && node->input()->size() > 0) {
        this->_isDeductable = true;

        auto block = new ContextPrototype(nullptr, this->id(), false);

        for (auto v : _dimensions) block->getAxis()->emplace_back(v);

        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
          for (int e = 0; e < (int)node->extraParams()->size(); e++) {
            block->getTArguments()->emplace_back(static_cast<double>(node->extraParams()->Get(e)));
          }

        if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
          for (int e = 0; e < (int)node->extraBools()->size(); e++) {
            block->getBArguments()->push_back(node->extraBools()->Get(e));
          }

        if (node->extraInteger() != nullptr && node->extraInteger()->size() > 0)
          for (int e = 0; e < (int)node->extraInteger()->size(); e++) {
            block->getIArguments()->emplace_back(node->extraInteger()->Get(e));
          }

        if (node->extraTypes() != nullptr && node->extraTypes()->size() > 0) {
          for (int e = 0; e < (int)node->extraTypes()->size(); e++) {
            block->getDArguments()->emplace_back((sd::DataType)node->extraTypes()->Get(e));
          }
        }

        NDArray _scalar = NDArrayFactory::create(0.0f);
        this->setContextPrototype(block);
        this->setCustomOp(Node::buildOpByType(_opType, (int)node->input()->size(), (int)block->getIArguments()->size(),
                                              (int)block->getTArguments()->size(), (int)_opNum, &_scalar));
        block->setOpDescriptor(this->getCustomOp()->getOpDescriptor());
      } else if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
        this->_isDeductable = true;

        auto block = new ContextPrototype(nullptr, this->id(), false);

        for (size_t e = 0; e < this->input()->size(); e++) {
          block->inputs()->emplace_back(this->input()->at(e));
        }

        // there's no other IArgs in legacy options, actually
        for (auto v : _dimensions) block->getAxis()->emplace_back(v);

        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
          for (int e = 0; e < (int)node->extraParams()->size(); e++) {
            block->getTArguments()->emplace_back(static_cast<double>(node->extraParams()->Get(e)));
          }

        if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
          for (int e = 0; e < (int)node->extraBools()->size(); e++) {
            block->getBArguments()->push_back(node->extraBools()->Get(e));
          }

        if (node->extraInteger() != nullptr && node->extraInteger()->size() > 0)
          for (int e = 0; e < (int)node->extraInteger()->size(); e++) {
            block->getIArguments()->emplace_back(node->extraInteger()->Get(e));
          }

        if (node->extraTypes() != nullptr && node->extraTypes()->size() > 0) {
          for (int e = 0; e < (int)node->extraTypes()->size(); e++) {
            block->getDArguments()->emplace_back((sd::DataType)node->extraTypes()->Get(e));
          }
        }

        this->setContextPrototype(block);
        NDArray _scalar = NDArrayFactory::create(0.0f);

        this->setCustomOp(Node::buildOpByType(_opType, (int)node->inputPaired()->size(),
                                              (int)block->getIArguments()->size(), (int)block->getTArguments()->size(),
                                              (int)_opNum, &_scalar));
        block->setOpDescriptor(this->getCustomOp()->getOpDescriptor());
      }
    } else if (this->_opType == ::graph::OpType_CUSTOM) {
      auto op = sd::ops::OpRegistrator::getInstance().getOperation(this->opNum());
      if (op == nullptr) {
        sd_verbose("Can't find operation: %lld\n", this->opNum());
        THROW_EXCEPTION("Can't find requested operation");
      }

      auto block = new ContextPrototype(nullptr, this->id());

      for (size_t e = 0; e < this->input()->size(); e++) {
        block->inputs()->emplace_back(this->input()->at(e));
      }

      if (node->extraInteger() != nullptr)
        for (uint32_t e = 0; e < node->extraInteger()->size(); e++) {
          auto v = node->extraInteger()->Get(e);
          // FIXME: remove this static_cast, iArgs should be sd::LongType
          block->getIArguments()->emplace_back(static_cast<int>(v));
        }

      if (node->extraParams() != nullptr)
        for (uint32_t e = 0; e < node->extraParams()->size(); e++)
          block->getTArguments()->emplace_back(static_cast<double>(node->extraParams()->Get(e)));

      if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
        for (int e = 0; e < (int)node->extraBools()->size(); e++) {
          block->getBArguments()->push_back(node->extraBools()->Get(e));
        }

      if (node->extraTypes() != nullptr && node->extraTypes()->size() > 0) {
        for (int e = 0; e < (int)node->extraTypes()->size(); e++) {
          block->getDArguments()->emplace_back((sd::DataType)node->extraTypes()->Get(e));
        }
      }

      for (auto v : _dimensions) block->getAxis()->emplace_back(v);

      this->setContextPrototype(block);
      this->setCustomOp(op);
      block->setOpDescriptor(this->getCustomOp()->getOpDescriptor());
    }
  } else {
    // empty dynamic node, tests probably
  }
}

sd::DataType Node::dataType() { return _dataType; }

ContextPrototype* Node::protoContext() { return _protoContext; }

sd::graph::Node::~Node() {
  if (_extraParams != nullptr) delete[] _extraParams;

  if (_dim != nullptr) delete[] _dim;


  if (_isDeductable && _customOp != nullptr) {
    Node::deleteOpByType(_opType, _customOp);
  }
}

int sd::graph::Node::getRewindNode() { return _rewindNode; }

void sd::graph::Node::setRewindNode(int nodeId) { _rewindNode = nodeId; }

std::pair<int, int>& sd::graph::Node::getRewindLayer() { return _rewindLayer; };

void sd::graph::Node::setRewindLayer(int layerId, int stepId) {
  _rewindLayer.first = layerId;
  _rewindLayer.second = stepId;
}

bool sd::graph::Node::equals(Node* other) {
  if (_opType == other->_opType && _dataType == other->_dataType && _opNum == other->_opNum) return true;

  return false;
}

void sd::graph::Node::deleteOpByType(::graph::OpType opType, void* op) {
  switch (opType) {
    case ::graph::OpType_PAIRWISE:
      delete reinterpret_cast<sd::ops::LegacyPairwiseTransformOp*>(op);
      break;
    case ::graph::OpType_PAIRWISE_BOOL:
      delete reinterpret_cast<sd::ops::LegacyPairwiseTransformBoolOp*>(op);
      break;
    case ::graph::OpType_TRANSFORM_STRICT:
      delete reinterpret_cast<sd::ops::LegacyTransformStrictOp*>(op);
      break;
    case ::graph::OpType_TRANSFORM_SAME:
      delete reinterpret_cast<sd::ops::LegacyTransformSameOp*>(op);
      break;
    case ::graph::OpType_TRANSFORM_FLOAT:
      delete reinterpret_cast<sd::ops::LegacyTransformFloatOp*>(op);
      break;
    case ::graph::OpType_TRANSFORM_BOOL:
      delete reinterpret_cast<sd::ops::LegacyTransformBoolOp*>(op);
      break;
    case ::graph::OpType_SCALAR:
      delete reinterpret_cast<sd::ops::LegacyScalarOp*>(op);
      break;
    case ::graph::OpType_SCALAR_BOOL:
      delete reinterpret_cast<sd::ops::LegacyScalarBoolOp*>(op);
      break;
    case ::graph::OpType_REDUCE_3:
      delete reinterpret_cast<sd::ops::LegacyReduce3Op*>(op);
      break;
    case ::graph::OpType_REDUCE_SAME:
      delete reinterpret_cast<sd::ops::LegacyReduceSameOp*>(op);
      break;
    case ::graph::OpType_REDUCE_FLOAT:
      delete reinterpret_cast<sd::ops::LegacyReduceFloatOp*>(op);
      break;
    case ::graph::OpType_REDUCE_LONG:
      delete reinterpret_cast<sd::ops::LegacyReduceLongOp*>(op);
      break;
    case ::graph::OpType_REDUCE_BOOL:
      delete reinterpret_cast<sd::ops::LegacyReduceBoolOp*>(op);
      break;
    case ::graph::OpType_INDEX_REDUCE:
      delete reinterpret_cast<sd::ops::LegacyIndexReduceOp*>(op);
      break;
    case ::graph::OpType_SUMMARYSTATS:
      delete reinterpret_cast<sd::ops::LegacyStatsOp*>(op);
      break;
    case ::graph::OpType_RANDOM:
      delete reinterpret_cast<sd::ops::LegacyRandomOp*>(op);
      break;
    case ::graph::OpType_BROADCAST:
      delete reinterpret_cast<sd::ops::LegacyBroadcastOp*>(op);
      break;
    case ::graph::OpType_BROADCAST_BOOL:
      delete reinterpret_cast<sd::ops::LegacyBroadcastBoolOp*>(op);
      break;
    case ::graph::OpType_CUSTOM:
      delete reinterpret_cast<sd::ops::DeclarableOp*>(op);
      break;
    default:
      THROW_EXCEPTION("Bad opType passed in");
  }
}

sd::ops::DeclarableOp* sd::graph::Node::buildOpByType(::graph::OpType opType, int numInputs, int numIArgs, int numTArgs,
                                                      int opNum, NDArray* scalar) {
  switch (opType) {
    case ::graph::OpType_PAIRWISE:
      return new sd::ops::LegacyPairwiseTransformOp(opNum);
    case ::graph::OpType_PAIRWISE_BOOL:
      return new sd::ops::LegacyPairwiseTransformBoolOp(opNum);
    case ::graph::OpType_TRANSFORM_STRICT:
      return new sd::ops::LegacyTransformStrictOp(opNum);
    case ::graph::OpType_TRANSFORM_SAME:
      return new sd::ops::LegacyTransformSameOp(opNum);
    case ::graph::OpType_TRANSFORM_FLOAT:
      return new sd::ops::LegacyTransformFloatOp(opNum);
    case ::graph::OpType_TRANSFORM_BOOL:
      return new sd::ops::LegacyTransformBoolOp(opNum);
    case ::graph::OpType_SCALAR:
      return scalar == nullptr ? new sd::ops::LegacyScalarOp(opNum) : new sd::ops::LegacyScalarOp(opNum, *scalar);
    case ::graph::OpType_SCALAR_BOOL:
      return scalar == nullptr ? new sd::ops::LegacyScalarBoolOp(opNum)
                               : new sd::ops::LegacyScalarBoolOp(opNum, *scalar);
    case ::graph::OpType_REDUCE_3:
      return new sd::ops::LegacyReduce3Op(opNum);
    case ::graph::OpType_REDUCE_SAME:
      return new sd::ops::LegacyReduceSameOp(opNum);
    case ::graph::OpType_REDUCE_FLOAT:
      return new sd::ops::LegacyReduceFloatOp(opNum);
    case ::graph::OpType_REDUCE_LONG:
      return new sd::ops::LegacyReduceLongOp(opNum);
    case ::graph::OpType_REDUCE_BOOL:
      return new sd::ops::LegacyReduceBoolOp(opNum);
    case ::graph::OpType_INDEX_REDUCE:
      return new sd::ops::LegacyIndexReduceOp(opNum);
    case ::graph::OpType_SUMMARYSTATS:
      return new sd::ops::LegacyStatsOp(opNum);
    case ::graph::OpType_RANDOM:
      return new sd::ops::LegacyRandomOp(opNum);
    case ::graph::OpType_BROADCAST:
      return new sd::ops::LegacyBroadcastOp(opNum);
    case ::graph::OpType_BROADCAST_BOOL:
      return new sd::ops::LegacyBroadcastBoolOp(opNum);
    default:
      THROW_EXCEPTION("Bad opType passed in");
  }

  return nullptr;
}

bool Node::isDeductable() { return _isDeductable; }

void Node::setDeductable(bool reallyDeductable) { _isDeductable = reallyDeductable; }

Node* Node::clone() {
  if (this->_customOp && this->_opType == ::graph::OpType_CUSTOM) {
    auto clone = new Node(this->_customOp, _id);
    clone->pullValues(this);
    return clone;
  } else {
    auto clone = new Node(_opType, _opNum, _id);

    clone->pullValues(this);

    // op time
    if (!_isDeductable)
      clone->_customOp = _customOp;
    else {
      auto c = dynamic_cast<sd::ops::LegacyOp*>(_customOp);
      clone->_customOp = c->clone();
    }

    return clone;
  }
}
}  // namespace graph
}  // namespace sd
