/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <graph/Node.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/LegacyTransformSameOp.h>
#include <ops/declarable/LegacyTransformFloatOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyReduceSameOp.h>
#include <ops/declarable/LegacyReduceFloatOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyStatsOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <ops/declarable/LegacyReduce3Op.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/LegacyOp.h>
#include <ops/declarable/LegacyReduceLongOp.h>
#include <ops/declarable/LegacyReduceBoolOp.h>
#include <ops/declarable/LegacyBroadcastBoolOp.h>
#include <ops/declarable/LegacyScalarBoolOp.h>
#include <ops/declarable/LegacyPairwiseTransformBoolOp.h>
#include <ops/declarable/LegacyTransformStrictOp.h>
#include <ops/declarable/LegacyTransformBoolOp.h>
#include <graph/FlatUtils.h>
#include <NDArrayFactory.h>
#include <exceptions/precondition_exception.h>

namespace nd4j {
    namespace graph {
        void nd4j::graph::Node::setOuterTime(Nd4jLong time){
//            if (hasBlockAttached())
//                _block->setOuterTime(time);
        }

        void nd4j::graph::Node::setInnerTime(Nd4jLong time){
//            if (hasBlockAttached())
//                _block->setInnerTime(time);
        }

        void nd4j::graph::Node::setEmbeddedGraph(nd4j::graph::Graph* graph) {
            _embeddedGraph = graph;
        }

        void nd4j::graph::Node::setParentGraph(nd4j::graph::Graph* graph) {
            _parentGraph = graph;
        }

        nd4j::graph::Graph* nd4j::graph::Node::embeddedGraph() {
            return _embeddedGraph;
        }

        nd4j::graph::Graph* nd4j::graph::Node::parentGraph() {
            return _parentGraph;
        }

        bool nd4j::graph::Node::hasGraphEmbedded() {
            return _embeddedGraph != nullptr;
        }

        void nd4j::graph::Node::markInplace(bool reallyInplace) {
            _isInplace = reallyInplace;
            if (_protoContext != nullptr) {
                _protoContext->markInplace(reallyInplace);
            }
        }

        OpClass nd4j::graph::Node::getOpClass() {
            return _opClass;
        }

        bool nd4j::graph::Node::hasBlockAttached() {
            return _protoContext != nullptr;
        }

        bool nd4j::graph::Node::isInplace() {
            return _isInplace;
        }

        bool nd4j::graph::Node::isDivergencePoint() {
            if (hasCustomOp()) {
                return _customOp->getOpDescriptor()->isDivergent();
            } else if (opType() == OpType_LOGIC && opNum() == 30)
                return true;
            else
                return false;
        }

        void nd4j::graph::Node::setActive(bool reallyActive) {
            _active = reallyActive;
        }

        bool nd4j::graph::Node::isActive() {
            return _active;
        }

        Nd4jLong Node::getFrameId() {
            return _frameId;
        }

        void Node::setFrameId(Nd4jLong frameId) {
            _frameId = frameId;
        }

        ContextPrototype * nd4j::graph::Node::getContextPrototype() {
            if (_protoContext == nullptr)
                _protoContext = new ContextPrototype(this->getCustomOp() != nullptr ? this->getCustomOp()->getOpDescriptor() : nullptr, this->id());
            if (_protoContext->inputs()->empty()) {
                for (int e = 0; e < this->input()->size(); e++) {
                    _protoContext->inputs()->emplace_back(this->input()->at(e));
                }
            }
            return _protoContext;
        }

        void nd4j::graph::Node::setContextPrototype(ContextPrototype *block) {
            if (_protoContext != nullptr)
                throw std::runtime_error("Block already exists");

            _protoContext = block;
        }

        void nd4j::graph::Node::setId(int id) {
            _id = id;
        }

        nd4j::ops::DeclarableOp* nd4j::graph::Node::getCustomOp() {
            return _customOp;
        }

        void nd4j::graph::Node::setCustomOp(nd4j::ops::DeclarableOp *customOp) {
            _customOp = customOp;

            // divergent ops (Switch etc) are always inplace, they don't allocate anything
            if (_customOp != nullptr && customOp->getOpDescriptor()->isDivergent())
                _isInplace = true;
        }

        bool nd4j::graph::Node::hasCustomOp() {
            return _customOp != nullptr;
        }

        std::string * nd4j::graph::Node::name() {
            return this->getName();
        }

        std::string * nd4j::graph::Node::getName() {
            return &_name;
        }

        void nd4j::graph::Node::setName(const std::string& name) {
            _name = name.c_str();
        }

        void nd4j::graph::Node::setName(std::string *name) {
            _name = *name;
        }

        double nd4j::graph::Node::scalar() {
            return  _scalar.e<double>(0);
        };

        void nd4j::graph::Node::pickInput(std::pair<int,int>& pair) {
            _input.push_back(pair);
        }

        void nd4j::graph::Node::pickInput(int inputId, int outputId) {
            std::pair<int,int> p(inputId,outputId);
            pickInput(p);
        }

        void nd4j::graph::Node::pickInput(int inputId) {
            pickInput(inputId, 0);

            if (inputId < 0)
                _hasExternalInputs = true;
            else
                _hasInternalInputs = true;
        }

        void nd4j::graph::Node::pickExternalOutput(int outputId) {
            std::pair<int, int> pair(outputId, 0);
            _output.push_back(pair);

            _hasExternalOutputs = true;
        }

        void nd4j::graph::Node::pickOutputOnce(int outputId) {
            std::pair<int, int> pair(outputId, 0);
            if (std::find(_output.begin(), _output.end(), pair) == _output.end())
                pickOutput(outputId);
        }

        void nd4j::graph::Node::pickOutput(int nodeId, int outputId) {
            std::pair<int, int> pair(nodeId, outputId);
            _output.emplace_back(pair);
        }

        void nd4j::graph::Node::pickOutput(int outputId) {
            std::pair<int, int> pair(outputId, 0);
            _output.emplace_back(pair);

            if (outputId < 0)
                _hasExternalOutputs = true;
            else
                _hasInternalOutputs = true;
        }

        int * nd4j::graph::Node::getDimensionsPtr() {
            return _dim;
        }

        std::vector<int> * nd4j::graph::Node::getDimensions() {
            return &_dimensions;
        }

        int nd4j::graph::Node::getLayer() {
            return _layer;
        }

        void nd4j::graph::Node::setLayer(int layer) {
            _layer = layer;
        }

        bool nd4j::graph::Node::hasExternalOutputs() {
            return _hasExternalOutputs;
        }

        bool nd4j::graph::Node::hasExternalInputs() {
            return _hasExternalInputs;
        }

        bool nd4j::graph::Node::hasInternalOutputs() {
            return _hasInternalOutputs;
        }

        bool nd4j::graph::Node::hasInternalInputs() {
            return _hasInternalInputs;
        }

        bool nd4j::graph::Node::isMultiInput() {
            return _input.size() > 1;
        }

        bool nd4j::graph::Node::isMultiOutput() {
            return _output.size() > 1;
        }

        double * nd4j::graph::Node::extraParams() {
            return _extraParams;
        }

        int Node::totalReferences() {
            return _referencedBy.size();
        }

        void Node::addReference(int nodeId) {
            _referencedBy.emplace_back(nodeId);
        }

        nd4j::graph::OpType nd4j::graph::Node::opType() {
            return _opType;
        }

        int nd4j::graph::Node::id() {
            return _id;
        }

        Nd4jLong nd4j::graph::Node::opNum() {
            return _opNum;
        }

        std::vector<std::pair<int,int>> *nd4j::graph::Node::input() {
            return &_input;
        }

        std::vector<std::pair<int, int>> *nd4j::graph::Node::output() {
            return &_output;
        }

        bool Node::isScoped() {
            return _scope_id != 0;
        }

        void Node::setScopeInfo(int id, const char* name) {
            _scope_id = id;

            if (name != nullptr)
                _scope_name = name;
        }

        int Node::scopeId() {
            return _scope_id;
        }

        std::string* Node::scopeName() {
            return &_scope_name;
        }

        template <typename T>
        Node* Node::asT() {
            auto node = this->clone();
            node->_dataType = DataTypeUtils::fromT<T>();
            return node;
        }
        BUILD_SINGLE_TEMPLATE(template Node* Node::asT, (), LIBND4J_TYPES);

        nd4j::graph::Node::Node(nd4j::ops::DeclarableOp *customOp, int id, std::initializer_list<int> input, std::initializer_list<int> output,  std::initializer_list<int> dimensions, float scalar, std::initializer_list<double> tArgs, std::initializer_list<int> iArgs) {
            this->_opType = OpType_CUSTOM;
            this->_id = id;
            this->_opNum = customOp->getOpHash();
            this->_extraParams = nullptr;
            this->_dataType = nd4j::DataType::FLOAT32; // float as default
            this->_dim = nullptr;
            this->_customOp = customOp;

            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;

            _scalar = NDArrayFactory::create(scalar);

            for (auto i: input)
                pickInput(i);

            for (auto o: output)
                pickOutput(o);

            if (dimensions.size() > 0) {
                _dim = new int[dimensions.size()];
                int cnt = 0;
                for (auto d: dimensions) {
                    _dimensions.push_back(d);
                    _dim[cnt++] = d;
                }
            }

            auto block = new ContextPrototype(this->getCustomOp()->getOpDescriptor(), this->id(), false);

            for (auto v: dimensions)
                block->getAxis()->emplace_back(v);

            for (auto v: iArgs)
                block->getIArguments()->emplace_back(v);

            for (auto v: tArgs)
                block->getTArguments()->emplace_back(v);

            this->setContextPrototype(block);
        }

        void nd4j::graph::Node::setOpType(OpType opType) {
            this->_opType = opType;
        }

        nd4j::graph::Node::Node(OpType opType, int opNum, int id, std::initializer_list<int> input, std::initializer_list<int> output, std::initializer_list<int> dimensions, float scalar, std::initializer_list<double> tArgs, std::initializer_list<int> iArgs) {
            this->_opType = opType;
            this->_id = id;
            this->_opNum = opNum;
            this->_extraParams = nullptr;
            this->_dataType = nd4j::DataType::FLOAT32; // float as default
            this->_dim = nullptr;

            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;

            _scalar = NDArrayFactory::create(scalar);

            for (auto i: input)
                pickInput(i);

            for (auto o: output)
                pickOutput(o);

            if (dimensions.size() > 0) {
                _dim = new int[dimensions.size()];
                int cnt = 0;
                for (auto d: dimensions) {
                    _dimensions.push_back(d);
                    _dim[cnt++] = d;
                }
            }

            // these ops allow in-place execution by design
            if (opType == OpType_TRANSFORM_SAME || opType == OpType_TRANSFORM_FLOAT || opType == OpType_TRANSFORM_STRICT || opType == OpType_TRANSFORM_BOOL || opType == OpType_SCALAR || opType == OpType_BROADCAST) {
                if (_output.size() <= 1) {
                    _isInplace = true;
                }
                _opClass = OpClass_TRANSFORM;
            } else if (opType == OpType_REDUCE_SAME || opType == OpType_REDUCE_FLOAT || opType == OpType_REDUCE_BOOL || opType == OpType_REDUCE_LONG || opType == OpType_SUMMARYSTATS) {
                _opClass = OpClass_REDUCTION;
            }


            if (opType == OpType_BROADCAST ||
                    opType == OpType_BROADCAST_BOOL ||
                    opType == OpType_INDEX_REDUCE ||
                    opType == OpType_SUMMARYSTATS ||
                    opType == OpType_REDUCE_BOOL ||
                    opType == OpType_REDUCE_SAME ||
                    opType == OpType_REDUCE_FLOAT ||
                    opType == OpType_REDUCE_3 ||
                    opType == OpType_TRANSFORM_STRICT ||
                    opType == OpType_TRANSFORM_SAME ||
                    opType == OpType_TRANSFORM_FLOAT ||
                    opType == OpType_TRANSFORM_BOOL ||
                    opType == OpType_RANDOM ||
                    opType == OpType_PAIRWISE ||
                    opType == OpType_PAIRWISE_BOOL ||
                    opType == OpType_SCALAR_BOOL ||
                    opType == OpType_SCALAR) {

                this->_isDeductable = true;

                auto block = new ContextPrototype(nullptr, this->id(), false);

                for (auto v: dimensions)
                    block->getAxis()->emplace_back(v);

                for (auto v: iArgs)
                    block->getIArguments()->emplace_back(v);

                for (auto v: tArgs)
                    block->getTArguments()->emplace_back(v);

                this->setContextPrototype(block);
                this->setCustomOp(Node::buildOpByType(opType, (int) input.size(), (int) block->getIArguments()->size(), (int) block->getTArguments()->size(), opNum, &_scalar));
                block->setOpDescriptor(this->getCustomOp()->getOpDescriptor());
            } else if (opType == OpType_CUSTOM) {
                if (this->getCustomOp()) {
                    auto block = new ContextPrototype(this->getCustomOp()->getOpDescriptor(), this->id(), false);

                    for (auto v: dimensions)
                        block->getAxis()->emplace_back(v);

                    for (auto v: iArgs)
                        block->getIArguments()->emplace_back(v);

                    for (auto v: tArgs)
                        block->getTArguments()->emplace_back(v);

                    this->setContextPrototype(block);
                } else throw std::runtime_error("wrong custom operation given");
            }
        };

        nd4j::graph::Node::Node(const nd4j::graph::FlatNode *node) {
            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;
            _extraParams = nullptr;
            _dim = nullptr;
            _dataType = nd4j::DataType::FLOAT32; // float as default
            if (node->scope_id() != 0)
                this->_scope_id = node->scope_id();

            if (node->scope_name() != nullptr && node->scope_name()->size() > 0)
                this->_scope_name = node->scope_name()->str();

            if (node->scalar() != nullptr) {
                auto scalar = nd4j::graph::FlatUtils::fromFlatArray(node->scalar());
                _scalar = *scalar;
                delete scalar;
            }

            if (node != nullptr) {
                this->_id = node->id();
                //this->_dataType = DataTypeUtils::fromFlatDataType(node->dataType());
                this->_opNum = node->opNum();
                this->_opType = node->opType();

                if (node->name() != nullptr && node->name()->c_str() != nullptr) {
                    this->_name = node->name()->str();
                }

                if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
                    for (int e = 0; e < (int) node->inputPaired()->size(); e++) {
                        auto pair = node->inputPaired()->Get(e);
                        pickInput(pair->first(), pair->second());
                    }
                } else if (node->input() != nullptr && node->input()->size() > 0) {
                    for (int e = 0; e < (int) node->input()->size(); e++)
                        pickInput(node->input()->Get(e));
                } else {
                    if (this->opType() != OpType_LOGIC) {
                        if (this->_name.size() > 0) {
                            nd4j_debug("Node [%i:<%s>] has no inputs defined\n", this->_id, this->_name.c_str());
                        } else {
                            nd4j_debug("Node [%i:<noname>] has no inputs defined\n", this->_id);
                        }
                    }
                }

                /*
                if (node->output() != nullptr)
                    for (int e = 0; e < (int) node->output()->size(); e++) {
                        auto oid = node->output()->Get(e);
                        if (oid != this->_id && oid != 0) {
                            nd4j_verbose("Picking output: %i\n", node->output()->Get(e));
                            pickOutput(oid);
                        }
                    }
                */


                if (node->extraParams() != nullptr && node->extraParams()->size() > 0) {
                    _extraParams = new double[node->extraParams()->size()];
                    for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                        _extraParams[e] = static_cast<double>(node->extraParams()->Get(e));
                    }
                }

                if (node->dimensions() != nullptr && node->dimensions()->size() > 0) {
                    _dim = new int[node->dimensions()->size()];
                    for (int e = 0; e < (int) node->dimensions()->size(); e++) {
                        _dimensions.emplace_back(node->dimensions()->Get(e));
                        _dim[e] = node->dimensions()->Get(e);
                    }
                }

                if (this->opType() == OpType_LOGIC && this->opNum() == 100L) {
                    if (node->extraInteger()->size() < 1) {
                        nd4j_printf("Node_%i is type of Enter, but has no FrameID defined\n", this->id());
                        throw std::runtime_error("Enter node must have FrameID specified");
                    }

                    this->setFrameId(node->extraInteger()->Get(0));
                }


                // these ops allow in-place execution by design
                if (_opType == OpType_BROADCAST ||
                    _opType == OpType_BROADCAST_BOOL ||
                        _opType == OpType_INDEX_REDUCE ||
                        _opType == OpType_SUMMARYSTATS ||
                        _opType == OpType_REDUCE_BOOL ||
                        _opType == OpType_REDUCE_SAME ||
                        _opType == OpType_REDUCE_FLOAT ||
                        _opType == OpType_REDUCE_3 ||
                        _opType == OpType_TRANSFORM_STRICT ||
                        _opType == OpType_TRANSFORM_SAME ||
                        _opType == OpType_TRANSFORM_FLOAT ||
                        _opType == OpType_TRANSFORM_BOOL ||
                        _opType == OpType_RANDOM ||
                        _opType == OpType_PAIRWISE ||
                        _opType == OpType_PAIRWISE_BOOL ||
                        _opType == OpType_SCALAR_BOOL ||
                        _opType == OpType_SCALAR) {

                    if (_output.size() <= 1) {
                        _isInplace = true;
                    }

                    if (node->input() != nullptr && node->input()->size() > 0) {
                        this->_isDeductable = true;

                        auto block = new ContextPrototype(nullptr, this->id(), false);


                        for (auto v: _dimensions)
                            block->getAxis()->emplace_back(v);

                        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
                            for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                                block->getTArguments()->emplace_back(static_cast<double>(node->extraParams()->Get(e)));
                            }

                        if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
                            for (int e = 0; e < (int) node->extraBools()->size(); e++) {
                                block->getBArguments()->push_back(node->extraBools()->Get(e));
                            }

                        if (node->extraInteger() != nullptr && node->extraInteger()->size() > 0)
                            for (int e = 0; e < (int) node->extraInteger()->size(); e++) {
                                block->getIArguments()->emplace_back(node->extraInteger()->Get(e));
                            }

                        this->setContextPrototype(block);
                        this->setCustomOp(Node::buildOpByType(_opType, (int) node->input()->size(), (int) block->getIArguments()->size(), (int) block->getTArguments()->size(), (int) _opNum, &_scalar));
                        block->setOpDescriptor(this->getCustomOp()->getOpDescriptor());
                    } else if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
                        this->_isDeductable = true;

                        auto block = new ContextPrototype(nullptr, this->id(), false);

                        for (int e = 0; e < this->input()->size(); e++) {
                            block->inputs()->emplace_back(this->input()->at(e));
                        }

                        // there's no other IArgs in legacy options, actually
                        for (auto v: _dimensions)
                            block->getAxis()->emplace_back(v);

                        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
                            for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                                block->getTArguments()->emplace_back(static_cast<double>(node->extraParams()->Get(e)));
                            }

                        if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
                            for (int e = 0; e < (int) node->extraBools()->size(); e++) {
                                block->getBArguments()->push_back(node->extraBools()->Get(e));
                            }

                        if (node->extraInteger() != nullptr && node->extraInteger()->size() > 0)
                            for (int e = 0; e < (int) node->extraInteger()->size(); e++) {
                                block->getIArguments()->emplace_back(node->extraInteger()->Get(e));
                            }

                        this->setContextPrototype(block);

                        this->setCustomOp(Node::buildOpByType(_opType, (int) node->inputPaired()->size(), (int) block->getIArguments()->size(), (int) block->getTArguments()->size(), (int) _opNum, &_scalar));
                        block->setOpDescriptor(this->getCustomOp()->getOpDescriptor());
                    }
                } else if (this->_opType == OpType_CUSTOM) {
                        auto op = nd4j::ops::OpRegistrator::getInstance()->getOperation(this->opNum());
                        if (op == nullptr) {
                            nd4j_verbose("Can't find operation: %lld\n", this->opNum());
                            throw std::runtime_error("Can't find requested operation");
                        }

                        auto block = new ContextPrototype(nullptr, this->id());

                        for (int e = 0; e < this->input()->size(); e++) {
                            block->inputs()->emplace_back(this->input()->at(e));
                        }

                        if (node->extraInteger() != nullptr)
                            for (uint32_t e = 0; e < node->extraInteger()->size(); e++) {
                                auto v = node->extraInteger()->Get(e);
                                // FIXME: remove this static_cast, iArgs should be Nd4jLong
                                block->getIArguments()->emplace_back(static_cast<int>(v));
                            }

                        if (node->extraParams() != nullptr)
                            for (uint32_t e = 0; e < node->extraParams()->size(); e++)
                                block->getTArguments()->emplace_back(static_cast<double>(node->extraParams()->Get(e)));

                        if (node->extraBools() != nullptr && node->extraBools()->size() > 0)
                            for (int e = 0; e < (int) node->extraBools()->size(); e++) {
                                block->getBArguments()->push_back(node->extraBools()->Get(e));
                            }

                        for (auto v: _dimensions)
                            block->getAxis()->emplace_back(v);

                        this->setContextPrototype(block);
                        this->setCustomOp(op);
                        block->setOpDescriptor(this->getCustomOp()->getOpDescriptor());
                }
            } else {
                // empty dynamic node, tests probably
            }
        }

        nd4j::DataType Node::dataType() {
            return _dataType;
        }

        ContextPrototype* Node::protoContext() {
            return _protoContext;
        }

        nd4j::graph::Node::~Node() {
            if (_extraParams != nullptr)
                delete[] _extraParams;

            if (_dim != nullptr)
                delete[] _dim;

            if (_protoContext != nullptr)
                delete _protoContext;

            if (_isDeductable && _customOp != nullptr)
                delete _customOp;
        }

        int nd4j::graph::Node::getRewindNode() {
            return _rewindNode;
        }

        void nd4j::graph::Node::setRewindNode(int nodeId) {
            _rewindNode = nodeId;
        }

        std::pair<int, int>& nd4j::graph::Node::getRewindLayer() {
            return _rewindLayer;
        };

        void nd4j::graph::Node::setRewindLayer(int layerId, int stepId) {
            _rewindLayer.first = layerId;
            _rewindLayer.second = stepId;
        }

        bool nd4j::graph::Node::equals(Node *other) {
            if (_opType == other->_opType && _dataType == other->_dataType && _opNum == other->_opNum)
                return true;

            return false;
        }

        nd4j::ops::DeclarableOp* nd4j::graph::Node::buildOpByType(OpType opType, int numInputs,  int numIArgs, int numTArgs, int opNum, NDArray *scalar) {
            switch (opType) {
                case OpType_PAIRWISE:
                    return new nd4j::ops::LegacyPairwiseTransformOp(opNum);
                case OpType_PAIRWISE_BOOL:
                    return new nd4j::ops::LegacyPairwiseTransformBoolOp(opNum);
                case OpType_TRANSFORM_STRICT:
                    return new nd4j::ops::LegacyTransformStrictOp(opNum);
                case OpType_TRANSFORM_SAME:
                    return new nd4j::ops::LegacyTransformSameOp(opNum);
                case OpType_TRANSFORM_FLOAT:
                    return new nd4j::ops::LegacyTransformFloatOp(opNum);
                case OpType_TRANSFORM_BOOL:
                    return new nd4j::ops::LegacyTransformBoolOp(opNum);
                case OpType_SCALAR:
                    return scalar == nullptr ? new nd4j::ops::LegacyScalarOp(opNum) : new nd4j::ops::LegacyScalarOp(opNum, *scalar);
                case OpType_SCALAR_BOOL:
                    return scalar == nullptr ? new nd4j::ops::LegacyScalarBoolOp(opNum) : new nd4j::ops::LegacyScalarBoolOp(opNum, *scalar);
                case OpType_REDUCE_3:
                    return new nd4j::ops::LegacyReduce3Op(opNum);
                case OpType_REDUCE_SAME:
                    return new nd4j::ops::LegacyReduceSameOp(opNum);
                case OpType_REDUCE_FLOAT:
                    return new nd4j::ops::LegacyReduceFloatOp(opNum);
                case OpType_REDUCE_LONG:
                    return new nd4j::ops::LegacyReduceLongOp(opNum);
                case OpType_REDUCE_BOOL:
                    return new nd4j::ops::LegacyReduceBoolOp(opNum);
                case OpType_INDEX_REDUCE:
                    return new nd4j::ops::LegacyIndexReduceOp(opNum);
                case OpType_SUMMARYSTATS:
                    return new nd4j::ops::LegacyStatsOp(opNum);
                case OpType_RANDOM:
                    return new nd4j::ops::LegacyRandomOp(opNum);
                case OpType_BROADCAST:
                    return new nd4j::ops::LegacyBroadcastOp(opNum);
                case OpType_BROADCAST_BOOL:
                    return new nd4j::ops::LegacyBroadcastBoolOp(opNum);
                default:
                    throw std::runtime_error("Bad opType passed in");
            }
        }

        bool Node::isDeductable() {
            return _isDeductable;
        }

        void Node::setDeductable(bool reallyDeductable) {
            _isDeductable = reallyDeductable;
        }


        Node* Node::clone() {
            if (this->_customOp && this->_opType == nd4j::graph::OpType_CUSTOM) {
                auto clone = new Node(this->_customOp, _id);
                clone->pullValues(this);
                return clone;
            }
            else {
            auto clone = new Node(_opType, _opNum, _id);

            clone->pullValues(this);

            // op time
            if (!_isDeductable)
                clone->_customOp = _customOp;
            else {
                auto c = dynamic_cast<nd4j::ops::LegacyOp*>(_customOp);
                clone->_customOp = c->clone();
            }

            return clone;
            }
        }


        Node::Node(Variable *variable) {
            samediff::precondition_exception::check(variable != nullptr, "Node: variable is null");
            samediff::precondition_exception::check(variable->id() < 0, "Node: variable id must be negative");
            samediff::precondition_exception::check(variable->getName() != nullptr, "Node: variable must have symbolic name");

            _opType = OpType_VARIABLE;
            _id = variable->id();
            _name = *variable->getName();
        }
    }
}
