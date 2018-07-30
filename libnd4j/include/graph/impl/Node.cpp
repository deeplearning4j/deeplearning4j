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
#include <ops/declarable/LegacyTransformOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyReduceOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyStatsOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <ops/declarable/LegacyReduce3Op.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/LegacyOp.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        void nd4j::graph::Node<T>::setOuterTime(Nd4jLong time){
//            if (hasBlockAttached())
//                _block->setOuterTime(time);
        }

        template <typename T>
        void nd4j::graph::Node<T>::setInnerTime(Nd4jLong time){
//            if (hasBlockAttached())
//                _block->setInnerTime(time);
        }

        template <typename T>
        void nd4j::graph::Node<T>::setGraph(nd4j::graph::Graph<T>* graph) {
            _graph = graph;
        }

        template <typename T>
        nd4j::graph::Graph<T>* nd4j::graph::Node<T>::getGraph() {
            return _graph;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasGraphEmbedded() {
            return _graph != nullptr;
        }

        template <typename T>
        void nd4j::graph::Node<T>::markInplace(bool reallyInplace) {
            _isInplace = reallyInplace;
            if (_protoContext != nullptr) {
                _protoContext->markInplace(reallyInplace);
            }
        }

        template <typename T>
        OpClass nd4j::graph::Node<T>::getOpClass() {
            return _opClass;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasBlockAttached() {
            return _protoContext != nullptr;
        }



        template <typename T>
        bool nd4j::graph::Node<T>::isInplace() {
            return _isInplace;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::isDivergencePoint() {
            if (hasCustomOp()) {
                return _customOp->getOpDescriptor()->isDivergent();
            } else if (opType() == OpType_LOGIC && opNum() == 30)
                return true;
            else
                return false;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setActive(bool reallyActive) {
            _active = reallyActive;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::isActive() {
            return _active;
        }

        template<typename T>
        Nd4jLong Node<T>::getFrameId() {
            return _frameId;
        }

        template<typename T>
        void Node<T>::setFrameId(Nd4jLong frameId) {
            _frameId = frameId;
        }

        template <typename T>
        ContextPrototype<T> * nd4j::graph::Node<T>::getContextPrototype() {
            if (_protoContext == nullptr)
                _protoContext = new ContextPrototype<T>(this->id());
            if (_protoContext->inputs()->empty()) {
                for (int e = 0; e < this->input()->size(); e++) {
                    _protoContext->inputs()->emplace_back(this->input()->at(e));
                }
            }
            return _protoContext;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setContextPrototype(ContextPrototype<T> *block) {
            if (_protoContext != nullptr)
                throw std::runtime_error("Block already exists");

            _protoContext = block;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setId(int id) {
            _id = id;
        }

        template <typename T>
        nd4j::ops::DeclarableOp<T>* nd4j::graph::Node<T>::getCustomOp() {
            return _customOp;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setCustomOp(nd4j::ops::DeclarableOp<T> *customOp) {
            _customOp = customOp;

            // divergent ops (Switch etc) are always inplace, they don't allocate anything
            if (_customOp != nullptr && customOp->getOpDescriptor()->isDivergent())
                _isInplace = true;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasCustomOp() {
            return _customOp != nullptr;
        }

        template <typename T>
        std::string * nd4j::graph::Node<T>::name() {
            return this->getName();
        }

        template <typename T>
        std::string * nd4j::graph::Node<T>::getName() {
            return &_name;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setName(const std::string& name) {
            _name = name.c_str();
        }

        template <typename T>
        void nd4j::graph::Node<T>::setName(std::string *name) {
            _name = *name;
        }

        template <typename T>
        T nd4j::graph::Node<T>::scalar() {
            return (T) _scalar;
        };

        template <typename T>
        void nd4j::graph::Node<T>::pickInput(std::pair<int,int>& pair) {
            _input.push_back(pair);
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickInput(int inputId, int outputId) {
            std::pair<int,int> p(inputId,outputId);
            pickInput(p);
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickInput(int inputId) {
            pickInput(inputId, 0);

            if (inputId < 0)
                _hasExternalInputs = true;
            else
                _hasInternalInputs = true;
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickExternalOutput(int outputId) {
            std::pair<int, int> pair(outputId, 0);
            _output.push_back(pair);

            _hasExternalOutputs = true;
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickOutputOnce(int outputId) {
            std::pair<int, int> pair(outputId, 0);
            if (std::find(_output.begin(), _output.end(), pair) == _output.end())
                pickOutput(outputId);
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickOutput(int nodeId, int outputId) {
            std::pair<int, int> pair(nodeId, outputId);
            _output.emplace_back(pair);
        }

        template <typename T>
        void nd4j::graph::Node<T>::pickOutput(int outputId) {
            std::pair<int, int> pair(outputId, 0);
            _output.emplace_back(pair);

            if (outputId < 0)
                _hasExternalOutputs = true;
            else
                _hasInternalOutputs = true;
        }

        template <typename T>
        int * nd4j::graph::Node<T>::getDimensionsPtr() {
            return _dim;
        }

        template <typename T>
        std::vector<int> * nd4j::graph::Node<T>::getDimensions() {
            return &_dimensions;
        }

        template <typename T>
        int nd4j::graph::Node<T>::getLayer() {
            return _layer;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setLayer(int layer) {
            _layer = layer;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasExternalOutputs() {
            return _hasExternalOutputs;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasExternalInputs() {
            return _hasExternalInputs;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasInternalOutputs() {
            return _hasInternalOutputs;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::hasInternalInputs() {
            return _hasInternalInputs;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::isMultiInput() {
            return _input.size() > 1;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::isMultiOutput() {
            return _output.size() > 1;
        }

        template <typename T>
        T * nd4j::graph::Node<T>::extraParams() {
            return _extraParams;
        }

        template <typename T>
        int Node<T>::totalReferences() {
            return _referencedBy.size();
        }

        template <typename T>
        void Node<T>::addReference(int nodeId) {
            _referencedBy.emplace_back(nodeId);
        }

        template <typename T>
        nd4j::graph::OpType nd4j::graph::Node<T>::opType() {
            return _opType;
        }

        template <typename T>
        int nd4j::graph::Node<T>::id() {
            return _id;
        }

        template <typename T>
        Nd4jLong nd4j::graph::Node<T>::opNum() {
            return _opNum;
        }

        template <typename T>
        std::vector<std::pair<int,int>> *nd4j::graph::Node<T>::input() {
            return &_input;
        }

        template <typename T>
        std::vector<std::pair<int, int>> *nd4j::graph::Node<T>::output() {
            return &_output;
        }

        template <typename T>
        bool Node<T>::isScoped() {
            return _scope_id != 0;
        }

        template <typename T>
        void Node<T>::setScopeInfo(int id, const char* name) {
            _scope_id = id;

            if (name != nullptr)
                _scope_name = name;
        }

        template <typename T>
        int Node<T>::scopeId() {
            return _scope_id;
        }

        template <typename T>
        std::string* Node<T>::scopeName() {
            return &_scope_name;
        }

        template <typename T>
        nd4j::graph::Node<T>::Node(OpType opType, int opNum, int id, std::initializer_list<int> input, std::initializer_list<int> output, std::initializer_list<int> dimensions, float scalar, std::initializer_list<T> tArgs, std::initializer_list<int> iArgs) {
            this->_opType = opType;
            this->_id = id;
            this->_opNum = opNum;
            this->_extraParams = nullptr;
            this->_dim = nullptr;

            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;

            _scalar = scalar;

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
            if (opType == OpType_TRANSFORM || opType == OpType_SCALAR || opType == OpType_BROADCAST) {
                if (_output.size() <= 1) {
                    _isInplace = true;
                }
                _opClass = OpClass_TRANSFORM;
            } else if (opType == OpType_ACCUMULATION || opType == OpType_SUMMARYSTATS) {
                _opClass = OpClass_REDUCTION;
            }


            if (opType == OpType_BROADCAST ||
                    opType == OpType_INDEX_ACCUMULATION ||
                    opType == OpType_SUMMARYSTATS ||
                    opType == OpType_ACCUMULATION ||
                    opType == OpType_ACCUMULATION3 ||
                    opType == OpType_TRANSFORM ||
                    opType == OpType_RANDOM ||
                    opType == OpType_PAIRWISE ||
                    opType == OpType_SCALAR) {

                this->_isDeductable = true;

                auto block = new ContextPrototype<T>(this->id(), false);

                // there's no other IArgs in legacy options, actually
                for (auto v: dimensions)
                    block->getIArguments()->emplace_back(v);

                for (auto v: iArgs)
                    block->getIArguments()->emplace_back(v);

                for (auto v: tArgs)
                    block->getTArguments()->emplace_back(v);

                this->setContextPrototype(block);
                this->setCustomOp(Node<T>::buildOpByType(opType, (int) input.size(), (int) block->getIArguments()->size(), (int) block->getTArguments()->size(), opNum, scalar));
            } else if (opType == OpType_CUSTOM) {
                auto block = new ContextPrototype<T>(this->id(), false);

                for (auto v: iArgs)
                    block->getIArguments()->emplace_back(v);

                for (auto v: tArgs)
                    block->getTArguments()->emplace_back(v);

                this->setContextPrototype(block);
            }
        };

        template <typename T>
        nd4j::graph::Node<T>::Node(const nd4j::graph::FlatNode *node) {
            _hasExternalInputs = false;
            _hasExternalOutputs = false;
            _hasInternalInputs = false;
            _hasInternalOutputs = false;
            _extraParams = nullptr;
            _dim = nullptr;

            if (node->scope_id() != 0)
                this->_scope_id = node->scope_id();

            if (node->scope_name() != nullptr && node->scope_name()->size() > 0)
                this->_scope_name = node->scope_name()->str();


            _scalar = node->scalar();

            if (node != nullptr) {
                this->_id = node->id();
                this->_dataType = node->dataType();
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
                            nd4j_printf("Node [%i:<%s>] do not have any inputs defined\n", this->_id, this->_name.c_str());
                        } else {
                            nd4j_printf("Node [%i:<noname>] do not have any inputs defined\n", this->_id);
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
                    _extraParams = new T[node->extraParams()->size()];
                    for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                        _extraParams[e] = static_cast<T>(node->extraParams()->Get(e));
                    }
                }

                if (node->dimensions() != nullptr && node->dimensions()->size() > 0) {
                    _dim = new int[node->dimensions()->size()];
                    for (int e = 0; e < (int) node->dimensions()->size(); e++) {
                        _dimensions.push_back(node->dimensions()->Get(e));
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
                if (this->_opType == OpType_TRANSFORM || this->_opType == OpType_SCALAR || this->_opType == OpType_BROADCAST || this->_opType == OpType_RANDOM || this->_opType == OpType_ACCUMULATION || this->_opType == OpType_ACCUMULATION3 || this->_opType == OpType_PAIRWISE || this->_opType == OpType_SUMMARYSTATS || this->_opType == OpType_INDEX_ACCUMULATION) {
                    if (_output.size() <= 1) {
                        _isInplace = true;
                    }

                    if (node->input() != nullptr && node->input()->size() > 0) {
                        this->_isDeductable = true;

                        auto block = new ContextPrototype<T>(this->id(), false);

                        // there's no other IArgs in legacy options, actually
                        for (auto v: _dimensions)
                            block->getIArguments()->emplace_back(v);

                        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
                            for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                                block->getTArguments()->emplace_back(static_cast<T>(node->extraParams()->Get(e)));
                            }

                        this->setContextPrototype(block);
                        this->setCustomOp(Node<T>::buildOpByType(_opType, (int) node->input()->size(), (int) block->getIArguments()->size(), (int) block->getTArguments()->size(), (int) _opNum, _scalar));
                    } else if (node->inputPaired() != nullptr && node->inputPaired()->size() > 0) {
                        this->_isDeductable = true;

                        auto block = new ContextPrototype<T>(this->id(), false);

                        for (int e = 0; e < this->input()->size(); e++) {
                            block->inputs()->emplace_back(this->input()->at(e));
                        }

                        // there's no other IArgs in legacy options, actually
                        for (auto v: _dimensions)
                            block->getIArguments()->emplace_back(v);

                        if (node->extraParams() != nullptr && node->extraParams()->size() > 0)
                            for (int e = 0; e < (int) node->extraParams()->size(); e++) {
                                block->getTArguments()->emplace_back(static_cast<T>(node->extraParams()->Get(e)));
                            }

                        this->setContextPrototype(block);

                        this->setCustomOp(Node<T>::buildOpByType(_opType, (int) node->inputPaired()->size(), (int) block->getIArguments()->size(), (int) block->getTArguments()->size(), (int) _opNum, _scalar));
                    }
                } else if (this->_opType == OpType_CUSTOM) {
                    if (sizeof(T) == 4) {
                        auto op = nd4j::ops::OpRegistrator::getInstance()->template getOperationT<T>(this->opNum());
                        if (op == nullptr) {
                            nd4j_verbose("Can't find operation: %lld\n", this->opNum());
                            throw std::runtime_error("Can't find requested operation");
                        }

                        auto block = new ContextPrototype<T>(this->id());

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
                                block->getTArguments()->emplace_back(static_cast<T>(node->extraParams()->Get(e)));

                        this->setContextPrototype(block);

                        this->setCustomOp(op);
                    }
                }
            } else {
                // empty dynamic node, tests probably
            }
        }

        template <typename T>
        DataType Node<T>::dataType() {
            return _dataType;
        }

        template <typename T>
        ContextPrototype<T>* Node<T>::protoContext() {
            return _protoContext;
        }

        template <typename T>
        nd4j::graph::Node<T>::~Node() {
            if (_extraParams != nullptr)
                delete[] _extraParams;

            if (_dim != nullptr)
                delete[] _dim;

            if (_protoContext != nullptr)
                delete _protoContext;

            if (_isDeductable && _customOp != nullptr)
                delete _customOp;
        }

        template <typename T>
        int nd4j::graph::Node<T>::getRewindNode() {
            return _rewindNode;
        }

        template <typename T>
        void nd4j::graph::Node<T>::setRewindNode(int nodeId) {
            _rewindNode = nodeId;
        }

        template <typename T>
        std::pair<int, int>& nd4j::graph::Node<T>::getRewindLayer() {
            return _rewindLayer;
        };

        template <typename T>
        void nd4j::graph::Node<T>::setRewindLayer(int layerId, int stepId) {
            _rewindLayer.first = layerId;
            _rewindLayer.second = stepId;
        }

        template <typename T>
        bool nd4j::graph::Node<T>::equals(Node *other) {
            if (_opType == other->_opType && _dataType == other->_dataType && _opNum == other->_opNum)
                return true;

            return false;
        }

        template <typename T>
        nd4j::ops::DeclarableOp<T>* nd4j::graph::Node<T>::buildOpByType(OpType opType, int numInputs,  int numIArgs, int numTArgs, int opNum, T scalar) {
            switch (opType) {
                case OpType_PAIRWISE:
                    return new nd4j::ops::LegacyPairwiseTransformOp<T>(opNum);
                case OpType_TRANSFORM:
                    return new nd4j::ops::LegacyTransformOp<T>(opNum);
                case OpType_SCALAR:
                    return new nd4j::ops::LegacyScalarOp<T>(opNum, scalar);
                case OpType_ACCUMULATION3:
                    return new nd4j::ops::LegacyReduce3Op<T>(opNum);
                case OpType_ACCUMULATION:
                    return new nd4j::ops::LegacyReduceOp<T>(opNum);
                case OpType_INDEX_ACCUMULATION:
                    return new nd4j::ops::LegacyIndexReduceOp<T>(opNum);
                case OpType_SUMMARYSTATS:
                    return new nd4j::ops::LegacyStatsOp<T>(opNum);
                case OpType_RANDOM:
                    return new nd4j::ops::LegacyRandomOp<T>(opNum);
                case OpType_BROADCAST:
                    return new nd4j::ops::LegacyBroadcastOp<T>(opNum);
                default:
                    throw std::runtime_error("Bad opType passed in");
            }
        }

        template <typename T>
        bool Node<T>::isDeductable() {
            return _isDeductable;
        }

        template <typename T>
        void Node<T>::setDeductable(bool reallyDeductable) {
            _isDeductable = reallyDeductable;
        }

        template <typename T>
        template <typename N>
        Node<N>* Node<T>::asT() {
            auto clone = new Node<N>(_opType, _opNum, _id);

            clone->pullValues(this);

            if (!_isDeductable && this->_customOp != nullptr)
                clone->setCustomOp(OpRegistrator::getInstance()->getOperationT<N>(this->_customOp->getOpHash()));
            else if (_customOp != nullptr) {
                // this->setCustomOp(Node<T>::buildOpByType(opType, (int) input.size(), (int) block->getIArguments()->size(), (int) block->getTArguments()->size(), opNum, scalar));
                auto op = clone->buildOpByType(_opType, clone->input()->size(), clone->getContextPrototype()->getIArguments()->size(), clone->getContextPrototype()->getTArguments()->size(), _opNum, clone->scalar());
                clone->setCustomOp(op);
            }

            return clone;
        }

        template <typename T>
        Node<T>* Node<T>::clone() {
            auto clone = new Node<T>(_opType, _opNum, _id);

            clone->pullValues(this);

            // op time
            if (!_isDeductable)
                clone->_customOp = _customOp;
            else {
                auto c = dynamic_cast<nd4j::ops::LegacyOp<T>*>(_customOp);
                clone->_customOp = c->clone();
            }

            return clone;
        }

        template class ND4J_EXPORT Node<float>;
        template class ND4J_EXPORT Node<float16>;
        template class ND4J_EXPORT Node<double>;


        template Node<float>* Node<float>::asT<float>();
        template Node<float16>* Node<float>::asT<float16>();
        template Node<double>* Node<float>::asT<double>();

        template Node<float>* Node<float16>::asT<float>();
        template Node<float16>* Node<float16>::asT<float16>();
        template Node<double>* Node<float16>::asT<double>();

        template Node<float>* Node<double>::asT<float>();
        template Node<float16>* Node<double>::asT<float16>();
        template Node<double>* Node<double>::asT<double>();
    }
}
