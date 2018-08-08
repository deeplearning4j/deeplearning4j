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

#include <graph/Graph.h>
#include <helpers/EnumUtils.h>
#include <graph/FlatUtils.h>
#include <NativeOps.h>
#include <vector>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/OpRegistrator.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        std::vector<Node<T>*>* Graph<T>::getAllNodes() {
            return &_handles;
        }

        template <typename T>
        std::vector<Variable<T>*>* Graph<T>::getPlaceholders() {
            return _variableSpace->getPlaceholders();
        }

        template <typename T>
        int Graph<T>::numberOfPlaceholders() {
            return _variableSpace->numberOfPlaceholders();
        };

        template <typename T>
        Nd4jLong Graph<T>::estimateRequiredMemory() {

            Nd4jLong result = 0L;
            Nd4jLong lastStep = 0L;

            std::vector<Nd4jLong *> shapes;
            std::map<std::pair<int,int>, Nd4jLong*> shapesMap;

            int cntFD = 0;

            // we loop in similar way to execution
            for (int l = 0; l < (int) _onion->size(); l++) {
                int layerSize = _onion->count(l) == 1 ? _onion->at(l)->size() : 0;


                for (int n = 0; n < layerSize; n++) {
                    Node<T>* node = _onion->at(l)->at(n);

                    /*
                     * Limited number of options here:
                     *
                     * 1) Op is inplace, so adds nothing to total
                     * 2) Op is not inplace, and 1:1 transform
                     * 3) Op is reduction (i.e. sum)
                     * 4) Op is multiplicator (i.e. im2col)
                     */
                    if (node->hasCustomOp()) {
                        //if (node->isInplace()) {
                        //    continue;
                        //}


                        nd4j_debug("Trying estimation [%i] on [%s]\n", node->id(), node->getCustomOp()->getOpName()->c_str());

                        auto op = node->getCustomOp();
                        auto in = node->input()->at(0);

                        auto block = node->getContextPrototype();
                        std::vector<Nd4jLong*> inputShapes;
                        int *oldShape;
                        for (auto v: *node->input()) {
                            nd4j_debug("     inputs for estimation are are: %i:%i\n", v.first, v.second);
                            if (v.first < 0) {
                                inputShapes.push_back(_variableSpace->getVariable(v.first)->getNDArray()->getShapeInfo());
                            } else {
                                inputShapes.push_back(shapesMap.at(v));
                            }
                        }

                        Context<T> ctx(block, _variableSpace);

                        ShapeList inSha(inputShapes);
                        auto outSha = op->calculateOutputShape(&inSha, ctx);

                        int cnt = 0;
                        for (auto newShape: *outSha->asVector()) {
                            std::pair<int, int> pairAddr(node->id(), cnt++);
                            std::pair<std::pair<int, int>, Nd4jLong*> pairShape(pairAddr, newShape);

                            shapesMap.insert(pairShape);

                            if (!block->isInplace() && !node->isInplace())
                                result += shape::length(newShape) * sizeof(T);

                            shapes.push_back(newShape);
                        }

                        delete outSha;
                    } else if (node->getOpClass() == OpClass_TRANSFORM) {
                        auto vec = node->input();

                        auto in = node->input()->at(0);
                        if (in.first < 0) {

                            auto x = _variableSpace->getVariable(in);
                            auto z = _variableSpace->getVariable(node->id());

                            auto newShape = new Nd4jLong[shape::shapeInfoLength(x->getNDArray()->getShapeInfo())];
                            memcpy(newShape, x->getNDArray()->getShapeInfo(), shape::shapeInfoByteLength(x->getNDArray()->getShapeInfo()));

                            std::pair<int, int> pairAddr(node->id(), 0);
                            std::pair<std::pair<int, int>, Nd4jLong*> pairShape(pairAddr, newShape);

                            shapesMap.insert(pairShape);

                            if (!node->isInplace())
                                result += shape::length(newShape) * sizeof(T);

                            shapes.push_back(newShape);
                        } else {
                            auto prevShape = shapesMap.at(in);

                            auto newShape = new Nd4jLong[shape::shapeInfoLength(prevShape)];
                            memcpy(newShape, prevShape, shape::shapeInfoByteLength(prevShape));

                            std::pair<int, int> pairAddr(node->id(), 0);
                            std::pair<std::pair<int, int>, Nd4jLong*> pairShape(pairAddr, newShape);

                            shapesMap.insert(pairShape);

                            if (!node->isInplace())
                                result += shape::length(newShape) * sizeof(T);

                            shapes.push_back(newShape);
                        }

                    } else if (node->getOpClass() == OpClass_REDUCTION) {
                        Nd4jLong *newShape = nullptr;

                        // if that's scalar output - we don't give a fuck about previous node
                        if (node->getDimensions()->size() == 0 || (node->getDimensions()->size() == 1 && node->getDimensions()->at(0) == MAX_INT)) {
                            newShape = new Nd4jLong[8];

                            newShape[0] = 2;
                            newShape[1] = 1;
                            newShape[2] = 1;
                            newShape[3] = 1;
                            newShape[4] = 1;
                            newShape[5] = 0;
                            newShape[6] = 1;
                            newShape[7] = 99;

                        } else {
                            auto in = node->input()->at(0);

                            Nd4jLong *oldShape = nullptr;
                            // calculate tads here
                            if (in.first < 0) {
                                auto x = _variableSpace->getVariable(in)->getNDArray();

                                oldShape = x->getShapeInfo();
                            } else {

                                oldShape = shapesMap.at(in);
                            }

                            //shape::TAD tad(oldShape, node->getDimensions()->data(), node->getDimensions()->size());
                            Nd4jLong numTads = shape::tadLength(oldShape, node->getDimensions()->data(), node->getDimensions()->size());
                            auto shape = new Nd4jLong[2]{1, (int) numTads};
                            newShape = shape::shapeBuffer(2, shape);
                        }

                        std::pair<int, int> pairAddr(node->id(), 0);
                        std::pair<std::pair<int, int>, Nd4jLong*> pairShape(pairAddr, newShape);

                        shapesMap.insert(pairShape);

                        result += shape::length(newShape) * sizeof(T);

                        shapes.push_back(newShape);
                    } else if (node->getOpClass() == OpClass_MULTIPLICATOR) {
                        // can't be in non-special op
                    }


                    cntFD++;
                }
            }

            // this is the only place where we deallocate shapes.
            if (_variableSpace->workspace() == nullptr)
                for (auto v: shapes)
                    delete[] v;

            return result;
        }

        template <typename T>
        void Graph<T>::pushToOutputOnce(int id) {
            if (std::find(_output.begin(), _output.end(), id) == _output.end())
                _output.emplace_back(id);
        }

        template <typename T>
        void Graph<T>::addOutput(int id) {
            if (_configuration->_outputMode == OutputMode_EXPLICIT || _configuration->_outputMode == OutputMode_EXPLICIT_AND_IMPLICIT)
                pushToOutputOnce(id);
        }

        template <typename T>
        ExecutorConfiguration * Graph<T>::getExecutorConfiguration() {
            return _configuration;
        }

        template <typename T>
        std::vector<Variable<T> *> * Graph<T>::fetchOutputs() {
            auto res = new std::vector<Variable<T> *>();

            nd4j_debug("Graph output size: %i\n", _output.size());
            for (int e = 0; e < (int) _output.size(); e++) {
                nd4j_debug("Output node: %i\n", _output.at(e));
                res->push_back(_variableSpace->getVariable(_output.at(e)));
            }

            return res;
        }

        template <typename T>
        std::map<int, Node<T> *> * Graph<T>::getMapped() {
            return _mapped;
        }

        template <typename T>
        std::map<int, std::vector<Node<T> *> *>* Graph<T>::getOnion() {
            return _onion;
        }

        template <typename T>
        void Graph<T>::injectNode(Node<T> *node) {
            if (node->getLayer() < 0)
                throw std::runtime_error("Only nodes with non-negative layer defined can be inserted");

            std::pair<int, Node<T> *> pair(node->id(), node);
            if (_mapped->count(pair.first) > 0)
                return;

            nd4j_debug("Node_%i mapped to layer_%i\n", node->id(), node->getLayer());


            _onion->at(node->getLayer())->push_back(node);
            _mapped->insert(pair);

            //_unmapped.erase(node->id());
        }

        template <typename T>
        void Graph<T>::expandOnion(int newLayer) {
            if (_onion->count(newLayer) > 0)
                return;

            std::vector<Node<T> *> *rootList = new std::vector<Node<T> *>();
            std::pair<int, std::vector<Node<T> *>*> pair(newLayer, rootList);
            _onion->insert(pair);
        }

        template <typename T>
        VariableSpace<T> * Graph<T>::getVariableSpace() {
            return _variableSpace;
        }

        template <typename T>
        Graph<T>::~Graph() {
            for (auto &v: *_mapped)
                delete v.second;

            for (auto &v: _unmapped)
                delete v.second;

            for (auto &v: *_onion)
                delete v.second;


            for (auto v: _scopes)
                delete v;

            delete _mapped;
            delete _nodes;
            delete _variableSpace;
            delete _onion;
            delete _configuration;


            // delete _onion content here
        }

        template <typename T>
        void Graph<T>::addNode(Node<T> *node) {
            _built.store(false);

            if (node->opType() == OpType_LOGIC) {
                // nd4j_debug("Adding LogicOp [%i]\n", node->opNum());
                // SCOPE
                if (node->opNum() == 10) {
                    auto scope = new Scope<T>(node->id(), node->getName() != nullptr ? node->getName()->c_str() : "");
                    _mappedScopes[node->id()] = scope;
                    _scopes.push_back(scope);
                }
            }

            auto cname = node->getName() == nullptr ? nullptr : node->getName()->c_str();
            auto nodeState = new Variable<T>(nullptr, cname, node->id());
            if (node->getName() != nullptr)
                nodeState->setName(node->getName());


            if (node->isInplace());
                    nodeState->markRemovable(false);

            _handles.push_back(node);


            _nodes->emplace_back(node->id());

            // storing node state now
            _variableSpace->putVariable(node->id(), nodeState);

            // here we're filling our blocks with future variables
            if (node->opType() == OpType_LOGIC && node->opNum() == 0) {
                // filling while
                int inputs = node->input()->size();
                for (int e = 0; e < inputs - 2; e++){
                    auto deepVar = new Variable<T>(nullptr, nullptr, node->id(), e);

                    std::pair<int,int> id(node->id(), e);
                    _variableSpace->putVariable(id, deepVar);
                }

            } else if (node->hasCustomOp()) {
                // custom ops require Block inside. but we'll set it inside buildGraph

                // TODO: we want to change this, to make blocks thread-local/session-local
                ContextPrototype<T>* block = nullptr;

                if (!node->hasBlockAttached()) {
                    block = new ContextPrototype<T>(node->id());
                    node->setContextPrototype(block);
                } else
                    block = node->getContextPrototype();


                if (!block->hasVariablesFilled()) {

                    for (uint32_t e = 0; e < node->input()->size(); e++) {
                        auto p = node->input()->at(e);

                        block->pickInput(p);
                    }
                }

                // and might have > 1 output
                if (node->getCustomOp()->getOpDescriptor()->getNumberOfOutputs() > 1) {
                    for (int e = 1; e < node->getCustomOp()->getOpDescriptor()->getNumberOfOutputs(); e++) {
                        auto deepVar = new Variable<T>(nullptr, nullptr, node->id());
                        //deepVar->setId(node->id());
                        deepVar->setId(node->id(), e);
                        if (node->isInplace())
                            deepVar->markRemovable(false);

                        std::pair<int,int> id(node->id(), e);
                        _variableSpace->putVariable(id, deepVar);
                    }
                } else {
                    // we need to check, if we should propagate output of this variable somewhere
                    for (int e = 0; e < node->output()->size(); e++) {
                        auto out = node->output()->at(e);
                        if (out.first < 0) {
                            nd4j_debug("Node [%i] will be propagating its output to Variable [%i]\n", node->id(), out.first);
                            auto extVar = _variableSpace->getVariable(out);
                            if (extVar->hasNDArray()) {
                                nodeState->setNDArray(extVar->getNDArray());
                                nodeState->markRemovable(false);
                            }
                        }
                    }
                }


                // adjust possible externals
                /*
                for (int e = 0; e < node->output()->size(); e++) {
                    auto out = node->output()->at(e);
                    std::pair<int, int> pair(node->id(), e);
                    if (out < 0) {
                        auto locVar = _variableSpace->getVariable(pair);
                        locVar->markRemovable(false);
                        auto extVar = _variableSpace->getVariable(out);
                        locVar->setNDArray(extVar->getNDArray());
                    }
                }
                */
            }

            // we're saving only ops that have internal outpus here
            if (_configuration->_outputMode == OutputMode_VARIABLE_SPACE)
                if (node->hasInternalOutputs())
                    pushToOutputOnce(node->id());

            // if outputs are undefined, we have to auto-create variable
            if (node->output()->size() == 0 || (node->output()->size() == 1 && node->output()->at(0).first == 0)){
                Variable<T>* var;
                if (!_variableSpace->hasVariable(node->id())) {
                    var = new Variable<T>();
                } else {
                    var = _variableSpace->getVariable(node->id());
                }
                // nd4j_logger("Adding auto output variable; Output size: %i\n", node->output()->size());

                var->setId(node->id());
                var->setName(node->getName());
                _variableSpace->putOutputVariable(var);
                //node->pickExternalOutput(var->id());

                // we're pushing this variable to output
                /*
                if (_configuration->_outputMode == OutputMode_IMPLICIT ||
                    _configuration->_outputMode == OutputMode_EXPLICIT_AND_IMPLICIT ||
                    _configuration->_outputMode == OutputMode_VARIABLE_SPACE)
                    pushToOutputOnce(var->id());
*/
                this->_autos.push_back(var->id());
//                assert(node->hasExternalOutputs());
//        }
            } else if (node->hasExternalOutputs()) {
                // TODO: we might want this behavior configurable!
                nd4j_logger("Adding specific output variable: Outputs: %i; HasInternal: %i;\n", node->output()->size(), node->hasInternalOutputs());

                // we're pushing this node to output only
                if ((!node->hasInternalOutputs() && (_configuration->_outputMode == OutputMode_IMPLICIT || _configuration->_outputMode == OutputMode_EXPLICIT_AND_IMPLICIT)) ) {
                    for (int e = 0;  e < (int) node->output()->size(); e++) {
                        if (node->output()->at(e).first < 0)
                            pushToOutputOnce(node->output()->at(e).first);
                    }

                    nd4j_logger("Loop finished: %i outputs now\n", this->_output.size());
                }
            }

            // ops that are tied to specific scope are never placed into the structure.
            if (node->isScoped()) {
                if (_mappedScopes.count(node->scopeId()) < 1) {
                    nd4j_printf("Requested scope [%i/%s] wasn't created yet\n", node->scopeId(), node->scopeName()->c_str());
                    throw std::invalid_argument("Unknown scope requested");
                }

                Scope<T>* scope = _mappedScopes.at(node->scopeId());
                scope->push_back(node);

                return;
            }

            std::pair<int, Node<T> *> pair(node->id(), node);
            // nd4j_debug("Adding node_%i\n", node->id());
            // if model has only external variables as input - it goes to first layer, no matter what.
            if (node->hasExternalInputs() && !node->hasInternalInputs()) {
                node->setLayer(0);

                injectNode(node);

                // nd4j_logger("A Node_%i mapped to layer_%i; Output: %i;\n", node->id(), node->getLayer(), node->output()->at(0));
                
                return;
            } else {
                // in some cases we're able to put stuff immediately
                if (node->hasInternalInputs() && !node->hasExternalInputs() && node->input()->size() == 1) {

                    bool automapAllowed = true;
                    for (int e = 0; e < node->input()->size(); e++) {
                        auto cInput = node->input()->at(e);
                        int cFirst = cInput.first;
                        if (_mapped->count(cFirst) == 0) {
                            automapAllowed = false;
                            break;
                        }
                    }

                    // we only can put single input nodes, whose outputs were not mapped yet
                    //if (_mapped->count(node->input()->at(0).first) == 1 && (node->output()->size() == 0 || _mapped->count(node->output()->at(0).first) == 0)) {
                    if (automapAllowed) {
                        auto parent = _mapped->at(node->input()->at(0).first);
                        int nLayer = parent->getLayer() + 1;

                        expandOnion(nLayer);
                        node->setLayer(nLayer);
                        injectNode(node);

                        nd4j_logger("Node_%i mapped to layer_%i; Output: %i;\n", node->id(), node->getLayer(), node->output()->at(0));

                        return;
                    }
                } /*else if (node->opType() == OpType_LOGIC && node->opNum() == 10) {
                    // Scopes are just being added. They won't be executed on their own anyway.

                    int nLayer = _onion->size();                    

                    expandOnion(nLayer);
                    node->setLayer(nLayer);
                    injectNode(node);

                    nd4j_logger("Node_%i mapped Scope to layer_%i; Output: %i;\n", node->id(), node->getLayer(), node->output()->at(0));

                    return;
                }
*/
                // otherwise we're putting it to unmapped space for further sorting
                _unmapped.insert(pair);
                _unmappedMap.emplace_back(pair.first);
                nd4j_debug("adding: %i\n", pair.first);
            }
        }

        template <typename T>
        Nd4jStatus Graph<T>::buildGraph() {
            if (_built.load()) {
                prepareOutputs();
                return ND4J_STATUS_OK;
            }

            typename std::map<int, Node<T> *>::iterator fit;
            int cnts = 0;
            for ( fit = _unmapped.begin(); fit != _unmapped.end(); fit++ ) {
                int tK = fit->first;
                int tF = _unmappedMap.at(cnts++);
            }

            int buildCnt = 0;
            int buildLimit = _unmapped.size() * 2;
            while (_unmapped.size() > 0) {

                int sz = _unmapped.size();
                int sf = _unmappedMap.size();

                std::vector<int> queue;

                // first pass for unmapped nodes, we try to build tale here
                typename std::map<int, Node<T> *>::iterator it;
                int cntf = 0;
                nd4j_debug("-----------\n","");
                for ( it = _unmapped.begin(); it != _unmapped.end(); it++ ) {
                    auto node = it->second;
                    int tK = it->first;
                    int tF = _unmappedMap.at(cntf++);

                    //nd4j_printf("tK: %i; tF: %i\n", tK, tF);
                //for (int f = 0; f < sz; f++) {
                //    auto node = _unmapped.at(_unmappedMap.at(f));
                

                    // single-input node
                    if (node->input()->size() == 1) {

                        if (node->getName() == nullptr) {
                            nd4j_debug("Trying SI Node_%i\n", node->id());
                        } else {
                            nd4j_debug("Trying SI Node_%i:[%s]\n", node->id(), node->getName()->c_str());
                        }

                        int iNode = node->input()->at(0).first;
                        if (iNode < 0 || _variableSpace->hasExternalVariable(iNode)) {
                            // this is external variable, should we check, who's the last user of this variable?
                            int lastLayer = _onion->size();
                            expandOnion(lastLayer);

                            node->setLayer(lastLayer);
                            this->injectNode(node);

                            if (node->hasCustomOp()) {
                                ContextPrototype<T>* block = nullptr;

                                if (!node->hasBlockAttached()) {
                                    block = new ContextPrototype<T>(node->id());
                                    node->setContextPrototype(block);
                                } else
                                    block = node->getContextPrototype();


                                if (!block->hasVariablesFilled()) {

                                    for (int e = 0; e < node->input()->size(); e++) {
                                        auto p = node->input()->at(e);

                                        block->pickInput(p);
                                    }
                                }
                            }
                        } else if (_mapped->count(iNode) > 0) {
                            int maxLayer = _mapped->at(iNode)->getLayer() + 1;

                            node->setLayer(maxLayer);
                            if (_onion->count(maxLayer) == 0)
                                expandOnion(maxLayer);

                            this->injectNode(node);
                            queue.emplace_back(node->id());

                            if (node->hasCustomOp()) {
                                ContextPrototype<T>* block = nullptr;

                                if (!node->hasBlockAttached()) {
                                    block = new ContextPrototype<T>(node->id());
                                    node->setContextPrototype(block);
                                } else
                                    block = node->getContextPrototype();


                                if (!block->hasVariablesFilled()) {

                                    for (uint32_t e = 0; e < node->input()->size(); e++) {
                                        auto p = node->input()->at(e);

                                        block->pickInput(p);
                                    }
                                }
                            }
                        } else
                            continue;

                        //_unmapped.erase(node->id());
                        queue.emplace_back(node->id());
                    } else {
                        // multi-input node
                        if (node->getName() == nullptr) {
                            nd4j_debug("Trying MI Node_%i\n", node->id());
                        } else {
                            std::string np = *(node->getName());
                            nd4j_debug("Trying MI Node_%i:[%s]\n", node->id(), node->getName()->c_str());
                        }

                        int maxLayer = 0;
                        bool breaker = false;
                        for (unsigned int e = 0; e < node->input()->size(); e++) {
                            int nodeId = node->input()->at(e).first;

                            // if input node wasn't mapped yet - we'll have skip it in this round
                            if (_mapped->count(nodeId) == 1) {
                                auto iNode = _mapped->at(nodeId);

                                if (maxLayer < iNode->getLayer())
                                    maxLayer = iNode->getLayer();
                            } else 
                                if (node->opType() == OpType_LOGIC) {
                                    // just allow it?
                            } else // checking if that's static variable
                                if (nodeId > 0 && !_variableSpace->hasExternalVariable(nodeId)) {
                                    breaker = true;
                                    break;
                            }
                        }

                        if (breaker)
                            continue;

                        maxLayer++;
                        if (_onion->count(maxLayer) == 0)
                            expandOnion(maxLayer);

                        node->setLayer(maxLayer);
                        injectNode(node);
                        queue.emplace_back(node->id());

                        if (node->hasCustomOp()) {
                            ContextPrototype<T>* block = nullptr;

                            if (!node->hasBlockAttached()) {
                                block = new ContextPrototype<T>(node->id());
                                node->setContextPrototype(block);
                            } else
                                block = node->getContextPrototype();

                            if (!block->hasVariablesFilled()) {

                                for (uint32_t e = 0; e < node->input()->size(); e++) {
                                    auto p = node->input()->at(e);

                                    block->pickInput(p);
                                }
                            }
                        }
                    }
                }

                for (auto &v: queue)
                    _unmapped.erase(v);

                // second pass is mover, we'll be moving onion layers around here
                buildCnt++;
                if (buildCnt > buildLimit) {
                    nd4j_printf("Unable to build graph, probably unmapped nodes, or something: %i nodes left\n", _unmapped.size());
                    for (auto v: _unmapped) {
                        Node<T>* node = v.second;
                        nd4j_printf("Unmapped node: [%i]\n", node->id());
                    }

                    throw std::runtime_error("Unable to build graph");
                }
            }

            if (_unmapped.size() == 0)
                _built.store(true);

            prepareOutputs();

            return ND4J_STATUS_OK;
        }

        template <typename T>
        void Graph<T>::tagInplaceNodes() {
            // just calling, in case it wasn't built before
            if (!_built.load())
                this->buildGraph();

            bool buildRef = false;

            // checking for non-refenenced nodes
            for (auto v: *_nodes) {
                // skipping unmapped nodes
                if (_mapped->count(v) == 0)
                    continue;

                Node<T>* node = _mapped->at(v);
                if (node->totalReferences() == 0) {
                    buildRef = true;
                    break;
                }
            }

            if (buildRef) {
                for (auto v: *_nodes) {
                    // skipping unmapped nodes
                    if (_mapped->count(v) == 0)
                        continue;

                    Node<T>* node = _mapped->at(v);
                    auto inputs = node->input();
                    for (auto &t: *inputs) {
                        if (_mapped->count(t.first) == 0)
                            continue;

                        Node<T>* inode = _mapped->at(t.first);
                        inode->addReference(node->id());
                    }
                }
            }


            for (auto v: *_nodes) {
                // skipping unmapped nodes
                if (_mapped->count(v) == 0)
                    continue;

                Node<T>* node = _mapped->at(v);
                
                /**
                 * Node can be inplace if 2 requirements met:
                 * 1) current node allows in-place modification
                 * 2) source node has only 1 output
                 */                

                // checking for first requirement first
                if (node->getCustomOp() != nullptr)
                    if (node->getCustomOp()->getOpDescriptor()->allowsInplace()){
                        bool singleInput = true;
                        auto inputs = node->input();
                        for (auto &t: *inputs) {
                            if (_mapped->count(t.first) == 0)
                                continue;

                            Node<T>* inode = _mapped->at(t.first);

                            int output_size = inode->output()->size();

                            // checking for second requirement: inputNode must not be used as input anywhere
                            if (inode->totalReferences() > 1) {
                                singleInput = false;
                                break;
                            }
                        }

                        node->markInplace(singleInput);
                    }
            }
        }

        template <typename T>
        void Graph<T>::prepareOutputs() {
            // if we're dumping everything out there - we'll add external variables as well
            if (_configuration->_outputMode == OutputMode_VARIABLE_SPACE) {
                auto ext = _variableSpace->getExternalVariables();
                nd4j_verbose("Number of external variables: %i\n", ext->size())
                for (unsigned int e = 0; e < ext->size(); e++) {
                    pushToOutputOnce(ext->at(e)->id());
                }

                for (auto v: *_nodes) {
                    if (_mapped->count(v) == 0)
                        continue;

                    Node<T>* node = _mapped->at(v);

                    if (std::find(_output.begin(), _output.end(), node->id()) == _output.end())
                        _output.emplace_back(node->id());
                }

            } else if (_configuration->_outputMode == OutputMode_IMPLICIT) {
                // we're adding final nodes of the graph. those, not used as input anywhere
                nd4j_debug("Paring nodes... \n", "");

                if (Environment::getInstance()->isDebugAndVerbose()) {
                    // nd4j_printv("current _output", _output);
                }
                //_output.clear();

                for (auto v: *_nodes) {
                    // we should check for scopes, and other possible non-mapped stuff
                    if (_mapped->count(v) == 0)
                        continue;

                    Node<T>* node = _mapped->at(v);
                    if (node->name() != nullptr) {
                        nd4j_debug("Node %i; Name: [%s]\n", v, node->name()->c_str());
                    } else {
                        nd4j_debug("Node %i\n", v);
                    }

                    // updating outputs now
                    for (int e = 0; e < node->input()->size(); e++) {
                        auto inP = node->input()->at(e);

                        // input can be variable, or node. we only care about nodes
                        if (_mapped->count(inP.first) > 0) {
                            _mapped->at(inP.first)->pickOutputOnce(v);
                        }
                    }
                }
                // at this point all nodes have filled inputs/outputs, so we know nodes that do not have any connected outputs

                for (auto v: *_nodes) {
                    // we should check for scopes, and other possible non-mapped stuff
                    if (_mapped->count(v) == 0)
                        continue;

                    Node<T>* node = _mapped->at(v);

                    if (!node->hasInternalOutputs()) {
                        if (node->name() != nullptr) {
                            nd4j_debug("Output node found: [%i:<%s>]\n", v, node->name()->c_str());
                        } else {
                            nd4j_debug("Output node found: [%i]\n", v);
                        }

                        // FIXME: we don't really need search here.

                        if (std::find(_output.begin(), _output.end(), node->id()) == _output.end())
                            _output.emplace_back(node->id());
                    } else if (Environment::getInstance()->isDebugAndVerbose()) {
                        nd4j_debug("Node [%i:<%s>] has %i outputs announced:\n", v, node->name()->c_str(), node->output()->size());
                        printf("{");
                        for (auto s : *node->output()) {
                            printf("[%i:%i], ", s.first, s.second);
                        }
                        printf("}\n");
                        fflush(stdout);
                    }
                }
            }
        }

        template <typename T>
        Graph<T>::Graph(const FlatGraph *flatGraph, VariableSpace<T> *variableSpace) {
            this->_onion = new std::map<int, std::vector<Node<T> *> *>();
            this->_mapped = new std::map<int, Node<T> *> ();
            this->_nodes = new std::vector<int>();
            this->_variableSpace = variableSpace == nullptr ? new VariableSpace<T>() : variableSpace;
            bool trusted = flatGraph != nullptr;

            // creating RNG for this instance
#ifndef __CUDABLAS__
            // FIXME: we temporary skip this random init for CUDA
            NativeOps nativeOps;
            auto buffer = new uint64_t[1000000];
            auto rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 119, 1000000, (Nd4jPointer) buffer);
            this->_variableSpace->setRNG(rng);
#endif

            // add 0 layer
            this->expandOnion(0);

            // if there was no exec configuration in flatgraph - create default one
            if (flatGraph != nullptr && flatGraph->configuration() != nullptr) {
                _configuration = new ExecutorConfiguration(flatGraph->configuration());
            } else
                _configuration = new ExecutorConfiguration();

            // if memory reqs were set - initialize workspace
            if (_configuration->_footprintForward > 0) {
                nd4j::memory::Workspace *workspace = this->_variableSpace->workspace();
                workspace->expandBy(_configuration->_footprintForward);
            }

            // parsing variables here
            if (flatGraph != nullptr && flatGraph->variables() != nullptr && flatGraph->variables()->size() > 0) {
                for (unsigned int e = 0; e < flatGraph->variables()->size(); e++) {
                    auto flatVar = flatGraph->variables()->Get(e);

                    auto var = new Variable<T>(flatVar);
                    std::pair<int, int> pair(flatVar->id()->first(), flatVar->id()->second());
                    _variableSpace->putVariable(pair, var);

                    // if that's VariableSpace mode - we're pushing it to _output
                    if (_configuration->_outputMode == OutputMode_VARIABLE_SPACE)
                        pushToOutputOnce(var->id());

                }
            }

            // at this point we expect all variables are already registered
            // we're saving outputs only if explicit mode is set
            if (_configuration->_outputMode == OutputMode_EXPLICIT || _configuration->_outputMode == OutputMode_EXPLICIT_AND_IMPLICIT) {
                if (flatGraph != nullptr && flatGraph->outputs() != nullptr) {
                    for (unsigned int e = 0; e < flatGraph->outputs()->size(); e++) {
                        auto out = flatGraph->outputs()->Get(e);
                        std::pair<int, int> vp(out->first(), out->second());
                        if (!_variableSpace->hasVariable(vp)) {
                            nd4j_verbose("Non-existent variable requested: %i\n", out);
                            throw std::runtime_error("Non-existent variable requested");
                        }

                        // TODO: fix this .first
                        pushToOutputOnce(vp.first);
                    }
                }
            }

            // rolling through nodes
            if (flatGraph != nullptr && flatGraph->nodes() != nullptr && flatGraph->nodes()->size() > 0) {
                for (unsigned int e = 0; e < flatGraph->nodes()->size(); e++) {
                    auto node = flatGraph->nodes()->Get(e);

                    if (node->output() == nullptr || node->output()->size() == 0) {
                        nd4j_verbose("Orphan node detected: %i; AutoOutput to be considered\n", node->id());
                    }

                    nd4j_debug("Node name: [%s]\n", node->name()->c_str());
                    auto nnode = new Node<T>(node);
                    expandOnion(e);
                    nnode->setLayer(e);
                    this->addNode(nnode);
                    injectNode(nnode);
                    _unmapped.erase(nnode->id());
                }

                _built = true;
            }

            /**
             *  we allow in-place execution optimizations ONLY if 2 requirements met:
             *  1) this is FeedForward pass ONLY
             *  2) OPTIMIZED mode is set, so no intermediate results are going to be used
             */
            if (_configuration->_direction == Direction_FORWARD_ONLY && _configuration->_outputMode == OutputMode_OPTIMIZED)
                this->tagInplaceNodes();
        }


/**
 * This method returns number of root nodes in this graph
 * @return
 */
        template <typename T>
        int Graph<T>::rootNodes() {
            return this->_onion->at(0)->size();
        }

/**
 * This method returns total number of nodes in this graph
 * @return
 */
        template <typename T>
        int Graph<T>::totalNodes() {
            if (_built != true)
                buildGraph();

            return _mapped->size();
        }

        template <typename T>
        Nd4jStatus Graph<T>::validate() {
            if (!_built) {
                _mutexPreprocessing.lock();
                if (!_built) {
                    this->buildGraph();
                }
                _mutexPreprocessing.unlock();
            }

            if (_built != true)
                return ND4J_STATUS_BAD_GRAPH;

            return ND4J_STATUS_OK;
        };

        template <typename T>
        void Graph<T>::printOutNode(Node<T>* node) {
            nd4j_printf("%i. ", node->id());
            switch(node->opType()) {
                case OpType_CUSTOM: {
                    printf("%s; ", node->getCustomOp()->getOpName()->c_str());
                }
                    break;
                case OpType_LOGIC: {
                    printf("%s; ", EnumUtils::_LogicOpToString(node->opNum()));
                }
                    break;
                default: {
                    printf("%s:{%i}; ", EnumUtils::_OpTypeToString(node->opType()), (int) node->opNum());
                }
            }

            nd4j_printf("Inputs: [", "");
            //auto block = node->getBlock();
            for (int e = 0; e < node->input()->size(); e++) {

                auto in = node->input()->at(e);
                printf("{%i:%i}", in.first, in.second);
                if (e < node->input()->size() - 1)
                    nd4j_printf(", ", "");
            }
            nd4j_printf("]; \n", "");

//            printf("\n");
            fflush(stdout);
        }

        template <typename T>
        void Graph<T>::printOut() {
            buildGraph();
            nd4j_printf("\nPrinting out Graph...\n", "");
            
            int opCnt = 0;
            for (int l = 0; l < _onion->size(); l++) {
                int layerSize = _onion->count(l) == 1 ? _onion->at(l)->size() : 0;

                for (int n = 0; n < layerSize; n++) {
                    Node<T>* node = _onion->at(l)->at(n);

                    // we're skipping Scopes here
                    if (node->opType() == OpType_LOGIC && node->opNum() == 10)
                        continue;

                    printOutNode(node);
                }
            }


            nd4j_printf("\nPrinting out Scopes...\n","");
            for (int s = 0; s < _scopes.size(); s++) {
                Scope<T>* scope = _scopes.at(s);
                nd4j_printf("Scope %i:<%s>:\n", scope->id(), scope->name()->c_str());

                for (int n = 0; n < scope->nodes()->size(); n++) {
                    Node<T>* node = scope->nodes()->at(n);
                    printOutNode(node);
                }
            }

            fflush(stdout);
        }

        template <typename T>
        Nd4jStatus Graph<T>::validateNode(Node<T> *node) {
            // TODO: to be implemented
            return ND4J_STATUS_OK;
        }

        template <typename T>
        std::vector<OpDescriptor> Graph<T>::getOperations() {
            buildGraph();
            // nd4j_printf("\nRetrieving ops from the Graph and collect them...\n", "");
            std::vector<OpDescriptor> res;

            int opCnt = 0;
            for (int l = 0; l < _onion->size(); l++) {
                int layerSize = _onion->count(l) == 1 ? _onion->at(l)->size() : 0;

                for (int n = 0; n < layerSize; n++) {
                    Node<T>* node = _onion->at(l)->at(n);
                    if (node->name() == nullptr) continue;
                    OpDescriptor* pOpDescriptor = nullptr;
                    std::string opNameStr; //node->name();
                    int numInputs = 0;
                    int numOutputs = 0;

                    switch(node->opType()) {
                        case OpType_CUSTOM: {
                            pOpDescriptor = node->getCustomOp()->getOpDescriptor();
                        }
                        break;
                        case OpType_LOGIC: {
                            opNameStr = std::string(EnumUtils::_LogicOpToString(node->opNum()));
                        }
                        break;
                        default: {
                            opNameStr = std::string(EnumUtils::_OpTypeToString(node->opType()))+"{" + OpRegistrator::getInstance()->local_to_string<int>((int) node->opNum()) + "}";
                        }
                    }

                    if (node->input())
                        numInputs = node->input()->size();

                    if (node->output())
                        numOutputs = node->output()->size();
                    bool inplace = node->isInplace();

                    //OpDescriptor opDescriptor(numInputs, numOutputs, opNameStr, inplace);

                    // we're skipping Scopes here
                    if (node->opType() == OpType_LOGIC && node->opNum() == 10)
                        continue;
                    if (pOpDescriptor)
                        res.emplace_back(*pOpDescriptor);
                    else
                        res.emplace_back(OpDescriptor(numInputs, numOutputs, opNameStr, inplace));
                }
            }


            // nd4j_printf("\nCollecting out Scopes...\n","");
            for (int s = 0; s < _scopes.size(); s++) {
                Scope<T>* scope = _scopes.at(s);
                // nd4j_printf("Scope %i:<%s>:\n", scope->id(), scope->name()->c_str());

                for (int n = 0; n < scope->nodes()->size(); n++) {
                    Node<T>* node = scope->nodes()->at(n);
                    //printOutNode(node);
                    if (node->name() == nullptr) continue;
                    std::string opNameStr; //node->name();
                    OpDescriptor* pOpDescriptor = nullptr;
                    int numInputs = 0;
                    int numOutputs = 0;

                    switch(node->opType()) {
                        case OpType_CUSTOM: {
                            pOpDescriptor = node->getCustomOp()->getOpDescriptor();
                        }
                        break;
                        case OpType_LOGIC: {
                            opNameStr = std::string(EnumUtils::_LogicOpToString(node->opNum()));
                        }
                        break;
                        default: {
                            opNameStr = std::string(EnumUtils::_OpTypeToString(node->opType()))+"{" + OpRegistrator::getInstance()->local_to_string<int>((int) node->opNum()) + "}";
                        }
                    }

                    if (node->input())
                        numInputs = node->input()->size();

                    if (node->output())
                        numOutputs = node->output()->size();
                    bool inplace = node->isInplace();

                    if (pOpDescriptor != nullptr)
                        res.emplace_back(*pOpDescriptor);
                    else
                        res.emplace_back(OpDescriptor(numInputs, numOutputs, opNameStr, inplace));
                }
            }

            return res;
        }

        template <typename T>
        Scope<T> *Graph<T>::scopeById(int id) {
            if (_mappedScopes.count(id) == 0) {
                nd4j_printf("Requested Scope [%i] doesn't exist\n", id);
                throw std::runtime_error("Non-existent Scope was requested");
            }

            return _mappedScopes.at(id);
        }

        template <typename T>
        void Graph<T>::forgetVariableSpace() {
            _variableSpace = nullptr;
        }

        template <typename T>
        void Graph<T>::replaceState(VariableSpace<T> *state, ExecutorConfiguration *configuration) {
            delete _variableSpace;
            delete _configuration;

            _variableSpace = state;
            _configuration = configuration;
        }

        template <typename T>
        template <typename N>
        Graph<N>* Graph<T>::asT() {
            auto clone = new Graph<N>();

            clone->replaceState(this->_variableSpace->template asT<N>(), this->_configuration->clone());

            clone->template pullState<T>(this);

            return clone;
        }

        template <typename T>
        Graph<T>* Graph<T>::clone() {
            auto clone = new Graph<T>();

            clone->replaceState(this->_variableSpace->clone(), this->_configuration->clone());

            // transfer nodes
            for (int e = 0; e < _nodes->size(); e++)
                clone->_nodes->emplace_back(_nodes->at(e));

            // transfer outputs
            for (auto v: _output)
                clone->_output.emplace_back(v);

            // transfer autos
            for (auto v: _autos)
                clone->_autos.emplace_back(v);

            // transfer scopes
            for (auto &v: _mappedScopes) {
                auto scp = v.second->clone();
                clone->_mappedScopes[v.first] = scp;
                clone->_scopes.emplace_back(scp);
            }

            // transfer mapped nodes
            for (auto &v: *_onion) {
                auto vec = clone->_onion->count(v.first) > 0 ? clone->_onion->at(v.first) : new std::vector<Node<T>*>();


                // cloning actual nodes
                auto ovec = (*_onion)[v.first];
                for (auto x: *(ovec)) {
                    auto n = x->clone();
                    vec->emplace_back(n);
                    _handles.emplace_back(n);
                    (*clone->_mapped)[n->id()] = n;
                }

                if (clone->_onion->count(v.first) < 1)
                    (*clone->_onion)[v.first] = vec;
            }

            // transfer mapped nodes
            for (auto &v: _unmapped)
                clone->_unmapped[v.first] = v.second->clone();

            clone->_built.store(_built.load());

            return clone;
        }

        template <typename T>
        bool Graph<T>::hasNode(int id) {
            return _mapped->count(id) > 0;
        }

        template <typename T>
        Node<T>* Graph<T>::nodeById(int id) {
            return _mapped->at(id);
        }

        template <typename T>
        bool Graph<T>::hasScope(int id) {
            return _mappedScopes.count(id) > 0;
        }

        template <typename T>
        Nd4jLong Graph<T>::hashCode() {
            if (!_built.load())
                this->buildGraph();

            Nd4jLong hash = 0L;
            std::string localStamp;
            /**
             * Plan is:
             * 1) get shapes of existing variables
             * 2) get hash codes of individual ops
             * 3) optionally: get node names, if they are defined
             * 4) use long hash on that
             */

            // FIXME: remove once additional dtypes added
            if (sizeof(T) == 8) {
                localStamp += "DOUBLE";
            } else if (sizeof(T) == 4) {
                localStamp += "FLOAT";
            } else if (sizeof(T) == 2) {
                localStamp += "HALF";
            }

            int cnt = 0;
            /*
            if (_variableSpace != nullptr) {
                // loop over existing variables
                for (auto v: *(_variableSpace->handles())) {
                    if (v->hasNDArray()) {
                        NDArray<T> *arr = v->getNDArray();
                        if (arr != nullptr && arr->nonNull()) {
                            auto shape = arr->getShapeAsVector();
                            auto string = ShapeUtils<T>::shapeAsString(shape);
                            localStamp += string;
                        }
                    }
                }
            }
            */

            // loop over nodes in graph
            for (auto &v: *_mapped) {
                Node<T> *node = v.second;

                // optional part: node names
                if (!node->name()->empty()) {
                    localStamp += *(node->name());
                }
            }


            hash = HashHelper::getInstance()->getLongHash(localStamp);        

            nd4j_debug("Graph hash: %lld\n", hash);

            return hash;
        }

        template class ND4J_EXPORT Graph<float>;
        template class ND4J_EXPORT Graph<float16>;
        template class ND4J_EXPORT Graph<double>;
        template class ND4J_EXPORT Graph<int>;
        template class ND4J_EXPORT Graph<Nd4jLong>;


        template Graph<float>* Graph<float>::asT<float>();
        template Graph<float16>* Graph<float>::asT<float16>();
        template Graph<double>* Graph<float>::asT<double>();
        template Graph<int>* Graph<float>::asT<int>();
        template Graph<Nd4jLong>* Graph<float>::asT<Nd4jLong>();

        template Graph<float>* Graph<float16>::asT<float>();
        template Graph<float16>* Graph<float16>::asT<float16>();
        template Graph<double>* Graph<float16>::asT<double>();
        template Graph<int>* Graph<float16>::asT<int>();
        template Graph<Nd4jLong>* Graph<float16>::asT<Nd4jLong>();

        template Graph<float>* Graph<double>::asT<float>();
        template Graph<float16>* Graph<double>::asT<float16>();
        template Graph<double>* Graph<double>::asT<double>();
        template Graph<int>* Graph<double>::asT<int>();
        template Graph<Nd4jLong>* Graph<double>::asT<Nd4jLong>();

        template Graph<float>* Graph<int>::asT<float>();
        template Graph<float16>* Graph<int>::asT<float16>();
        template Graph<double>* Graph<int>::asT<double>();
        template Graph<int>* Graph<int>::asT<int>();
        template Graph<Nd4jLong>* Graph<int>::asT<Nd4jLong>();

        template Graph<float>* Graph<Nd4jLong>::asT<float>();
        template Graph<float16>* Graph<Nd4jLong>::asT<float16>();
        template Graph<double>* Graph<Nd4jLong>::asT<double>();
        template Graph<int>* Graph<Nd4jLong>::asT<int>();
        template Graph<Nd4jLong>* Graph<Nd4jLong>::asT<Nd4jLong>();
    }
}

