//
// @author raver119@gmail.com
//

#include <graph/Graph.h>

namespace nd4j {
    namespace graph {

        template <typename T>
        std::vector<nd4j::graph::Node<T>*>* nd4j::graph::Graph<T>::getAllNodes() {
            return &_handles;
        }

        template <typename T>
        std::vector<nd4j::graph::Variable<T>*>* nd4j::graph::Graph<T>::getPlaceholders() {
            return _variableSpace->getPlaceholders();
        }

        template <typename T>
        int nd4j::graph::Graph<T>::numberOfPlaceholders() {
            return _variableSpace->numberOfPlaceholders();
        };

        template <typename T>
        Nd4jIndex nd4j::graph::Graph<T>::estimateRequiredMemory() {

            Nd4jIndex result = 0L;
            Nd4jIndex lastStep = 0L;

            std::vector<int *> shapes;
            std::map<std::pair<int,int>, int*> shapesMap;

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

                        auto block = node->getBlock();
                        std::vector<int*> inputShapes;
                        int *oldShape;
                        for (auto v: *node->input()) {
                            nd4j_debug("     inputs for estimation are are: %i:%i\n", v.first, v.second);
                            if (v.first < 0) {
                                inputShapes.push_back(_variableSpace->getVariable(v.first)->getNDArray()->getShapeInfo());
                            } else {
                                inputShapes.push_back(shapesMap.at(v));
                                shape::printShapeInfoLinear(shapesMap.at(v));
                            }
                        }

                        ShapeList inSha(inputShapes);
                        auto outSha = op->calculateOutputShape(&inSha, *block);

                        int cnt = 0;
                        for (auto newShape: *outSha->asVector()) {
                            std::pair<int, int> pairAddr(node->id(), cnt++);
                            std::pair<std::pair<int, int>, int *> pairShape(pairAddr, newShape);

                            shapesMap.insert(pairShape);

                            if (!block->isInplace() && !node->isInplace())
                                result += shape::length(newShape) * sizeof(T);

                            shape::printShapeInfoLinear(newShape);

                            shapes.push_back(newShape);
                        }

                        delete outSha;
                    } else if (node->getOpClass() == OpClass_TRANSFORM) {
                        auto vec = node->input();

                        auto in = node->input()->at(0);
                        if (in.first < 0) {

                            auto x = _variableSpace->getVariable(in);
                            auto z = _variableSpace->getVariable(node->id());

                            int *newShape = new int[shape::shapeInfoLength(x->getNDArray()->getShapeInfo())];
                            memcpy(newShape, x->getNDArray()->getShapeInfo(), shape::shapeInfoByteLength(x->getNDArray()->getShapeInfo()));

                            std::pair<int, int> pairAddr(node->id(), 0);
                            std::pair<std::pair<int, int>, int *> pairShape(pairAddr, newShape);

                            shapesMap.insert(pairShape);

                            if (!node->isInplace())
                                result += shape::length(newShape) * sizeof(T);

                            shapes.push_back(newShape);
                        } else {
                            auto prevShape = shapesMap.at(in);

                            int *newShape = new int[shape::shapeInfoLength(prevShape)];
                            memcpy(newShape, prevShape, shape::shapeInfoByteLength(prevShape));

                            std::pair<int, int> pairAddr(node->id(), 0);
                            std::pair<std::pair<int, int>, int *> pairShape(pairAddr, newShape);

                            shapesMap.insert(pairShape);

                            if (!node->isInplace())
                                result += shape::length(newShape) * sizeof(T);

                            shapes.push_back(newShape);
                        }

                    } else if (node->getOpClass() == OpClass_REDUCTION) {
                        int *newShape = nullptr;

                        // if that's scalar output - we don't give a fuck about previous node
                        if (node->getDimensions()->size() == 0 || (node->getDimensions()->size() == 1 && node->getDimensions()->at(0) == MAX_INT)) {
                            newShape = new int[8];

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

                            int *oldShape = nullptr;
                            // calculate tads here
                            if (in.first < 0) {
                                auto x = _variableSpace->getVariable(in)->getNDArray();

                                oldShape = x->getShapeInfo();
                            } else {

                                oldShape = shapesMap.at(in);
                            }

                            //shape::TAD tad(oldShape, node->getDimensions()->data(), node->getDimensions()->size());
                            Nd4jIndex numTads = shape::tadLength(oldShape, node->getDimensions()->data(), node->getDimensions()->size());
                            int *shape = new int[2]{1, (int) numTads};
                            newShape = shape::shapeBuffer(2, shape);
                        }

                        std::pair<int, int> pairAddr(node->id(), 0);
                        std::pair<std::pair<int, int>, int *> pairShape(pairAddr, newShape);

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
            for (auto v: shapes)
                delete[] v;

            return result;
        }

        template <typename T>
        void nd4j::graph::Graph<T>::pushToOutputOnce(int32_t id) {
            if (std::find(_output.begin(), _output.end(), id) == _output.end())
                _output.emplace_back(id);
        }

        template <typename T>
        void nd4j::graph::Graph<T>::addOutput(int32_t id) {
            if (_configuration->_outputMode == OutputMode_EXPLICIT || _configuration->_outputMode == OutputMode_EXPLICIT_AND_IMPLICIT)
                pushToOutputOnce(id);
        }

        template <typename T>
        nd4j::graph::ExecutorConfiguration * nd4j::graph::Graph<T>::getExecutorConfiguration() {
            return _configuration;
        }

        template <typename T>
        std::vector<nd4j::graph::Variable<T> *> * nd4j::graph::Graph<T>::fetchOutputs() {
            auto res = new std::vector<nd4j::graph::Variable<T> *>();

            for (int e = 0; e < (int) _output.size(); e++) {
                res->push_back(_variableSpace->getVariable(_output.at(e)));
            }

            return res;
        }

        template <typename T>
        std::map<int32_t, nd4j::graph::Node<T> *> * nd4j::graph::Graph<T>::getMapped() {
            return _mapped;
        }

        template <typename T>
        std::map<int, std::vector<nd4j::graph::Node<T> *> *>* nd4j::graph::Graph<T>::getOnion() {
            return _onion;
        }

        template <typename T>
        void nd4j::graph::Graph<T>::injectNode(nd4j::graph::Node<T> *node) {
            if (node->getLayer() < 0)
                throw std::runtime_error("Only nodes with non-negative layer defined can be inserted");

            printf("Node_%i mapped to layer_%i\n", node->id(), node->getLayer());
            fflush(stdout);

            std::pair<int32_t, nd4j::graph::Node<T> *> pair(node->id(), node);
            _onion->at(node->getLayer())->push_back(node);
            _mapped->insert(pair);
        }

        template <typename T>
        void nd4j::graph::Graph<T>::expandOnion(int newLayer) {
            std::vector<nd4j::graph::Node<T> *> *rootList = new std::vector<nd4j::graph::Node<T> *>();
            std::pair<int, std::vector<nd4j::graph::Node<T> *>*> pair(newLayer, rootList);
            _onion->insert(pair);
        }

        template <typename T>
        nd4j::graph::VariableSpace<T> * nd4j::graph::Graph<T>::getVariableSpace() {
            return _variableSpace;
        }

        template <typename T>
        nd4j::graph::Graph<T>::~Graph() {
            for (auto v: *_mapped)
                delete v.second;

            for (auto v: _unmapped)
                delete v.second;

            for (auto v: *_onion) {
                delete v.second;
            }

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
        void nd4j::graph::Graph<T>::addNode(nd4j::graph::Node<T> *node) {
            _built.store(false);

            if (node->opType() == OpType_LOGIC) {
                nd4j_debug("Adding LogicOp [%i]\n", node->opNum());
                // SCOPE
                if (node->opNum() == 10) {
                    auto scope = new Scope<T>(node->id(), node->getName()->c_str());
                    _mappedScopes[node->id()] = scope;
                }
            }

            auto cname = node->getName() == nullptr ? nullptr : node->getName()->c_str();
            auto nodeState = new Variable<T>(nullptr, cname, node->id());
            if (node->getName() != nullptr)
                nodeState->setName(node->getName());


            if (node->isInplace());
                    nodeState->markRemovable(false);

            _handles.push_back(node);

            // storing node state now
            _variableSpace->putVariable(node->id(), nodeState);

            if (node->hasCustomOp()) {
                // custom ops require Block inside. but we'll set it inside buildGraph

                // TODO: we want to change this, to make blocks thread-local/session-local
                Block<T>* block = nullptr;

                if (!node->hasBlockAttached()) {
                    block = new Block<T>(node->id(), _variableSpace);
                    node->setBlock(block);
                } else
                    block = node->getBlock();


                if (!block->hasVariablesFilled()) {
                    block->setVariableSpace(_variableSpace);

                    for (uint32_t e = 0; e < node->input()->size(); e++) {
                        auto var = _variableSpace->getVariable(node->input()->at(e));

                        block->getVariables()->push_back(var);
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
                        if (out < 0) {
                            nd4j_debug("Node [%i] will be propagating its output to Variable [%i]\n", node->id(), out);
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
            if (node->output()->size() == 0 || (node->output()->size() == 1 && node->output()->at(0) == 0)){
                Variable<T>* var;
                if (!_variableSpace->hasVariable(node->id())) {
                    var = new Variable<T>();
                } else {
                    var = _variableSpace->getVariable(node->id());
                }
                nd4j_logger("Adding auto output variable; Output size: %i\n", node->output()->size());

                var->setId(node->id());
                var->setName(node->getName());
                _variableSpace->putOutputVariable(var);
                node->pickExternalOutput(var->id());

                // we're pushing this variable to output
                if (_configuration->_outputMode == OutputMode_IMPLICIT ||
                    _configuration->_outputMode == OutputMode_EXPLICIT_AND_IMPLICIT ||
                    _configuration->_outputMode == OutputMode_VARIABLE_SPACE)
                    pushToOutputOnce(var->id());

                this->_autos.push_back(var->id());
                assert(node->hasExternalOutputs());
//        }
            } else if (node->hasExternalOutputs()) {
                // TODO: we might want this behavior configurable!
                nd4j_logger("Adding specific output variable: Outputs: %i; HasInternal: %i;\n", node->output()->size(), node->hasInternalOutputs());

                // we're pushing this node to output only
                if ((!node->hasInternalOutputs() && (_configuration->_outputMode == OutputMode_IMPLICIT || _configuration->_outputMode == OutputMode_EXPLICIT_AND_IMPLICIT)) ) {
                    for (int e = 0;  e < (int) node->output()->size(); e++) {
                        if (node->output()->at(e) < 0)
                            pushToOutputOnce(node->output()->at(e));
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

            std::pair<int32_t, nd4j::graph::Node<T> *> pair(node->id(), node);
            // if model has only external variables as input - it goes to first layer, no matter what.
            if (node->hasExternalInputs() && !node->hasInternalInputs()) {
                node->setLayer(0);

                _onion->at(0)->push_back(node);
                _mapped->insert(pair);

                nd4j_logger("A Node_%i mapped to layer_%i; Output: %i;\n", node->id(), node->getLayer(), node->output()->at(0));
            } else {
                // in some cases we're able to put stuff immediately
                if (node->hasInternalInputs() && !node->hasExternalInputs() && node->input()->size() == 1) {

                    // we only can put single input nodes, whose outputs were not mapped yet
                    if (_mapped->count(node->input()->at(0).first) == 1 && (node->output()->size() == 0 || _mapped->count(node->output()->at(0)) == 0)) {
                        auto parent = _mapped->at(node->input()->at(0).first);
                        int nLayer = parent->getLayer() + 1;
                        if (_onion->count(nLayer) != 1) {
                            expandOnion(nLayer);
                        }

                        node->setLayer(nLayer);
                        _onion->at(nLayer)->push_back(node);
                        _mapped->insert(pair);

                        nd4j_logger("B Node_%i mapped to layer_%i; Output: %i;\n", node->id(), node->getLayer(), node->output()->at(0));

                        return;
                    }
                }

                // otherwise we're putting it to unmapped space for further sorting
                _unmapped.insert(pair);
            }
        }

        template <typename T>
        Nd4jStatus nd4j::graph::Graph<T>::buildGraph() {
            while (_unmapped.size() > 0) {

                // first pass for unmapped nodes, we try to build tale here
                typename std::map<int32_t, nd4j::graph::Node<T> *>::iterator it;
                for ( it = _unmapped.begin(); it != _unmapped.end(); it++ ) {
                    auto node = it->second;

                    // single-input node
                    if (node->input()->size() == 1) {

                        nd4j_logger("Trying SI Node_%i\n", node->id());


                        int iNode = node->input()->at(0).first;
                        if (_mapped->count(iNode) > 0) {
                            int maxLayer = _mapped->at(iNode)->getLayer() + 1;

                            node->setLayer(maxLayer);
                            if (_onion->count(maxLayer) == 0)
                                expandOnion(maxLayer);

                            this->injectNode(node);

                            if (node->hasCustomOp()) {
                                Block<T>* block = nullptr;

                                if (!node->hasBlockAttached()) {
                                    block = new Block<T>(node->id(), _variableSpace);
                                    node->setBlock(block);
                                } else
                                    block = node->getBlock();


                                if (!block->hasVariablesFilled()) {
                                    block->setVariableSpace(_variableSpace);

                                    for (uint32_t e = 0; e < node->input()->size(); e++) {
                                        auto var = _variableSpace->getVariable(node->input()->at(e));

                                        block->getVariables()->emplace_back(var);
                                    }
                                }
                            }
                        } else
                            continue;

                        _unmapped.erase(node->id());
                    } else {
                        // multi-input node
                        nd4j_logger("Trying MI Node_%i\n", node->id());

                        int maxLayer = 0;
                        for (unsigned int e = 0; e < node->input()->size(); e++) {
                            int nodeId = node->input()->at(e).first;

                            // if input node wasn't mapped yet - we'll have skip it in this round
                            if (_mapped->count(nodeId) == 1) {
                                auto iNode = _mapped->at(nodeId);

                                if (maxLayer < iNode->getLayer())
                                    maxLayer = iNode->getLayer();
                            } else
                                continue;
                        }

                        maxLayer++;
                        if (_onion->count(maxLayer) == 0)
                            expandOnion(maxLayer);

                        node->setLayer(maxLayer);
                        injectNode(node);

                        if (node->hasCustomOp()) {
                            Block<T>* block = nullptr;

                            if (!node->hasBlockAttached()) {
                                block = new Block<T>(node->id(), _variableSpace);
                                node->setBlock(block);
                            } else
                                block = node->getBlock();


                            if (!block->hasVariablesFilled()) {
                                block->setVariableSpace(_variableSpace);

                                for (uint32_t e = 0; e < node->input()->size(); e++) {
                                    auto var = _variableSpace->getVariable(node->input()->at(e));

                                    block->getVariables()->push_back(var);
                                }
                            }
                        }

                        _unmapped.erase(node->id());
                    }
                }

                // second pass is mover, we'll be moving onion layers around here
            }

            if (_unmapped.size() == 0)
                _built.store(true);

            // if we're dumping everything out there - we'll add external variables as well
            if (_configuration->_outputMode == OutputMode_VARIABLE_SPACE) {
                auto ext = _variableSpace->getExternalVariables();
                nd4j_verbose("Number of external variables: %i\n", ext->size())
                for (unsigned int e = 0; e < ext->size(); e++) {
                    pushToOutputOnce(ext->at(e)->id());
                }
            }

            return ND4J_STATUS_OK;
        }

        template <typename T>
        nd4j::graph::Graph<T>::Graph(const FlatGraph *flatGraph) {
            this->_onion = new std::map<int, std::vector<nd4j::graph::Node<T> *> *>();
            this->_mapped = new std::map<int32_t, nd4j::graph::Node<T> *> ();
            this->_nodes = new std::vector<int32_t>();
            this->_variableSpace = new VariableSpace<T>();

            // add 0 layer
            this->expandOnion(0);

            // if there was no exec configuration in flatgraph - create default one
            if (flatGraph != nullptr && flatGraph->configuration() != nullptr) {
                _configuration = new ExecutorConfiguration(flatGraph->configuration());
            } else
                _configuration = new ExecutorConfiguration();

            // parsing variables here
            if (flatGraph != nullptr && flatGraph->variables() != nullptr && flatGraph->variables()->size() > 0) {
                for (unsigned int e = 0; e < flatGraph->variables()->size(); e++) {
                    auto flatVar = flatGraph->variables()->Get(e);

                    auto var = new Variable<T>(flatVar);
                    nd4j_verbose("Registering variable: %i\n", var->id());
                    _variableSpace->putVariable(flatVar->id(), var);

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
                        if (!_variableSpace->hasVariable(out)) {
                            nd4j_verbose("Non-existent variable requested: %i\n", out);
                            throw "Non-existent variable requested";
                        }

                        pushToOutputOnce(out);
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

                    this->addNode(new Node<T>(node));
                }
            }
        }


/**
 * This method returns number of root nodes in this graph
 * @return
 */
        template <typename T>
        int nd4j::graph::Graph<T>::rootNodes() {
            return this->_onion->at(0)->size();
        }

/**
 * This method returns total number of nodes in this graph
 * @return
 */
        template <typename T>
        int nd4j::graph::Graph<T>::totalNodes() {
            if (_built != true)
                buildGraph();

            return _mapped->size();
        }

        template <typename T>
        Nd4jStatus nd4j::graph::Graph<T>::validate() {
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
        Nd4jStatus nd4j::graph::Graph<T>::validateNode(nd4j::graph::Node<T> *node) {
            // TODO: to be implemented
            return ND4J_STATUS_OK;
        }

        template <typename T>
        Scope<T> *Graph<T>::scopeById(int id) {
            if (_mappedScopes.count(id) == 0) {
                nd4j_printf("Requested Scope [%i] doesn't exist\n", id);
                throw "Non-existent Scope was requested";
            }

            return _mappedScopes.at(id);
        }

        template class ND4J_EXPORT Graph<float>;
        //template class ND4J_EXPORT Graph<float16>;
        //template class ND4J_EXPORT Graph<double>;
    }
}

