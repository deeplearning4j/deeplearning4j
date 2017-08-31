//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GRAPH_H
#define LIBND4J_GRAPH_H

#include <list>
#include <algorithm>
#include <map>
//#include <NDArray.h>
#include <graph/Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/generated/config_generated.h>
#include <ExecutorConfiguration.h>

/*
template<typename K, typename V>
using MapIterator = typename std::map<K,V>::iterator;
*/

namespace nd4j {
    namespace graph {

        template <typename T>
        class Graph {
        protected:
            ExecutorConfiguration *_configuration;
            VariableSpace<T> *_variableSpace;

            // vector holds ID's of top nodes only
            std::vector<int32_t > *_nodes;
            std::map<int32_t, nd4j::graph::Node<T> *> *_mapped;

            std::map<int, std::vector<nd4j::graph::Node<T> *> *> *_onion;
            std::map<int32_t, nd4j::graph::Node<T> *> _unmapped;

            std::mutex _mutexPreprocessing;
            std::atomic<bool> _built;


            std::vector<int32_t> _output;
            std::vector<int32_t> _autos;

            Nd4jStatus validateNode(nd4j::graph::Node<T> *node);

            void expandOnion(int newLayer);

            void injectNode(nd4j::graph::Node<T> *node);

            void pushToOutputOnce(int32_t id);
        public:
            Graph(const FlatGraph *flatGraph = nullptr);

            ~Graph();

            // method that'll print out graph
            Nd4jStatus validate();

            // this method will build structured representation of graph
            Nd4jStatus buildGraph();

            int rootNodes();
            int totalNodes();

            /**
             * This method returns pointer to thread_local VariableSpace
             * @return
             */
            nd4j::graph::VariableSpace<T> *getVariableSpace();

            /**
             * This method adds given node to the graph
             *
             * @param node
             */
            void addNode(nd4j::graph::Node<T> *node);

            /**
             * This method returns layered representation of the graph
             *
             * @return
             */
            std::map<int, std::vector<nd4j::graph::Node<T> *> *> *getOnion();

            /**
             * This method returns map of all nodes of the graph
             * @return
             */
            std::map<int32_t, nd4j::graph::Node<T> *> *getMapped();

            /**
             * This method returns outputs of of this graph
             * @return
             */
            std::vector<nd4j::graph::Variable<T> *> *fetchOutputs();

            /**
             * This method returns pointer to ExecutorConfiguration
             *
             * @return
             */
            ExecutorConfiguration *getExecutorConfiguration();

            /**
             * This method adds specified node (by ID) to de
             * @param id
             */
            void addOutput(int32_t id);
        };
    }
}

template <typename T>
void nd4j::graph::Graph<T>::pushToOutputOnce(int32_t id) {
    if (std::find(_output.begin(), _output.end(), id) == _output.end())
        _output.push_back(id);
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

    for (int e = 0; e < _output.size(); e++) {
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
    delete _mapped;
    delete _nodes;
    delete _variableSpace;
    delete _onion;

    // delete _onion content here
}

template <typename T>
void nd4j::graph::Graph<T>::addNode(nd4j::graph::Node<T> *node) {
    _built.store(false);

    auto nodeState = new Variable<T>();
    if (node->getName() != nullptr)
        nodeState->setName(node->getName());

    // storing node state now
    _variableSpace->putVariable(node->id(), nodeState);

    if (node->hasCustomOp()) {
        // custom ops require Block inside. but we'll set it inside buildGraph

        // and might have > 1 output
        if (node->getCustomOp()->getOpDescriptor()->getNumberOfOutputs() > 1) {
            for (int e = 0; e < node->getCustomOp()->getOpDescriptor()->getNumberOfOutputs(); e++) {
                auto deepVar = new Variable<T>();
                //deepVar->setId(node->id(), e);

                std::pair<int,int> id(node->id(), e);
                _variableSpace->putVariable(id, deepVar);
            }
        }
    }

    // we're saving only ops that have internal outpus here
    if (_configuration->_outputMode == OutputMode_VARIABLE_SPACE)
        if (node->hasInternalOutputs())
            pushToOutputOnce(node->id());

    // if outputs are undefined, we have to auto-create variable
    if (node->output()->size() == 0 || (node->output()->size() == 1 && node->output()->at(0) == 0)){
        nd4j_verbose("Adding auto output variable; Output size: %i\n", node->output()->size());
        auto var = new Variable<T>();
        var->setId(node->id());
        var->setName(node->getName());
        _variableSpace->putOutputVariable(var);
        node->pickExternalOutput(var->id());

        // we're pushing this variable to output
        if (_configuration->_outputMode == OutputMode_IMPLICIT || _configuration->_outputMode == OutputMode_EXPLICIT_AND_IMPLICIT || _configuration->_outputMode == OutputMode_VARIABLE_SPACE)
            pushToOutputOnce(var->id());

        this->_autos.push_back(var->id());
        assert(node->hasExternalOutputs());
    } else if (node->hasExternalOutputs()) {
        // TODO: we might want this behavior configurable!
        nd4j_verbose("Adding specific output variable: Outputs: %i; HasInternal: %i;\n", node->output()->size(), node->hasInternalOutputs())

        // we're pushing this node to output only
        if ((!node->hasInternalOutputs() && (_configuration->_outputMode == OutputMode_IMPLICIT || _configuration->_outputMode == OutputMode_EXPLICIT_AND_IMPLICIT)) ) {
            for (int e = 0; e < node->output()->size(); e++) {
                if (node->output()->at(e) < 0)
                    pushToOutputOnce(node->output()->at(e));
            }

            nd4j_printf("Loop finished: %i outputs now\n", this->_output.size());
        }
    }

    std::pair<int32_t, nd4j::graph::Node<T> *> pair(node->id(), node);
    // if model has only external variables as input - it goes to first layer, no matter what.
    if (node->hasExternalInputs() && !node->hasInternalInputs()) {
        node->setLayer(0);

        _onion->at(0)->push_back(node);
        _mapped->insert(pair);

        nd4j_verbose("A Node_%i mapped to layer_%i; Output: %i;\n", node->id(), node->getLayer(), node->output()->at(0));
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

                nd4j_verbose("B Node_%i mapped to layer_%i; Output: %i;\n", node->id(), node->getLayer(), node->output()->at(0));

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

                nd4j_verbose("Trying SI Node_%i\n", node->id());


                int iNode = node->input()->at(0).first;
                if (_mapped->count(iNode) > 0) {
                    int maxLayer = _mapped->at(iNode)->getLayer() + 1;

                    node->setLayer(maxLayer);
                    if (_onion->count(maxLayer) == 0)
                        expandOnion(maxLayer);

                    this->injectNode(node);
                } else
                    continue;

                _unmapped.erase(node->id());
            } else {
                // multi-input node
                nd4j_verbose("Trying MI Node_%i\n", node->id());

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
                    auto block = new Block<T>(node->id(), _variableSpace);
                    node->setBlock(block);

                    for (uint32_t e = 0; e < node->input()->size(); e++) {
                        auto var = _variableSpace->getVariable(node->input()->at(e));

                        block->getVariables().push_back(var);
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

        // flag to be raised if there's nodes without output being set
        bool outputPassNeeded = false;

        for (unsigned int e = 0; e < flatGraph->nodes()->size(); e++) {
            auto node = flatGraph->nodes()->Get(e);

            if (node->output() == nullptr || node->output()->size() == 0) {
                outputPassNeeded = true;
                nd4j_printf("Orphan node detected: %i; AutoOutput to be considered\n", node->id());
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
}

#endif //LIBND4J_GRAPH_H
