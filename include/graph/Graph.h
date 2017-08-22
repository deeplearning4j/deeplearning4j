//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GRAPH_H
#define LIBND4J_GRAPH_H

#include <list>
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
        public:
            Graph(const FlatGraph *flatGraph = nullptr);

            ~Graph();

            // method that'll print out graph
            Nd4jStatus validate();

            // this method will build structured representation of graph
            Nd4jStatus buildGraph();

            int rootNodes();
            int totalNodes();

            nd4j::graph::VariableSpace<T> *getVariableSpace();

            void addNode(nd4j::graph::Node<T> *node);

            std::map<int, std::vector<nd4j::graph::Node<T> *> *> *getOnion();
            std::map<int32_t, nd4j::graph::Node<T> *> *getMapped();

            std::vector<nd4j::graph::Variable<T> *> *fetchOutputs();
        };
    }
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

    // if outputs are undefined, we have to auto-create variable
    if (node->output()->size() == 0 || (node->output()->size() == 1 && node->output()->at(0) == 0)){
        nd4j_verbose("Adding auto output variable; Output size: %i\n", node->output()->size());
        auto var = new Variable<T>();
        _variableSpace->putOutputVariable(var);
        node->pickOutput(var->id());

        this->_output.push_back(var->id());
        this->_autos.push_back(var->id());
        assert(node->hasExternalOutputs());
    } else if (node->hasExternalOutputs()) {
        // TODO: we might want this behavior configurable!
        nd4j_verbose("Adding specific output variable: Outputs: %i\n", node->output()->size())

        for (int e = 0; e < node->output()->size(); e++) {
            if (node->output()->at(e) < 0)
                this->_output.push_back(node->output()->at(e));
        }
    }

    std::pair<int32_t, nd4j::graph::Node<T> *> pair(node->id(), node);
    // if model has only external variables as input - it goes to first layer, no matter what.
    if (node->hasExternalInputs() && !node->hasInternalInputs()) {
        node->setLayer(0);

        _onion->at(0)->push_back(node);
        _mapped->insert(pair);

        printf("A Node_%i mapped to layer_%i; Output: %i;\n", node->id(), node->getLayer(), node->output()->at(0));
        fflush(stdout);
    } else {
        // in some cases we're able to put stuff immediately
        if (node->hasInternalInputs() && !node->hasExternalInputs() && node->input()->size() == 1) {

            // we only can put single input nodes, whose outputs were not mapped yet
            if (_mapped->count(node->input()->at(0)) == 1 && (node->output()->size() == 0 || _mapped->count(node->output()->at(0)) == 0)) {
                auto parent = _mapped->at(node->input()->at(0));
                int nLayer = parent->getLayer() + 1;
                if (_onion->count(nLayer) != 1) {
                    expandOnion(nLayer);
                }

                node->setLayer(nLayer);
                _onion->at(nLayer)->push_back(node);
                _mapped->insert(pair);
                printf("B Node_%i mapped to layer_%i; Output: %i;\n", node->id(), node->getLayer(), node->output()->at(0));
                fflush(stdout);

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

                printf("Trying SI Node_%i\n", node->id());
                fflush(stdout);

                int iNode = node->input()->at(0);
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
                printf("Trying MI Node_%i\n", node->id());
                fflush(stdout);

                int maxLayer = 0;
                for (int e = 0; e < node->input()->size(); e++) {
                    int nodeId = node->input()->at(e);

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

                _unmapped.erase(node->id());
            }
        }


        // second pass is mover, we'll be moving onion layers around here
    }

    if (_unmapped.size() == 0)
        _built.store(true);
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
        for (int e = 0; e < flatGraph->variables()->size(); e++) {
            auto flatVar = flatGraph->variables()->Get(e);

            auto var = new Variable<T>(flatVar);
            nd4j_verbose("Registering variable: %i\n", var->id());
            _variableSpace->putVariable(flatVar->id(), var);
        }
    }

    // at this point we expect all variables are already registered
    if (flatGraph != nullptr && flatGraph->outputs() != nullptr) {
        for (int e = 0; e < flatGraph->outputs()->size(); e++) {
            auto out = flatGraph->outputs()->Get(e);
            if (!_variableSpace->hasVariable(out)) {
                nd4j_verbose("Non-existent variable requested: %i\n", out);
                throw "Non-existent variable requested";
            }

            _output.push_back(out);
        }
    }

    // rolling through nodes
    if (flatGraph != nullptr && flatGraph->nodes() != nullptr && flatGraph->nodes()->size() > 0) {

        // flag to be raised if there's nodes without output being set
        bool outputPassNeeded = false;

        for (int e = 0; e < flatGraph->nodes()->size(); e++) {
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
